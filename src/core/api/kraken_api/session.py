"""
Module pour la gestion des sessions et de la persistance des données.
"""

from typing import Dict, Any, Optional
import json
import os
import logging
from datetime import datetime, timedelta
import asyncio
from pathlib import Path

from .exceptions import SessionError


class KrakenSession:
    """
    Gestionnaire de sessions pour l'API Kraken.
    """

    def __init__(self, session_dir: Optional[str] = None):
        """
        Initialise le gestionnaire de sessions.

        Args:
            session_dir: Répertoire pour stocker les sessions
        """
        self.logger = logging.getLogger(__name__ + '.KrakenSession')

        # Configuration du répertoire de session
        self.session_dir = session_dir or os.path.join(
            os.path.expanduser('~'), '.kraken_api')
        self.session_file = os.path.join(self.session_dir, 'session.json')

        # Créer le répertoire s'il n'existe pas
        Path(self.session_dir).mkdir(parents=True, exist_ok=True)

        # Charger la session existante
        self.session_data = self._load_session()

    def _load_session(self) -> Dict[str, Any]:
        """
        Charge les données de session.

        Returns:
            Données de session
        """
        try:
            if os.path.exists(self.session_file):
                with open(self.session_file, 'r') as f:
                    data = json.load(f)

                # Validation des données
                if not isinstance(data, dict):
                    raise ValueError(
                        "Les données de session doivent être un dictionnaire")

                # Nettoyage des données expirées
                now = datetime.now()
                cleaned_data = {}

                for key, value in data.items():
                    if isinstance(value, dict) and 'expiry' in value:
                        expiry = datetime.fromisoformat(value['expiry'])
                        if expiry > now:
                            cleaned_data[key] = value
                    else:
                        cleaned_data[key] = value

                return cleaned_data

            return {}

        except Exception as e:
            self.logger.warning(
                f"Erreur lors du chargement de la session: {str(e)}")
            return {}

    def _save_session(self) -> None:
        """
        Sauvegarde les données de session.
        """
        try:
            with open(self.session_file, 'w') as f:
                json.dump(self.session_data, f, indent=4, default=str)
        except Exception as e:
            self.logger.error(
                f"Erreur lors de la sauvegarde de la session: {str(e)}")
            raise SessionError(
                f"Erreur lors de la sauvegarde de la session: {str(e)}")

    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """
        Stocke une valeur dans la session avec une durée de vie optionnelle.

        Args:
            key: Clé de stockage
            value: Valeur à stocker
            ttl: Durée de vie en secondes (None pour non expirant)
        """
        if ttl is not None:
            expiry = datetime.now() + timedelta(seconds=ttl)
            self.session_data[key] = {
                'value': value,
                'expiry': expiry.isoformat()
            }
        else:
            self.session_data[key] = value

        self._save_session()

    def get(self, key: str, default: Optional[Any] = None) -> Any:
        """
        Récupère une valeur de la session.

        Args:
            key: Clé de stockage
            default: Valeur par défaut si la clé n'existe pas

        Returns:
            Valeur stockée ou valeur par défaut
        """
        if key not in self.session_data:
            return default

        value = self.session_data[key]

        if isinstance(value, dict) and 'expiry' in value:
            expiry = datetime.fromisoformat(value['expiry'])
            if expiry < datetime.now():
                self.logger.debug(f"Session expired for key: {key}")
                del self.session_data[key]
                self._save_session()
                return default

            return value['value']

        return value

    def delete(self, key: str) -> None:
        """
        Supprime une valeur de la session.

        Args:
            key: Clé à supprimer
        """
        if key in self.session_data:
            del self.session_data[key]
            self._save_session()

    def clear(self) -> None:
        """
        Supprime toutes les données de session.
        """
        self.session_data = {}
        self._save_session()

    def cleanup(self) -> None:
        """
        Supprime les entrées expirées.
        """
        now = datetime.now()
        cleaned_data = {}

        for key, value in self.session_data.items():
            if isinstance(value, dict) and 'expiry' in value:
                expiry = datetime.fromisoformat(value['expiry'])
                if expiry > now:
                    cleaned_data[key] = value
            else:
                cleaned_data[key] = value

        self.session_data = cleaned_data
        self._save_session()

    async def cleanup_periodic(self, interval: int = 3600) -> None:
        """
        Nettoie périodiquement les entrées expirées.

        Args:
            interval: Intervalle en secondes entre les nettoyages
        """
        while True:
            await asyncio.sleep(interval)
            self.cleanup()

    def get_session_stats(self) -> Dict[str, Any]:
        """
        Récupère les statistiques de session.

        Returns:
            Dictionnaire avec les statistiques
        """
        stats = {
            'total_items': len(
                self.session_data),
            'directory': self.session_dir,
            'file': self.session_file,
            'last_modified': datetime.fromtimestamp(
                os.path.getmtime(
                    self.session_file)).isoformat() if os.path.exists(
                self.session_file) else None}

        return stats

    def __contains__(self, key: str) -> bool:
        """Vérifie si une clé existe dans la session."""
        return key in self.session_data

    def __getitem__(self, key: str) -> Any:
        """Récupère une valeur de la session."""
        return self.get(key)

    def __setitem__(self, key: str, value: Any) -> None:
        """Stocke une valeur dans la session."""
        self.set(key, value)

    def __delitem__(self, key: str) -> None:
        """Supprime une valeur de la session."""
        self.delete(key)

    def __len__(self) -> int:
        """Renvoie le nombre d'éléments dans la session."""
        return len(self.session_data)

    def __iter__(self):
        """Itère sur les clés de la session."""
        return iter(self.session_data)

    def __repr__(self) -> str:
        """Représentation de la session."""
        return f"KrakenSession({len(self.session_data)} items)"
