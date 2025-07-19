"""
Module pour la gestion du cache dans l'API Kraken.
"""

from typing import Dict, Any, Optional
from datetime import datetime, timedelta
import logging


class KrakenCache:
    def __init__(self, ttl: int = 60):
        """
        Initialise le cache.

        Args:
            ttl: Durée de vie des entrées en secondes
        """
        self.cache = {}
        self.ttl = ttl
        self.logger = logging.getLogger(__name__ + '.KrakenCache')

    def get(self, key: str) -> Optional[Dict[str, Any]]:
        """
        Récupère une valeur du cache.

        Args:
            key: Clé du cache

        Returns:
            Valeur du cache ou None si non trouvée ou expirée
        """
        if key not in self.cache:
            return None

        entry = self.cache[key]

        # Vérification de l'expiration
        if datetime.now() > entry['expires']:
            self.logger.debug(f"Cache expired for key: {key}")
            del self.cache[key]
            return None

        self.logger.debug(f"Cache hit for key: {key}")
        return entry['value']

    def set(self, key: str, value: Dict[str, Any]) -> None:
        """
        Stocke une valeur dans le cache.

        Args:
            key: Clé du cache
            value: Valeur à stocker
        """
        self.cache[key] = {
            'value': value,
            'expires': datetime.now() + timedelta(seconds=self.ttl)
        }
        self.logger.debug(f"Cache stored for key: {key}")

    def clear(self) -> None:
        """Supprime toutes les entrées du cache."""
        self.cache.clear()
        self.logger.info("Cache cleared")

    def cleanup(self) -> None:
        """Supprime les entrées expirées."""
        now = datetime.now()
        expired_keys = [k for k, v in self.cache.items() if v['expires'] < now]

        for key in expired_keys:
            del self.cache[key]
            self.logger.debug(f"Expired cache entry removed: {key}")

    def stats(self) -> Dict[str, Any]:
        """
        Renvoie les statistiques du cache.

        Returns:
            Dictionnaire avec:
            - size: Nombre d'entrées
            - ttl: Durée de vie actuelle
            - next_cleanup: Prochaine purge prévue
        """
        return {'size': len(self.cache), 'ttl': self.ttl, 'next_cleanup': min(
            v['expires'] for v in self.cache.values()) if self.cache else None}
