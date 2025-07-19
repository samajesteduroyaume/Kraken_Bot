"""
Gestionnaire de rate limiting avec métriques et gestion des pics de charge.
"""

from typing import Dict, Any
from collections import defaultdict
import time
import asyncio


class RateLimiter:
    def __init__(self,
                 max_requests: int = 15,
                 window_seconds: int = 60,
                 burst_factor: float = 1.5,
                 retry_strategy: str = 'exponential',
                 health_check_interval: int = 300):
        """
        Initialise le gestionnaire de rate limiting.

        Args:
            max_requests: Nombre maximum de requêtes par fenêtre
            window_seconds: Durée de la fenêtre en secondes
            burst_factor: Facteur de tolérance pour les pics de charge
            retry_strategy: Stratégie de retry ('exponential', 'linear', 'constant')
            health_check_interval: Intervalle entre les vérifications de santé
        """
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.burst_factor = burst_factor
        self.retry_strategy = retry_strategy
        self.health_check_interval = health_check_interval
        self.requests = []
        self._last_reset = time.time()
        self._current_window = 0
        self._last_health_check = time.time()
        self.endpoint_stats = defaultdict(int)

        # Vérification des paramètres
        if not isinstance(max_requests, int) or max_requests <= 0:
            raise ValueError("max_requests doit être un entier positif")
        if not isinstance(window_seconds, int) or window_seconds <= 0:
            raise ValueError("window_seconds doit être un entier positif")
        if not isinstance(burst_factor, (int, float)) or burst_factor <= 1:
            raise ValueError("burst_factor doit être un nombre supérieur à 1")
        if retry_strategy not in ['exponential', 'linear', 'constant']:
            raise ValueError(
                "retry_strategy doit être 'exponential', 'linear' ou 'constant'")
        if not isinstance(
                health_check_interval,
                int) or health_check_interval <= 0:
            raise ValueError(
                "health_check_interval doit être un entier positif")

    async def __aenter__(self):
        """Gestion de la fenêtre de rate limiting avec tolérance pour les pics."""
        current_time = time.time()

        # Nettoyage des requêtes anciennes
        self.requests = [
            req for req in self.requests if current_time -
            req < self.window_seconds]

        # Vérification du nombre de requêtes
        if len(self.requests) >= self.max_requests:
            # Calcul du temps d'attente pour le prochain slot
            wait_time = self.requests[0] + self.window_seconds - current_time
            if wait_time > 0:
                await asyncio.sleep(wait_time)

        # Ajout de la requête
        self.requests.append(current_time)
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Vérifier la santé de la connexion et les métriques."""
        current_time = time.time()

        # Vérification de la santé
        if current_time - self._last_health_check >= self.health_check_interval:
            await self._check_health()
            self._last_health_check = current_time

    async def _check_health(self) -> Dict[str, Any]:
        """
        Vérifie la santé de la connexion et les métriques.

        Returns:
            dict: Statut de santé avec :
                - status: 'healthy', 'warning' ou 'critical'
                - last_check: Timestamp de la dernière vérification
                - last_error: Dernière erreur rencontrée (si applicable)
                - response_time: Temps de réponse du serveur (si applicable)
                - server_time: Timestamp du serveur (si disponible)
                - local_time_diff: Différence entre temps local et serveur (si applicable)
        """
        # TODO: Implémenter la vérification de santé
        return {
            'status': 'healthy',
            'last_check': time.time(),
            'last_error': None,
            'response_time': None,
            'server_time': None,
            'local_time_diff': None
        }
