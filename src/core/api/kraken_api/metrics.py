"""
Module pour la surveillance et les métriques de l'API Kraken.
"""

from typing import Dict, Any
import time
import logging
from datetime import datetime
from collections import defaultdict


class KrakenMetrics:
    def __init__(self, client):
        self.client = client
        self.logger = logging.getLogger(__name__ + '.KrakenMetrics')

        # Métriques globales
        self.metrics = {
            'total_requests': 0,
            'total_success': 0,
            'total_errors': 0,
            'total_cache_hits': 0,
            'total_cache_misses': 0,
            'total_rate_limit_hits': 0,
            'total_retries': 0,
            'total_time': 0,
            'avg_time': 0,
            'requests_by_method': defaultdict(int),
            'errors_by_type': defaultdict(int),
            'response_times': [],
            'last_reset': time.time()
        }

        # Métriques par endpoint
        self.endpoint_stats = defaultdict(lambda: {
            'requests': 0,
            'success': 0,
            'errors': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'avg_time': 0,
            'last_error': None,
            'last_request_time': None,
            'last_response_time': None,
            'rate_limit': 0,
            'retry_count': 0,
            'errors_by_type': defaultdict(int)
        })

        # Métriques de rate limiting
        self.rate_limit_stats = {
            'total_hits': 0,
            'total_blocked': 0,
            'current_window': 0,
            'requests_in_window': 0,
            'last_reset': time.time()
        }

        # Métriques de cache
        self.cache_stats = {
            'size': 0,
            'hits': 0,
            'misses': 0,
            'hit_rate': 0,
            'expiry_count': 0,
            'avg_ttl': 0
        }

        # Métriques de websocket
        self.ws_stats = {
            'connections': 0,
            'disconnections': 0,
            'messages_received': 0,
            'messages_sent': 0,
            'errors': 0,
            'pings_sent': 0,
            'pings_received': 0,
            'latency': [],
            'subscriptions': defaultdict(int)
        }

    def record_request(
            self,
            method: str,
            endpoint: str,
            success: bool,
            duration: float) -> None:
        """
        Enregistre une requête dans les métriques.

        Args:
            method: Méthode HTTP
            endpoint: Point de terminaison
            success: Si la requête a réussi
            duration: Durée de la requête en secondes
        """
        self.metrics['total_requests'] += 1
        self.metrics['requests_by_method'][method] += 1
        self.metrics['total_time'] += duration
        self.metrics['avg_time'] = self.metrics['total_time'] / \
            self.metrics['total_requests']
        self.metrics['response_times'].append(duration)

        endpoint_stats = self.endpoint_stats[endpoint]
        endpoint_stats['requests'] += 1
        endpoint_stats['last_request_time'] = datetime.now()

        if success:
            self.metrics['total_success'] += 1
            endpoint_stats['success'] += 1
        else:
            self.metrics['total_errors'] += 1
            endpoint_stats['errors'] += 1

    def record_error(
            self,
            error_type: str,
            endpoint: str,
            error_message: str) -> None:
        """
        Enregistre une erreur dans les métriques.

        Args:
            error_type: Type d'erreur
            endpoint: Point de terminaison
            error_message: Message d'erreur
        """
        self.metrics['total_errors'] += 1
        self.metrics['errors_by_type'][error_type] += 1

        endpoint_stats = self.endpoint_stats[endpoint]
        endpoint_stats['errors'] += 1
        endpoint_stats['errors_by_type'][error_type] += 1
        endpoint_stats['last_error'] = error_message

    def record_cache_hit(self, endpoint: str) -> None:
        """
        Enregistre un hit de cache.

        Args:
            endpoint: Point de terminaison
        """
        self.metrics['total_cache_hits'] += 1
        self.cache_stats['hits'] += 1
        self.cache_stats['hit_rate'] = self.cache_stats['hits'] / \
            (self.cache_stats['hits'] + self.cache_stats['misses'])

        self.endpoint_stats[endpoint]['cache_hits'] += 1

    def record_cache_miss(self, endpoint: str) -> None:
        """
        Enregistre un miss de cache.

        Args:
            endpoint: Point de terminaison
        """
        self.metrics['total_cache_misses'] += 1
        self.cache_stats['misses'] += 1
        self.cache_stats['hit_rate'] = self.cache_stats['hits'] / \
            (self.cache_stats['hits'] + self.cache_stats['misses'])

        self.endpoint_stats[endpoint]['cache_misses'] += 1

    def record_rate_limit(self, blocked: bool) -> None:
        """
        Enregistre une utilisation du rate limiting.

        Args:
            blocked: Si la requête a été bloquée
        """
        self.rate_limit_stats['total_hits'] += 1
        self.rate_limit_stats['requests_in_window'] += 1

        if blocked:
            self.rate_limit_stats['total_blocked'] += 1
            self.metrics['total_rate_limit_hits'] += 1

    def record_ws_event(self, event_type: str, data: Any) -> None:
        """
        Enregistre un événement websocket.

        Args:
            event_type: Type d'événement
            data: Données de l'événement
        """
        if event_type == 'connect':
            self.ws_stats['connections'] += 1
        elif event_type == 'disconnect':
            self.ws_stats['disconnections'] += 1
        elif event_type == 'message':
            self.ws_stats['messages_received'] += 1
        elif event_type == 'send':
            self.ws_stats['messages_sent'] += 1
        elif event_type == 'ping':
            self.ws_stats['pings_sent'] += 1
        elif event_type == 'pong':
            self.ws_stats['pings_received'] += 1
            if isinstance(data, float):
                self.ws_stats['latency'].append(data)
        elif event_type == 'error':
            self.ws_stats['errors'] += 1
        elif event_type == 'subscribe':
            self.ws_stats['subscriptions'][data] += 1

    def get_metrics(self) -> Dict[str, Any]:
        """
        Récupère toutes les métriques.

        Returns:
            Dictionnaire avec toutes les métriques
        """
        return {
            'global': dict(self.metrics),
            'endpoints': dict(self.endpoint_stats),
            'rate_limit': dict(self.rate_limit_stats),
            'cache': dict(self.cache_stats),
            'websocket': dict(self.ws_stats)
        }

    def reset_metrics(self) -> None:
        """Réinitialise toutes les métriques."""
        self.metrics = {
            'total_requests': 0,
            'total_success': 0,
            'total_errors': 0,
            'total_cache_hits': 0,
            'total_cache_misses': 0,
            'total_rate_limit_hits': 0,
            'total_retries': 0,
            'total_time': 0,
            'avg_time': 0,
            'requests_by_method': defaultdict(int),
            'errors_by_type': defaultdict(int),
            'response_times': [],
            'last_reset': time.time()
        }

        self.endpoint_stats = defaultdict(lambda: {
            'requests': 0,
            'success': 0,
            'errors': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'avg_time': 0,
            'last_error': None,
            'last_request_time': None,
            'last_response_time': None,
            'rate_limit': 0,
            'retry_count': 0,
            'errors_by_type': defaultdict(int)
        })

        self.rate_limit_stats = {
            'total_hits': 0,
            'total_blocked': 0,
            'current_window': 0,
            'requests_in_window': 0,
            'last_reset': time.time()
        }

        self.cache_stats = {
            'size': 0,
            'hits': 0,
            'misses': 0,
            'hit_rate': 0,
            'expiry_count': 0,
            'avg_ttl': 0
        }

        self.ws_stats = {
            'connections': 0,
            'disconnections': 0,
            'messages_received': 0,
            'messages_sent': 0,
            'errors': 0,
            'pings_sent': 0,
            'pings_received': 0,
            'latency': [],
            'subscriptions': defaultdict(int)
        }

    def get_stats_summary(self) -> str:
        """
        Récupère un résumé des statistiques sous forme de chaîne.

        Returns:
            Résumé des statistiques
        """
        metrics = self.get_metrics()

        summary = (
            f"=== Statistiques Kraken API ===\n"
            f"Requêtes totales: {metrics['global']['total_requests']}\n"
            f"Succès: {metrics['global']['total_success']}\n"
            f"Erreurs: {metrics['global']['total_errors']}\n"
            f"Cache hits: {metrics['global']['total_cache_hits']}\n"
            f"Cache misses: {metrics['global']['total_cache_misses']}\n"
            f"Rate limit hits: {metrics['global']['total_rate_limit_hits']}\n"
            f"\n=== Rate Limiting ===\n"
            f"Total hits: {metrics['rate_limit']['total_hits']}\n"
            f"Blocked: {metrics['rate_limit']['total_blocked']}\n"
            f"\n=== Cache ===\n"
            f"Hit rate: {metrics['cache']['hit_rate']:.2%}\n"
            f"Size: {metrics['cache']['size']}\n"
            f"\n=== WebSocket ===\n"
            f"Connections: {metrics['websocket']['connections']}\n"
            f"Messages: {metrics['websocket']['messages_received']}\n"
            f"Errors: {metrics['websocket']['errors']}\n"
        )

        return summary
