import logging
from typing import Dict, Any
from src.core.config import Config
import json
from datetime import datetime
import asyncio
from src.cache.redis_cache import RedisCache

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class RedisMonitor:
    """Moniteur Redis pour suivre l'état et les performances."""

    def __init__(self, config: Config):
        """Initialise le moniteur Redis."""
        self.config = config
        self.redis_cache = RedisCache(config)
        self.redis = self.redis_cache.client  # Accès direct au client natif si besoin

        self.alert_thresholds = {
            'memory_usage': 80,  # % de mémoire utilisé
            'cache_hits': 90,   # % de hits dans le cache
            'latency': 100,     # ms
            'connections': 1000  # Nombre maximum de connexions
        }

    async def monitor(self):
        """Monitore Redis et envoie des alertes si nécessaire."""
        while True:
            try:
                # Récupérer les métriques Redis
                if self.redis is None:
                    self.redis = self.redis_cache.client
                info = self.redis.info() if self.redis else {}

                # Vérifier les métriques
                self._check_memory_usage(info)
                self._check_cache_hits(info)
                self._check_latency(info)
                self._check_connections(info)

                # Attendre avant la prochaine vérification
                interval = getattr(self.config, 'MONITOR_INTERVAL', 60)
                if not isinstance(interval, (int, float)):
                    interval = 60
                await asyncio.sleep(interval)

            except Exception as e:
                logger.error(f"Erreur lors du monitoring Redis: {str(e)}")
                await asyncio.sleep(60)

    def _check_memory_usage(self, info: Dict[str, Any]):
        """Vérifie l'utilisation de la mémoire."""
        try:
            used_memory = info.get('used_memory', 0)
            total_memory = info.get('total_system_memory', 1)
            memory_percent = (used_memory / total_memory) * 100

            if memory_percent > self.alert_thresholds['memory_usage']:
                self._send_alert(
                    'MEMORY_USAGE',
                    f"Utilisation de la mémoire à {memory_percent:.1f}%"
                )

        except Exception as e:
            logger.error(
                f"Erreur lors de la vérification de la mémoire: {str(e)}")

    def _check_cache_hits(self, info: Dict[str, Any]):
        """Vérifie le taux de hits dans le cache."""
        try:
            hits = info.get('keyspace_hits', 0)
            misses = info.get('keyspace_misses', 0)
            total = hits + misses

            if total > 0:
                hit_rate = (hits / total) * 100
                if hit_rate < self.alert_thresholds['cache_hits']:
                    self._send_alert(
                        'CACHE_HITS',
                        f"Taux de hits dans le cache à {hit_rate:.1f}%"
                    )

        except Exception as e:
            logger.error(f"Erreur lors de la vérification des hits: {str(e)}")

    def _check_latency(self, info: Dict[str, Any]):
        """Vérifie la latence."""
        try:
            latency = info.get('instantaneous_ops_per_sec', 0)
            if latency > self.alert_thresholds['latency']:
                self._send_alert(
                    'LATENCY',
                    f"Latence élevée: {latency}ms"
                )

        except Exception as e:
            logger.error(
                f"Erreur lors de la vérification de la latence: {str(e)}")

    def _check_connections(self, info: Dict[str, Any]):
        """Vérifie le nombre de connexions."""
        try:
            connected_clients = info.get('connected_clients', 0)
            if connected_clients > self.alert_thresholds['connections']:
                self._send_alert(
                    'CONNECTIONS',
                    f"Nombre de connexions élevé: {connected_clients}"
                )

        except Exception as e:
            logger.error(
                f"Erreur lors de la vérification des connexions: {str(e)}")

    def _send_alert(self, alert_type: str, message: str):
        """Envoie une alerte."""
        try:
            # Publier l'alerte sur le canal Redis
            if self.redis:
                self.redis.publish('redis:alerts', json.dumps({
                    'timestamp': datetime.now().isoformat(),
                    'type': alert_type,
                    'message': message
                }))

            # Enregistrer l'alerte
            self._log_alert(alert_type, message)

        except Exception as e:
            logger.error(f"Erreur lors de l'envoi de l'alerte: {str(e)}")

    def _log_alert(self, alert_type: str, message: str):
        """Enregistre une alerte dans les logs."""
        logger.warning(f"ALERT ({alert_type}): {message}")


def setup_redis_monitor(config: Config) -> RedisMonitor:
    """Configure et retourne un moniteur Redis."""
    return RedisMonitor(config)


async def start_redis_monitor(config: Config):
    """Démarre le moniteur Redis."""
    monitor = setup_redis_monitor(config)
    await monitor.monitor()
