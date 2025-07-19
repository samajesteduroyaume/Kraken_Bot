import logging
try:
    import redis
except ImportError:
    raise ImportError(
        "Le module 'redis' n'est pas installé. Installez-le avec 'pip install redis'.")
from typing import Any, Optional, Dict, List
from src.core.config import Config
import json

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class RedisCache:
    """Cache Redis pour le bot de trading."""

    def __init__(self, config: Config):
        """Initialise le cache Redis."""
        self.config = config
        self.client = None

        # Configuration Redis
        self.redis_config = getattr(config, 'redis_config', None)
        if self.redis_config is None:
            raise AttributeError(
                "La configuration Redis n'est pas définie dans l'objet Config. Ajoutez une propriété 'redis_config' ou adaptez la classe Config.")

        self._connect()

    def _connect(self):
        """Établir la connexion Redis."""
        if not self.client:
            if self.redis_config is None:
                raise AttributeError(
                    "La configuration Redis n'est pas définie dans l'objet Config. Ajoutez une propriété 'redis_config' ou adaptez la classe Config.")
            try:
                self.client = redis.Redis(
                    host=self.redis_config['host'],
                    port=self.redis_config['port'],
                    db=self.redis_config['db'],
                    password=self.redis_config.get('password')
                )
                self.client.ping()
            except redis.exceptions.AuthenticationError:
                self.client = redis.Redis(
                    host=self.redis_config['host'],
                    port=self.redis_config['port'],
                    db=self.redis_config['db']
                )
                self.client.ping()
            except redis.ConnectionError as e:
                logger.error(f"Impossible de se connecter à Redis: {str(e)}")
                self.client = None
                raise
            except Exception as e:
                logger.error(f"Erreur lors de la connexion à Redis: {str(e)}")
                self.client = None
                raise
            logger.info("Connexion à Redis établie avec succès")

    def _ensure_connected(self):
        if self.client is None:
            self._connect()
        if self.client is None:
            raise ConnectionError("Client Redis non initialisé.")

    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Stocke une valeur dans Redis avec une durée de vie."""
        try:
            self._ensure_connected()
            if self.redis_config is None:
                raise AttributeError(
                    "La configuration Redis n'est pas définie dans l'objet Config. Ajoutez une propriété 'redis_config' ou adaptez la classe Config.")
            if not isinstance(value, (str, bytes)):
                value = json.dumps(value)
            if ttl is None:
                ttl = self.redis_config['ttl']
            self.client.set(key, value, ex=ttl)
            return True
        except Exception as e:
            logger.error(f"Erreur lors du stockage de la clé {key}: {str(e)}")
            return False

    def get(self, key: str) -> Optional[Any]:
        """Récupère une valeur depuis Redis."""
        try:
            self._ensure_connected()
            value = self.client.get(key)
            if value is None:
                return None
            try:
                return json.loads(value)
            except (json.JSONDecodeError, TypeError):
                return value.decode(
                    'utf-8') if isinstance(value, bytes) else value
        except Exception as e:
            logger.error(
                f"Erreur lors de la récupération de la clé {key}: {str(e)}")
            return None

    def delete(self, key: str) -> bool:
        """Supprime une clé de Redis."""
        try:
            self._ensure_connected()
            self.client.delete(key)
            return True
        except Exception as e:
            logger.error(
                f"Erreur lors de la suppression de la clé {key}: {str(e)}")
            return False

    def exists(self, key: str) -> bool:
        """Vérifie si une clé existe dans Redis."""
        try:
            self._ensure_connected()
            return self.client.exists(key) == 1
        except Exception as e:
            logger.error(
                f"Erreur lors de la vérification de la clé {key}: {str(e)}")
            return False

    def set_market_data(self,
                        symbol: str,
                        data: Dict[str,
                                   Any],
                        ttl: Optional[int] = None) -> bool:
        key = f"market:data:{symbol}"
        return self.set(key, data, ttl)

    def get_market_data(self, symbol: str) -> Optional[Dict[str, Any]]:
        key = f"market:data:{symbol}"
        return self.get(key)

    def set_indicator(
            self,
            symbol: str,
            indicator: str,
            value: Any,
            ttl: Optional[int] = None) -> bool:
        key = f"indicators:{symbol}:{indicator}"
        return self.set(key, value, ttl)

    def get_indicator(self, symbol: str, indicator: str) -> Optional[Any]:
        key = f"indicators:{symbol}:{indicator}"
        return self.get(key)

    def set_position(self,
                     position_id: str,
                     data: Dict[str,
                                Any],
                     ttl: Optional[int] = None) -> bool:
        key = f"positions:{position_id}"
        return self.set(key, data, ttl)

    def get_position(self, position_id: str) -> Optional[Dict[str, Any]]:
        key = f"positions:{position_id}"
        return self.get(key)

    def set_config(
            self,
            key: str,
            value: Any,
            ttl: Optional[int] = None) -> bool:
        key = f"config:{key}"
        return self.set(key, value, ttl)

    def get_config(self, key: str) -> Optional[Any]:
        key = f"config:{key}"
        return self.get(key)

    def publish(self, channel: str, message: Any) -> bool:
        try:
            self._ensure_connected()
            if not isinstance(message, (str, bytes)):
                message = json.dumps(message)
            self.client.publish(channel, message)
            return True
        except Exception as e:
            logger.error(
                f"Erreur lors de la publication sur {channel}: {str(e)}")
            return False

    def subscribe(self, channels: List[str]):
        try:
            self._ensure_connected()
            pubsub = self.client.pubsub()
            pubsub.subscribe(channels)
            return pubsub
        except Exception as e:
            logger.error(f"Erreur lors de l'abonnement aux canaux: {str(e)}")
            raise


def setup_redis_cache(config: Config) -> RedisCache:
    """Configure et retourne une instance du cache Redis."""
    return RedisCache(config)


def is_redis_available(config: Config) -> bool:
    """Vérifie si Redis est disponible et configuré."""
    redis_conf = getattr(config, 'redis_config', None)
    if not redis_conf or not redis_conf.get('enabled', False):
        return False
    try:
        cache = RedisCache(config)
        cache._ensure_connected()
        return cache.client.ping() if cache.client else False
    except Exception as e:
        logger.error(f"Erreur lors de la vérification de Redis: {str(e)}")
        return False
