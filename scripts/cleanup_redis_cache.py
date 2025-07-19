import os
import logging
import asyncio
import json
from datetime import datetime, timedelta
from typing import Dict, Any
from src.cache.redis_cache import RedisCache
from src.core.config import Config

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class RedisCacheCleaner:
    """Nettoyeur de cache Redis."""
    
    def __init__(self, config: Config):
        """Initialise le nettoyeur de cache."""
        self.config = config
        self.redis_cache = RedisCache(config)
        self.redis = self.redis_cache.client
        
        self.cleanup_patterns = {
            'market_data': 'market:data:*',
            'indicators': 'indicators:*',
            'positions': 'positions:*',
            'config': 'config:*'
        }
        
        self.cleanup_intervals = {
            'market_data': 3600,  # 1 heure
            'indicators': 7200,  # 2 heures
            'positions': 86400,  # 24 heures
            'config': 604800     # 7 jours
        }
        
    async def cleanup(self):
        """Nettoie le cache Redis périodiquement."""
        while True:
            try:
                # Nettoyer chaque type de données
                for data_type, pattern in self.cleanup_patterns.items():
                    await self._cleanup_data_type(data_type, pattern)
                    
                # Attendre avant la prochaine vérification
                await asyncio.sleep(3600)  # 1 heure
                
            except Exception as e:
                logger.error(f"Erreur lors du nettoyage du cache: {str(e)}")
                await asyncio.sleep(3600)
    
    async def _cleanup_data_type(self, data_type: str, pattern: str):
        """Nettoie un type spécifique de données."""
        try:
            # Récupérer les clés correspondant au pattern
            if self.redis is None:
                self.redis = self.redis_cache.client
            keys = self.redis.keys(pattern) if self.redis else []
            if not keys:
                return
                
            # Vérifier l'expiration de chaque clé
            current_time = datetime.now()
            for key in keys:
                ttl = self.redis.ttl(key) if self.redis else None
                if ttl == -1 and self.redis:
                    self.redis.delete(key)
                    logger.info(f"Supprimé {key} (pas de TTL)")
                    continue
                if ttl == -2:
                    continue
                # Vérifier si la clé est trop ancienne
                # On suppose que la valeur stockée est un JSON avec un champ 'timestamp' ou similaire
                value = self.redis.get(key) if self.redis else None
                if value is None:
                    continue
                try:
                    # S'assurer que value est bien un str ou bytes décodable
                    if isinstance(value, bytes):
                        value = value.decode('utf-8')
                    data = json.loads(value)
                    timestamp = data.get('timestamp')
                    if timestamp:
                        key_time = datetime.fromisoformat(timestamp)
                        if (current_time - key_time) > timedelta(seconds=self.cleanup_intervals[data_type]) and self.redis:
                            self.redis.delete(key)
                            logger.info(f"Supprimé {key} (trop ancien)")
                except Exception:
                    pass
        except Exception as e:
            logger.error(f"Erreur lors du nettoyage des données {data_type}: {str(e)}")
    
    def cleanup_now(self, data_type: str = None):
        """Nettoie immédiatement le cache."""
        try:
            import asyncio
            loop = asyncio.get_event_loop()
            if data_type:
                pattern = self.cleanup_patterns.get(data_type)
                if pattern:
                    loop.run_until_complete(self._cleanup_data_type(data_type, pattern))
                else:
                    logger.warning(f"Type de données {data_type} non reconnu")
            else:
                # Nettoyer tous les types
                for dt, pattern in self.cleanup_patterns.items():
                    loop.run_until_complete(self._cleanup_data_type(dt, pattern))
        except Exception as e:
            logger.error(f"Erreur lors du nettoyage immédiat: {str(e)}")

def setup_redis_cleaner(config: Config) -> RedisCacheCleaner:
    """Configure et retourne un nettoyeur de cache Redis."""
    return RedisCacheCleaner(config)

async def start_cache_cleanup(config: Config):
    """Démarre le nettoyage du cache Redis."""
    cleaner = setup_redis_cleaner(config)
    await cleaner.cleanup()

if __name__ == "__main__":
    # Charger la configuration
    config = Config()
    
    # Configurer le logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Démarrer le nettoyage
    cleaner = setup_redis_cleaner(config)
    asyncio.run(cleaner.cleanup())
