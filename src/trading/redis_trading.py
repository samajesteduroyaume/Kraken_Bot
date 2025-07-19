import logging
import asyncio
from typing import Dict, Any, Optional, List
from datetime import datetime
from src.cache.redis_cache import RedisCache
from src.core.api.kraken import KrakenAPI
from src.core.trading import TradingManager
from src.core.config import Config

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class RedisTradingManager(TradingManager):
    """Gestionnaire de trading utilisant Redis pour le cache."""

    def __init__(self, config: Config, loop: asyncio.AbstractEventLoop = None):
        """Initialise le gestionnaire de trading Redis."""
        # Créer un event loop si aucun n'est fourni
        self.loop = loop or asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)

        # Initialiser l'API Kraken avec la configuration de test
        api_config = {
            'credentials': {
                'api_key': 'test_key',
                'api_secret': 'test_secret'
            },
            'base_url': 'https://api.kraken.com',
            'version': '0',
            'timeout': config.api.timeout,
            'max_retries': 3,
            'retry_delay': 0.5,
            'cache_ttl': 60,
            'rate_limit': {
                'enabled': True,
                'window': 30,
                'limit': 50
            }
        }

        # Initialiser l'API Kraken avec le loop
        api = KrakenAPI(api_config, loop=self.loop)

        # Initialiser le trader parent
        super().__init__(
            api=api,
            config=config.trading
        )

        self.redis = RedisCache(config)
        self.cache_ttl = getattr(
            config, 'redis_config', {}).get(
            'ttl', 3600)  # 1 heure par défaut

    async def get_market_data(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Récupère les données de marché, en utilisant le cache Redis."""
        try:
            # Vérifier si les données sont dans le cache
            cached_data = self.redis.get_market_data(symbol)
            if cached_data:
                logger.debug(
                    f"Données de marché pour {symbol} trouvées dans le cache")
                return cached_data

            # Si pas dans le cache, récupérer depuis l'API
            api = KrakenAPI(self.config)
            data = await api.get_market_data(symbol)

            if data:
                # Mettre en cache les données
                self.redis.set_market_data(symbol, data, self.cache_ttl)
                logger.debug(f"Données de marché pour {symbol} mises en cache")

            return data

        except Exception as e:
            logger.error(
                f"Erreur lors de la récupération des données de marché: {str(e)}")
            return None

    async def get_indicator(
            self,
            symbol: str,
            indicator: str) -> Optional[Any]:
        """Récupère une valeur d'indicateur technique, en utilisant le cache Redis."""
        try:
            # Vérifier si l'indicateur est dans le cache
            cached_value = self.redis.get_indicator(symbol, indicator)
            if cached_value is not None:
                logger.debug(
                    f"Indicateur {indicator} pour {symbol} trouvé dans le cache")
                return cached_value

            # Si pas dans le cache, calculer
            value = await self.calculate_indicator(symbol, indicator)
            if value is not None:
                # Mettre en cache
                self.redis.set_indicator(
                    symbol, indicator, value, self.cache_ttl)
                logger.debug(
                    f"Indicateur {indicator} pour {symbol} mis en cache")

            return value

        except Exception as e:
            logger.error(
                f"Erreur lors de la récupération de l'indicateur {indicator}: {str(e)}")
            return None

    async def calculate_indicator(
            self,
            symbol: str,
            indicator: str) -> Optional[Any]:
        """Calcule une valeur d'indicateur technique."""
        try:
            # Récupérer les données de marché
            market_data = await self.get_market_data(symbol)
            if not market_data:
                return None

            # Calculer l'indicateur selon le type
            if indicator == 'RSI':
                return self._calculate_rsi(market_data['prices'])
            elif indicator == 'MACD':
                return self._calculate_macd(market_data['prices'])
            elif indicator == 'BB':
                return self._calculate_bollinger_bands(market_data['prices'])

            return None

        except Exception as e:
            logger.error(
                f"Erreur lors du calcul de l'indicateur {indicator}: {str(e)}")
            return None

    def _calculate_rsi(self, prices: List[float]) -> float:
        """Calcule l'RSI sur les dernières données."""
        period = 14
        if len(prices) < period:
            return 50.0  # Neutre si pas assez de données

        delta = [prices[i] - prices[i - 1] for i in range(1, len(prices))]
        gain = [d if d > 0 else 0 for d in delta[-period:]]
        loss = [-d if d < 0 else 0 for d in delta[-period:]]

        avg_gain = sum(gain) / period
        avg_loss = sum(loss) / period

        if avg_loss == 0:
            return 100.0

        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))

    def _calculate_macd(self, prices: List[float]) -> Dict[str, float]:
        """Calcule le MACD."""
        if len(prices) < 26:
            return {
                'macd': 0.0,
                'signal': 0.0,
                'histogram': 0.0
            }

        fast_ema = self._calculate_ema(prices, 12)
        slow_ema = self._calculate_ema(prices, 26)
        macd = fast_ema - slow_ema
        signal = self._calculate_ema([macd], 9)
        histogram = macd - signal

        return {
            'macd': macd,
            'signal': signal,
            'histogram': histogram
        }

    def _calculate_bollinger_bands(
            self, prices: List[float]) -> Dict[str, float]:
        """Calcule les bandes de Bollinger."""
        period = 20
        if len(prices) < period:
            return {
                'upper': 0.0,
                'middle': 0.0,
                'lower': 0.0
            }

        sma = sum(prices[-period:]) / period
        std_dev = (
            sum([(x - sma) ** 2 for x in prices[-period:]]) / period) ** 0.5

        return {
            'upper': sma + (2 * std_dev),
            'middle': sma,
            'lower': sma - (2 * std_dev)
        }

    def _calculate_ema(self, prices: List[float], period: int) -> float:
        """Calcule l'EMA."""
        if len(prices) < period:
            return sum(prices) / len(prices)

        multiplier = 2 / (period + 1)
        ema = sum(prices[-period:]) / period

        for price in prices[-period + 1:]:
            ema = (price - ema) * multiplier + ema

        return ema

    async def get_position(self, position_id: str) -> Optional[Dict[str, Any]]:
        """Récupère les informations sur une position."""
        try:
            position = self.redis.get_position(position_id)
            if position:
                return position

            # Si pas en cache, récupérer depuis la base de données
            position = await super().get_position(position_id)
            if position:
                self.redis.set_position(position_id, position, self.cache_ttl)

            return position

        except Exception as e:
            logger.error(
                f"Erreur lors de la récupération de la position {position_id}: {str(e)}")
            return None

    async def update_position(self, position_id: str,
                              data: Dict[str, Any]) -> bool:
        """Met à jour une position et met à jour le cache."""
        try:
            # Mettre à jour dans la base de données
            success = await super().update_position(position_id, data)
            if success:
                # Mettre à jour le cache
                self.redis.set_position(position_id, data, self.cache_ttl)

            return success

        except Exception as e:
            logger.error(
                f"Erreur lors de la mise à jour de la position {position_id}: {str(e)}")
            return False

    async def publish_trade_signal(self, symbol: str, signal: Dict[str, Any]):
        """Publie un signal de trading via Redis Pub/Sub."""
        try:
            # Publier le signal sur le canal de trading
            self.redis.publish('trading:signals', {
                'timestamp': datetime.now().isoformat(),
                'symbol': symbol,
                'signal': signal
            })

        except Exception as e:
            logger.error(f"Erreur lors de la publication du signal: {str(e)}")


def setup_redis_trading_manager(config: Dict[str, Any]) -> RedisTradingManager:
    """Configure et retourne un gestionnaire de trading avec Redis."""
    return RedisTradingManager(config)


class TradingSignalSubscriber:
    """Abonné aux signaux de trading."""

    def __init__(self, config: Dict[str, Any]):
        self.redis = RedisCache(config)
        self.subscriber = None

    async def subscribe(self):
        """S'abonne aux signaux de trading."""
        try:
            self.subscriber = self.redis.subscribe(['trading:signals'])

            while True:
                message = await self.subscriber.get_message()
                if message and message['type'] == 'message':
                    signal = json.loads(message['data'])
                    await self.handle_signal(signal)

        except Exception as e:
            logger.error(f"Erreur lors de l'abonnement aux signaux: {str(e)}")
            raise

    async def handle_signal(self, signal: Dict[str, Any]):
        """Gère un signal de trading."""
        try:
            logger.info(f"Reçu signal pour {signal['symbol']}")

            # Récupérer les données de marché
            market_data = await self.redis.get_market_data(signal['symbol'])
            if not market_data:
                logger.warning(
                    f"Pas de données de marché pour {signal['symbol']}")
                return

            # Vérifier les conditions de trading
            if self._check_trading_conditions(signal, market_data):
                # Exécuter le trade
                await self.execute_trade(signal)

        except Exception as e:
            logger.error(f"Erreur lors du traitement du signal: {str(e)}")

    def _check_trading_conditions(
            self, signal: Dict[str, Any], market_data: Dict[str, Any]) -> bool:
        """Vérifie les conditions de trading."""
        # Implémenter les conditions de trading ici
        return True

    async def execute_trade(self, signal: Dict[str, Any]):
        """Exécute un trade basé sur le signal."""
        try:
            KrakenAPI(self.config)
            # Implémenter l'exécution du trade ici

        except Exception as e:
            logger.error(f"Erreur lors de l'exécution du trade: {str(e)}")
