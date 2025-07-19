"""
Module d'interface pour l'API Kraken.
"""

from typing import Dict, List, Optional, Any, Callable
from .kraken_api.client import KrakenAPI
from .kraken_api.orders import KrakenOrders
from .kraken_api.websocket import KrakenWebSocket
from .kraken_api.endpoints import KrakenEndpoints
from .kraken_api.metrics import KrakenMetrics
from .kraken_api.cache import KrakenCache
from .kraken_api.ratelimiter import RateLimiter
from .kraken_api.config import KrakenConfig
from .kraken_api.validators import KrakenValidator
from .kraken_modules.market_analysis import KrakenMarketAnalysis
from .kraken_modules.position_management import KrakenPositionManagement
from .kraken_modules.event_manager import KrakenEventManager
from .kraken_modules.real_time_data import KrakenRealTimeData

class KrakenInterface:
    def __init__(self, api_key: Optional[str] = None, api_secret: Optional[str] = None):
        """Initialise l'interface Kraken."""
        self._client = KrakenAPI(api_key, api_secret)
        self._orders = KrakenOrders(self._client)
        self._websocket = KrakenWebSocket()
        self._endpoints = KrakenEndpoints(self._client)
        self._metrics = KrakenMetrics()
        self._cache = KrakenCache()
        self._ratelimiter = RateLimiter()
        self._config = KrakenConfig()
        self._validator = KrakenValidator()
        self.logger = self._client.logger

        # Initialisation des modules
        self._market_analysis = KrakenMarketAnalysis(self._validator, self._endpoints, self._metrics)
        self._position_management = KrakenPositionManagement(self._validator, self._endpoints, self._metrics)
        self._event_manager = KrakenEventManager(self._validator, self._endpoints, self._metrics, self._websocket)
        self._real_time_data = KrakenRealTimeData(self._validator, self._endpoints, self._metrics, self._websocket)

    async def initialize(self) -> None:
        """Initialise les connexions et les configurations."""
        await self._client.initialize()
        await self._websocket.initialize()
        await self._cache.initialize()
        await self._ratelimiter.initialize()

    async def close(self) -> None:
        """Ferme toutes les connexions."""
        await self._client.close()
        await self._websocket.close()
        await self._cache.close()
        await self._ratelimiter.close()

    # Méthodes de trading
    async def create_market_order(self, pair: str, type: str, volume: float) -> Dict:
        """Crée un ordre market."""
        return await self._orders.create_market_order(pair, type, volume)

    async def create_limit_order(self, pair: str, type: str, price: float, volume: float) -> Dict:
        """Crée un ordre limit."""
        return await self._orders.create_limit_order(pair, type, price, volume)

    async def add_margin_order(self, pair: str, type: str, ordertype: str, leverage: int, price: float, volume: float) -> Dict:
        """Crée un ordre avec marge."""
        return await self._orders.add_margin_order(pair, type, ordertype, leverage, price, volume)

    async def add_order(self, pair: str, type: str, ordertype: str, price: Optional[float] = None, volume: Optional[float] = None, **kwargs) -> Dict:
        """Crée un ordre personnalisé."""
        return await self._orders.add_order(pair, type, ordertype, price, volume, **kwargs)

    # Méthodes de gestion des ordres
    async def get_open_orders(self, trades: bool = False, userref: Optional[int] = None) -> Dict:
        """Récupère les ordres ouverts."""
        return await self._orders.get_open_orders(trades, userref)

    async def get_closed_orders(self, trades: bool = False, userref: Optional[int] = None) -> Dict:
        """Récupère les ordres fermés."""
        return await self._orders.get_closed_orders(trades, userref)

    async def get_order_info(self, txid: str) -> Dict:
        """Récupère les informations d'un ordre."""
        return await self._orders.get_order_info(txid)

    async def get_trades_history(self, trades: bool = False, userref: Optional[int] = None) -> Dict:
        """Récupère l'historique des trades."""
        return await self._orders.get_trades_history(trades, userref)

    async def get_trades_info(self, txid: str) -> Dict:
        """Récupère les informations d'un trade."""
        return await self._orders.get_trades_info(txid)

    async def cancel_order(self, txid: str) -> Dict:
        """Annule un ordre."""
        return await self._orders.cancel_order(txid)

    async def cancel_all_orders(self) -> Dict:
        """Annule tous les ordres."""
        return await self._orders.cancel_all_orders()

    # Méthodes de gestion des positions et marges
    async def get_open_positions(self, txid: Optional[str] = None) -> Dict:
        """Récupère les positions ouvertes."""
        return await self._orders.get_open_positions(txid)

    async def get_margin_position(self, pair: str) -> Dict:
        """Récupère une position avec marge."""
        return await self._orders.get_margin_position(pair)

    async def get_margin_info(self, pair: Optional[str] = None) -> Dict:
        """Récupère les informations sur la marge."""
        return await self._orders.get_margin_info(pair)

    async def get_trade_volume(self, pair: Optional[str] = None) -> Dict:
        """Récupère le volume de trading."""
        return await self._orders.get_trade_volume(pair)

    # Méthodes de données de marché
    async def get_ticker(self, pair: str) -> Dict:
        """Récupère le ticker."""
        return await self._endpoints.get_ticker(pair)

    async def get_ohlc(self, pair: str, interval: int = 1, since: Optional[int] = None) -> Dict:
        """Récupère les données OHLC (Open, High, Low, Close)."""
        return await self._endpoints.get_ohlc(pair, interval, since)

    async def get_orderbook(self, pair: str, count: int = 100) -> Dict:
        """Récupère le carnet d'ordres."""
        return await self._endpoints.get_orderbook(pair, count)

    async def get_recent_spread(self, pair: str, since: Optional[int] = None) -> Dict:
        """Récupère le spread récent."""
        return await self._endpoints.get_recent_spread(pair, since)

    # Méthodes de compte
    async def get_balance(self) -> Dict:
        """Récupère le solde."""
        return await self._endpoints.get_balance()

    async def get_trade_balance(self, asset: str = 'ZUSD') -> Dict:
        """Récupère le solde de trading."""
        return await self._endpoints.get_trade_balance(asset)

    # Méthodes d'information
    async def get_assets(self, asset: Optional[str] = None) -> Dict:
        """Récupère les actifs."""
        return await self._endpoints.get_assets(asset)

    async def get_asset_pairs(self, pair: Optional[str] = None) -> Dict:
        """Récupère les paires d'actifs."""
        return await self._endpoints.get_asset_pairs(pair)
        
    async def get_tradable_asset_pairs(self, info: Optional[str] = None, pair: Optional[str] = None) -> Dict:
        """
        Récupère les informations sur les paires de trading disponibles sur Kraken.
        
        Args:
            info: Type d'information ('info', 'leverage', 'fees', 'margin')
            pair: Paire de trading spécifique (optionnel)
            
        Returns:
            Dictionnaire avec les informations sur les paires de trading
            
        Raises:
            ValueError: Si les paramètres sont invalides
            KrakenAPIError: Si l'API retourne une erreur
        """
        return await self._endpoints.get_tradable_asset_pairs(info, pair)

    async def get_server_time(self) -> Dict:
        """Récupère l'heure du serveur."""
        return await self._endpoints.get_server_time()

    async def get_system_status(self) -> Dict:
        """Récupère le statut du système."""
        return await self._endpoints.get_system_status()

    # Méthodes de surveillance
    async def get_metrics(self) -> Dict:
        """Récupère les métriques globales."""
        return self._metrics.get_metrics()

    async def get_endpoint_metrics(self, endpoint: str) -> Dict:
        """Récupère les métriques par endpoint."""
        return self._metrics.get_endpoint_metrics(endpoint)

    async def get_request_stats(self) -> Dict:
        """Récupère les statistiques des requêtes."""
        return self._metrics.get_request_stats()

    async def reset_metrics(self) -> None:
        """Réinitialise les métriques."""
        self._metrics.reset()

    # Méthodes de cache
    async def cache_get(self, key: str) -> Any:
        """Récupère une valeur du cache."""
        return await self._cache.get(key)

    async def cache_set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Stocke une valeur dans le cache."""
        await self._cache.set(key, value, ttl)

    async def cache_clear(self) -> None:
        """Efface le cache."""
        await self._cache.clear()

    async def cache_info(self) -> Dict:
        """Récupère les informations du cache."""
        return await self._cache.info()

    # Méthodes de rate limiting
    async def get_rate_limit_info(self) -> Dict:
        """Récupère les informations sur les limites de débit."""
        return {
            'max_requests': self._ratelimiter.max_requests,
            'window_seconds': self._ratelimiter.window_seconds,
            'burst_factor': self._ratelimiter.burst_factor,
            'current_window': self._ratelimiter._current_window,
            'requests_this_window': len(self._ratelimiter.requests),
            'endpoint_stats': dict(self._ratelimiter.endpoint_stats)
        }

    async def set_rate_limit(self, max_requests: int, window_seconds: int, burst_factor: float = 1.5) -> None:
        """Configure les limites de débit."""
        self._ratelimiter.max_requests = max_requests
        self._ratelimiter.window_seconds = window_seconds
        self._ratelimiter.burst_factor = burst_factor

    async def reset_rate_limit(self) -> None:
        """Réinitialise les limites de débit."""
        self._ratelimiter.requests = []
        self._ratelimiter._last_reset = time.time()
        self._ratelimiter._current_window = 0
        self._ratelimiter.endpoint_stats.clear()

    async def wait_for_limit(self, endpoint: str) -> None:
        """Attend si nécessaire pour respecter les limites de débit."""
        await self._ratelimiter.wait_for_limit(endpoint)

    async def check_limit(self, endpoint: str) -> bool:
        """Vérifie si une requête peut être effectuée."""
        return self._ratelimiter.check_limit(endpoint)

    # Méthodes de configuration
    async def load_config(self, config_path: str = None) -> None:
        """Charge la configuration."""
        await self._config.load_config(config_path)

    async def get_config(self) -> Dict:
        """Récupère la configuration."""
        return self._config.get_config()

    async def set_config(self, config: Dict) -> None:
        """Définit la configuration."""
        self._config.set_config(config)

    async def save_config(self, config_path: str = None) -> None:
        """Sauvegarde la configuration."""
        await self._config.save_config(config_path)

    async def get_api_credentials(self) -> Dict:
        """Récupère les identifiants API."""
        return self._config.get_api_credentials()

    async def set_api_credentials(self, api_key: str, api_secret: str) -> None:
        """Définit les identifiants API."""
        self._config.set_api_credentials(api_key, api_secret)

    # Méthodes de gestion de l'environnement
    async def get_environment(self) -> str:
        """Récupère l'environnement actuel."""
        return self._config.get_environment()

    async def set_environment(self, env: str) -> None:
        """Change l'environnement (production/sandbox)."""
        self._config.set_environment(env)

    async def get_api_url(self) -> str:
        """Récupère l'URL de l'API selon l'environnement."""
        return self._config.get_api_url()

    async def get_ws_url(self) -> str:
        """Récupère l'URL WebSocket selon l'environnement."""
        return self._config.get_ws_url()

    # Méthodes de validation
    async def validate_pair(self, pair: str) -> bool:
        """Valide une paire de trading."""
        return self._validator.validate_pair(pair)

    async def validate_timestamp(self, timestamp: int) -> bool:
        """Valide un timestamp."""
        return self._validator.validate_timestamp(timestamp)

    async def validate_interval(self, interval: int) -> bool:
        """Valide un intervalle de temps."""
        return self._validator.validate_interval(interval)

    async def validate_price(self, price: float) -> bool:
        """Valide un prix."""
        return self._validator.validate_price(price)

    async def validate_volume(self, volume: float) -> bool:
        """Valide un volume."""
        return self._validator.validate_volume(volume)

    async def validate_order_type(self, ordertype: str) -> bool:
        """Valide un type d'ordre."""
        return self._validator.validate_order_type(ordertype)

    async def validate_params(self, params: Dict) -> bool:
        """Valide les paramètres d'une requête."""
        return self._validator.validate_params(params)

    # Méthodes de débogage et logging
    async def set_log_level(self, level: str) -> None:
        """Définit le niveau de logging."""
        self.logger.setLevel(level)

    async def get_log_level(self) -> str:
        """Récupère le niveau de logging actuel."""
        return self.logger.getEffectiveLevel()

    async def get_log_handlers(self) -> List:
        """Récupère les handlers de logging actuels."""
        return self.logger.handlers

    async def add_log_handler(self, handler: Any) -> None:
        """Ajoute un handler de logging."""
        self.logger.addHandler(handler)

    async def remove_log_handler(self, handler: Any) -> None:
        """Supprime un handler de logging."""
        self.logger.removeHandler(handler)

    async def clear_log_handlers(self) -> None:
        """Supprime tous les handlers de logging."""
        for handler in self.logger.handlers[:]:
            self.logger.removeHandler(handler)

    # Méthodes de gestion des erreurs
    async def get_last_error(self) -> Optional[Dict]:
        """Récupère la dernière erreur."""
        return self._metrics.get_last_error()

    async def get_error_history(self) -> List[Dict]:
        """Récupère l'historique des erreurs."""
        return self._metrics.get_error_history()

    async def get_error_stats(self) -> Dict:
        """Récupère les statistiques d'erreurs."""
        return self._metrics.get_error_stats()

    async def reset_error_stats(self) -> None:
        """Réinitialise les statistiques d'erreurs."""
        self._metrics.reset_error_stats()

    # Méthodes de gestion des événements (via event_manager)
    async def add_event_handler(self, event_type: str, handler: Callable) -> None:
        """Ajoute un handler pour un type d'événement."""
        await self._event_manager.add_event_handler(event_type, handler)

    async def remove_event_handler(self, event_type: str, handler: Callable) -> None:
        """Supprime un handler d'événement."""
        await self._event_manager.remove_event_handler(event_type, handler)

    async def emit_event(self, event_type: str, data: Any) -> None:
        """Émet un événement."""
        await self._event_manager.emit_event(event_type, data)

    async def get_event_handlers(self) -> Dict:
        """Récupère les handlers d'événements."""
        return await self._event_manager.get_event_handlers()

    async def clear_event_handlers(self) -> None:
        """Supprime tous les handlers d'événements."""
        await self._event_manager.clear_event_handlers()

    # Méthodes de données en temps réel (via real_time_data)
    async def subscribe_real_time_data(self, pair: str, channels: List[str]) -> None:
        """S'abonne aux données en temps réel."""
        await self._real_time_data.subscribe(pair, channels)

    async def unsubscribe_real_time_data(self, pair: str, channels: List[str]) -> None:
        """Se désabonne des données en temps réel."""
        await self._real_time_data.unsubscribe(pair, channels)

    async def get_real_time_subscriptions(self) -> Dict:
        """Récupère les abonnements en temps réel."""
        return await self._real_time_data.get_subscriptions()

    async def get_real_time_data(self, pair: str, channel: str) -> Optional[Dict]:
        """Récupère les données en temps réel."""
        return await self._real_time_data.get_data(pair, channel)

    async def get_real_time_trades(self, pair: str) -> List:
        """Récupère les trades en temps réel."""
        return await self._real_time_data.get_trades(pair)

    async def get_real_time_spread(self, pair: str) -> Dict:
        """Récupère le spread en temps réel."""
        return await self._real_time_data.get_spread(pair)

    async def get_real_time_orderbook(self, pair: str) -> Dict:
        """Récupère le carnet d'ordres en temps réel."""
        return await self._real_time_data.get_orderbook(pair)

    async def get_real_time_ticker(self, pair: str) -> Dict:
        """Récupère le ticker en temps réel."""
        return await self._real_time_data.get_ticker(pair)

    # Méthodes de gestion des sessions
    async def get_session_info(self) -> Dict:
        """Récupère les informations sur la session actuelle."""
        return {
            'api_key': bool(self._client.api_key),
            'api_secret': bool(self._client.api_secret),
            'connected': self._client.is_connected,
            'websocket_connected': self._websocket.is_connected,
            'last_request': self._metrics.get_last_request_time(),
            'last_response': self._metrics.get_last_response_time(),
            'total_requests': self._metrics.get_total_requests(),
            'success_rate': self._metrics.get_success_rate()
        }

    async def refresh_session(self) -> None:
        """Rafraîchit la session HTTP."""
        await self._client.close()
        await self._client.initialize()
        await self.initialize()

    async def get_connection_stats(self) -> Dict:
        """Récupère les statistiques de connexion."""
        return {
            'http': {
                'connected': self._client.is_connected,
                'total_requests': self._metrics.get_total_requests(),
                'success_rate': self._metrics.get_success_rate(),
                'avg_response_time': self._metrics.get_avg_response_time()
            },
            'websocket': {
                'connected': self._websocket.is_connected,
                'subscriptions': await self.get_real_time_subscriptions(),
                'ping_interval': self._websocket.ping_interval,
                'ping_timeout': self._websocket.ping_timeout
            }
        }

    async def set_connection_timeout(self, timeout: int) -> None:
        """Configure le timeout de connexion HTTP."""
        if timeout < 1:
            raise ValueError("Le timeout doit être supérieur à 0")
        self._client.timeout = timeout

    async def get_connection_timeout(self) -> int:
        """Récupère le timeout de connexion HTTP."""
        return self._client.timeout

    # Méthodes de gestion des pics de charge
    async def set_burst_limit(self, endpoint: str, max_burst: int, duration: int) -> None:
        """Configure la limite de pic pour un endpoint."""
        self._ratelimiter.set_burst_limit(endpoint, max_burst, duration)

    async def get_burst_stats(self, endpoint: str) -> Dict:
        """Récupère les statistiques de pic pour un endpoint."""
        return self._ratelimiter.get_burst_stats(endpoint)

    async def set_endpoint_rate_limit(self, endpoint: str, max_requests: int, window_seconds: int) -> None:
        """Configure la limite de débit pour un endpoint spécifique."""
        self._ratelimiter.set_endpoint_rate_limit(endpoint, max_requests, window_seconds)

    async def get_endpoint_rate_limit(self, endpoint: str) -> Dict:
        """Récupère la configuration de la limite de débit pour un endpoint."""
        return self._ratelimiter.get_endpoint_rate_limit(endpoint)

    async def set_global_rate_limit(self, max_requests: int, window_seconds: int) -> None:
        """Configure la limite de débit globale."""
        self._ratelimiter.set_global_rate_limit(max_requests, window_seconds)

    async def get_global_rate_limit(self) -> Dict:
        """Récupère la configuration de la limite de débit globale."""
        return {
            'max_requests': self._ratelimiter.max_requests,
            'window_seconds': self._ratelimiter.window_seconds,
            'burst_factor': self._ratelimiter.burst_factor
        }


        # Méthodes de gestion des callbacks WebSocket
async def set_ws_message_handler(self, handler: Callable) -> None:
    """Configure le handler de messages WebSocket."""
    self._websocket.message_handler = handler

async def get_ws_message_handler(self) -> Optional[Callable]:
    """Récupère le handler de messages WebSocket actuel."""
    return self._websocket.message_handler

async def set_ws_error_handler(self, handler: Callable) -> None:
    """Configure le handler d'erreurs WebSocket."""
    self._websocket.error_handler = handler

async def get_ws_error_handler(self) -> Optional[Callable]:
    """Récupère le handler d'erreurs WebSocket actuel."""
    return self._websocket.error_handler

async def set_ws_close_handler(self, handler: Callable) -> None:
    """Configure le handler de fermeture WebSocket."""
    self._websocket.close_handler = handler

async def get_ws_close_handler(self) -> Optional[Callable]:
    """Récupère le handler de fermeture WebSocket actuel."""
    return self._websocket.close_handler

# Méthodes utilitaires
async def format_price(self, price: float, pair: str) -> str:
    """Formate un prix selon la paire."""
    pair_info = await self._endpoints.get_asset_pairs(pair)
    decimals = pair_info[pair]['pair_decimals']
    return f"{price:.{decimals}f}"

async def format_volume(self, volume: float, pair: str) -> str:
    """Formate un volume selon la paire."""
    pair_info = await self._endpoints.get_asset_pairs(pair)
    decimals = pair_info[pair]['lot_decimals']
    return f"{volume:.{decimals}f}"

async def calculate_order_value(self, price: float, volume: float) -> float:
    """Calcule la valeur d'un ordre."""
    return price * volume

async def get_min_order_size(self, pair: str) -> float:
    """Récupère la taille minimale d'ordre pour une paire."""
    pair_info = await self._endpoints.get_asset_pairs(pair)
    return pair_info[pair]['ordermin']

async def get_trading_fees(self, pair: str) -> Dict:
    """Récupère les frais de trading pour une paire."""
    pair_info = await self._endpoints.get_asset_pairs(pair)
    return {
        'maker': pair_info[pair]['fees'][0][1],
        'taker': pair_info[pair]['fees'][0][0]
    }

async def get_available_balance(self, asset: str) -> float:
    """Récupère le solde disponible pour trading."""
    balance = await self.get_balance()
    return float(balance[asset]['available'])

async def get_total_balance(self, asset: str) -> float:
    """Récupère le solde total."""
    balance = await self.get_balance()
    return float(balance[asset]['balance'])

async def validate_credentials(self) -> bool:
    """Valide les identifiants API."""
    try:
        await self._client.get_server_time()
        return True
    except Exception:
        return False

async def ping(self) -> None:
    """Envoie un ping au serveur."""
    await self._websocket.ping()

async def close_websocket(self) -> None:
    """Ferme la connexion WebSocket."""
    await self._websocket.close()

async def set_reconnect_config(self, max_attempts: int, delay: int) -> None:
    """Configure la stratégie de reconnexion."""
    self._websocket.max_reconnect_attempts = max_attempts
    self._websocket.reconnect_delay = delay

async def get_reconnect_config(self) -> Dict:
    """Récupère la configuration de reconnexion."""
    return {
        'max_attempts': self._websocket.max_reconnect_attempts,
        'delay': self._websocket.reconnect_delay
    }

async def get_connection_status(self) -> Dict:
    """Récupère le statut de la connexion WebSocket."""
    return {
        'connected': self._websocket.is_connected,
        'running': self._websocket.is_running,
        'reconnect_attempts': self._websocket.reconnect_attempts,
        'max_reconnect_attempts': self._websocket.max_reconnect_attempts,
        'ping_interval': self._websocket.ping_interval,
        'ping_timeout': self._websocket.ping_timeout
    }

async def get_subscriptions(self) -> Dict:
    """Récupère les abonnements actuels."""
    return await self._websocket.get_subscriptions()

async def get_ws_config(self) -> Dict:
    """Récupère la configuration WebSocket."""
    return {
        'url': self._websocket.url,
        'ping_interval': self._websocket.ping_interval,
        'ping_timeout': self._websocket.ping_timeout,
        'max_reconnect_attempts': self._websocket.max_reconnect_attempts,
        'reconnect_delay': self._websocket.reconnect_delay
    }

async def set_ws_config(self, config: Dict) -> None:
    """Configure les paramètres WebSocket."""
    self._websocket.configure(**config)

async def get_ws_stats(self) -> Dict:
    """Récupère les statistiques WebSocket."""
    return {
        'messages_received': self._websocket.messages_received,
        'errors': self._websocket.errors,
        'last_message_time': self._websocket.last_message_time,
        'last_error_time': self._websocket.last_error_time
    }

async def reset_ws_stats(self) -> None:
    """Réinitialise les statistiques WebSocket."""
    self._websocket.reset_stats()

async def get_ws_subscriptions(self) -> Dict:
    """Récupère les abonnements WebSocket."""
    return self._websocket.subscriptions

async def get_ws_messages(self) -> List:
    """Récupère les messages WebSocket en attente."""
    return self._websocket.messages

async def clear_ws_messages(self) -> None:
    """Supprime les messages WebSocket en attente."""
    self._websocket.messages.clear()

async def get_ws_error_count(self) -> int:
    """Récupère le nombre d'erreurs WebSocket."""
    return self._websocket.error_count

async def reset_ws_error_count(self) -> None:
    """Réinitialise le compteur d'erreurs WebSocket."""
    self._websocket.error_count = 0

async def get_ws_last_error(self) -> Optional[Dict]:
    """Récupère la dernière erreur WebSocket."""
    return self._websocket.last_error

async def get_ws_last_message(self) -> Optional[Dict]:
    """Récupère le dernier message WebSocket."""
    return self._websocket.last_message

async def get_ws_message_count(self) -> int:
    """Récupère le nombre de messages reçus."""
    return len(self._websocket.messages)

async def get_ws_message_queue(self) -> List:
    """Récupère la file d'attente des messages WebSocket."""
    return self._websocket.message_queue

async def clear_ws_message_queue(self) -> None:
    """Supprime la file d'attente des messages WebSocket."""
    self._websocket.message_queue.clear()

async def get_ws_message_handler_stats(self) -> Dict:
    """Récupère les statistiques du handler de messages."""
    return {
        'messages_handled': self._websocket.messages_handled,
        'errors': self._websocket.message_handler_errors,
        'last_error': self._websocket.last_message_handler_error
    }

    async def reset_ws_message_queue_processing_time_stats(self) -> None:
        """Réinitialise les statistiques de temps de traitement de la file d'attente des messages."""
        self._websocket.message_queue_processing_time_min = float('inf')
        self._websocket.message_queue_processing_time_max = 0.0
        self._websocket.message_queue_processing_time_avg = 0.0

    async def get_ws_message_queue_latency_min(self) -> float:
        """Récupère la latence minimale de la file d'attente des messages."""
        return self._websocket.message_queue_latency_min

    async def reset_ws_message_queue_latency_min(self) -> None:
        """Réinitialise la latence minimale de la file d'attente des messages."""
        self._websocket.message_queue_latency_min = float('inf')

    async def get_ws_reconnect_count(self) -> int:
        """Récupère le nombre de reconnexions."""
        return self._websocket.reconnect_count

    async def reset_ws_reconnect_count(self) -> None:
        """Réinitialise le compteur de reconnexions."""
        self._websocket.reconnect_count = 0

    async def get_ws_heartbeat_stats(self) -> Dict:
        """Récupère les statistiques du heartbeat."""
        return {
            'last_heartbeat': self._websocket.last_heartbeat,
            'missed_heartbeats': self._websocket.missed_heartbeats,
            'heartbeat_interval': self._websocket.heartbeat_interval
        }

    async def reset_ws_heartbeat_stats(self) -> None:
        """Réinitialise les statistiques du heartbeat."""
        self._websocket.last_heartbeat = None
        self._websocket.missed_heartbeats = 0

    async def get_ws_subscription_stats(self) -> Dict:
        """Récupère les statistiques des abonnements."""
        return {
            'total_subscriptions': len(self._websocket.subscriptions),
            'active_subscriptions': len([s for s in self._websocket.subscriptions.values() if s['active']]),
            'subscription_errors': self._websocket.subscription_errors
        }

    async def reset_ws_subscription_stats(self) -> None:
        """Réinitialise les statistiques des abonnements."""
        self._websocket.subscription_errors = 0

    async def get_ws_message_latency(self) -> float:
        """Récupère la latence moyenne des messages."""
        return self._websocket.message_latency

    async def reset_ws_message_latency(self) -> None:
        """Réinitialise la latence des messages."""
        self._websocket.message_latency = 0.0

    async def get_ws_message_throughput(self) -> float:
        """Récupère le débit des messages."""
        return self._websocket.message_throughput

    async def reset_ws_message_throughput(self) -> None:
        """Réinitialise le débit des messages."""
        self._websocket.message_throughput = 0.0

    async def get_ws_error_rate(self) -> float:
        """Récupère le taux d'erreurs."""
        return self._websocket.error_rate

    async def reset_ws_error_rate(self) -> None:
        """Réinitialise le taux d'erreurs."""
        self._websocket.error_rate = 0.0

    async def get_ws_message_loss_rate(self) -> float:
        """Récupère le taux de perte de messages."""
        return self._websocket.message_loss_rate

    async def reset_ws_message_loss_rate(self) -> None:
        """Réinitialise le taux de perte de messages."""
        self._websocket.message_loss_rate = 0.0

    async def get_ws_reconnect_stats(self) -> Dict:
        """Récupère les statistiques de reconnexion."""
        return {
            'total_reconnects': self._websocket.reconnect_count,
            'failed_reconnects': self._websocket.failed_reconnects,
            'last_reconnect_time': self._websocket.last_reconnect_time
        }

    async def reset_ws_reconnect_stats(self) -> None:
        """Réinitialise les statistiques de reconnexion."""
        self._websocket.reconnect_count = 0
        self._websocket.failed_reconnects = 0
        self._websocket.last_reconnect_time = None

    async def get_ws_subscription_success_rate(self) -> float:
        """Récupère le taux de succès des abonnements."""
        return self._websocket.subscription_success_rate

    async def reset_ws_subscription_success_rate(self) -> None:
        """Réinitialise le taux de succès des abonnements."""
        self._websocket.subscription_success_rate = 0.0

    async def get_ws_message_delivery_rate(self) -> float:
        """Récupère le taux de livraison des messages."""
        return self._websocket.message_delivery_rate

    async def reset_ws_message_delivery_rate(self) -> None:
        """Réinitialise le taux de livraison des messages."""
        self._websocket.message_delivery_rate = 0.0

    async def get_ws_message_processing_time(self) -> float:
        """Récupère le temps de traitement des messages."""
        return self._websocket.message_processing_time

    async def reset_ws_message_processing_time(self) -> None:
        """Réinitialise le temps de traitement des messages."""
        self._websocket.message_processing_time = 0.0

    async def get_ws_message_queue_size(self) -> int:
        """Récupère la taille de la file d'attente des messages."""
        return len(self._websocket.message_queue)

    async def reset_ws_message_queue_size(self) -> None:
        """Réinitialise la taille de la file d'attente des messages."""
        self._websocket.message_queue.clear()

    async def get_ws_message_queue_latency(self) -> float:
        """Récupère la latence de la file d'attente des messages."""
        return self._websocket.message_queue_latency

    async def reset_ws_message_queue_latency(self) -> None:
        """Réinitialise la latence de la file d'attente des messages."""
        self._websocket.message_queue_latency = 0.0

    async def get_ws_message_queue_throughput(self) -> float:
        """Récupère le débit de la file d'attente des messages."""
        return self._websocket.message_queue_throughput

    async def reset_ws_message_queue_throughput(self) -> None:
        """Réinitialise le débit de la file d'attente des messages."""
        self._websocket.message_queue_throughput = 0.0

    async def get_ws_message_queue_error_rate(self) -> float:
        """Récupère le taux d'erreurs de la file d'attente des messages."""
        return self._websocket.message_queue_error_rate

    async def reset_ws_message_queue_error_rate(self) -> None:
        """Réinitialise le taux d'erreurs de la file d'attente des messages."""
        self._websocket.message_queue_error_rate = 0.0

    async def get_ws_message_queue_loss_rate(self) -> float:
        """Récupère le taux de perte de la file d'attente des messages."""
        return self._websocket.message_queue_loss_rate

    async def reset_ws_message_queue_loss_rate(self) -> None:
        """Réinitialise le taux de perte de la file d'attente des messages."""
        self._websocket.message_queue_loss_rate = 0.0

    async def get_ws_message_queue_reconnect_rate(self) -> float:
        """Récupère le taux de reconnexion de la file d'attente des messages."""
        return self._websocket.message_queue_reconnect_rate

    async def reset_ws_message_queue_reconnect_rate(self) -> None:
        """Réinitialise le taux de reconnexion de la file d'attente des messages."""
        self._websocket.message_queue_reconnect_rate = 0.0

    async def get_ws_message_queue_subscription_rate(self) -> float:
        """Récupère le taux d'abonnement de la file d'attente des messages."""
        return self._websocket.message_queue_subscription_rate

    async def reset_ws_message_queue_subscription_rate(self) -> None:
        """Réinitialise le taux d'abonnement de la file d'attente des messages."""
        self._websocket.message_queue_subscription_rate = 0.0

    async def get_ws_message_queue_delivery_rate(self) -> float:
        """Récupère le taux de livraison de la file d'attente des messages."""
        return self._websocket.message_queue_delivery_rate

    async def reset_ws_message_queue_delivery_rate(self) -> None:
        """Réinitialise le taux de livraison de la file d'attente des messages."""
        self._websocket.message_queue_delivery_rate = 0.0

    async def get_ws_message_queue_processing_time(self) -> float:
        """Récupère le temps de traitement de la file d'attente des messages."""
        return self._websocket.message_queue_processing_time

    async def reset_ws_message_queue_processing_time(self) -> None:
        """Réinitialise le temps de traitement de la file d'attente des messages."""
        self._websocket.message_queue_processing_time = 0.0

    async def get_ws_message_queue_latency_stats(self) -> Dict:
        """Récupère les statistiques de latence de la file d'attente des messages."""
        return {
            'min': self._websocket.message_queue_latency_min,
            'max': self._websocket.message_queue_latency_max,
            'avg': self._websocket.message_queue_latency_avg
        }

    async def reset_ws_message_queue_latency_stats(self) -> None:
        """Réinitialise les statistiques de latence de la file d'attente des messages."""
        self._websocket.message_queue_latency_min = float('inf')
        self._websocket.message_queue_latency_max = 0.0
        self._websocket.message_queue_latency_avg = 0.0

    async def get_ws_message_queue_throughput_stats(self) -> Dict:
        """Récupère les statistiques de débit de la file d'attente des messages."""
        return {
            'min': self._websocket.message_queue_throughput_min,
            'max': self._websocket.message_queue_throughput_max,
            'avg': self._websocket.message_queue_throughput_avg
        }

    async def reset_ws_message_queue_throughput_stats(self) -> None:
        """Réinitialise les statistiques de débit de la file d'attente des messages."""
        self._websocket.message_queue_throughput_min = float('inf')
        self._websocket.message_queue_throughput_max = 0.0
        self._websocket.message_queue_throughput_avg = 0.0

    async def get_ws_message_queue_error_rate_stats(self) -> Dict:
        """Récupère les statistiques de taux d'erreurs de la file d'attente des messages."""
        return {
            'min': self._websocket.message_queue_error_rate_min,
            'max': self._websocket.message_queue_error_rate_max,
            'avg': self._websocket.message_queue_error_rate_avg
        }

    async def reset_ws_message_queue_error_rate_stats(self) -> None:
        """Réinitialise les statistiques de taux d'erreurs de la file d'attente des messages."""
        self._websocket.message_queue_error_rate_min = float('inf')
        self._websocket.message_queue_error_rate_max = 0.0
        self._websocket.message_queue_error_rate_avg = 0.0

    async def get_ws_message_queue_loss_rate_stats(self) -> Dict:
        """Récupère les statistiques de taux de perte de la file d'attente des messages."""
        return {
            'min': self._websocket.message_queue_loss_rate_min,
            'max': self._websocket.message_queue_loss_rate_max,
            'avg': self._websocket.message_queue_loss_rate_avg
        }

    async def reset_ws_message_queue_loss_rate_stats(self) -> None:
        """Réinitialise les statistiques de taux de perte de la file d'attente des messages."""
        self._websocket.message_queue_loss_rate_min = float('inf')
        self._websocket.message_queue_loss_rate_max = 0.0
        self._websocket.message_queue_loss_rate_avg = 0.0

    async def get_ws_message_queue_reconnect_rate_stats(self) -> Dict:
        """Récupère les statistiques de taux de reconnexion de la file d'attente des messages."""
        return {
            'min': self._websocket.message_queue_reconnect_rate_min,
            'max': self._websocket.message_queue_reconnect_rate_max,
            'avg': self._websocket.message_queue_reconnect_rate_avg
        }

    async def reset_ws_message_queue_reconnect_rate_stats(self) -> None:
        """Réinitialise les statistiques de taux de reconnexion de la file d'attente des messages."""
        self._websocket.message_queue_reconnect_rate_min = float('inf')
        self._websocket.message_queue_reconnect_rate_max = 0.0
        self._websocket.message_queue_reconnect_rate_avg = 0.0

    async def get_ws_message_queue_subscription_rate_stats(self) -> Dict:
        """Récupère les statistiques de taux d'abonnement de la file d'attente des messages."""
        return {
            'min': self._websocket.message_queue_subscription_rate_min,
            'max': self._websocket.message_queue_subscription_rate_max,
            'avg': self._websocket.message_queue_subscription_rate_avg
        }

    async def reset_ws_message_queue_subscription_rate_stats(self) -> None:
        """Réinitialise les statistiques de taux d'abonnement de la file d'attente des messages."""
        self._websocket.message_queue_subscription_rate_min = float('inf')
        self._websocket.message_queue_subscription_rate_max = 0.0
        self._websocket.message_queue_subscription_rate_avg = 0.0

    async def get_ws_message_queue_delivery_rate_stats(self) -> Dict:
        """Récupère les statistiques de taux de livraison de la file d'attente des messages."""
        return {
            'min': self._websocket.message_queue_delivery_rate_min,
            'max': self._websocket.message_queue_delivery_rate_max,
            'avg': self._websocket.message_queue_delivery_rate_avg
        }

    async def reset_ws_message_queue_delivery_rate_stats(self) -> None:
        """Réinitialise les statistiques de taux de livraison de la file d'attente des messages."""
        self._websocket.message_queue_delivery_rate_min = float('inf')
        self._websocket.message_queue_delivery_rate_max = 0.0
        self._websocket.message_queue_delivery_rate_avg = 0.0

    async def get_ws_message_queue_processing_time_stats(self) -> Dict:
        """Récupère les statistiques de temps de traitement de la file d'attente des messages."""
        return {
            'min': self._websocket.message_queue_processing_time_min,
            'max': self._websocket.message_queue_processing_time_max,
            'avg': self._websocket.message_queue_processing_time_avg
        }

    async def reset_ws_message_queue_processing_time_stats(self) -> None:
        """Réinitialise les statistiques de temps de traitement de la file d'attente des messages."""
        self._websocket.message_queue_processing_time_min = float('inf')
        self._websocket.message_queue_processing_time_max = 0.0
        self._websocket.message_queue_processing_time_avg = 0.0

    async def get_ws_message_queue_latency_min(self) -> float:
        """Récupère la latence minimale de la file d'attente des messages."""
        return self._websocket.message_queue_latency_min

    async def reset_ws_message_queue_latency_min(self) -> None:
        """Réinitialise la latence minimale de la file d'attente des messages."""
        self._websocket.message_queue_latency_min = float('inf')

    async def get_ws_message_queue_latency_max(self) -> float:
        """Récupère la latence maximale de la file d'attente des messages."""
        return self._websocket.message_queue_latency_max

    async def reset_ws_message_queue_latency_max(self) -> None:
        """Réinitialise la latence maximale de la file d'attente des messages."""
        self._websocket.message_queue_latency_max = 0.0

    async def get_ws_message_queue_latency_avg(self) -> float:
        """Récupère la latence moyenne de la file d'attente des messages."""
        return self._websocket.message_queue_latency_avg

    async def reset_ws_message_queue_latency_avg(self) -> None:
        """Réinitialise la latence moyenne de la file d'attente des messages."""
        self._websocket.message_queue_latency_avg = 0.0

    async def get_ws_message_queue_throughput_min(self) -> float:
        """Récupère le débit minimal de la file d'attente des messages."""
        return self._websocket.message_queue_throughput_min

    async def reset_ws_message_queue_throughput_min(self) -> None:
        """Réinitialise le débit minimal de la file d'attente des messages."""
        self._websocket.message_queue_throughput_min = float('inf')

    async def get_ws_message_queue_throughput_max(self) -> float:
        """Récupère le débit maximal de la file d'attente des messages."""
        return self._websocket.message_queue_throughput_max

    async def reset_ws_message_queue_throughput_max(self) -> None:
        """Réinitialise le débit maximal de la file d'attente des messages."""
        self._websocket.message_queue_throughput_max = 0.0

    async def get_ws_message_queue_throughput_avg(self) -> float:
        """Récupère le débit moyen de la file d'attente des messages."""
        return self._websocket.message_queue_throughput_avg

    async def reset_ws_message_queue_throughput_avg(self) -> None:
        """Réinitialise le débit moyen de la file d'attente des messages."""
        self._websocket.message_queue_throughput_avg = 0.0

    async def get_ws_message_queue_error_rate_min(self) -> float:
        """Récupère le taux d'erreurs minimal de la file d'attente des messages."""
        return self._websocket.message_queue_error_rate_min

    async def reset_ws_message_queue_error_rate_min(self) -> None:
        """Réinitialise le taux d'erreurs minimal de la file d'attente des messages."""
        self._websocket.message_queue_error_rate_min = float('inf')

    async def get_ws_message_queue_error_rate_max(self) -> float:
        """Récupère le taux d'erreurs maximal de la file d'attente des messages."""
        return self._websocket.message_queue_error_rate_max

    async def reset_ws_message_queue_error_rate_max(self) -> None:
        """Réinitialise le taux d'erreurs maximal de la file d'attente des messages."""
        self._websocket.message_queue_error_rate_max = 0.0

    async def get_ws_message_queue_error_rate_avg(self) -> float:
        """Récupère le taux d'erreurs moyen de la file d'attente des messages."""
        return self._websocket.message_queue_error_rate_avg

    async def reset_ws_message_queue_error_rate_avg(self) -> None:
        """Réinitialise le taux d'erreurs moyen de la file d'attente des messages."""
        self._websocket.message_queue_error_rate_avg = 0.0

    async def get_ws_message_queue_loss_rate_min(self) -> float:
        """Récupère le taux de perte minimal de la file d'attente des messages."""
        return self._websocket.message_queue_loss_rate_min

    async def reset_ws_message_queue_loss_rate_min(self) -> None:
        """Réinitialise le taux de perte minimal de la file d'attente des messages."""
        self._websocket.message_queue_loss_rate_min = float('inf')

    async def get_ws_message_queue_loss_rate_max(self) -> float:
        """Récupère le taux de perte maximal de la file d'attente des messages."""
        return self._websocket.message_queue_loss_rate_max

    async def reset_ws_message_queue_loss_rate_max(self) -> None:
        """Réinitialise le taux de perte maximal de la file d'attente des messages."""
        self._websocket.message_queue_loss_rate_max = 0.0

    async def get_ws_message_queue_loss_rate_avg(self) -> float:
        """Récupère le taux de perte moyen de la file d'attente des messages."""
        return self._websocket.message_queue_loss_rate_avg

    async def reset_ws_message_queue_loss_rate_avg(self) -> None:
        """Réinitialise le taux de perte moyen de la file d'attente des messages."""
        self._websocket.message_queue_loss_rate_avg = 0.0

    async def get_ws_message_queue_reconnect_rate_min(self) -> float:
        """Récupère le taux de reconnexion minimal de la file d'attente des messages."""
        return self._websocket.message_queue_reconnect_rate_min

    async def reset_ws_message_queue_reconnect_rate_min(self) -> None:
        """Réinitialise le taux de reconnexion minimal de la file d'attente des messages."""
        self._websocket.message_queue_reconnect_rate_min = float('inf')

    async def get_ws_message_queue_reconnect_rate_max(self) -> float:
        """Récupère le taux de reconnexion maximal de la file d'attente des messages."""
        return self._websocket.message_queue_reconnect_rate_max

    async def reset_ws_message_queue_reconnect_rate_max(self) -> None:
        """Réinitialise le taux de reconnexion maximal de la file d'attente des messages."""
        self._websocket.message_queue_reconnect_rate_max = 0.0

    async def get_ws_message_queue_reconnect_rate_avg(self) -> float:
        """Récupère le taux de reconnexion moyen de la file d'attente des messages."""
        return self._websocket.message_queue_reconnect_rate_avg

    async def reset_ws_message_queue_reconnect_rate_avg(self) -> None:
        """Réinitialise le taux de reconnexion moyen de la file d'attente des messages."""
        self._websocket.message_queue_reconnect_rate_avg = 0.0

    async def get_ws_message_queue_subscription_rate_min(self) -> float:
        """Récupère le taux d'abonnement minimal de la file d'attente des messages."""
        return self._websocket.message_queue_subscription_rate_min

    async def reset_ws_message_queue_subscription_rate_min(self) -> None:
        """Réinitialise le taux d'abonnement minimal de la file d'attente des messages."""
        self._websocket.message_queue_subscription_rate_min = float('inf')

    async def get_ws_message_queue_subscription_rate_max(self) -> float:
        """Récupère le taux d'abonnement maximal de la file d'attente des messages."""
        return self._websocket.message_queue_subscription_rate_max

    async def reset_ws_message_queue_subscription_rate_max(self) -> None:
        """Réinitialise le taux d'abonnement maximal de la file d'attente des messages."""
        self._websocket.message_queue_subscription_rate_max = 0.0

    async def get_ws_message_queue_subscription_rate_avg(self) -> float:
        """Récupère le taux d'abonnement moyen de la file d'attente des messages."""
        return self._websocket.message_queue_subscription_rate_avg

    async def reset_ws_message_queue_subscription_rate_avg(self) -> None:
        """Réinitialise le taux d'abonnement moyen de la file d'attente des messages."""
        self._websocket.message_queue_subscription_rate_avg = 0.0

    async def get_ws_message_queue_delivery_rate_min(self) -> float:
        """Récupère le taux de livraison minimal de la file d'attente des messages."""
        return self._websocket.message_queue_delivery_rate_min

    async def reset_ws_message_queue_delivery_rate_min(self) -> None:
        """Réinitialise le taux de livraison minimal de la file d'attente des messages."""
        self._websocket.message_queue_delivery_rate_min = float('inf')

    async def get_ws_message_queue_delivery_rate_max(self) -> float:
        """Récupère le taux de livraison maximal de la file d'attente des messages."""
        return self._websocket.message_queue_delivery_rate_max

    async def reset_ws_message_queue_delivery_rate_max(self) -> None:
        """Réinitialise le taux de livraison maximal de la file d'attente des messages."""
        self._websocket.message_queue_delivery_rate_max = 0.0

    async def get_ws_message_queue_delivery_rate_avg(self) -> float:
        """Récupère le taux de livraison moyen de la file d'attente des messages."""
        return self._websocket.message_queue_delivery_rate_avg

    async def reset_ws_message_queue_delivery_rate_avg(self) -> None:
        """Réinitialise le taux de livraison moyen de la file d'attente des messages."""
        self._websocket.message_queue_delivery_rate_avg = 0.0

    async def get_ws_message_queue_processing_time_min(self) -> float:
        """Récupère le temps de traitement minimal de la file d'attente des messages."""
        return self._websocket.message_queue_processing_time_min

    async def reset_ws_message_queue_processing_time_min(self) -> None:
        """Réinitialise le temps de traitement minimal de la file d'attente des messages."""
        self._websocket.message_queue_processing_time_min = float('inf')

    async def get_ws_message_queue_processing_time_max(self) -> float:
        """Récupère le temps de traitement maximal de la file d'attente des messages."""
        return self._websocket.message_queue_processing_time_max

    async def reset_ws_message_queue_processing_time_max(self) -> None:
        """Réinitialise le temps de traitement maximal de la file d'attente des messages."""
        self._websocket.message_queue_processing_time_max = 0.0

    async def get_ws_message_queue_processing_time_avg(self) -> float:
        """Récupère le temps de traitement moyen de la file d'attente des messages."""
        return self._websocket.message_queue_processing_time_avg

    async def reset_ws_message_queue_processing_time_avg(self) -> None:
        """Réinitialise le temps de traitement moyen de la file d'attente des messages."""
        self._websocket.message_queue_processing_time_avg = 0.0
