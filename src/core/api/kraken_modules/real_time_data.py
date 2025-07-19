from typing import Dict, List, Optional, Any
from ..kraken_api.validators import KrakenValidator
from ..kraken_api.endpoints import KrakenEndpoints
from ..kraken_api.metrics import KrakenMetrics
from ..kraken_api.websocket import KrakenWebSocket

class KrakenRealTimeData:
    def __init__(self, validator: KrakenValidator, endpoints: KrakenEndpoints, metrics: KrakenMetrics, websocket: KrakenWebSocket):
        self._validator = validator
        self._endpoints = endpoints
        self._metrics = metrics
        self._websocket = websocket

    async def subscribe_market_data(self, pair: str, channels: List[str]) -> None:
        """S'abonne aux données de marché en temps réel."""
        for channel in channels:
            await self._websocket.subscribe(channel, pair)

    async def unsubscribe_market_data(self, pair: str, channels: List[str]) -> None:
        """Se désabonne des données de marché en temps réel."""
        for channel in channels:
            await self._websocket.unsubscribe(channel, pair)

    async def get_market_data_subscriptions(self) -> Dict:
        """Récupère les abonnements aux données de marché."""
        return self._websocket.get_subscriptions()

    async def get_market_data(self, pair: str, channel: str) -> Optional[Dict]:
        """Récupère les données de marché en temps réel."""
        return self._websocket.get_market_data(pair, channel)

    async def get_recent_trades(self, pair: str, since: Optional[int] = None) -> Dict:
        """Récupère les trades récents."""
        return await self._websocket.get_trades(pair, since)

    async def get_recent_spread(self, pair: str, since: Optional[int] = None) -> Dict:
        """Récupère le spread récent."""
        return await self._websocket.get_spread(pair, since)

    async def get_orderbook_depth(self, pair: str, count: int = 100) -> Dict:
        """Récupère la profondeur du carnet d'ordres."""
        return await self._websocket.get_orderbook(pair, count)

    async def get_ticker_data(self, pair: str) -> Dict:
        """Récupère les données du ticker."""
        return await self._websocket.get_ticker(pair)
