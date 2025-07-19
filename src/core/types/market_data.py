from typing import List, TypedDict
from datetime import datetime
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .market_types import PriceLevel, Candle, MarketStats
    from .types import Trade

__all__ = ['MarketData', 'OrderBook']


class OrderBook(TypedDict):
    """Carnet d'ordres."""
    asks: List['PriceLevel']
    bids: List['PriceLevel']
    timestamp: datetime


class MarketData(TypedDict, total=False):
    """Données de marché complètes."""
    symbol: str
    candles: List['Candle']
    order_book: OrderBook
    stats: 'MarketStats'
    trades: List['Trade']
    analysis: dict  # Ajouté, facultatif
