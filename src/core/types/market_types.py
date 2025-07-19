from typing import List, TypedDict
from decimal import Decimal
from datetime import datetime

__all__ = ['PriceLevel', 'Candle', 'MarketStats', 'OrderBook', 'MarketData']

# Utilisation d'annotations forward pour Trade
Trade = 'Trade'

# Types de base pour les données de marché


class PriceLevel(TypedDict):
    """Niveau de prix dans un carnet d'ordres."""
    price: Decimal
    amount: Decimal


class Candle(TypedDict):
    """Bougie de marché."""
    timestamp: datetime
    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal
    volume: Decimal


class MarketStats(TypedDict):
    """Statistiques de marché."""
    symbol: str
    last_price: Decimal
    volume_24h: Decimal
    volume_change_24h: Decimal
    price_change_24h: Decimal
    price_change_percent_24h: Decimal
    high_24h: Decimal
    low_24h: Decimal
    timestamp: datetime


class OrderBook(TypedDict):
    """Carnet d'ordres."""
    asks: List['PriceLevel']
    bids: List['PriceLevel']
    timestamp: datetime


class MarketData(TypedDict):
    """Données de marché complètes."""
    symbol: str
    candles: List['Candle']
    order_book: OrderBook
    stats: MarketStats
    trades: List[Trade]  # Utilisation d'une annotation forward pour Trade
