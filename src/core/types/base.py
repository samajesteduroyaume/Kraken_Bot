"""
Base types for the trading bot.
"""

from typing import TypedDict, List
from datetime import datetime


class PairMetrics(TypedDict):
    """Metrics for a trading pair."""
    momentum: float
    volatility: float
    volume: float
    score: float


class TradingPair(TypedDict):
    """Trading pair information."""
    base: str
    quote: str
    name: str
    volume: float
    price: float
    metrics: PairMetrics


class Trade(TypedDict):
    """Trade information."""
    pair: str
    side: str
    price: float
    amount: float
    timestamp: datetime
    status: str
    id: str


class Order(TypedDict):
    """Order information."""
    pair: str
    side: str
    type: str
    price: float
    amount: float
    status: str
    id: str
    trades: List[Trade]


class Position(TypedDict):
    """Position information."""
    pair: str
    amount: float
    entry_price: float
    current_price: float
    pnl: float
    status: str


class MarketData(TypedDict):
    """Market data for a pair."""
    pair: str
    price: float
    volume: float
    timestamp: datetime
    metrics: PairMetrics
