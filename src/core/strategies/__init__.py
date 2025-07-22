"""
Package de gestion des stratégies de trading.

Ce package contient les différentes implémentations de stratégies,
le gestionnaire de stratégies et les types associés.
"""

from .base_strategy import BaseStrategy
from .trend_following import TrendFollowingStrategy
from .mean_reversion import MeanReversionStrategy
from .momentum import MomentumStrategy
from .manager import StrategyManager
from .types import TradingSignal, SignalAction, StrategyType, StrategyConfig, MarketData, Indicators

__all__ = [
    'BaseStrategy',
    'TrendFollowingStrategy',
    'MeanReversionStrategy',
    'MomentumStrategy',
    'StrategyManager',
    'TradingSignal',
    'SignalAction',
    'StrategyType',
    'StrategyConfig',
    'MarketData',
    'Indicators'
]
