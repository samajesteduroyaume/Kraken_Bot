"""
Package contenant les différentes stratégies de trading pour le bot Kraken.
"""
from .base_strategy import BaseStrategy
from .momentum_strategy import MomentumStrategy
from .mean_reversion_strategy import MeanReversionStrategy
from .breakout_strategy import BreakoutStrategy
from .grid_strategy import GridStrategy
from .swing_strategy import SwingStrategy

__all__ = [
    'BaseStrategy',
    'MomentumStrategy',
    'MeanReversionStrategy',
    'BreakoutStrategy',
    'GridStrategy',
    'SwingStrategy'
]
