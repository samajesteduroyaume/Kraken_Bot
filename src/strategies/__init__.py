"""
Package contenant les différentes stratégies de trading pour le bot Kraken.

Ce package a été réorganisé avec une architecture plus modulaire :
- core/ : Contient les implémentations des stratégies principales
- indicators/ : Contient les indicateurs techniques réutilisables
- config/ : Contient les configurations des stratégies
"""
from .base_strategy import BaseStrategy
from .core import TrendFollowingStrategy, MeanReversionStrategy
from .config import get_config

__all__ = [
    'BaseStrategy',
    'TrendFollowingStrategy',
    'MeanReversionStrategy',
    'get_config'
]
