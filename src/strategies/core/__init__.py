"""
Module principal pour les stratégies de trading unifiées.

Ce package contient les implémentations des stratégies de trading avancées,
construites sur une architecture commune et modulaire.
"""

from .trend_following import TrendFollowingStrategy
from .mean_reversion import MeanReversionStrategy

__all__ = ['TrendFollowingStrategy', 'MeanReversionStrategy']
