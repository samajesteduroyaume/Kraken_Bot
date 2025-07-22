"""
Module de configuration principal pour le bot de trading Kraken.

Ce package contient les configurations pour les différentes parties du système,
y compris les stratégies, les paramètres de trading, et la gestion des risques.
"""
from .strategy_config import (
    StrategyType,
    BaseStrategyConfig,
    TrendFollowingConfig,
    MeanReversionConfig,
    MomentumConfig
)
from .strategy_config_impl import StrategyConfig

# Alias pour la rétrocompatibilité
Config = StrategyConfig

__all__ = [
    'Config',  # Alias pour StrategyConfig
    'StrategyType',
    'BaseStrategyConfig',
    'TrendFollowingConfig',
    'MeanReversionConfig',
    'MomentumConfig',
    'StrategyConfig'
]
