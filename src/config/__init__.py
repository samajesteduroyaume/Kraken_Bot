"""
Package de configuration - Contient les fichiers de configuration du projet.
"""

from .trading_pairs_config import (
    TradingPairConfig,
    CorrelationAnalyzer,
    get_recommended_pairs,
    filter_correlated_pairs,
    get_pair_config,
    MAJOR_PAIRS,
    ALL_PAIR_SYMBOLS
)

__all__ = [
    'DB_CONFIG',
    'KRAKEN_CONFIG',
    'ML_CONFIG',
    'LOG_CONFIG',
    'TradingPairConfig',
    'CorrelationAnalyzer',
    'get_recommended_pairs',
    'filter_correlated_pairs',
    'get_pair_config',
    'MAJOR_PAIRS',
    'ALL_PAIR_SYMBOLS',
    'ALTCOINS',
    'DEFI_PAIRS'
]
