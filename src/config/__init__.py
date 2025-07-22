"""
Package de configuration - Contient les fichiers de configuration du projet.
"""

from .trading_pairs_config import (
    get_trading_pairs,
    get_all_trading_pairs,
    initialize
)

# Les configurations de stratégie ont été déplacées dans src/core/config

__all__ = [
    'get_trading_pairs',
    'get_all_trading_pairs',
    'initialize'
]
