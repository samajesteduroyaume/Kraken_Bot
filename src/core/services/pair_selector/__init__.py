"""
Module principal du sélecteur de paires de trading.

Ce package fournit des fonctionnalités pour analyser, valider et sélectionner des paires 
de trading basées sur divers indicateurs techniques et métriques de marché.
"""

from .core import PairSelector
from .models import PairAnalysis
from .validators import validate_pair_format

__all__ = [
    'PairSelector',
    'PairAnalysis',
    'validate_pair_format',
]
