"""
Package config - Contient les configurations pour les modèles ML.
"""

from .model_config import (
    RANDOM_FOREST_DEFAULT_PARAMS,
    XGBOOST_DEFAULT_PARAMS,
    NEURAL_NETWORK_DEFAULT_PARAMS,
    LSTM_DEFAULT_PARAMS
)

__all__ = [
    'RANDOM_FOREST_DEFAULT_PARAMS',
    'XGBOOST_DEFAULT_PARAMS',
    'NEURAL_NETWORK_DEFAULT_PARAMS',
    'LSTM_DEFAULT_PARAMS'
]
