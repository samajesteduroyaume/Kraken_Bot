"""
Package ml - Contient les composants d'apprentissage automatique.
"""

from .predictor import MLPredictor
from .trainer import train_model

# Alias pour la compatibilité avec le code existant
MarketPredictor = MLPredictor

__all__ = ['MLPredictor', 'MarketPredictor', 'train_model']
