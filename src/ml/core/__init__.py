"""
Package core - Contient les classes de base et les interfaces pour les modèles ML.
"""

from .base_model import BaseModel
from .model_factory import ModelFactory
from .predictor import MLPredictor

__all__ = ['BaseModel', 'ModelFactory', 'MLPredictor']
