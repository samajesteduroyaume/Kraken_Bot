"""
Package ml - Module d'apprentissage automatique pour le bot de trading Kraken.

Ce package fournit des fonctionnalités de prédiction de tendance de marché en utilisant
divers algorithmes d'apprentissage automatique, ainsi que des outils pour le prétraitement
des données et l'évaluation des modèles.

Classes principales:
    MLPredictor: Classe principale pour la prédiction de tendance de marché.
    MarketPredictor: Alias pour MLPredictor (rétrocompatibilité).

Sous-modules:
    core: Classes de base et interfaces pour les modèles
    models: Implémentations des modèles
    utils: Utilitaires pour le prétraitement et l'évaluation
    config: Configuration des modèles
"""

from .predictor import MLPredictor

# Alias pour la compatibilité avec le code existant
MarketPredictor = MLPredictor

__all__ = [
    'MLPredictor',
    'MarketPredictor'
]
