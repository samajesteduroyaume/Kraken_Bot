"""
Module de gestion des risques pour le bot de trading Kraken.

Ce package contient les composants pour la gestion des risques, y compris la gestion du levier,
le calcul de la taille des positions, et d'autres fonctionnalités liées à la gestion du risque.
"""
from .leverage_manager import LeverageManager, LeverageStrategy

__all__ = ['LeverageManager', 'LeverageStrategy']
