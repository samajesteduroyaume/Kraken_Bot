"""
Package de trading avancé avec séparation des responsabilités.

Ce package contient les modules spécialisés pour le trading algorithmique :
- signals : Génération et gestion des signaux de trading
- execution : Exécution des ordres et gestion des positions
- analysis : Analyse technique et indicateurs
- risk : Gestion des risques et du levier
"""

from .base_trader import BaseTrader
from .signals import SignalGenerator
from .execution import OrderExecutor
from .analysis import MarketAnalyzer
from .risk import RiskManager
from .advanced_trader import AdvancedAITrader as TradingManager

__all__ = [
    'BaseTrader',
    'SignalGenerator',
    'OrderExecutor',
    'MarketAnalyzer',
    'RiskManager',
    'TradingManager']
