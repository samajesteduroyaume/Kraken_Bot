"""
Package backtesting - Système de backtesting pour les stratégies de trading.
"""

from .backtester import Backtester

__all__ = [
    'Backtester',
    'MarketData',
    'TradeSignal',
    'Trade',
    'Position',
    'TradingConfig',
    'TradingMetrics']
