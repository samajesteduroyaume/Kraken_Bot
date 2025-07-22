"""
Configuration des stratégies de trading.

Ce module contient les configurations par défaut pour les différentes stratégies
de trading implémentées dans le système.
"""
from typing import Dict, Any

# Configuration de base commune à toutes les stratégies
BASE_CONFIG = {
    'portfolio_value': 10000.0,
    'risk_per_trade': 0.01,  # 1% du portefeuille par trade
    'max_position_size': 0.15,  # 15% du portefeuille max par position
    'timeframes': ['15m', '1h', '4h', '1d'],
    'real_time': True,
    'volatility_adjusted_risk': True,
    'dynamic_position_sizing': True,
    'max_drawdown': 0.10,  # 10% de drawdown maximum
}

# Configuration pour la stratégie de suivi de tendance
TREND_FOLLOWING_CONFIG = {
    **BASE_CONFIG,
    'strategy_type': 'trend_following',
    'rsi_period': 14,
    'rsi_overbought': 70,
    'rsi_oversold': 30,
    'macd_fast': 12,
    'macd_slow': 26,
    'macd_signal': 9,
    'atr_period': 14,
    'bb_period': 20,
    'bb_std': 2.0,
}

# Configuration pour la stratégie de retour à la moyenne
MEAN_REVERSION_CONFIG = {
    **BASE_CONFIG,
    'strategy_type': 'mean_reversion',
    'rsi_period': 14,
    'rsi_overbought': 70,
    'rsi_oversold': 30,
    'stoch_period': 14,
    'stoch_overbought': 80,
    'stoch_oversold': 20,
    'bb_period': 20,
    'bb_std': 2.0,
}

def get_config(strategy_type: str) -> Dict[str, Any]:
    """
    Récupère la configuration par défaut pour un type de stratégie.
    
    Args:
        strategy_type: Type de stratégie ('trend_following' ou 'mean_reversion')
        
    Returns:
        Dictionnaire de configuration
    """
    configs = {
        'trend_following': TREND_FOLLOWING_CONFIG,
        'mean_reversion': MEAN_REVERSION_CONFIG,
    }
    return configs.get(strategy_type, BASE_CONFIG).copy()
