"""
Configuration pour la stratégie de momentum avancée.
"""
from decimal import Decimal

# Configuration de base de la stratégie
MOMENTUM_STRATEGY_CONFIG = {
    # Paramètres généraux
    'name': 'AdvancedMomentum',
    'enabled': True,
    'risk_per_trade': 0.01,  # 1% de risque par trade
    'max_position_size': 0.15,  # 15% du portefeuille max par position
    'portfolio_value': 100000.0,  # Valeur initiale du portefeuille
    'timeframes': ['15m', '1h', '4h', '1d'],  # Timeframes à analyser
    
    # Gestion du risque avancée
    'volatility_adjusted_risk': True,
    'dynamic_position_sizing': True,
    'max_drawdown': 0.10,  # 10% de drawdown maximum
    'risk_reward_ratio': 2.0,  # Ratio risque/rendement cible
    
    # Configuration des indicateurs
    'rsi': {
        'period': 14,
        'overbought': 70,
        'oversold': 30,
        'threshold': 50
    },
    'macd': {
        'fast': 12,
        'slow': 26,
        'signal': 9
    },
    'ema': {
        'fast': 9,
        'medium': 21,
        'slow': 50
    },
    'sma': {
        'fast': 20,
        'medium': 50,
        'slow': 200
    },
    'bbands': {
        'window': 20,
        'window_dev': 2.0
    },
    'ichimoku': {
        'conversion': 9,
        'base': 26,
        'lagging_span': 52,
        'displacement': 26
    },
    'stoch': {
        'k': 14,
        'd': 3,
        'smooth': 3
    },
    'adx': {
        'window': 14,
        'threshold': 25
    },
    'atr': {
        'window': 14
    },
    'roc': {
        'period': 14
    },
    'volume_ma': {
        'period': 20
    },
    'mfi': {
        'window': 14
    },
    'kc': {
        'window': 20,
        'window_atr': 10,
        'mult': 2.0
    },
    'vwap': {
        'window': 14
    },
    
    # Paramètres de confirmation
    'confirmation_period': 3,
    'trend_strength_threshold': 0.7,
    'min_volume_multiplier': 1.0,
    'required_confirmations': 2,
    'min_adx_strength': 25,
    
    # Paramètres de trading
    'stop_loss_pct': 0.02,  # 2% de stop loss
    'take_profit_pct': 0.04,  # 4% de take profit
    'trailing_stop': True,
    'trailing_stop_pct': 0.01,  # 1% de trailing stop
    'max_trades_per_day': 5,
    'max_open_trades': 3,
    
    # Paramètres de backtest
    'initial_balance': 10000.0,
    'commission': 0.001,  # 0.1% de commission par trade
    'slippage': 0.0005,  # 0.05% de slippage
}
