"""
Configuration pour la stratégie de Swing Trading.
"""
from typing import Dict, Any, List

def get_swing_config() -> Dict[str, Any]:
    """
    Retourne la configuration par défaut pour la stratégie de Swing Trading.
    
    Returns:
        Dict[str, Any]: Configuration de la stratégie
    """
    return {
        'symbol': 'BTC/EUR',
        'timeframes': ['15m', '1h', '4h'],
        'indicators': {
            'rsi': {'period': 14},
            'macd': {'fast_period': 12, 'slow_period': 26, 'signal_period': 9},
            'atr': {'period': 14, 'multiplier': 2.0},
            'sma': {'short_period': 50, 'long_period': 200},
            'ema': {'short_period': 9, 'medium_period': 21, 'long_period': 50},
            'bollinger_bands': {'window': 20, 'window_dev': 2.0},
            'mfi': {'period': 14},
            'adx': {'period': 14},
            'roc': {'period': 14},
            'stoch_rsi': {'rsi_period': 14, 'stoch_period': 14, 'k': 3, 'd': 3}
        },
        'signal_generation': {
            'required_timeframe_confirmation': 2,
            'min_signal_strength': 0.6,
            'volume_filter': {'enabled': True, 'min_volume_ratio': 1.2},
            'trend_filter': {'enabled': True, 'min_trend_strength': 0.6},
            'volatility_filter': {'enabled': True, 'max_atr_ratio': 0.02}
        },
        'risk_management': {
            'stop_loss_atr_multiplier': 2.0,
            'take_profit_ratio': 2.0,
            'max_position_size': 0.15,
            'risk_per_trade': 0.01,
            'max_drawdown': 0.1,
            'trailing_stop': {'enabled': True, 'activation': 0.5, 'distance': 0.3}
        },
        'entry_rules': {
            'rsi_overbought': 70,
            'rsi_oversold': 30,
            'macd_histogram_threshold': 0.0,
            'price_above_ema': True,
            'volume_increase': 1.5
        },
        'exit_rules': {
            'trailing_stop': True,
            'profit_target': 2.0,
            'stop_loss': 1.0,
            'time_exit': '4h'  # Sortie après une période définie
        },
        'position_sizing': {
            'method': 'fixed_fractional',
            'max_risk_per_trade': 0.01,
            'max_portfolio_risk': 0.1,
            'max_position_size': 0.15
        },
        'advanced': {
            'use_multiple_timeframes': True,
            'use_volume_profile': True,
            'use_order_flow': False,
            'use_market_profile': False,
            'use_machine_learning': False,
            'ml_model_path': 'models/swing_model.pkl'
        },
        'notifications': {
            'enabled': True,
            'on_signal': True,
            'on_entry': True,
            'on_exit': True,
            'on_stop_loss': True,
            'on_take_profit': True
        },
        'backtest': {
            'initial_balance': 10000,
            'commission': 0.001,
            'slippage': 0.0005,
            'start_date': '2023-01-01',
            'end_date': '2023-12-31',
            'warmup_period': 100
        }
    }
