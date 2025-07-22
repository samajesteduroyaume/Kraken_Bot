"""
Configuration de test autonome pour la stratégie de momentum.

Ce fichier contient une version autonome de la configuration de la stratégie de momentum
pour les tests unitaires, sans dépendances externes.
"""
from decimal import Decimal
from typing import Dict, Any, List

# Configuration de base de la stratégie
MOMENTUM_STRATEGY_CONFIG = {
    # Paramètres généraux
    "strategy_name": "MomentumStrategy",
    "enabled": True,
    "is_real_time": True,
    "timeframes": ["15m", "1h", "4h", "1d"],
    "symbols": ["BTC/USD", "ETH/USD"],
    
    # Paramètres des indicateurs
    "indicators": {
        "rsi": {
            "period": 14,
            "overbought": 70,
            "oversold": 30
        },
        "macd": {
            "fast_period": 12,
            "slow_period": 26,
            "signal_period": 9
        },
        "ema": {
            "short_period": 9,
            "medium_period": 21,
            "long_period": 50
        },
        "bollinger_bands": {
            "period": 20,
            "std_dev": 2.0
        },
        "adx": {
            "period": 14,
            "adx_threshold": 25
        },
        "atr": {
            "period": 14
        },
        "stochastic": {
            "k_period": 14,
            "d_period": 3,
            "smooth_k": 3,
            "overbought": 80,
            "oversold": 20
        },
        "mfi": {
            "period": 14,
            "overbought": 80,
            "oversold": 20
        },
        "roc": {
            "period": 14,
            "threshold": 2.0
        }
    },
    
    # Paramètres de génération de signaux
    "signal_generation": {
        "required_timeframe_confirmation": 2,
        "min_signal_strength": 0.5,
        "volume_filter": {
            "enabled": True,
            "min_volume_btc": 1.0,
            "volume_period": 20
        },
        "trend_filter": {
            "enabled": True,
            "min_trend_strength": 0.5
        },
        "volatility_filter": {
            "enabled": True,
            "min_atr_percent": 0.5,
            "max_atr_percent": 5.0
        },
        "divergence_filter": {
            "enabled": True,
            "min_divergence_strength": 0.5
        }
    },
    
    # Paramètres de gestion du risque
    "risk_management": {
        "max_position_size_pct": 5.0,  # Pourcentage du portefeuille par trade
        "max_drawdown_pct": 10.0,      # Drawdown maximum autorisé
        "stop_loss_type": "atr",       # Type de stop-loss: 'fixed', 'atr', 'percentage'
        "stop_loss_atr_multiplier": 2.0,
        "stop_loss_pct": 2.0,          # Si type = 'percentage'
        "take_profit_ratio": 2.0,      # Ratio take-profit/stop-loss
        "trailing_stop": {
            "enabled": True,
            "activation_pct": 1.0,
            "trail_pct": 0.5
        },
        "position_sizing": {
            "method": "fixed_fractional",  # 'fixed_size', 'fixed_fractional', 'kelly'
            "risk_per_trade_pct": 1.0,     # Pourcentage du capital à risquer par trade
            "max_leverage": 5.0            # Effet de levier maximum
        }
    },
    
    # Paramètres de backtesting
    "backtest": {
        "initial_balance": 10000.0,
        "commission_pct": 0.1,        # Commission en pourcentage
        "slippage_pct": 0.05,         # Slippage moyen en pourcentage
        "start_date": "2022-01-01",   # Date de début du backtest
        "end_date": "2023-01-01"      # Date de fin du backtest
    },
    
    # Paramètres de journalisation
    "logging": {
        "level": "INFO",
        "log_to_file": True,
        "log_file": "momentum_strategy.log"
    },
    
    # Paramètres avancés
    "advanced": {
        "use_ml": False,              # Activer l'intégration du machine learning
        "ml_model_path": "models/momentum_model.pkl",
        "use_heikin_ashi": False,     # Utiliser les chandeliers Heikin-Ashi
        "use_renko": False,           # Utiliser les briques Renko
        "use_volume_profile": False,   # Utiliser le profil de volume
        "use_order_flow": False       # Utiliser l'analyse du flux d'ordres
    }
}

# Classes utilitaires pour les tests
class TradeSignal:
    """Classe représentant un signal de trading pour les tests."""
    
    def __init__(self, symbol: str, direction: int, strength: float, 
                 price: Decimal, stop_loss: Decimal, take_profit: Decimal):
        self.symbol = symbol
        self.direction = direction  # 1 pour achat, -1 pour vente
        self.strength = strength    # Force du signal (0.0 à 1.0)
        self.price = price
        self.stop_loss = stop_loss
        self.take_profit = take_profit


class SignalStrength:
    """Énumération des forces de signal pour les tests."""
    WEAK = 0.3
    MODERATE = 0.6
    STRONG = 0.9
