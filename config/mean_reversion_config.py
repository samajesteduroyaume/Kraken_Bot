"""
Configuration pour la stratégie de retour à la moyenne (Mean Reversion).

Ce fichier contient tous les paramètres de configuration pour la stratégie de mean reversion,
y compris les paramètres des indicateurs techniques, la gestion des risques, et les paramètres
de trading.
"""
from decimal import Decimal

# Configuration de base de la stratégie
MEAN_REVERSION_STRATEGY_CONFIG = {
    # Paramètres généraux
    "strategy_name": "MeanReversionStrategy",
    "enabled": True,
    "is_real_time": True,
    "timeframes": ["15m", "1h", "4h", "1d"],
    "symbols": ["BTC/USD", "ETH/USD"],
    
    # Paramètres des indicateurs
    "indicators": {
        # Bandes de Bollinger
        "bollinger_bands": {
            "window": 20,
            "window_dev": 2.0,
            "overbought_threshold": 0.95,  # Niveau supérieur des bandes pour surachat
            "oversold_threshold": 0.05     # Niveau inférieur des bandes pour survente
        },
        
        # Indice de Force Relative (RSI)
        "rsi": {
            "period": 14,
            "overbought": 70,
            "oversold": 30
        },
        
        # Moyenne Mobile Simple (SMA)
        "sma": {
            "short_period": 20,
            "medium_period": 50,
            "long_period": 200
        },
        
        # Taux de Changement (ROC)
        "roc": {
            "period": 14,
            "threshold": 2.0
        },
        
        # Indicateur de Développement de la Tendance (ADX)
        "adx": {
            "period": 14,
            "adx_threshold": 25  # Seuil minimum de tendance
        },
        
        # Bande de Keltner
        "keltner_channels": {
            "ema_period": 20,
            "atr_period": 10,
            "multiplier": 2.0
        },
        
        # Indice de Flux de Marché (MFI)
        "mfi": {
            "period": 14,
            "overbought": 80,
            "oversold": 20
        }
    },
    
    # Paramètres de génération de signaux
    "signal_generation": {
        "required_timeframe_confirmation": 2,  # Nombre de timeframes nécessaires pour confirmer un signal
        "min_signal_strength": 0.5,           # Force minimale du signal pour prendre une position
        
        # Filtres
        "volume_filter": {
            "enabled": True,
            "min_volume_btc": 1.0,    # Volume minimum en BTC pour considérer un signal
            "volume_period": 20       # Période pour le calcul du volume moyen
        },
        
        "trend_filter": {
            "enabled": True,
            "min_trend_strength": 0.5  # Force minimale de la tendance
        },
        
        "volatility_filter": {
            "enabled": True,
            "min_atr_percent": 0.5,   # Volatilité minimale (en % du prix)
            "max_atr_percent": 5.0    # Volatilité maximale (en % du prix)
        }
    },
    
    # Paramètres de gestion du risque
    "risk_management": {
        "max_position_size_pct": 5.0,  # Pourcentage maximum du portefeuille par trade
        "max_drawdown_pct": 10.0,      # Drawdown maximum autorisé
        "stop_loss_type": "atr",       # Type de stop-loss: 'fixed', 'atr', 'percentage'
        "stop_loss_atr_multiplier": 2.0,
        "stop_loss_pct": 2.0,          # Si type = 'percentage'
        "take_profit_ratio": 2.0,      # Ratio take-profit/stop-loss
        "trailing_stop": {
            "enabled": True,
            "activation_pct": 1.0,     # Pourcentage de profit pour activer le trailing stop
            "trail_pct": 0.5          # Pourcentage de suivi du prix
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
        "log_file": "mean_reversion_strategy.log"
    },
    
    # Paramètres avancés
    "advanced": {
        "use_ml": False,              # Activer l'intégration du machine learning
        "ml_model_path": "models/mean_reversion_model.pkl",
        "use_heikin_ashi": False,     # Utiliser les chandeliers Heikin-Ashi
        "use_renko": False,           # Utiliser les briques Renko
        "use_volume_profile": False,  # Utiliser le profil de volume
        "use_order_flow": False       # Utiliser l'analyse du flux d'ordres
    }
}
