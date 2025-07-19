from typing import Dict, Any


class ModuleConfig:
    """Configuration des modules du bot."""

    @staticmethod
    def get_module_config() -> Dict[str, Dict[str, Any]]:
        """Retourne la configuration des modules."""
        return {
            "database": {
                "enabled": True,
                "connection": "postgresql",
                "pool_size": 10,
                "retry_attempts": 3,
                "retry_delay": 5
            },
            "api": {
                "enabled": True,
                "retry_attempts": 3,
                "retry_delay": 5,
                "timeout": 30,
                "rate_limit": {
                    "enabled": True,
                    "window": 60,
                    "limit": 100
                }
            },
            "ml": {
                "enabled": True,
                "model_dir": "ml_models",
                "cache_dir": "ml_cache",
                "training": {
                    "batch_size": 32,
                    "epochs": 100,
                    "validation_split": 0.2
                },
                "prediction": {
                    "batch_size": 1,
                    "window_size": 60,
                    "features": ["close", "volume", "rsi", "macd"]
                }
            },
            "trading": {
                "enabled": True,
                "backtesting": {
                    "enabled": False,
                    "start_date": "2023-01-01",
                    "end_date": "2023-12-31",
                    "initial_balance": 10000.0
                },
                "live": {
                    "enabled": True,
                    "max_positions": 5,
                    "max_leverage": 5.0,
                    "position_adjustment": "dynamic"
                }
            },
            "monitoring": {
                "enabled": True,
                "metrics": {
                    "enabled": True,
                    "interval": 60,
                    "providers": ["prometheus", "grafana"]
                },
                "alerts": {
                    "enabled": True,
                    "providers": ["slack", "email"],
                    "thresholds": {
                        "error_rate": 0.05,
                        "latency": 1000,
                        "memory_usage": 0.8
                    }
                }
            },
            "logging": {
                "enabled": True,
                "level": "INFO",
                "handlers": ["console", "file", "json", "error"],
                "rotation": {
                    "enabled": True,
                    "interval": "daily",
                    "keep_days": 30
                }
            }
        }
