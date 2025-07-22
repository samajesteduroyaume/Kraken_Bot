"""
Adaptateur pour la compatibilité avec l'ancien système de configuration.

Ce module fournit une interface de compatibilité qui permet d'utiliser
la nouvelle configuration YAML avec l'ancien système basé sur des classes.
"""
import os
from typing import Any, Dict, Optional
from pathlib import Path
import sys

# Ajouter le répertoire racine au PYTHONPATH
root_dir = str(Path(__file__).parent.parent.parent)
if root_dir not in sys.path:
    sys.path.append(root_dir)

from src.config_manager import config as new_config

class ConfigAdapter:
    """Adaptateur pour la compatibilité avec l'ancien système de configuration."""
    
    def __init__(self):
        """Initialise l'adaptateur de configuration."""
        self.logger = self._get_logger()
        self._mapping = self._create_mapping()
        
        # Configurations principales
        self.api_config = self._get_api_config()
        self.trading_config = self._get_trading_config()
        self.db_config = self._get_db_config()
        self.ml_config = self._get_ml_config()
        self.log_config = self._get_log_config()
        self.credentials = self._get_credentials()
        self.reporting_config = self._get_reporting_config()
        self.redis_config = self._get_redis_config()
    
    def _get_logger(self):
        """Initialise et retourne un logger."""
        import logging
        return logging.getLogger('config_adapter')
    
    def _create_mapping(self) -> Dict[str, str]:
        """Crée le mapping entre les anciennes et les nouvelles clés de configuration."""
        return {
            # API Configuration
            'KRAKEN_API_KEY': 'api.api_key',
            'KRAKEN_API_SECRET': 'api.api_secret',
            'KRAKEN_API_URL': 'api.base_url',
            'KRAKEN_API_VERSION': 'api.version',
            
            # Trading Configuration
            'RISK_PER_TRADE': 'risk_management.max_risk_per_trade',
            'STOP_LOSS_PERCENT': 'risk_management.stop_loss_percent',
            'TAKE_PROFIT_PERCENT': 'risk_management.take_profit_percent',
            'MAX_POSITIONS': 'trading.max_positions',
            'MAX_DRAWDOWN': 'risk_management.max_drawdown',
            'MAX_HOLDING_TIME': 'trading.max_holding_time',
            'MAX_CONCURRENT_REQUESTS': 'trading.max_concurrent_requests',
            'ANALYSIS_TIMEOUT': 'trading.analysis_timeout',
            'MAX_PAIRS_TO_ANALYZE': 'trading.max_pairs_to_analyze',
            'EXCLUDE_ILLIQUID': 'trading.exclude_illiquid',
            'MIN_DAILY_TRADES': 'trading.min_daily_trades',
            'MIN_MARKET_CAP': 'trading.min_market_cap',
            'EXCLUDE_STABLECOINS': 'trading.exclude_stablecoins',
            'EXCLUDE_LEVERAGED': 'trading.exclude_leveraged',
            
            # Database Configuration
            'POSTGRES_USER': 'postgres.user',
            'POSTGRES_PASSWORD': 'postgres.password',
            'POSTGRES_HOST': 'postgres.host',
            'POSTGRES_PORT': 'postgres.port',
            'POSTGRES_DB': 'postgres.database',
            
            # Logging Configuration
            'LOG_LEVEL': 'log.level',
            'LOG_FILE': 'log.log_dir',
            
            # Reporting Configuration
            'REPORTING_ENABLED': 'monitoring.email.enabled',
            'REPORT_DIR': 'monitoring.report_dir',
            'REPORTING_EMAIL_ENABLED': 'monitoring.email.enabled',
            'REPORTING_EMAIL_RECIPIENTS': 'monitoring.email.to_emails',
            'REPORTING_DAILY': 'monitoring.alerts.daily_enabled',
            'REPORTING_WEEKLY': 'monitoring.alerts.weekly_enabled',
            'REPORTING_MONTHLY': 'monitoring.alerts.monthly_enabled',
        }
    
    def _get_api_config(self) -> Dict[str, Any]:
        """Retourne la configuration de l'API."""
        return {
            'api_key': new_config.get('api.api_key', ''),
            'api_secret': new_config.get('api.api_secret', ''),
            'base_url': new_config.get('api.base_url', 'https://api.kraken.com'),
            'version': new_config.get('api.version', 'v0'),
            'enable_rate_limit': new_config.get('api.enable_rate_limit', True),
            'timeout': new_config.get('api.timeout', 30000),
        }
    
    def _get_trading_config(self) -> Dict[str, Any]:
        """Retourne la configuration du trading."""
        return {
            'pairs': new_config.get('trading.pairs', []),
            'min_score': new_config.get('trading.min_score', 0.3),
            'min_volume': float(new_config.get('trading.min_volume', 1000000)),
            'risk_per_trade': float(new_config.get('risk_management.max_risk_per_trade', 0.02)),
            'stop_loss_percent': float(new_config.get('risk_management.stop_loss_percent', 0.02)),
            'take_profit_percent': float(new_config.get('risk_management.take_profit_percent', 0.04)),
            'max_positions': int(new_config.get('trading.max_positions', 5)),
            'max_drawdown': float(new_config.get('risk_management.max_drawdown', 0.1)),
            'max_holding_time': int(new_config.get('trading.max_holding_time', 86400)),
            'max_concurrent_requests': int(new_config.get('trading.max_concurrent_requests', 1)),
            'analysis_timeout': float(new_config.get('trading.analysis_timeout', 60.0)),
            'max_pairs_to_analyze': int(new_config.get('trading.max_pairs_to_analyze', 10)),
            'exclude_illiquid': bool(new_config.get('trading.exclude_illiquid', True)),
            'min_daily_trades': int(new_config.get('trading.min_daily_trades', 200)),
            'min_market_cap': float(new_config.get('trading.min_market_cap', 100000000)),
            'exclude_stablecoins': bool(new_config.get('trading.exclude_stablecoins', True)),
            'exclude_leveraged': bool(new_config.get('trading.exclude_leveraged', True)),
        }
    
    def _get_db_config(self) -> Dict[str, Any]:
        """Retourne la configuration de la base de données."""
        return {
            'user': new_config.get('postgres.user', 'kraken_bot'),
            'password': new_config.get('postgres.password', ''),
            'host': new_config.get('postgres.host', 'localhost'),
            'port': int(new_config.get('postgres.port', 5432)),
            'database': new_config.get('postgres.database', 'kraken_bot'),
            'enabled': bool(new_config.get('postgres.enabled', True)),
        }
    
    def _get_ml_config(self) -> Dict[str, Any]:
        """Retourne la configuration du machine learning."""
        return {
            'enabled': bool(new_config.get('ml.enabled', False)),
            'model_path': new_config.get('ml.model_path', 'models/'),
            'window_size': int(new_config.get('ml.window_size', 20)),
            'train_size': float(new_config.get('ml.train_size', 0.8)),
            'test_size': float(new_config.get('ml.test_size', 0.2)),
            'n_estimators': int(new_config.get('ml.n_estimators', 100)),
            'max_depth': int(new_config.get('ml.max_depth', 10)),
            'retrain_interval': int(new_config.get('ml.retrain_interval', 86400)),
        }
    
    def _get_log_config(self) -> Dict[str, Any]:
        """Retourne la configuration des logs."""
        return {
            'level': new_config.get('log.level', 'INFO'),
            'log_dir': new_config.get('log.log_dir', 'logs'),
            'max_bytes': int(new_config.get('log.max_bytes', 10485760)),  # 10MB
            'backup_count': int(new_config.get('log.backup_count', 5)),
            'console': bool(new_config.get('log.console', True)),
            'file': bool(new_config.get('log.file', True)),
            'format': new_config.get('log.format', 
                                   '%(asctime)s - %(name)s - %(levelname)s - %(message)s'),
            'date_format': new_config.get('log.date_format', '%Y-%m-%d %H:%M:%S'),
        }
    
    def _get_credentials(self) -> Dict[str, str]:
        """Retourne les informations d'identification."""
        return {
            'api_key': new_config.get('api.api_key', ''),
            'api_secret': new_config.get('api.api_secret', ''),
        }
    
    def _get_redis_config(self) -> Dict[str, Any]:
        """Récupère la configuration Redis."""
        try:
            return {
                'enabled': new_config.get('redis.enabled', False),
                'host': new_config.get('redis.host', 'localhost'),
                'port': int(new_config.get('redis.port', 6379)),
                'db': int(new_config.get('redis.db', 0)),
                'password': new_config.get('redis.password', ''),
                'socket_timeout': float(new_config.get('redis.socket_timeout', 5.0)),
                'socket_connect_timeout': float(new_config.get('redis.socket_connect_timeout', 5.0)),
                'socket_keepalive': bool(new_config.get('redis.socket_keepalive', True)),
                'retry_on_timeout': bool(new_config.get('redis.retry_on_timeout', True)),
                'max_connections': int(new_config.get('redis.max_connections', 10)),
                'health_check_interval': int(new_config.get('redis.health_check_interval', 30))
            }
        except Exception as e:
            self.logger.error(f"Erreur lors du chargement de la configuration Redis: {e}")
            return {}
    
    def _get_reporting_config(self) -> Dict[str, Any]:
        """Retourne la configuration des rapports."""
        try:
            return {
                'enabled': new_config.get('reporting.enabled', False),
                'report_dir': new_config.get('reporting.report_dir', 'reports'),
                'email_enabled': new_config.get('reporting.email.enabled', False),
                'email_recipients': new_config.get('reporting.email.recipients', []),
                'daily_report': new_config.get('reporting.daily_report', False),
                'weekly_report': new_config.get('reporting.weekly_report', True),
                'monthly_report': new_config.get('reporting.monthly_report', True)
            }
        except Exception as e:
            self.logger.error(f"Erreur lors du chargement de la configuration des rapports: {e}")
            return {}
    
    def get_config(self, key: str, default: Any = None) -> Any:
        """
        Récupère une valeur de configuration par sa clé.
        
        Args:
            key: La clé de configuration à récupérer
            default: Valeur par défaut à retourner si la clé n'existe pas
            
        Returns:
            La valeur de configuration ou la valeur par défaut
        """
        try:
            return self[key]
        except KeyError:
            return default
    
    def __getitem__(self, key: str) -> Any:
        """Permet d'accéder aux configurations comme à un dictionnaire."""
        # Parcourir tous les dictionnaires de configuration
        for config_name in [
            'api_config', 'trading_config', 'db_config', 
            'ml_config', 'log_config', 'credentials', 'reporting_config', 'redis_config'
        ]:
            config_dict = getattr(self, config_name, {})
            if key in config_dict:
                return config_dict[key]
        
        # Si la clé n'est pas trouvée, lever une KeyError
        raise KeyError(f"Configuration '{key}' introuvable")
    
    def get(self, key: str, default: Any = None) -> Any:
        """Méthode get compatible avec les dictionnaires."""
        try:
            return self[key]
        except KeyError:
            return default
    
    def load_environment(self) -> None:
        """Charge les variables d'environnement (compatibilité)."""
        # Cette méthode est conservée pour la compatibilité
        pass
    
    def get_environment_variable(self, var_name: str, default: str = '') -> str:
        """Récupère une variable d'environnement avec une valeur par défaut."""
        # Vérifier d'abord dans le mapping
        mapped_key = self._mapping.get(var_name)
        if mapped_key:
            value = new_config.get(mapped_key)
            if value is not None:
                return str(value)
        
        # Sinon, essayer de récupérer directement la variable d'environnement
        return os.getenv(var_name, default)


# Instance globale pour la compatibilité
Config = ConfigAdapter()
