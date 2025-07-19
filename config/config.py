"""
Configuration module for the trading bot.
"""

import os
from dataclasses import dataclass, field
from typing import Dict, Any

def get_database_config():
    """Retourne la configuration de la base de données."""
    return {
        'host': 'localhost',
        'port': 5432,
        'name': 'kraken_bot',
        'user': os.getenv('POSTGRES_USER', 'kraken_bot'),
        'password': os.getenv('POSTGRES_PASSWORD', '')
    }

@dataclass
class APIConfig:
    """Configuration de l'API Kraken."""
    key: str = ""
    secret: str = ""
    timeout: int = 30

@dataclass
class TradingConfig:
    """Configuration du trading."""
    max_pairs: int = 50
    min_volume: float = 100000
    volatility_window: int = 20
    momentum_window: int = 14
    max_trade_amount: float = 100.0
    min_trade_amount: float = 10.0
    initial_balance: float = 10000.0
    risk_percentage: float = 1.0
    max_daily_drawdown: float = 5.0
    max_leverage: float = 3.0
    simulation_mode: bool = False
    simulation_balance_btc: float = 0.0
    simulation_balance_eur: float = 1000.0
    
    def __post_init__(self):
        # Validation des valeurs
        if self.max_pairs < 1:
            self.max_pairs = 1
        if self.min_volume < 0:
            self.min_volume = 0
        if self.volatility_window < 1:
            self.volatility_window = 1
        if self.momentum_window < 1:
            self.momentum_window = 1
        if self.max_trade_amount < 0:
            self.max_trade_amount = 100.0
        if self.min_trade_amount < 0:
            self.min_trade_amount = 10.0
        if self.risk_percentage < 0 or self.risk_percentage > 100:
            self.risk_percentage = 1.0
        if self.max_daily_drawdown < 0 or self.max_daily_drawdown > 100:
            self.max_daily_drawdown = 5.0
        if self.max_leverage < 1:
            self.max_leverage = 1.0

@dataclass
class LoggingConfig:
    """Configuration du logging."""
    level: str = field(default_factory=lambda: os.getenv('LOG_LEVEL', 'INFO').upper())
    format: str = field(default_factory=lambda: os.getenv('LOG_FORMAT', '%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    log_dir: str = field(default_factory=lambda: os.path.expanduser(os.getenv('LOG_DIR', '~/kraken_bot_logs')))
    max_bytes: int = field(default_factory=lambda: int(os.getenv('LOG_MAX_SIZE', '10485760')))
    backup_count: int = field(default_factory=lambda: int(os.getenv('LOG_BACKUP_COUNT', '5')))
    
    def __post_init__(self):
        """Validation des valeurs après l'initialisation."""
        # Validation du niveau de log
        if self.level not in ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']:
            self.level = 'INFO'
        
        # Création du répertoire de logs
        os.makedirs(self.log_dir, exist_ok=True)
        
        # Validation de la taille maximale
        if self.max_bytes < 1024 * 1024:  # Minimum 1MB
            self.max_bytes = 1024 * 1024
        
        # Validation du nombre de backups
        if self.backup_count < 1:
            self.backup_count = 1

@dataclass
class Config:
    """Main configuration class."""
    db_config: Dict[str, Any] = field(default_factory=get_database_config)
    api: APIConfig = field(default_factory=lambda: APIConfig(
        key=os.getenv('KRAKEN_API_KEY', ''),
        secret=os.getenv('KRAKEN_API_SECRET', ''),
        timeout=int(os.getenv('KRAKEN_API_TIMEOUT', '30'))
    ))
    trading: TradingConfig = field(default_factory=lambda: TradingConfig(
        max_pairs=int(os.getenv('MAX_PAIRS', '50')),
        min_volume=float(os.getenv('MIN_VOLUME', '100000')),
        volatility_window=int(os.getenv('VOLATILITY_WINDOW', '20')),
        momentum_window=int(os.getenv('MOMENTUM_WINDOW', '14'))
    ))
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    
    @property
    def LOGGING(self):
        """Configuration logging."""
        return {
            'level': os.getenv('LOG_LEVEL', 'INFO'),
            'dir': os.getenv('LOG_DIR', '~/kraken_bot_logs'),
            'format': self.logging.format,
            'max_bytes': self.logging.max_bytes,
            'backup_count': self.logging.backup_count
        }
    
    @property
    def BACKUP(self):
        """Configuration backup."""
        return {
            'dir': os.getenv('BACKUP_DIR', '~/kraken_bot_backups'),
            'interval': os.getenv('BACKUP_INTERVAL', 'daily'),
            'max_files': int(os.getenv('BACKUP_MAX_FILES', '30'))
        }
    
    @property
    def REPORTING(self):
        """Configuration reporting."""
        return {
            'dir': os.getenv('REPORT_DIR', '~/kraken_bot_reports'),
            'format': os.getenv('REPORT_FORMAT', 'html'),
            'max_files': int(os.getenv('REPORT_MAX_FILES', '30'))
        }
    
    @property
    def EMAIL_REPORTING(self):
        """Configuration email reporting."""
        return {
            'enabled': os.getenv('EMAIL_REPORTING_ENABLED', 'false').lower() == 'true',
            'smtp_host': os.getenv('SMTP_HOST', 'localhost'),
            'smtp_port': int(os.getenv('SMTP_PORT', '587')),
            'smtp_user': os.getenv('SMTP_USER', ''),
            'smtp_password': os.getenv('SMTP_PASSWORD', ''),
            'from_email': os.getenv('FROM_EMAIL', ''),
            'to_emails': os.getenv('TO_EMAILS', '').split(',')
        }
    
    @property
    def REDIS(self):
        """Configuration Redis."""
        return {
            'enabled': os.getenv('REDIS_ENABLED', 'false').lower() == 'true',
            'host': os.getenv('REDIS_HOST', 'localhost'),
            'port': int(os.getenv('REDIS_PORT', '6379')),
            'db': int(os.getenv('REDIS_DB', '0')),
            'password': os.getenv('REDIS_PASSWORD'),
            'ttl': int(os.getenv('REDIS_TTL', '3600'))
        }
    
    @property
    def redis_config(self):
        """Alias pour la configuration Redis (compatibilité)."""
        return self.REDIS
    
    @property
    def AWS(self):
        """Configuration AWS."""
        return {
            'enabled': os.getenv('AWS_ENABLED', 'false').lower() == 'true',
            'access_key': os.getenv('AWS_ACCESS_KEY_ID'),
            'secret_key': os.getenv('AWS_SECRET_ACCESS_KEY'),
            'region': os.getenv('AWS_REGION'),
            'bucket': os.getenv('AWS_BUCKET_NAME')
        }
    
    def __post_init__(self):
        """Initialisation des valeurs après la création de l'objet."""
        # Assurez-vous que le répertoire de logs existe
        os.makedirs(self.logging.log_dir, exist_ok=True)
        # Assurez-vous que le répertoire de backup existe
        os.makedirs(os.path.expanduser(self.BACKUP.get('dir', '~/kraken_bot_backups')), exist_ok=True)
        # Assurez-vous que le répertoire de rapports existe
        os.makedirs(os.path.expanduser(self.REPORTING.get('dir', '~/kraken_bot_reports')), exist_ok=True)
        
    def load_from_env(self, env: Dict[str, str]):
        """Load configuration from environment variables."""
        # Update database configuration
        self.db_config['host'] = env.get('POSTGRES_HOST', self.db_config.get('host', 'localhost'))
        self.db_config['port'] = int(env.get('POSTGRES_PORT', str(self.db_config.get('port', 5432))))
        self.db_config['name'] = env.get('POSTGRES_DB', self.db_config.get('name', 'kraken_bot'))
        self.db_config['user'] = env.get('POSTGRES_USER', self.db_config.get('user', 'kraken_bot'))
        self.db_config['password'] = env.get('POSTGRES_PASSWORD', self.db_config.get('password', ''))
        
        # Update API configuration
        self.api.key = env.get('KRAKEN_API_KEY', self.api.key)
        self.api.secret = env.get('KRAKEN_API_SECRET', self.api.secret)
        self.api.timeout = int(env.get('KRAKEN_API_TIMEOUT', str(self.api.timeout)))
