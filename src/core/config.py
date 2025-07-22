from typing import Dict, Any
import os
from dotenv import load_dotenv
import logging


def load_env():
    current = os.path.abspath(os.path.dirname(__file__))
    while True:
        env_path = os.path.join(current, '.env')
        if os.path.exists(env_path):
            load_dotenv(env_path, override=True)
            break
        parent = os.path.dirname(current)
        if parent == current:
            break
        current = parent


load_env()


class Config:
    """Configuration du bot de trading."""

    def __init__(self):
        """Initialise la configuration."""
        self.logger = logging.getLogger('config')
        self.load_environment()

    def __getitem__(self, key):
        """Permet d'accéder aux configurations comme à un dictionnaire."""
        # Recherche dans toutes les configurations
        for config in [
            'api_config',
            'trading_config',
            'db_config',
            'ml_config',
                'log_config']:
            config_dict = getattr(self, config, {})
            if key in config_dict:
                return config_dict[key]

        # Si la clé n'est pas trouvée, vérifier si c'est un attribut direct
        if hasattr(self, key):
            return getattr(self, key)

        raise KeyError(f"Clé non trouvée: {key}")

    def get(self, key, default=None):
        """Méthode get compatible avec les dictionnaires."""
        try:
            return self[key]
        except KeyError:
            return default

    def load_environment(self):
        """Charge les variables d'environnement."""
        try:
            # Charger les variables d'environnement depuis .env
            env_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), '.env')
            load_dotenv(env_path)
            self.logger.info(f"Configuration chargée depuis: {env_path}. POSTGRES_USER={os.environ.get('POSTGRES_USER')}")

            # Configuration reporting
            self.reporting_config = {
                'enabled': os.getenv('REPORTING_ENABLED', 'true').lower() == 'true',
                'report_dir': os.getenv('REPORT_DIR', 'reports'),
                'email_enabled': os.getenv('REPORTING_EMAIL_ENABLED', 'false').lower() == 'true',
                'email_recipients': os.getenv('REPORTING_EMAIL_RECIPIENTS', '').split(','),
                'daily_report': os.getenv('REPORTING_DAILY', 'true').lower() == 'true',
                'weekly_report': os.getenv('REPORTING_WEEKLY', 'true').lower() == 'true',
                'monthly_report': os.getenv('REPORTING_MONTHLY', 'true').lower() == 'true'
            }

            # Configuration credentials
            self.credentials = {
                'api_key': os.getenv('KRAKEN_API_KEY'),
                'api_secret': os.getenv('KRAKEN_API_SECRET')
            }

            # Configuration API Kraken
            self.api_config = {
                'api_key': os.getenv('KRAKEN_API_KEY'),
                'api_secret': os.getenv('KRAKEN_API_SECRET'),
                'base_url': os.getenv(
                    'KRAKEN_API_URL',
                    'https://api.kraken.com'),
                'version': os.getenv(
                    'KRAKEN_API_VERSION',
                    'v0')}

            # Configuration de trading
            self.trading_config = {
                'risk_per_trade': float(os.getenv('RISK_PER_TRADE', '0.02')),
                'stop_loss_percent': float(os.getenv('STOP_LOSS_PERCENT', '0.02')),
                'take_profit_percent': float(os.getenv('TAKE_PROFIT_PERCENT', '0.04')),
                'max_positions': int(os.getenv('MAX_POSITIONS', '5')),
                'max_drawdown': float(os.getenv('MAX_DRAWDOWN', '0.1')),
                # 24h en secondes
                'max_holding_time': int(os.getenv('MAX_HOLDING_TIME', '86400'))
            }

            # Configuration de la base de données
            self.db_config = {
                'user': os.getenv('POSTGRES_USER', 'postgres'),
                'password': os.getenv('POSTGRES_PASSWORD', ''),
                'name': os.getenv('POSTGRES_DB', 'kraken_bot'),
                'host': os.getenv('POSTGRES_HOST', 'localhost'),
                'port': int(os.getenv('POSTGRES_PORT', '5432'))
            }

            # Configuration ML
            self.ml_config = {
                'window_size': int(os.getenv('ML_WINDOW_SIZE', '20')),
                'train_size': float(os.getenv('ML_TRAIN_SIZE', '0.8')),
                'n_estimators': int(os.getenv('ML_N_ESTIMATORS', '100')),
                'max_depth': int(os.getenv('ML_MAX_DEPTH', '10'))
            }

            # Configuration des logs
            self.log_config = {
                'log_dir': os.getenv('LOG_DIR', 'logs'),
                # 10MB
                'max_bytes': int(os.getenv('LOG_MAX_BYTES', '10485760')),
                'backup_count': int(os.getenv('LOG_BACKUP_COUNT', '5'))
            }

            # Vérifier les configurations
            self.validate_config()

        except Exception as e:
            self.logger.error(
                f"Erreur lors du chargement de la configuration: {str(e)}")
            raise

    def validate_config(self):
        """Valide la configuration."""
        # Vérifier les configurations requises
        required_configs = {
            'api_config': ['api_key', 'api_secret'],
            'db_config': ['user', 'password', 'name', 'host', 'port']
        }

        for config_name, required_fields in required_configs.items():
            config = getattr(self, f"{config_name}", {})
            missing = [
                field for field in required_fields if not config.get(field)]
            if missing:
                raise ValueError(
                    f"Configuration manquante dans {config_name}: {', '.join(missing)}")

        # Vérifier les valeurs numériques
        numeric_configs = {
            'trading_config': [
                'risk_per_trade',
                'stop_loss_percent',
                'take_profit_percent',
                'max_positions',
                'max_drawdown',
                'max_holding_time'],
            'ml_config': [
                'window_size',
                'train_size',
                'n_estimators',
                'max_depth'],
            'log_config': [
                'max_bytes',
                'backup_count']}

        for config_name, fields in numeric_configs.items():
            config = getattr(self, f"{config_name}", {})
            for field in fields:
                value = config.get(field)
                if value is None or (
                    isinstance(
                        value,
                        str) and not value.strip()):
                    raise ValueError(
                        f"Valeur manquante pour {field} dans {config_name}")

        # Vérifier les valeurs
        if self.trading_config['risk_per_trade'] <= 0 or self.trading_config['risk_per_trade'] > 1:
            raise ValueError("Le risque par trade doit être entre 0 et 1")

        if self.trading_config['stop_loss_percent'] <= 0:
            raise ValueError("Le stop loss doit être supérieur à 0")

        if self.trading_config['max_positions'] <= 0:
            raise ValueError(
                "Le nombre maximum de positions doit être supérieur à 0")

    def get_config(self, section: str) -> Dict[str, Any]:
        """
        Récupère une section de configuration.

        Args:
            section: Section de configuration (api, trading, db, ml, log, reporting)

        Returns:
            Dictionnaire de configuration
        """
        try:
            if section == 'api':
                return self.api_config
            elif section == 'trading':
                return self.trading_config
            elif section == 'db':
                return self.db_config
            elif section == 'ml':
                return self.ml_config
            elif section == 'log':
                return self.log_config
            elif section == 'reporting':
                return self.reporting_config
            else:
                raise ValueError(
                    f"Section de configuration invalide: {section}")

        except Exception as e:
            self.logger.error(
                f"Erreur lors de la récupération de la configuration: {str(e)}")
            raise

    @property
    def redis_config(self):
        """Configuration Redis."""
        return {
            'enabled': os.getenv('REDIS_ENABLED', 'true').lower() == 'true',
            'host': os.getenv('REDIS_HOST', 'localhost'),
            'port': int(os.getenv('REDIS_PORT', '6379')),
            'db': int(os.getenv('REDIS_DB', '0')),
            'password': os.getenv('REDIS_PASSWORD', None),
            'ttl': int(os.getenv('REDIS_TTL', '3600'))
        }
