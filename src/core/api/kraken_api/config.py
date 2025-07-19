"""
Module de configuration pour l'API Kraken.
"""

from typing import Dict, Any, Optional
import os
import logging
from dotenv import load_dotenv
import json

from .exceptions import ConfigurationError


class KrakenConfig:
    """
    Gestionnaire de configuration pour l'API Kraken.
    """

    DEFAULT_CONFIG = {
        'base_url': 'https://api.kraken.com',
        'version': '0',
        'timeout': 10,
        'max_retries': 5,
        'retry_delay': 0.5,
        'cache_ttl': 60,
        'log_level': 'INFO',
        'rate_limit': {
            'enabled': True,
            'window': 30,
            'limit': 50
        },
        'credentials': {
            'api_key': None,
            'api_secret': None
        },
        'environments': {
            'production': {
                'base_url': 'https://api.kraken.com',
                'version': '0',
                'timeout': 10,
                'max_retries': 5,
                'retry_delay': 0.5,
                'cache_ttl': 60,
                'rate_limit': {
                    'enabled': True,
                    'window': 30,
                    'limit': 50
                }
            },
            'sandbox': {
                'base_url': 'https://api-sandbox.kraken.com',
                'version': '0',
                'timeout': 10,
                'max_retries': 5,
                'retry_delay': 0.5,
                'cache_ttl': 60,
                'rate_limit': {
                    'enabled': True,
                    'window': 30,
                    'limit': 50
                }
            }
        }
    }

    def __init__(
            self,
            env: Optional[str] = None,
            config_file: Optional[str] = None):
        """
        Initialise le gestionnaire de configuration.

        Args:
            env: Nom de l'environnement (production, sandbox, etc.)
            config_file: Chemin vers le fichier de configuration
        """
        self.logger = logging.getLogger(__name__ + '.KrakenConfig')

        # Charger les variables d'environnement
        load_dotenv()

        # Déterminer l'environnement
        self.env = env or os.getenv('KRAKEN_ENV', 'production')
        self.logger.info(f"Using environment: {self.env}")

        # Charger la configuration
        self.config = self._load_config(config_file)

    def _load_config(
            self, config_file: Optional[str] = None) -> Dict[str, Any]:
        """
        Charge la configuration depuis différents sources.

        Args:
            config_file: Chemin vers le fichier de configuration

        Returns:
            Configuration chargée
        """
        # Charger la configuration par défaut
        config = dict(self.DEFAULT_CONFIG)

        # Surcharge avec la configuration de l'environnement spécifique
        if self.env in config['environments']:
            env_config = config['environments'][self.env]
            config.update(env_config)

        # Surcharge avec le fichier de configuration
        if config_file and os.path.exists(config_file):
            try:
                with open(config_file, 'r') as f:
                    file_config = json.load(f)
                    config.update(file_config)
            except Exception as e:
                self.logger.error(
                    f"Erreur lors du chargement du fichier de configuration: {str(e)}")
                raise ConfigurationError(
                    f"Erreur lors du chargement du fichier de configuration: {str(e)}")

        # Surcharge avec les variables d'environnement
        env_vars = {
            'base_url': os.getenv('KRAKEN_BASE_URL'),
            'version': os.getenv('KRAKEN_VERSION'),
            'timeout': os.getenv('KRAKEN_TIMEOUT'),
            'max_retries': os.getenv('KRAKEN_MAX_RETRIES'),
            'retry_delay': os.getenv('KRAKEN_RETRY_DELAY'),
            'cache_ttl': os.getenv('KRAKEN_CACHE_TTL'),
            'log_level': os.getenv('KRAKEN_LOG_LEVEL'),
            'api_key': os.getenv('KRAKEN_API_KEY'),
            'api_secret': os.getenv('KRAKEN_API_SECRET'),
            'rate_limit_enabled': os.getenv('KRAKEN_RATE_LIMIT_ENABLED'),
            'rate_limit_window': os.getenv('KRAKEN_RATE_LIMIT_WINDOW'),
            'rate_limit_limit': os.getenv('KRAKEN_RATE_LIMIT_LIMIT')
        }

        for key, value in env_vars.items():
            if value is not None:
                if key in ['api_key', 'api_secret']:
                    # Injecter les clés API dans la section credentials
                    if 'credentials' not in config:
                        config['credentials'] = {}
                    config['credentials'][key] = value
                elif key in [
                    'timeout',
                    'max_retries',
                    'retry_delay',
                    'cache_ttl',
                    'rate_limit_window',
                    'rate_limit_limit'
                ]:
                    try:
                        config[key] = float(value) if '.' in value else int(value)
                    except ValueError:
                        self.logger.warning(
                            f"Valeur invalide pour {key}: {value}")
                elif key == 'rate_limit_enabled':
                    config['rate_limit']['enabled'] = value.lower() == 'true'
                else:
                    config[key] = value

        # Validation de la configuration
        self._validate_config(config)

        return config

    def _validate_config(self, config: Dict[str, Any]) -> None:
        """
        Valide la configuration.

        Args:
            config: Configuration à valider

        Raises:
            ConfigurationError: Si la configuration est invalide
        """
        required_keys = [
            'base_url',
            'version',
            'timeout',
            'max_retries',
            'retry_delay',
            'cache_ttl',
            'log_level',
            'credentials',
            'rate_limit']

        for key in required_keys:
            if key not in config:
                raise ConfigurationError(f"Configuration manquante: {key}")

        # Validation des types
        if not isinstance(config['timeout'], (int, float)
                          ) or config['timeout'] <= 0:
            raise ConfigurationError("timeout doit être un nombre positif")

        if not isinstance(
                config['max_retries'],
                int) or config['max_retries'] < 0:
            raise ConfigurationError("max_retries doit être un entier positif")

        if not isinstance(
            config['retry_delay'],
            (int,
             float)) or config['retry_delay'] < 0:
            raise ConfigurationError("retry_delay doit être un nombre positif")

        if not isinstance(config['cache_ttl'], (int, float)
                          ) or config['cache_ttl'] <= 0:
            raise ConfigurationError("cache_ttl doit être un nombre positif")

        # Validation des clés API
        if not config['credentials']['api_key'] or not config['credentials']['api_secret']:
            self.logger.warning("Les clés API ne sont pas configurées")

        # Validation du rate limiting
        if not isinstance(config['rate_limit']['enabled'], bool):
            raise ConfigurationError("rate_limit.enabled doit être un booléen")

        if not isinstance(
            config['rate_limit']['window'],
            (int,
             float)) or config['rate_limit']['window'] <= 0:
            raise ConfigurationError(
                "rate_limit.window doit être un nombre positif")

        if not isinstance(
            config['rate_limit']['limit'],
            (int,
             float)) or config['rate_limit']['limit'] <= 0:
            raise ConfigurationError(
                "rate_limit.limit doit être un nombre positif")

    def get(self, key: str, default=None) -> Any:
        """
        Récupère une valeur de configuration.

        Args:
            key: Clé de la configuration
            default: Valeur par défaut si la clé n'existe pas

        Returns:
            Valeur de la configuration
        """
        return self.config.get(key, default)

    def set(self, key: str, value: Any) -> None:
        """
        Définit une valeur de configuration.

        Args:
            key: Clé de la configuration
            value: Nouvelle valeur
        """
        self.config[key] = value

    def get_credentials(self) -> Dict[str, str]:
        """
        Récupère les informations d'authentification.

        Returns:
            Dictionnaire avec api_key et api_secret
        """
        return self.config['credentials']

    def get_rate_limit(self) -> Dict[str, Any]:
        """
        Récupère les paramètres de rate limiting.

        Returns:
            Dictionnaire avec les paramètres de rate limiting
        """
        return self.config['rate_limit']

    def get_environment(self) -> str:
        """
        Récupère l'environnement actuel.

        Returns:
            Nom de l'environnement
        """
        return self.env

    def get_config(self) -> Dict[str, Any]:
        """
        Récupère la configuration complète.

        Returns:
            Configuration complète
        """
        return dict(self.config)

    def get_summary(self) -> str:
        """
        Récupère un résumé formaté de la configuration.

        Returns:
            Résumé formaté
        """
        summary = (
            f"=== Configuration Kraken API ===\n"
            f"Environnement: {self.env}\n"
            f"Base URL: {self.config['base_url']}\n"
            f"Version: {self.config['version']}\n"
            f"Timeout: {self.config['timeout']}s\n"
            f"Max retries: {self.config['max_retries']}\n"
            f"Cache TTL: {self.config['cache_ttl']}s\n"
            f"Rate limiting: {'enabled' if self.config['rate_limit']['enabled'] else 'disabled'}\n"
            f"Log level: {self.config['log_level']}\n")

        return summary
