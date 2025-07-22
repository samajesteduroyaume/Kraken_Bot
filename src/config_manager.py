"""
Gestionnaire de configuration pour le bot de trading Kraken.
Charge les paramètres depuis config.yaml et les variables d'environnement.
"""
import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
from dotenv import load_dotenv

class ConfigManager:
    """Classe pour gérer la configuration du bot de trading."""
    
    _instance = None
    
    def __new__(cls):
        """Implémentation du pattern Singleton."""
        if cls._instance is None:
            cls._instance = super(ConfigManager, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        """Initialise le gestionnaire de configuration."""
        if self._initialized:
            return
            
        # Charger les variables d'environnement depuis .env
        load_dotenv()
        
        # Charger la configuration YAML
        self.config_path = Path(__file__).parent.parent / 'config' / 'config_improved.yaml'
        self._load_config()
        
        # Surcharger avec les variables d'environnement
        self._override_with_env()
        
        self._initialized = True
    
    def _load_config(self) -> None:
        """Charge la configuration depuis le fichier YAML."""
        try:
            with open(self.config_path, 'r') as f:
                self.config = yaml.safe_load(f) or {}
        except FileNotFoundError:
            raise FileNotFoundError(f"Le fichier de configuration {self.config_path} est introuvable.")
        except yaml.YAMLError as e:
            raise ValueError(f"Erreur lors du chargement du fichier de configuration: {e}")
    
    def _override_with_env(self) -> None:
        """Surcharge la configuration avec les variables d'environnement."""
        # API Kraken
        if os.getenv('KRAKEN_API_KEY'):
            if 'api' not in self.config:
                self.config['api'] = {}
            self.config['api']['api_key'] = os.getenv('KRAKEN_API_KEY')
            self.config['api']['api_secret'] = os.getenv('KRAKEN_API_SECRET', '')
        
        # Base de données
        if os.getenv('POSTGRES_USER'):
            if 'postgres' not in self.config:
                self.config['postgres'] = {}
            self.config['postgres']['user'] = os.getenv('POSTGRES_USER')
            self.config['postgres']['password'] = os.getenv('POSTGRES_PASSWORD', '')
            self.config['postgres']['host'] = os.getenv('POSTGRES_HOST', 'localhost')
            self.config['postgres']['port'] = int(os.getenv('POSTGRES_PORT', '5432'))
            self.config['postgres']['database'] = os.getenv('POSTGRES_DB', 'kraken_bot')
        
        # Mode de trading
        if os.getenv('TRADING_MODE'):
            self.config['trading']['mode'] = os.getenv('TRADING_MODE', 'paper')
        
        # Niveau de log
        if os.getenv('LOG_LEVEL'):
            self.config['log']['level'] = os.getenv('LOG_LEVEL', 'INFO')
    
    def get(self, key: str, default: Any = None) -> Any:
        """Récupère une valeur de configuration par clé."""
        keys = key.split('.')
        value = self.config
        
        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default
    
    def get_section(self, section: str) -> Dict[str, Any]:
        """Récupère une section complète de la configuration."""
        return self.config.get(section, {})
    
    def update(self, key: str, value: Any) -> None:
        """Met à jour une valeur de configuration."""
        keys = key.split('.')
        config = self.config
        
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        config[keys[-1]] = value
    
    def reload(self) -> None:
        """Recharge la configuration depuis le fichier."""
        self._load_config()
        self._override_with_env()


# Instance globale
config = ConfigManager()

# Exemple d'utilisation:
# from config_manager import config
# api_key = config.get('api.api_key')
# db_config = config.get_section('postgres')
# config.update('trading.max_positions', 5)
