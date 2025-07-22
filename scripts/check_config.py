import os
import sys
import yaml
import logging
from pathlib import Path
from typing import Dict, Any
from src.core.config_adapter import Config  # Migré vers le nouvel adaptateur

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ConfigValidator:
    """Valideur de configuration pour le bot de trading."""
    
    REQUIRED_ENV_VARS = [
        'KRAKEN_API_KEY',
        'KRAKEN_API_SECRET',
        'POSTGRES_USER',
        'POSTGRES_PASSWORD',
        'POSTGRES_DB'
    ]
    
    def __init__(self):
        self.errors = []
        self.warnings = []
        
    def check_environment(self) -> bool:
        """Vérifie les variables d'environnement requises."""
        missing_vars = []
        empty_vars = []
        
        for var in self.REQUIRED_ENV_VARS:
            value = os.getenv(var)
            if not value:
                if var in os.environ:
                    empty_vars.append(var)
                else:
                    missing_vars.append(var)
        
        if missing_vars:
            self.errors.append(f"Variables d'environnement manquantes: {', '.join(missing_vars)}")
        if empty_vars:
            self.errors.append(f"Variables d'environnement vides: {', '.join(empty_vars)}")
        
        return not bool(self.errors)
    
    def check_config_file(self, config_path: str) -> bool:
        """Vérifie la configuration YAML."""
        try:
            if not os.path.isfile(config_path):
                # Essayer le chemin absolu si le relatif échoue
                abs_path = os.path.join('/app', config_path) if not config_path.startswith('/app') else config_path
                if os.path.isfile(abs_path):
                    config_path = abs_path
            with open(config_path, 'r') as file:
                config = yaml.safe_load(file)
                
            # Vérifier les sections requises
            required_sections = ['kraken', 'trading', 'risk_management']
            missing_sections = [s for s in required_sections if s not in config]
            if missing_sections:
                self.errors.append(f"Sections manquantes dans la configuration: {', '.join(missing_sections)}")
                return False
                
            # Vérifier les clés API
            if not config['kraken'].get('api_key') or not config['kraken'].get('api_secret'):
                self.errors.append("Les clés API Kraken doivent être définies dans le fichier de configuration")
                
            # Vérifier les paramètres de trading
            if not isinstance(config['trading'].get('pairs'), list) or not config['trading']['pairs']:
                self.errors.append("La liste des paires de trading doit être une liste non vide")
                
            # Vérifier les paramètres de gestion des risques
            if not isinstance(config['risk_management'].get('max_risk_per_trade'), (int, float)):
                self.errors.append("max_risk_per_trade doit être un nombre")
                
            return True
            
        except FileNotFoundError:
            self.errors.append(f"Fichier de configuration {config_path} non trouvé")
            return False
        except yaml.YAMLError as e:
            self.errors.append(f"Erreur dans le fichier YAML: {str(e)}")
            return False
    
    def check_database_connection(self, config: Config) -> bool:
        """Vérifie la connexion à la base de données."""
        try:
            # Ici, vous devriez implémenter une vérification de connexion à la base de données
            # Pour l'exemple, nous simulons une vérification
            if not config.db_config['user'] or not config.db_config['password']:
                self.errors.append("Impossible de se connecter à la base de données: identifiants manquants")
                return False
            return True
        except Exception as e:
            self.errors.append(f"Erreur lors de la connexion à la base de données: {str(e)}")
            return False
    
    def validate(self) -> bool:
        """Valide la configuration complète."""
        valid = True
        
        # Vérifier les variables d'environnement
        if not self.check_environment():
            valid = False
        
        # Vérifier le fichier de configuration
        config_path = os.getenv('CONFIG_FILE', 'config/config.yaml')
        if not self.check_config_file(config_path):
            valid = False
        
        # Vérifier la connexion à la base de données
        try:
            config = Config
            if not self.check_database_connection(config):
                valid = False
        except Exception as e:
            self.errors.append(f"Erreur lors de la création de la configuration: {str(e)}")
            valid = False
        
        # Afficher les résultats
        if self.errors:
            logger.error("Erreurs de configuration:")
            for error in self.errors:
                logger.error(f"- {error}")
        
        if self.warnings:
            logger.warning("Avertissements:")
            for warning in self.warnings:
                logger.warning(f"- {warning}")
        
        return valid

def main():
    """Point d'entrée du script."""
    validator = ConfigValidator()
    
    if validator.validate():
        logger.info("La configuration est valide!")
        sys.exit(0)
    else:
        logger.error("La configuration est invalide. Veuillez corriger les erreurs.")
        sys.exit(1)

if __name__ == "__main__":
    main()
