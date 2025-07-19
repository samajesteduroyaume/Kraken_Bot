import os
import sys
import yaml
import json
from pathlib import Path
import jsonschema
from jsonschema import validate
from typing import Dict, Any
import logging

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class YAMLValidator:
    """Valideur de fichiers YAML pour le bot de trading."""
    
    def __init__(self):
        self.errors = []
        self.warnings = []
        
    def get_schema(self) -> Dict[str, Any]:
        """Retourne le schéma JSON pour la validation."""
        return {
            "$schema": "http://json-schema.org/draft-07/schema#",
            "type": "object",
            "required": ["kraken", "trading", "risk_management"],
            "properties": {
                "kraken": {
                    "type": "object",
                    "required": ["api_key", "api_secret"],
                    "properties": {
                        "api_key": {"type": "string"},
                        "api_secret": {"type": "string"},
                        "api_timeout": {"type": "integer", "minimum": 1}
                    }
                },
                "trading": {
                    "type": "object",
                    "required": ["pairs", "timeframes"],
                    "properties": {
                        "pairs": {
                            "type": "array",
                            "items": {"type": "string"},
                            "minItems": 1
                        },
                        "timeframes": {
                            "type": "array",
                            "items": {"type": "string"},
                            "minItems": 1
                        },
                        "max_open_positions": {"type": "integer", "minimum": 1},
                        "max_position_size": {"type": "number", "minimum": 0.01}
                    }
                },
                "risk_management": {
                    "type": "object",
                    "required": ["max_risk_per_trade", "max_daily_drawdown"],
                    "properties": {
                        "max_risk_per_trade": {
                            "type": "number",
                            "minimum": 0,
                            "maximum": 100
                        },
                        "max_daily_drawdown": {
                            "type": "number",
                            "minimum": 0,
                            "maximum": 100
                        }
                    }
                }
            }
        }
    
    def validate_file(self, file_path: str) -> bool:
        """Valide un fichier YAML."""
        try:
            with open(file_path, 'r') as file:
                config = yaml.safe_load(file)
                
            # Vérifier le schéma
            schema = self.get_schema()
            validate(instance=config, schema=schema)
            
            # Vérifier les valeurs spécifiques
            if not self._validate_trading_values(config['trading']):
                return False
                
            if not self._validate_risk_values(config['risk_management']):
                return False
                
            logger.info(f"Fichier {file_path} valide")
            return True
            
        except FileNotFoundError:
            self.errors.append(f"Fichier {file_path} non trouvé")
            return False
        except yaml.YAMLError as e:
            self.errors.append(f"Erreur YAML dans {file_path}: {str(e)}")
            return False
        except jsonschema.exceptions.ValidationError as e:
            self.errors.append(f"Erreur de validation dans {file_path}: {str(e)}")
            return False
        except Exception as e:
            self.errors.append(f"Erreur lors de la validation de {file_path}: {str(e)}")
            return False
    
    def _validate_trading_values(self, trading_config: Dict[str, Any]) -> bool:
        """Valide les valeurs de la section trading."""
        valid = True
        
        # Vérifier les paires
        if not trading_config['pairs']:
            self.errors.append("La liste des paires de trading ne peut pas être vide")
            valid = False
            
        # Vérifier les timeframes
        valid_timeframes = ['1m', '5m', '15m', '30m', '1h', '4h', '1d', '1w']
        invalid_timeframes = [tf for tf in trading_config['timeframes'] if tf not in valid_timeframes]
        if invalid_timeframes:
            self.errors.append(f"Timeframes invalides: {', '.join(invalid_timeframes)}")
            valid = False
            
        return valid
    
    def _validate_risk_values(self, risk_config: Dict[str, Any]) -> bool:
        """Valide les valeurs de gestion des risques."""
        valid = True
        
        # Vérifier les pourcentages
        for key in ['max_risk_per_trade', 'max_daily_drawdown']:
            value = risk_config.get(key)
            if value is None:
                self.errors.append(f"{key} doit être défini")
                valid = False
            elif not (0 <= value <= 100):
                self.errors.append(f"{key} doit être entre 0 et 100")
                valid = False
                
        return valid
    
    def validate_directory(self, dir_path: str) -> Dict[str, bool]:
        """Valide tous les fichiers YAML dans un répertoire."""
        results = {}
        yaml_files = [f for f in Path(dir_path).glob('*.yaml')]
        
        for file in yaml_files:
            is_valid = self.validate_file(str(file))
            results[str(file)] = is_valid
            
        return results

def main():
    """Point d'entrée du script."""
    validator = YAMLValidator()
    
    # Vérifier les arguments
    if len(sys.argv) < 2:
        print("Usage: python validate_yaml.py <fichier.yaml | répertoire>")
        sys.exit(1)
    
    path = sys.argv[1]
    
    if os.path.isfile(path):
        if validator.validate_file(path):
            logger.info("La configuration est valide!")
            sys.exit(0)
        else:
            logger.error("La configuration est invalide.")
            for error in validator.errors:
                logger.error(f"- {error}")
            sys.exit(1)
    elif os.path.isdir(path):
        results = validator.validate_directory(path)
        
        valid = all(results.values())
        if valid:
            logger.info("Toutes les configurations sont valides!")
            sys.exit(0)
        else:
            logger.error("Certaines configurations sont invalides.")
            for file, is_valid in results.items():
                if not is_valid:
                    logger.error(f"- {file}")
            sys.exit(1)
    else:
        logger.error(f"{path} n'est ni un fichier ni un répertoire existant")
        sys.exit(1)

if __name__ == "__main__":
    main()
