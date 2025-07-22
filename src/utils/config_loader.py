"""
Module pour le chargement de la configuration de l'application.
"""
import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional

class ConfigLoader:
    """Classe utilitaire pour charger la configuration de l'application."""
    
    _instance = None
    _config: Dict[str, Any] = {}
    
    def __new__(cls):
        """Implémentation du pattern Singleton."""
        if cls._instance is None:
            cls._instance = super(ConfigLoader, cls).__new__(cls)
        return cls._instance
    
    @classmethod
    def load_config(cls, config_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Charge la configuration depuis un fichier YAML.
        
        Args:
            config_path: Chemin vers le fichier de configuration. Si non spécifié,
                        utilise le chemin par défaut.
                        
        Returns:
            Dictionnaire contenant la configuration chargée.
        """
        if not config_path:
            # Chemin par défaut relatif au répertoire du projet
            base_dir = Path(__file__).parent.parent
            config_path = base_dir / 'config' / 'config.yaml'
        
        try:
            with open(config_path, 'r') as f:
                cls._config = yaml.safe_load(f) or {}
            return cls._config
        except FileNotFoundError:
            print(f"Avertissement: Fichier de configuration non trouvé à {config_path}")
            return {}
        except yaml.YAMLError as e:
            print(f"Erreur lors du chargement de la configuration: {e}")
            return {}
    
    @classmethod
    def get_config(cls, key: str = None, default: Any = None) -> Any:
        """
        Récupère une valeur de configuration par sa clé.
        
        Args:
            key: Clé de configuration (notation par points pour les dictionnaires imbriqués)
            default: Valeur par défaut si la clé n'existe pas
            
        Returns:
            La valeur de configuration ou la valeur par défaut
        """
        if not cls._config:
            cls.load_config()
            
        if key is None:
            return cls._config
            
        # Gestion des clés imbriquées (ex: "database.host")
        keys = key.split('.')
        value = cls._config
        
        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default
    
    @classmethod
    def update_config(cls, updates: Dict[str, Any]) -> None:
        """
        Met à jour la configuration avec de nouvelles valeurs.
        
        Args:
            updates: Dictionnaire des mises à jour à appliquer
        """
        if not cls._config:
            cls.load_config()
        
        def update_nested(d: Dict, u: Dict) -> Dict:
            for k, v in u.items():
                if isinstance(v, dict):
                    d[k] = update_nested(d.get(k, {}), v)
                else:
                    d[k] = v
            return d
        
        cls._config = update_nested(cls._config, updates)
