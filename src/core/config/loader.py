"""
Chargeur de configuration pour les stratégies de trading.

Ce module fournit des utilitaires pour charger et valider la configuration
des stratégies à partir de fichiers YAML.
"""
import os
from typing import Dict, Any, Optional
import yaml
from pathlib import Path

from .strategy_config import StrategyConfig


def load_strategy_config(config_path: Optional[str] = None) -> StrategyConfig:
    """
    Charge la configuration des stratégies à partir d'un fichier YAML.
    
    Args:
        config_path: Chemin vers le fichier de configuration. Si non spécifié,
                    utilise le fichier de configuration par défaut.
                    
    Returns:
        Une instance de StrategyConfig avec la configuration chargée.
        
    Raises:
        FileNotFoundError: Si le fichier de configuration n'existe pas.
        ValidationError: Si la configuration n'est pas valide.
    """
    if config_path is None:
        # Chemin par défaut relatif au répertoire racine du projet
        default_path = Path(__file__).parent.parent.parent.parent / "config" / "strategies" / "default.yaml"
        config_path = str(default_path)
    
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Le fichier de configuration n'existe pas: {config_path}")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config_dict = yaml.safe_load(f) or {}
    
    # Créer une configuration avec les valeurs par défaut
    config = StrategyConfig()
    
    # Mettre à jour avec les valeurs du fichier
    if config_dict:
        config.update_from_dict(config_dict)
    
    return config
