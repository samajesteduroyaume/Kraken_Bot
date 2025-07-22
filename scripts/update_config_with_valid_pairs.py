"""
Script pour mettre à jour la configuration avec les paires valides identifiées.
"""
import json
import logging
from pathlib import Path

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_valid_pairs() -> list:
    """Charge la liste des paires valides depuis le fichier JSON dans le dossier data/."""
    try:
        with open('../data/valid_pairs.json', 'r') as f:
            data = json.load(f)
            return data.get('valid_pairs', [])
    except FileNotFoundError:
        logger.error("Fichier data/valid_pairs.json non trouvé. Assurez-vous d'exécuter le script depuis le répertoire racine du projet.")
        return []
    except Exception as e:
        logger.error(f"Erreur lors du chargement des paires valides: {e}")
        return []

def update_trading_pairs_config(valid_pairs: list):
    """Met à jour le fichier de configuration des paires de trading."""
    # Catégoriser les paires par liquidité (simplifié pour l'exemple)
    high_liquidity = []
    medium_liquidity = []
    low_liquidity = []
    
    for pair in valid_pairs:
        # Exemple de catégorisation basée sur les paires
        if any(asset in pair for asset in ['XBT', 'ETH', 'USDT', 'USDC']):
            high_liquidity.append(pair)
        elif any(asset in pair for asset in ['DAI', 'EUR']):
            medium_liquidity.append(pair)
        else:
            low_liquidity.append(pair)
    
    # Créer le contenu du fichier de configuration
    config_content = f'''"""
Configuration des paires de trading recommandées pour le bot Kraken.

Ce module définit les paires de trading recommandées avec leurs paramètres spécifiques.
Les paires sont organisées par catégories basées sur leur liquidité et leur volatilité.
"""

# Paires avec haute liquidité
HIGH_LIQUIDITY_PAIRS = {json.dumps(high_liquidity, indent=4)}

# Paires avec liquidité moyenne
MEDIUM_LIQUIDITY_PAIRS = {json.dumps(medium_liquidity, indent=4)}

# Paires avec faible liquidité
LOW_LIQUIDITY_PAIRS = {json.dumps(low_liquidity, indent=4)}

# Toutes les paires dans un dictionnaire
TRADING_PAIRS = {{
    'high_liquidity': HIGH_LIQUIDITY_PAIRS,
    'medium_liquidity': MEDIUM_LIQUIDITY_PAIRS,
    'low_liquidity': LOW_LIQUIDITY_PAIRS
}}

def get_trading_pairs() -> dict:
    """Retourne la configuration des paires de trading."""
    return TRADING_PAIRS
'''
    
    # Chemin vers le fichier de configuration
    config_path = Path("src/config/trading_pairs_config.py")
    
    try:
        # Sauvegarder l'ancien fichier
        if config_path.exists():
            backup_path = config_path.with_suffix('.py.bak')
            config_path.rename(backup_path)
            logger.info(f"Ancien fichier de configuration sauvegardé sous {backup_path}")
        
        # Écrire la nouvelle configuration
        with open(config_path, 'w') as f:
            f.write(config_content)
        
        logger.info(f"Configuration mise à jour avec succès dans {config_path}")
        
    except Exception as e:
        logger.error(f"Erreur lors de la mise à jour de la configuration: {e}")
        if 'backup_path' in locals() and backup_path.exists():
            backup_path.rename(config_path)
            logger.info("Restauration de l'ancienne configuration")

def main():
    logger.info("Démarrage de la mise à jour de la configuration...")
    
    # Charger les paires valides
    valid_pairs = load_valid_pairs()
    if not valid_pairs:
        logger.error("Aucune paire valide trouvée. Vérifiez le fichier valid_pairs.json")
        return
    
    logger.info(f"{len(valid_pairs)} paires valides chargées")
    
    # Mettre à jour la configuration
    update_trading_pairs_config(valid_pairs)
    
    logger.info("Mise à jour terminée avec succès!")

if __name__ == "__main__":
    main()
