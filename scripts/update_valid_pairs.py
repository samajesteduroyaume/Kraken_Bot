#!/usr/bin/env python3
"""
Script pour mettre à jour la liste des paires de trading valides sur Kraken.

Ce script va :
1. Se connecter à l'API Kraken
2. Récupérer la liste des paires disponibles
3. Vérifier quelles paires de la configuration sont valides
4. Mettre à jour le fichier de configuration avec uniquement les paires valides
"""

import asyncio
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Set, Any

# Ajouter le répertoire racine au chemin de recherche Python
root_dir = str(Path(__file__).parent.parent)
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Chemin vers le fichier de configuration des paires
CONFIG_PATH = Path("src/config/trading_pairs_config.py")

# Paires de trading actuelles (à mettre à jour avec celles du fichier de configuration)
CURRENT_PAIRS = {
    'high_liquidity': [
        "XBT/USD", "XBT/USDC", "XBT/USDT", "ETH/USD", "ETH/USDT", 
        "ETH/USDC", "XRP/USD", "SOL/USD", "ADA/USD", "DOT/USD"
    ],
    'medium_liquidity': [
        "LINK/USD", "MATIC/USD", "DOGE/USD", "AVAX/USD", "ATOM/USD"
    ],
    'low_liquidity': [
        "ALGO/USD", "FIL/USD", "LTC/USD", "UNI/USD", "AAVE/USD"
    ]
}

async def get_valid_pairs() -> Dict[str, List[str]]:
    """Récupère uniquement les paires valides depuis Kraken."""
    from src.core.market.available_pairs_refactored import AvailablePairs
    from src.core.api.kraken import KrakenAPI
    
    logger.info("Initialisation des paires disponibles...")
    
    try:
        # Initialiser l'API et les paires disponibles avec la méthode de fabrique
        logger.info("Chargement initial des paires via AvailablePairs.create()...")
        api = KrakenAPI()
        available_pairs = await AvailablePairs.create(api=api)
        
        # Vérifier que les données sont chargées
        if not hasattr(available_pairs, '_pairs_data') or not available_pairs._pairs_data:
            logger.error("Aucune donnée de paires disponible après l'initialisation.")
            return {}
        
        logger.info(f"{len(available_pairs._pairs_data)} paires chargées depuis Kraken")
        
    except Exception as e:
        logger.error(f"Erreur lors de l'initialisation des paires: {e}", exc_info=True)
        return {}
    
    # Vérifier quelles paires de la configuration sont valides
    valid_pairs = {
        'high_liquidity': [],
        'medium_liquidity': [],
        'low_liquidity': []
    }
    
    for category, pairs in CURRENT_PAIRS.items():
        logger.info(f"\nVérification des paires {category.replace('_', ' ')}:")
        
        for pair in pairs:
            try:
                # Essayer de normaliser la paire
                normalized = available_pairs.normalize_pair(pair)
                
                if normalized:
                    # Vérifier si la paire normalisée est supportée
                    is_supported = available_pairs.is_pair_supported(normalized)
                    
                    if is_supported:
                        valid_pairs[category].append(pair)
                        logger.info(f"✅ {pair} (normalisée: {normalized}) - VALIDE")
                    else:
                        logger.warning(f"❌ {pair} (normalisée: {normalized}) - NON SUPPORTÉE")
                else:
                    logger.warning(f"❌ {pair} - IMPOSSIBLE DE NORMALISER")
                    
            except Exception as e:
                logger.error(f"❌ Erreur avec la paire {pair}: {str(e)}")
    
    return valid_pairs

def update_config_file(valid_pairs: Dict[str, List[str]]) -> None:
    """Met à jour le fichier de configuration avec les paires valides."""
    if not CONFIG_PATH.exists():
        logger.error(f"Fichier de configuration non trouvé: {CONFIG_PATH}")
        return
    
    # Lire le contenu actuel du fichier
    with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Mettre à jour chaque catégorie de paires
    for category, pairs in valid_pairs.items():
        # Créer la chaîne de paires formatée
        pairs_str = '\n    "' + '",\n    "'.join(pairs) + '"' if pairs else ''
        
        # Construire le bloc de code à remplacer
        start_marker = f'# Paires avec {category.replace("_", " ")}'
        
        # Trouver la section à mettre à jour
        start_idx = content.find(start_marker)
        if start_idx == -1:
            logger.warning(f"Marqueur de catégorie non trouvé: {start_marker}")
            continue
            
        # Trouver le début du tableau (après le nom de la catégorie)
        array_start = content.find('[', start_idx)
        if array_start == -1:
            logger.warning(f"Début de tableau non trouvé pour la catégorie {category}")
            continue
            
        # Trouver la fin du tableau
        array_end = content.find(']', array_start) + 1
        if array_end == 0:
            logger.warning(f"Fin de tableau non trouvée pour la catégorie {category}")
            continue
        
        # Remplacer le contenu du tableau
        old_content = content[array_start:array_end]
        new_content = f'[{pairs_str}\n]' if pairs else '[]'
        
        content = content.replace(old_content, new_content)
        logger.info(f"Catégorie {category} mise à jour avec {len(pairs)} paires")
    
    # Écrire le contenu mis à jour
    backup_path = str(CONFIG_PATH) + '.bak'
    import shutil
    shutil.copy2(CONFIG_PATH, backup_path)
    logger.info(f"Sauvegarde de l'ancien fichier de configuration dans {backup_path}")
    
    with open(CONFIG_PATH, 'w', encoding='utf-8') as f:
        f.write(content)
    
    logger.info(f"Configuration mise à jour avec succès dans {CONFIG_PATH}")

async def main():
    """Fonction principale."""
    try:
        logger.info("Début de la mise à jour des paires valides...")
        
        # Récupérer les paires valides
        valid_pairs = await get_valid_pairs()
        
        # Afficher un résumé
        total_valid = sum(len(pairs) for pairs in valid_pairs.values())
        logger.info(f"\nRésumé des paires valides :")
        for category, pairs in valid_pairs.items():
            logger.info(f"- {category.replace('_', ' ').title()}: {len(pairs)} paires")
        
        if total_valid == 0:
            logger.error("Aucune paire valide trouvée. Vérifiez la connexion à l'API Kraken.")
            return
        
        # Demander confirmation avant de mettre à jour le fichier
        update = input(f"\nMettre à jour le fichier de configuration avec ces {total_valid} paires valides ? (o/n): ")
        if update.lower() == 'o':
            update_config_file(valid_pairs)
        else:
            logger.info("Mise à jour annulée par l'utilisateur")
    
    except Exception as e:
        logger.error(f"Erreur lors de la mise à jour des paires: {e}", exc_info=True)
    finally:
        logger.info("Terminé")

if __name__ == "__main__":
    asyncio.run(main())
