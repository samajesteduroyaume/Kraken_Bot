"""
Script pour corriger le fichier de cache des paires disponibles.

Ce script vérifie et corrige la structure du fichier de cache pour s'assurer
qu'il contient bien la clé 'pairs' attendue par AvailablePairs.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('fix_cache.log')
    ]
)
logger = logging.getLogger(__name__)

def fix_cache_file(cache_path: Path) -> bool:
    """
    Corrige le fichier de cache en s'assurant qu'il contient la clé 'pairs'.
    
    Args:
        cache_path: Chemin vers le fichier de cache
        
    Returns:
        bool: True si la correction a réussi, False sinon
    """
    try:
        # Lire le contenu actuel du fichier
        with open(cache_path, 'r', encoding='utf-8') as f:
            cache_data = json.load(f)
        
        logger.info(f"Contenu actuel du cache: {list(cache_data.keys())}")
        
        # Vérifier si la clé 'pairs' existe déjà
        if 'pairs' in cache_data:
            logger.info("La clé 'pairs' est déjà présente dans le cache, aucune correction nécessaire.")
            return True
        
        # Si la clé 'pairs' n'existe pas, essayer de la créer à partir des données existantes
        logger.warning("La clé 'pairs' est manquante, tentative de reconstruction...")
        
        # Si le cache contient des paires directement à la racine, les déplacer sous la clé 'pairs'
        if any(k for k in cache_data if k not in ['base_currencies', 'last_updated', 'metadata']):
            logger.info("Détection de paires à la racine, restructuration...")
            pairs = {k: v for k, v in cache_data.items() 
                    if k not in ['base_currencies', 'last_updated', 'metadata']}
            
            # Créer un nouveau dictionnaire avec les paires déplacées
            new_cache = {
                'pairs': pairs,
                'last_updated': cache_data.get('last_updated'),
                'base_currencies': cache_data.get('base_currencies', []),
                'metadata': cache_data.get('metadata', {})
            }
            
            # Sauvegarder la correction dans un fichier temporaire d'abord
            temp_file = str(cache_path) + '.fixed'
            with open(temp_file, 'w', encoding='utf-8') as f:
                json.dump(new_cache, f, indent=2, ensure_ascii=False)
            
            # Remplacer l'ancien fichier
            backup_file = str(cache_path) + '.bak'
            import shutil
            shutil.copy2(cache_path, backup_file)
            logger.info(f"Sauvegarde de l'ancien cache dans {backup_file}")
            
            shutil.move(temp_file, cache_path)
            logger.info("Fichier de cache corrigé avec succès.")
            return True
        
        # Si on ne peut pas récupérer les paires, on ne peut pas corriger
        logger.error("Impossible de déterminer la structure des paires dans le cache.")
        return False
        
    except json.JSONDecodeError as e:
        logger.error(f"Erreur de décodage JSON dans le fichier de cache: {e}")
    except Exception as e:
        logger.error(f"Erreur inattendue lors de la correction du cache: {e}")
    
    return False

if __name__ == "__main__":
    cache_path = Path("data/available_pairs_cache.json")
    
    if not cache_path.exists():
        logger.error(f"Le fichier de cache {cache_path} n'existe pas.")
        exit(1)
    
    logger.info(f"Vérification du fichier de cache: {cache_path}")
    
    if fix_cache_file(cache_path):
        logger.info("Le fichier de cache a été corrigé avec succès.")
    else:
        logger.error("Échec de la correction du fichier de cache.")
        logger.info("Suppression du fichier de cache pour forcer une régénération...")
        try:
            cache_path.unlink()
            logger.info("Le fichier de cache a été supprimé. Il sera régénéré au prochain démarrage.")
        except Exception as e:
            logger.error(f"Échec de la suppression du fichier de cache: {e}")
        
        logger.info("Veuillez relancer le script de test pour régénérer le cache.")
    
    logger.info("Terminé.")
