"""
Script pour forcer le rechargement des paires depuis l'API Kraken.

Ce script contourne le cache et télécharge directement les paires depuis l'API Kraken,
puis enregistre les données dans le cache local.
"""

import asyncio
import logging
import sys
from pathlib import Path

# Ajouter le répertoire parent au PYTHONPATH pour pouvoir importer les modules
sys.path.insert(0, str(Path(__file__).parent.absolute()))

from src.core.market.available_pairs import AvailablePairs

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('force_reload_pairs.log')
    ]
)
logger = logging.getLogger(__name__)

async def main():
    """Fonction principale qui force le rechargement des paires."""
    logger.info("Début du rechargement forcé des paires depuis l'API Kraken...")
    
    try:
        # Créer une instance de AvailablePairs
        pairs_manager = AvailablePairs()
        
        # Forcer le rechargement depuis l'API (ignorer le cache)
        logger.info("Téléchargement des paires depuis l'API Kraken...")
        await pairs_manager._fetch_from_api()
        
        # Sauvegarder dans le cache
        logger.info("Sauvegarde des données dans le cache local...")
        pairs_manager._save_to_cache()
        
        # Afficher un résumé
        logger.info("Rechargement terminé avec succès!")
        logger.info(f"Nombre de paires chargées: {len(pairs_manager._pairs_data)}")
        logger.info(f"Cache enregistré dans: {pairs_manager.CACHE_FILE.absolute()}")
        
        # Vérifier que les données sont bien chargées
        if pairs_manager._pairs_data:
            sample_pair = next(iter(pairs_manager._pairs_data.items()))
            logger.info(f"Exemple de paire chargée: {sample_pair[0]} -> {sample_pair[1].get('wsname')}")
        
        return True
        
    except Exception as e:
        logger.error(f"Erreur lors du rechargement des paires: {e}", exc_info=True)
        return False

if __name__ == "__main__":
    # Exécuter la fonction asynchrone
    success = asyncio.run(main())
    
    if success:
        print("\nRechargement des paires réussi!")
        print("Vous pouvez maintenant relancer le script de test.")
    else:
        print("\nÉchec du rechargement des paires. Consultez le fichier de log pour plus de détails.")
    
    input("\nAppuyez sur Entrée pour quitter...")
