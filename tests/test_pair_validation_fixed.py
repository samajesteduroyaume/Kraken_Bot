"""
Script de test amélioré pour la validation des paires Kraken.

Ce script résout les problèmes d'initialisation en s'assurant qu'une seule
instance de available_pairs est utilisée dans toute l'application.
"""

import asyncio
import logging
import sys
from pathlib import Path

# Ajouter le répertoire parent au PYTHONPATH pour pouvoir importer les modules du projet
sys.path.insert(0, str(Path(__file__).parent.absolute()))

# Configuration du logging avancée
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('pair_validation_test_fixed.log')
    ]
)
logger = logging.getLogger(__name__)

# Importer available_pairs depuis le module market
from src.core.market import available_pairs, initialize_available_pairs

# Liste des paires à tester avec différents formats
TEST_PAIRS = [
    # Formats avec séparateur
    "BTC/USD",
    "BTC-USD",
    "APE/EUR",
    "APE-EUR",
    "APENFT/USD",
    "APENFT-USD",
    
    # Formats natifs Kraken (sans séparateur)
    "XXBTZUSD",  # BTC/USD
    "XBTUSDT",    # BTC/USDT
    "APEEUR",     # APE/EUR
    "APENFTEUR",  # APENFT/EUR
    "APENFTUSD",  # APENFT/USD
    "XBTUSD",     # BTC/USD (format alternatif)
    "XXBTZEUR",   # BTC/EUR
    
    # Paires avec des formats spéciaux
    "XBTUSDT.M",  # Futures
    "XBTUSDT_240927",  # Futures avec date
    
    # Paires invalides (pour tester la détection d'erreur)
    "INVALID_PAIR",
    "BTCEUR",  # Format invalide (devrait être XBTEUR ou XXBTZEUR)
    "123/456",  # Format invalide
    "",  # Chaîne vide
    12345,  # Type invalide
]

async def initialize_application():
    """Initialise l'application et les dépendances."""
    logger.info("Initialisation de l'application...")
    
    try:
        # Initialiser available_pairs avec un timeout
        logger.info("Initialisation de available_pairs...")
        await asyncio.wait_for(initialize_available_pairs(), timeout=10.0)
        
        # Vérifier que l'initialisation a réussi
        if not available_pairs._initialized:  # pylint: disable=protected-access
            logger.error("Échec de l'initialisation de available_pairs")
            return False
            
        logger.info("available_pairs initialisé avec succès")
        logger.info(f"Nombre de paires chargées: {len(available_pairs._pairs_data)}")
        
        return True
        
    except asyncio.TimeoutError:
        logger.error("Le délai d'initialisation de available_pairs a expiré")
        return False
    except Exception as e:
        logger.error(f"Erreur lors de l'initialisation: {e}", exc_info=True)
        return False

async def test_pair_validation():
    """Teste la validation de différentes paires avec le validateur."""
    # Initialiser l'application
    if not await initialize_application():
        logger.error("Impossible d'initialiser l'application. Arrêt du test.")
        return
    
    # Importer le validateur après l'initialisation
    from src.core.api.kraken_api.validators import Validator
    from src.utils.pair_utils import normalize_pair_input
    
    # Créer une instance du validateur
    validator = Validator()
    
    print("\n=== Test de validation des paires ===\n")
    
    for pair in TEST_PAIRS:
        try:
            # Afficher le type de l'entrée pour le débogage
            print(f"Test de validation pour: {pair!r} (type: {type(pair).__name__})")
            
            # Tester la validation
            validator.validate_pair(pair)
            print(f"  ✅ Valide: {pair}")
            
            # Essayer de normaliser la paire
            try:
                normalized = normalize_pair_input(str(pair))
                print(f"  🔄 Normalisé en: {normalized}")
                
                # Afficher des informations supplémentaires sur la paire
                pair_info = available_pairs.get_pair_info(normalized)
                if pair_info:
                    print(f"  ℹ️  Détails: {pair_info.get('wsname', 'N/A')} (base: {pair_info.get('base')}, quote: {pair_info.get('quote')})")
                
            except Exception as e:
                print(f"  ❌ Échec de la normalisation: {e}")
                
        except Exception as e:
            print(f"  ❌ Échec de la validation: {e}")
        
        print("-" * 60)

if __name__ == "__main__":
    asyncio.run(test_pair_validation())
