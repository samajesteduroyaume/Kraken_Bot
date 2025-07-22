"""
Script de test pour la validation des paires Kraken.

Ce script permet de tester manuellement la validation des paires avec différents formats.
"""

import asyncio
import logging
import sys
from pathlib import Path

# Ajouter le répertoire parent au PYTHONPATH pour pouvoir importer les modules du projet
sys.path.append(str(Path(__file__).parent.absolute()))

# Configurer le logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('pair_validation_test.log')
    ]
)
logger = logging.getLogger(__name__)

# Importer available_pairs depuis le module market
from src.core.market import available_pairs, initialize_available_pairs

# Importer le validateur après avoir configuré le logging
from src.core.api.kraken_api.validators import Validator

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

async def test_pair_validation():
    """Teste la validation de différentes paires avec le validateur."""
    # Initialiser available_pairs avant de créer le validateur
    logger.info("Initialisation de available_pairs...")
    
    try:
        # Utiliser asyncio.wait_for pour ajouter un timeout
        try:
            # Essayer d'utiliser le cache d'abord
            logger.info("Tentative d'initialisation avec cache...")
            await asyncio.wait_for(initialize_available_pairs(), timeout=5.0)
            logger.info(f"available_pairs._initialized = {available_pairs._initialized}")
            logger.info(f"available_pairs._last_updated = {available_pairs._last_updated}")
            logger.info("available_pairs initialisé avec succès (depuis le cache)")
        except asyncio.TimeoutError:
            logger.warning("Le chargement depuis le cache a pris trop de temps, tentative sans cache...")
            # Réessayer sans utiliser le cache
            await asyncio.wait_for(initialize_available_pairs(), timeout=10.0)
            logger.info("available_pairs initialisé avec succès (depuis l'API)")
        
        # Vérifier que l'initialisation a réussi
        logger.info(f"Vérification de l'initialisation: _initialized={available_pairs._initialized}")
        if not available_pairs._initialized:
            logger.error("Échec de l'initialisation de available_pairs")
            logger.info(f"Contenu de available_pairs: {available_pairs.__dict__}")
            # Essayer de continuer malgré tout pour voir ce qui se passe
            logger.info("Tentative de continuation malgré l'échec d'initialisation...")
            # return  # Ne pas s'arrêter pour voir ce qui se passe
            
    except Exception as e:
        logger.error(f"Erreur lors de l'initialisation de available_pairs: {e}")
        logger.info("Utilisation d'un validateur avec des données partielles...")
    
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
                normalized = validator.normalize_pair_input(pair)
                print(f"  🔄 Normalisé en: {normalized}")
            except Exception as e:
                print(f"  ❌ Échec de la normalisation: {e}")
                
        except Exception as e:
            print(f"  ❌ Échec de la validation: {e}")
        
        print("-" * 60)

if __name__ == "__main__":
    asyncio.run(test_pair_validation())
