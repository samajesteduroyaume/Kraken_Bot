"""
Script de test d'intégration après la migration vers la nouvelle version d'AvailablePairs.

Ce script teste les fonctionnalités clés du système après la migration pour s'assurer que tout fonctionne correctement.
"""

import asyncio
import logging
import sys
from pathlib import Path

# Ajouter le répertoire parent au PYTHONPATH pour pouvoir importer les modules du projet
sys.path.insert(0, str(Path(__file__).parent.absolute()))

# Configuration du logging avancée
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('test_integration_after_migration.log')
    ]
)
logger = logging.getLogger(__name__)

# Importer les modules nécessaires
from src.core.market.available_pairs_refactored import available_pairs, AvailablePairs, KrakenAPIError
from src.utils.pair_utils import extract_pair_name, extract_pair_names, is_valid_pair, normalize_pair_input

# Liste des paires à tester avec différents formats
TEST_PAIRS = [
    # Formats avec séparateur
    "BTC/USD", "BTC-USD", "BTC_USDT", "APE/EUR", "APE-EUR", "APENFT/USD", "APENFT-USD",
    # Formats natifs Kraken (sans séparateur)
    "XXBTZUSD", "XBTUSDT", "APEEUR", "APENFTEUR", "APENFTUSD", "XBTUSD", "XXBTZEUR",
    # Paires avec des formats spéciaux
    "XBTUSDT.M", "XBTUSDT_240927",
    # Paires invalides
    "INVALID_PAIR", "BTCEUR", "123/456", ""
]

async def test_available_pairs_integration():
    """Teste l'intégration d'AvailablePairs avec les autres modules."""
    logger.info("=== Début du test d'intégration après migration ===")
    
    try:
        # 1. Tester l'initialisation
        logger.info("1. Test d'initialisation...")
        assert available_pairs is not None, "L'instance available_pairs n'est pas initialisée"
        await available_pairs.initialize()
        logger.info("✅ Initialisation réussie")
        
        # 2. Tester la normalisation des paires
        logger.info("\n2. Test de normalisation des paires...")
        for pair in TEST_PAIRS:
            try:
                normalized = available_pairs.normalize_pair(pair)
                is_supported = available_pairs.is_pair_supported(pair)
                
                logger.info(f"Paire: {pair!r}")
                logger.info(f"  Normalisée: {normalized}")
                logger.info(f"  Supportée: {'✅' if is_supported else '❌'}")
                
                if is_supported and normalized:
                    pair_info = available_pairs.get_pair_info(normalized)
                    if pair_info:
                        logger.info(f"  Détails: wsname={pair_info.get('wsname')}, "
                                   f"base={pair_info.get('base')}, quote={pair_info.get('quote')}")
                
            except Exception as e:
                logger.error(f"Erreur avec la paire {pair}: {e}")
            
            logger.info("-" * 60)
        
        # 3. Tester les fonctions utilitaires
        logger.info("\n3. Test des fonctions utilitaires...")
        
        # Test de extract_pair_name
        test_pair = "XBT/USD"
        extracted = extract_pair_name(test_pair)
        logger.info(f"extract_pair_name('{test_pair}') = {extracted}")
        assert extracted == "XBT/USD", f"Erreur avec extract_pair_name: {extracted} != 'XBT/USD'"
        
        # Test de extract_pair_names
        test_pairs = ["XBT/USD", "ETH/USD", "INVALID"]
        extracted_pairs = extract_pair_names(test_pairs)
        logger.info(f"extract_pair_names({test_pairs}) = {extracted_pairs}")
        assert len(extracted_pairs) >= 2, "extract_pair_names n'a pas retourné assez de paires valides"
        
        # Test de is_valid_pair
        valid = is_valid_pair("XBT/USD")
        logger.info(f"is_valid_pair('XBT/USD') = {valid}")
        assert valid, "XBT/USD devrait être une paire valide"
        
        invalid = is_valid_pair("INVALID_PAIR")
        logger.info(f"is_valid_pair('INVALID_PAIR') = {invalid}")
        assert not invalid, "INVALID_PAIR ne devrait pas être une paire valide"
        
        # Test de normalize_pair_input
        normalized = normalize_pair_input("btc-usd")
        logger.info(f"normalize_pair_input('btc-usd') = {normalized}")
        assert normalized == "XBT/USD", f"Erreur avec normalize_pair_input: {normalized} != 'XBT/USD'"
        
        logger.info("✅ Tous les tests des fonctions utilitaires ont réussi")
        
        # 4. Tester la récupération des paires par devise de cotation
        logger.info("\n4. Test de récupération des paires par devise de cotation...")
        for currency in ['USD', 'EUR', 'USDT', 'BTC']:
            pairs = available_pairs.get_available_pairs(currency)
            logger.info(f"Paires avec devise de cotation {currency}: {len(pairs)} trouvées")
            if pairs:
                logger.info(f"  Exemples: {pairs[:3]}..." if len(pairs) > 3 else f"  {pairs}")
            assert isinstance(pairs, list), f"La méthode get_available_pairs('{currency}') devrait retourner une liste"
        
        # 5. Tester le rafraîchissement du cache
        logger.info("\n5. Test de rafraîchissement du cache...")
        try:
            await available_pairs.refresh_cache()
            logger.info("✅ Le cache a été rafraîchi avec succès")
        except Exception as e:
            logger.error(f"❌ Erreur lors du rafraîchissement du cache: {e}")
            raise
        
        logger.info("\n✅ Tous les tests d'intégration ont été exécutés avec succès!")
        
    except Exception as e:
        logger.error(f"❌ Erreur lors des tests d'intégration: {e}", exc_info=True)
        raise

if __name__ == "__main__":
    asyncio.run(test_available_pairs_integration())
