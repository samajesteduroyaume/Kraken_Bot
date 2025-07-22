"""
Script de test pour la version refactorisée d'AvailablePairs.
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
        logging.FileHandler('test_available_pairs_refactored.log')
    ]
)
logger = logging.getLogger(__name__)

# Importer la version refactorisée
from src.core.market.available_pairs_refactored import AvailablePairs, available_pairs

# Liste des paires à tester avec différents formats
TEST_PAIRS = [
    # Formats avec séparateur
    "BTC/USD",
    "BTC-USD",
    "BTC_USDT",
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
    # "XBTUSDT.M",  # Futures (à décommenter si nécessaire)
    # "XBTUSDT_240927",  # Futures avec date (à décommenter si nécessaire)
    
    # Paires invalides (pour tester la détection d'erreur)
    "INVALID_PAIR",
    "BTCEUR",  # Format invalide (devrait être XBTEUR ou XXBTZEUR)
    "123/456",  # Format invalide
    "",  # Chaîne vide
    # 12345,  # Type invalide (désactivé pour éviter les erreurs dans ce test)
]

async def test_available_pairs():
    """Teste les fonctionnalités principales d'AvailablePairs."""
    logger.info("Début du test d'AvailablePairs (version refactorisée)")
    
    try:
        # Initialiser avec la méthode de fabrique recommandée
        logger.info("Initialisation avec AvailablePairs.create()...")
        pairs_manager = await AvailablePairs.create()
        
        # Vérifier que c'est bien un singleton
        assert pairs_manager is available_pairs, "La méthode create() devrait retourner l'instance singleton"
        
        # Tester la méthode is_pair_supported
        logger.info("\n=== Test de is_pair_supported ===")
        for pair in TEST_PAIRS:
            try:
                is_supported = pairs_manager.is_pair_supported(pair)
                print(f"{pair!r}: {'✅' if is_supported else '❌'} {'(supportée)' if is_supported else '(non supportée)'}")
                
                # Si la paire est supportée, tester la normalisation
                if is_supported:
                    normalized = pairs_manager.normalize_pair(pair)
                    print(f"  Normalisé en: {normalized}")
                    
                    # Afficher les informations sur la paire
                    pair_info = pairs_manager.get_pair_info(normalized)
                    if pair_info:
                        print(f"  Détails: wsname={pair_info.get('wsname')}, base={pair_info.get('base')}, quote={pair_info.get('quote')}")
            
            except Exception as e:
                print(f"Erreur avec la paire {pair}: {e}")
            
            print("-" * 60)
        
        # Tester la récupération des paires par devise de cotation
        logger.info("\n=== Test de get_available_pairs ===")
        for currency in ['USD', 'EUR', 'USDT', 'BTC']:
            pairs = pairs_manager.get_available_pairs(currency)
            print(f"Paires avec devise de cotation {currency}: {len(pairs)} trouvées")
            if pairs:
                print(f"  Exemples: {pairs[:3]}..." if len(pairs) > 3 else f"  {pairs}")
            print()
        
        # Tester les méthodes utilitaires
        logger.info("\n=== Test des méthodes utilitaires ===")
        print(f"Nombre total de paires: {len(pairs_manager.get_available_pairs())}")
        print(f"Devises de base: {sorted(pairs_manager.get_base_currencies())}")
        print(f"Devises de cotation: {sorted(pairs_manager.get_quote_currencies())}")
        
        # Tester la vérification des devises de cotation
        print("\nVérification des devises de cotation:")
        for currency in ['USD', 'EUR', 'USDT', 'BTC', 'ZUSD', 'ZEUR', 'INVALID']:
            is_supported = pairs_manager.is_quote_currency_supported(currency)
            print(f"  {currency}: {'✅' if is_supported else '❌'}")
        
        # Tester le rafraîchissement du cache
        logger.info("\n=== Test de refresh_cache ===")
        try:
            await pairs_manager.refresh_cache()
            print("✅ Le cache a été rafraîchi avec succès")
        except Exception as e:
            print(f"❌ Erreur lors du rafraîchissement du cache: {e}")
        
        logger.info("\n✅ Tous les tests ont été exécutés avec succès!")
        
    except Exception as e:
        logger.error(f"❌ Erreur lors de l'exécution des tests: {e}", exc_info=True)
        raise

if __name__ == "__main__":
    asyncio.run(test_available_pairs())
