#!/usr/bin/env python3
"""
Script de test pour la méthode normalize_pair du module available_pairs_refactored.
"""
import asyncio
import sys
from pathlib import Path

# Ajouter le répertoire src au PYTHONPATH
sys.path.append(str(Path(__file__).parent.parent))

from src.core.market.available_pairs_refactored import AvailablePairs, KrakenAPIError

# Liste des paires à tester avec leurs résultats attendus
TEST_PAIRS = [
    # Format standard avec tiret
    ("btc-usd", "XBT/USD"),
    ("eth-eur", "ETH/EUR"),
    ("xbt-eur", "XBT/EUR"),
    
    # Format standard avec slash
    ("BTC/USD", "XBT/USD"),
    ("XBT/EUR", "XBT/EUR"),
    ("ETH/EUR", "ETH/EUR"),
    
    # Format sans séparateur
    ("btcusd", "XBT/USD"),
    ("xbtusd", "XBT/USD"),
    ("etheur", "ETH/EUR"),
    
    # Format avec tiret bas
    ("btc_usd", "XBT/USD"),
    ("eth_eur", "ETH/EUR"),
    
    # Format Kraken natif
    ("XXBTZUSD", "XBT/USD"),
    ("XETHZEUR", "ETH/EUR"),
    
    # Variantes de casse
    ("Btc-Usd", "XBT/USD"),
    ("bTcEuR", "XBT/EUR"),
    
    # Cas spéciaux (DOGE/XDG)
    ("doge-usd", "XDG/USD"),
    ("xdg-usd", "XDG/USD"),
    ("dogeeur", "XDG/EUR"),
]

async def test_normalize_pair():
    """Teste la méthode normalize_pair avec différentes paires."""
    # Créer une instance d'AvailablePairs
    available_pairs = await AvailablePairs.create()
    
    # Tester chaque paire
    total = len(TEST_PAIRS)
    success = 0
    
    print("\n=== Début des tests de normalisation des paires ===\n")
    
    for input_pair, expected_output in TEST_PAIRS:
        try:
            # Normaliser la paire
            normalized = available_pairs.normalize_pair(input_pair)
            
            # Vérifier le résultat
            if normalized == expected_output:
                status = "✅ SUCCÈS"
                success += 1
            else:
                status = f"❌ ÉCHEC (obtenu: {normalized})"
                
            print(f"{status} - Entrée: '{input_pair}'", end="")
            if status.startswith("❌"):
                print(f" (attendu: '{expected_output}')")
            else:
                print()
                
        except Exception as e:
            print(f"❌ ERREUR - Entrée: '{input_pair}' - {str(e)}")
    
    # Afficher le résumé
    print(f"\n=== Résumé des tests ===")
    print(f"Total: {total}")
    print(f"Succès: {success}")
    print(f"Échecs: {total - success}")
    print(f"Taux de réussite: {(success/total)*100:.1f}%")
    
    return success == total

if __name__ == "__main__":
    try:
        asyncio.run(test_normalize_pair())
    except KrakenAPIError as e:
        print(f"Erreur lors de la récupération des paires depuis l'API Kraken: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Erreur inattendue: {e}")
        sys.exit(1)
