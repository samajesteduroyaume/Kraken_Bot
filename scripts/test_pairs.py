"""
Script pour tester la gestion des paires de trading.
"""
import asyncio
import sys
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Ajouter le répertoire racine au PYTHONPATH
root_dir = str(Path(__file__).parent.parent)
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from src.core.market.available_pairs import available_pairs
from src.core.api.kraken_api.client import KrakenAPI
from src.core.api.kraken_api.endpoints import KrakenEndpoints
from src.core.logging.logging import LoggerManager

# Configuration du logger
logger = LoggerManager().get_logger("test_pairs")

class PairTester:
    """Classe pour tester les paires de trading."""
    
    def __init__(self):
        """Initialise le testeur de paires."""
        self.api = KrakenAPI()
        self.endpoints = KrakenEndpoints(self.api)
        self.results: List[Dict] = []
    
    async def test_pair(self, pair: str) -> Dict:
        """Teste une paire spécifique et retourne les résultats."""
        result = {
            "pair": pair,
            "normalized": None,
            "pair_info": None,
            "ticker": None,
            "error": None,
            "valid": False
        }
        
        try:
            # Vérifier si la paire est supportée
            normalized = available_pairs.is_pair_supported(pair)
            result["normalized"] = normalized
            
            if normalized:
                # Récupérer les infos de la paire
                pair_info = available_pairs.get_pair_info(normalized)
                result["pair_info"] = pair_info
                
                # Essayer d'obtenir le ticker avec la version normalisée
                try:
                    ticker = await self.endpoints.get_ticker(normalized)
                    result["ticker"] = ticker
                    result["valid"] = True
                except Exception as e:
                    # Si ça échoue, essayer avec la paire originale
                    try:
                        ticker = await self.endpoints.get_ticker(pair)
                        result["ticker"] = ticker
                        result["valid"] = True
                    except Exception as e2:
                        result["error"] = f"API Error (normalized): {str(e)}\nAPI Error (original): {str(e2)}"
            else:
                result["error"] = "Paire non supportée"
                
        except Exception as e:
            result["error"] = f"Erreur lors du test de la paire: {str(e)}"
        
        self.results.append(result)
        return result
    
    def print_result(self, result: Dict):
        """Affiche les résultats d'un test de paire."""
        print(f"\n{'='*50}")
        print(f"Test de la paire: {result['pair']}")
        print(f"- Normalisée: {result['normalized']}")
        
        if result['pair_info']:
            print(f"- Info: {result['pair_info'].get('wsname', 'N/A')} (ID: {result['pair_info'].get('altname', 'N/A')})")
        
        if result['ticker']:
            print("- Ticker: Données disponibles")
            # Afficher seulement quelques champs importants du ticker
            ticker_data = result['ticker']
            first_key = next(iter(ticker_data))
            ticker_info = ticker_data[first_key]
            print(f"  - Dernier prix: {ticker_info.get('c', ['N/A'])[0]}")
            print(f"  - Volume 24h: {ticker_info.get('v', ['N/A'])[1]}")
        
        if result['error']:
            print(f"- ❌ Erreur: {result['error']}")
        
        print(f"- {'✅ Valide' if result['valid'] else '❌ Invalide'}")
    
    def print_summary(self):
        """Affiche un résumé des tests."""
        total = len(self.results)
        valid = sum(1 for r in self.results if r["valid"])
        invalid = total - valid
        
        print("\n" + "="*50)
        print("RÉSUMÉ DES TESTS")
        print("="*50)
        print(f"- Paires testées: {total}")
        print(f"- Paires valides: {valid} ({valid/max(1, total)*100:.1f}%)")
        print(f"- Paires invalides: {invalid} ({invalid/max(1, total)*100:.1f}%)")
        
        if invalid > 0:
            print("\nPaires invalides:")
            for result in self.results:
                if not result["valid"]:
                    print(f"- {result['pair']}: {result.get('error', 'Raison inconnue')}")
        
        # Afficher quelques statistiques
        print("\nStatistiques des paires :")
        print(f"- Nombre total de paires : {len(available_pairs.get_available_pairs())}")
        print(f"- Devises de base uniques : {len(available_pairs.get_base_currencies())}")
        print(f"- Devises de cotation uniques : {len(available_pairs.get_quote_currencies())}")
        
        # Afficher quelques exemples de paires valides
        valid_pairs = [r for r in self.results if r["valid"]]
        if valid_pairs:
            print("\nExemples de paires valides :")
            for result in valid_pairs[:5]:
                print(f"- {result['pair']} (normalisée: {result['normalized']})")
        
        # Afficher des suggestions pour les paires invalides
        invalid_pairs = [r for r in self.results if not r["valid"]]
        if invalid_pairs:
            print("\nSuggestions pour les paires invalides :")
            for result in invalid_pairs:
                pair = result["pair"]
                base, quote = pair.split('/') if '/' in pair else (pair[:-3], pair[-3:])
                
                # Vérifier si la devise de base existe
                base_exists = base in available_pairs.get_base_currencies()
                quote_exists = quote in available_pairs.get_quote_currencies()
                
                if not base_exists or not quote_exists:
                    print(f"- {pair}: ")
                    if not base_exists:
                        print(f"  ❌ Devise de base '{base}' non trouvée")
                        # Proposer des alternatives pour la devise de base
                        similar_bases = [b for b in available_pairs.get_base_currencies() 
                                      if b.startswith(base[:2]) or base in b]
                        if similar_bases:
                            print(f"  🔍 Suggestions pour {base}: {', '.join(similar_bases[:3])}")
                    
                    if not quote_exists:
                        print(f"  ❌ Devise de cotation '{quote}' non trouvée")
                        # Proposer des alternatives pour la devise de cotation
                        similar_quotes = [q for q in available_pairs.get_quote_currencies() 
                                       if q.startswith(quote[:2]) or quote in q]
                        if similar_quotes:
                            print(f"  🔍 Suggestions pour {quote}: {', '.join(similar_quotes[:3])}")
                else:
                    print(f"- {pair}: La paire n'existe pas mais les devises sont valides")
                    print(f"  ℹ️  Essayez de vérifier le format ou la casse (ex: XBT au lieu de BTC)")

async def main():
    """Fonction principale."""
    print("Initialisation du test des paires...")
    
    # Initialiser les paires disponibles
    print("\nInitialisation des paires disponibles...")
    await available_pairs.initialize()
    
    # Rafraîchir le cache
    print("\nRafraîchissement du cache...")
    success = available_pairs.refresh_cache()
    if not success:
        print("❌ Impossible de rafraîchir le cache")
        return
    
    # Créer le testeur de paires
    tester = PairTester()
    
    # Tester des paires spécifiques
    test_pairs = [
        # Paires qui devraient fonctionner
        "XBT/USD",  # Bitcoin/USD
        "ETH/USD",   # Ethereum/USD
        "XRP/EUR",   # Ripple/Euro
        "LTC/BTC",   # Litecoin/Bitcoin
        "DOT/USD",   # Polkadot/USD
        
        # Paires avec des problèmes potentiels
        "BTC/USD",   # Devrait être mappé à XBT/USD
        "DOGE/USD",  # Devrait être mappé à XDG/USD
        "AAVE/USD",  # Aave/USD
        "SOL/EUR",   # Solana/Euro
        "ADA/BTC",   # Cardano/Bitcoin
        
        # Paires potentiellement invalides
        "INVALID/USD",
        "BTC/INVALID",
        "DOG/USD",   # Devrait être DOGE
    ]
    
    # Tester chaque paire
    print("\nDébut des tests...")
    for pair in test_pairs:
        result = await tester.test_pair(pair)
        tester.print_result(result)
    
    # Afficher le résumé
    tester.print_summary()
    
    print("\nTest terminé !")

if __name__ == "__main__":
    asyncio.run(main())
