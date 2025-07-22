#!/usr/bin/env python3
"""
Script de test simplifié pour le module AvailablePairs.

Ce script peut être exécuté depuis la racine du projet avec :
PYTHONPATH=$PYTHONPATH:. python3 scripts/test_available_pairs.py
"""
import asyncio
import sys
import os
from pathlib import Path

# Ajouter le répertoire racine au PYTHONPATH
root_dir = Path(__file__).parent.parent
sys.path.insert(0, str(root_dir))

# Maintenant, nous pouvons importer nos modules
from src.core.api.kraken import KrakenAPI
from src.core.market.available_pairs import AvailablePairs

async def test_available_pairs():
    """Teste le chargement et l'utilisation du module AvailablePairs."""
    print("🔍 Test du module AvailablePairs")
    
    # Créer une instance de l'API Kraken
    print("🔄 Initialisation de l'API Kraken...")
    api = KrakenAPI()
    
    # Créer une instance de AvailablePairs
    print("🔄 Création de l'instance AvailablePairs...")
    pairs_manager = AvailablePairs(api=api)
    
    # Initialiser avec chargement depuis l'API
    print("📡 Chargement des paires depuis l'API Kraken...")
    await pairs_manager.initialize(use_cache=False)
    
    # Tester quelques fonctionnalités de base
    print("\n=== TESTS DE BASE ===")
    
    # Tester avec quelques paires connues
    test_pairs = ["XBT/USD", "ETH/USD", "XRP/USD", "SOL/USD", "ADA/USD"]
    
    for pair in test_pairs:
        is_supported = pairs_manager.is_pair_supported(pair)
        print(f"- {pair}: {'✅ Supportée' if is_supported else '❌ Non supportée'}")
        
        if is_supported:
            pair_info = pairs_manager.get_pair_info(pair)
            if pair_info:
                print(f"  - ID: {pair_info.get('wsname', 'N/A')}")
                print(f"  - Base: {pair_info.get('base', 'N/A')}")
                print(f"  - Quote: {pair_info.get('quote', 'N/A')}")
    
    # Afficher quelques statistiques
    print("\n=== STATISTIQUES ===")
    quote_currencies = sorted(pairs_manager.get_quote_currencies())
    print(f"Devises de cotation disponibles: {', '.join(quote_currencies[:10])}...")
    
    # Compter le nombre de paires par devise de cotation
    print("\nNombre de paires par devise de cotation (top 5):")
    quote_counts = {}
    for quote in quote_currencies:
        pairs = pairs_manager.get_available_pairs(quote_currency=quote)
        quote_counts[quote] = len(pairs)
    
    # Afficher les 5 principales devises par nombre de paires
    for quote, count in sorted(quote_counts.items(), key=lambda x: x[1], reverse=True)[:5]:
        print(f"- {quote}: {count} paires")
    
    # Tester la normalisation
    print("\n=== TEST DE NORMALISATION ===")
    test_cases = ["btcusd", "ETH-EUR", "xbtusdt", "SOLUSD"]
    
    for test_case in test_cases:
        normalized = pairs_manager.normalize_pair(test_case)
        print(f"- '{test_case}' → '{normalized}'")
        if normalized != test_case and pairs_manager.is_pair_supported(normalized):
            print(f"  ✓ Paire valide: {normalized}")
    
    print("\n✅ Tests terminés avec succès!")

if __name__ == "__main__":
    asyncio.run(test_available_pairs())
