#!/usr/bin/env python3
"""
Script utilitaire pour afficher les paires de trading disponibles sur Kraken.

Ce script utilise le module AvailablePairs pour récupérer et afficher les paires
supportées par Kraken, organisées par devise de cotation.
"""
import asyncio
import sys
import os
from pathlib import Path

# Ajouter le répertoire src au chemin Python
sys.path.append(str(Path(__file__).parent.parent / 'src'))

# Imports locaux
from core.api.kraken import KrakenAPI
from core.market.available_pairs import available_pairs
from config.trading_pairs_config import get_trading_pairs

async def main():
    """Fonction principale du script."""
    print("🔄 Initialisation de l'API Kraken...")
    api = KrakenAPI()
    
    print("📡 Récupération des paires disponibles depuis Kraken...")
    await available_pairs.initialize(api=api)
    
    # Afficher les informations de base
    print("\n" + "="*80)
    print(f"{'PAIRES DISPONIBLES SUR KRAKEN':^80}")
    print("="*80)
    
    # Afficher les devises de cotation disponibles
    quote_currencies = sorted(available_pairs.get_quote_currencies())
    print(f"\n📊 Devises de cotation disponibles ({len(quote_currencies)}):")
    print(", ".join(quote_currencies))
    
    # Afficher les paires par devise de cotation
    for quote in sorted(quote_currencies):
        pairs = available_pairs.get_available_pairs(quote_currency=quote)
        if not pairs:
            continue
            
        print(f"\n{'='*40}")
        print(f"{quote} ({len(pairs)} paires)")
        print("-"*40)
        
        # Afficher 5 paires par ligne
        for i in range(0, len(pairs), 5):
            line = [f"{p:12}" for p in pairs[i:i+5]]
            print("  ".join(line))
    
    # Afficher les paires configurées
    print("\n" + "="*80)
    print(f"{'PAIRES CONFIGURÉES':^80}")
    print("="*80)
    
    config_pairs = get_trading_pairs(validate=True)
    for category, pairs in config_pairs.items():
        print(f"\n{category.upper().replace('_', ' ')} ({len(pairs)} paires):")
        if pairs:
            print(", ".join(pairs))
    
    print("\n" + "="*80)
    print(f"{'TOTAL':^80}")
    print("="*80)
    print(f"Paires haute liquidité: {len(config_pairs['high_liquidity'])}")
    print(f"Paires liquidité moyenne: {len(config_pairs['medium_liquidity'])}")
    print(f"Paires faible liquidité: {len(config_pairs['low_liquidity'])}")
    print(f"Total: {sum(len(p) for p in config_pairs.values())} paires")

if __name__ == "__main__":
    asyncio.run(main())
