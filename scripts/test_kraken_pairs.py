#!/usr/bin/env python3
"""
Script autonome pour tester la récupération des paires de trading depuis l'API Kraken.

Ce script peut être exécuté directement depuis le répertoire du projet.
"""
import asyncio
import aiohttp
import json
import os
from pathlib import Path

# Configuration
KRAKEN_API_URL = "https://api.kraken.com/0/public/AssetPairs"
CACHE_FILE = Path("kraken_pairs_cache.json")

async def fetch_available_pairs(session):
    """Récupère toutes les paires disponibles depuis l'API Kraken."""
    async with session.get(KRAKEN_API_URL) as response:
        data = await response.json()
        if data.get('error'):
            print(f"Erreur de l'API Kraken: {data['error']}")
            return None
        return data.get('result', {})

def save_to_cache(data):
    """Sauvegarde les données dans un fichier cache."""
    with open(CACHE_FILE, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"Données sauvegardées dans {CACHE_FILE}")

def load_from_cache():
    """Charge les données depuis le cache si disponible."""
    if CACHE_FILE.exists():
        with open(CACHE_FILE, 'r') as f:
            return json.load(f)
    return None

def analyze_pairs(pairs_data):
    """Analyse et affiche des statistiques sur les paires disponibles."""
    if not pairs_data:
        print("Aucune donnée de paire disponible.")
        return
    
    # Extraire les paires et les organiser par devise de cotation
    quote_currencies = {}
    pair_count = 0
    
    for pair_id, pair_info in pairs_data.items():
        if not isinstance(pair_info, dict):
            continue
            
        quote = pair_info.get('quote')
        wsname = pair_info.get('wsname', '')
        
        if quote and wsname:
            if quote not in quote_currencies:
                quote_currencies[quote] = []
            quote_currencies[quote].append(wsname)
            pair_count += 1
    
    # Afficher les statistiques
    print(f"\n📊 STATISTIQUES DES PAIRES KRAKEN")
    print("=" * 50)
    print(f"Nombre total de paires uniques: {pair_count}")
    print(f"Nombre de devises de cotation: {len(quote_currencies)}")
    
    # Afficher les principales devises de cotation
    print("\nPrincipales devises de cotation (par nombre de paires):")
    sorted_quotes = sorted(quote_currencies.items(), key=lambda x: len(x[1]), reverse=True)
    
    for quote, pairs in sorted_quotes[:10]:  # Top 10
        print(f"- {quote}: {len(pairs)} paires")
    
    # Afficher quelques exemples de paires pour les principales devises
    print("\nExemples de paires par devise de cotation:")
    for quote, pairs in sorted_quotes[:3]:  # Top 3
        print(f"\n{quote} (exemples):")
        for i in range(0, min(5, len(pairs)), 2):
            print(f"  {pairs[i]:<15}", end="")
            if i+1 < len(pairs):
                print(f"  {pairs[i+1]:<15}", end="")
            print()
    
    return quote_currencies

async def main():
    """Fonction principale."""
    # Vérifier d'abord le cache
    pairs_data = load_from_cache()
    use_cache = pairs_data is not None
    
    if use_cache:
        print(f"Données chargées depuis le cache ({CACHE_FILE})")
    else:
        print("Connexion à l'API Kraken...")
        async with aiohttp.ClientSession() as session:
            pairs_data = await fetch_available_pairs(session)
            
        if pairs_data:
            save_to_cache(pairs_data)
    
    # Analyser et afficher les résultats
    if pairs_data:
        quote_currencies = analyze_pairs(pairs_data)
        
        # Demander à l'utilisateur s'il veut voir les paires pour une devise spécifique
        while True:
            print("\nEntrez une devise de cotation (ex: USD, EUR, XBT) ou 'q' pour quitter:")
            user_input = input("> ").strip().upper()
            
            if user_input.lower() == 'q':
                break
                
            if user_input in quote_currencies:
                pairs = sorted(quote_currencies[user_input])
                print(f"\nPaires disponibles en {user_input} ({len(pairs)}):")
                for i in range(0, len(pairs), 5):
                    print("  ".join(f"{p:<15}" for p in pairs[i:i+5]))
            else:
                print(f"Aucune paire trouvée pour la devise {user_input}")
                print("Devises disponibles:", ", ".join(sorted(quote_currencies.keys())[:20]) + "...")
    
    print("\nAu revoir!")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nArrêt demandé par l'utilisateur.")
    except Exception as e:
        print(f"\nErreur: {e}")
