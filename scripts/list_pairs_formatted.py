#!/usr/bin/env python3
"""
Script pour lister toutes les paires de trading disponibles sur Kraken de manière formatée.
"""
import asyncio
import sys
import os
import json
from typing import Dict, List, Optional
from collections import defaultdict

# Ajouter le répertoire src au chemin Python
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from core.api.kraken import KrakenAPI

async def get_available_pairs() -> Dict[str, List[str]]:
    """Récupère toutes les paires de trading disponibles sur Kraken."""
    try:
        # Initialiser l'API Kraken
        api = KrakenAPI()
        
        # Obtenir toutes les paires d'actifs
        print("⏳ Récupération des paires de trading depuis Kraken...")
        asset_pairs = await api.get_asset_pairs()
        
        if not asset_pairs or not isinstance(asset_pairs, dict):
            print("❌ Aucune paire de trading trouvée ou format de réponse inattendu")
            return {}
            
        return asset_pairs.get('result', {})
        
    except Exception as e:
        print(f"❌ Erreur lors de la récupération des paires : {str(e)}")
        return {}

def organize_pairs_by_quote(pairs_data: Dict) -> Dict[str, List[Dict]]:
    """Organise les paires par devise de cotation."""
    organized = defaultdict(list)
    
    for pair_id, pair_info in pairs_data.items():
        if not isinstance(pair_info, dict):
            continue
            
        wsname = pair_info.get('wsname', '')
        base = pair_info.get('base', '')
        quote = pair_info.get('quote', '')
        
        if not all([wsname, base, quote]):
            continue
            
        organized[quote].append({
            'id': pair_id,
            'wsname': wsname,
            'base': base,
            'quote': quote,
            'altname': pair_info.get('altname', '')
        })
    
    return dict(organized)

def display_pairs(organized_pairs: Dict[str, List[Dict]]):
    """Affiche les paires de manière organisée."""
    # Trier les devises de cotation
    sorted_quotes = sorted(organized_pairs.keys())
    
    print("\n" + "="*80)
    print(f"{'PAIRES DE TRADING DISPONIBLES SUR KRAKEN':^80}")
    print("="*80)
    
    total_pairs = sum(len(pairs) for pairs in organized_pairs.values())
    print(f"\nNombre total de paires : {total_pairs}\n")
    
    for quote in sorted_quotes:
        pairs = organized_pairs[quote]
        print(f"\n{'-'*40}")
        print(f"{quote} (x{len(pairs)} paires)")
        print("-"*40)
        
        # Trier les paires par nom de base
        sorted_pairs = sorted(pairs, key=lambda x: x['base'])
        
        # Afficher 5 paires par ligne
        for i in range(0, len(sorted_pairs), 5):
            line = []
            for pair in sorted_pairs[i:i+5]:
                display_name = pair['wsname'] or f"{pair['base']}/{pair['quote']}"
                line.append(f"{display_name:15}")
            print("  ".join(line))
    
    print("\n" + "="*80)
    print(f"{'FIN DE LA LISTE':^80}")
    print("="*80)

async def main():
    # Récupérer les paires
    pairs_data = await get_available_pairs()
    
    if not pairs_data:
        print("❌ Impossible de récupérer les paires de trading")
        return
    
    # Organiser les paires par devise de cotation
    organized_pairs = organize_pairs_by_quote(pairs_data)
    
    # Afficher les paires
    display_pairs(organized_pairs)
    
    # Sauvegarder les données brutes
    with open('kraken_pairs_organized.json', 'w') as f:
        json.dump(pairs_data, f, indent=2)
    print("\n📝 Données brutes sauvegardées dans 'kraken_pairs_organized.json'")

if __name__ == "__main__":
    asyncio.run(main())
