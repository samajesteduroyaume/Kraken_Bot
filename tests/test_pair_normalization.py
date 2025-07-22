#!/usr/bin/env python3
"""
Script de test pour vérifier la normalisation des paires de trading.
"""
import asyncio
import sys
from pathlib import Path

# Ajouter le répertoire src au PYTHONPATH
sys.path.append(str(Path(__file__).parent / 'src'))

from src.core.market import available_pairs

async def test_pair_normalization(pair_input):
    """Teste la normalisation d'une paire de trading."""
    print(f"\n🔍 Test de normalisation pour: {pair_input}")
    
    # Initialiser le marché
    await available_pairs().initialize(use_cache=True)
    
    # Normaliser la paire
    normalized = available_pairs.normalize_pair(pair_input)
    
    # Afficher le résultat
    if normalized:
        print(f"✅ Paire normalisée: {normalized}")
        
        # Vérifier si la paire normalisée existe dans les paires disponibles
        if normalized in available_pairs._wsname_to_pair_id:
            pair_id = available_pairs._wsname_to_pair_id[normalized]
            pair_info = available_pairs._pairs_data.get(pair_id, {})
            print(f"   - ID de paire: {pair_id}")
            print(f"   - Nom alternatif: {pair_info.get('altname', 'N/A')}")
            print(f"   - Statut: {pair_info.get('status', 'N/A')}")
            print(f"   - Base: {pair_info.get('base', 'N/A')}")
            print(f"   - Quote: {pair_info.get('quote', 'N/A')}")
        else:
            print(f"⚠️  La paire normalisée {normalized} n'a pas été trouvée dans les paires disponibles")
    else:
        print(f"❌ Impossible de normaliser la paire: {pair_input}")
        
        # Essayer de trouver des paires similaires
        print("\n🔍 Recherche de paires similaires...")
        pair_upper = pair_input.upper()
        similar_pairs = []
        
        for wsname in available_pairs._wsname_to_pair_id.keys():
            if pair_upper in wsname.replace('/', ''):
                similar_pairs.append(wsname)
                if len(similar_pairs) >= 5:  # Limiter à 5 résultats
                    break
        
        if similar_pairs:
            print("Paires similaires trouvées:")
            for i, pair in enumerate(similar_pairs, 1):
                print(f"  {i}. {pair}")
        else:
            print("Aucune paire similaire trouvée.")

async def main():
    # Paires à tester
    test_pairs = [
        # Paires qui posent problème
        'APE/EUR', 'APENFT/EUR', 'APENFT/USD',
        # Paires avec différents formats
        'APE-EUR', 'APE_EUR', 'APEEUR', 'APE EUR',
        'APENFT-EUR', 'APENFT_EUR', 'APENFTEUR', 'APENFT EUR',
        # Paires de référence
        'BTC/EUR', 'ETH/USD', 'XBT/USD'
    ]
    
    print("🔄 Test de normalisation des paires de trading\n")
    
    for pair in test_pairs:
        await test_pair_normalization(pair)
    
    # Demander à l'utilisateur d'entrer des paires personnalisées
    print("\n🔍 Entrez une paire de trading à tester (ou 'q' pour quitter):")
    while True:
        user_input = input("> ").strip()
        if user_input.lower() == 'q':
            break
        if user_input:
            await test_pair_normalization(user_input)
        print("\n🔍 Entrez une autre paire (ou 'q' pour quitter):")

if __name__ == "__main__":
    asyncio.run(main())
