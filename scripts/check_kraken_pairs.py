#!/usr/bin/env python3
"""
Script pour vérifier les paires disponibles sur Kraken et rechercher des actifs spécifiques.
"""
import asyncio
import aiohttp
import json
from typing import Dict, List, Optional
from urllib.parse import urljoin

class KrakenPairChecker:
    BASE_URL = "https://api.kraken.com/0/public/"
    
    def __init__(self):
        self.session = aiohttp.ClientSession()
    
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.session.close()
    
    async def get_asset_pairs(self) -> Dict:
        """Récupère toutes les paires d'actifs disponibles sur Kraken."""
        url = urljoin(self.BASE_URL, "AssetPairs")
        async with self.session.get(url) as response:
            response.raise_for_status()
            return await response.json()
    
    async def search_asset(self, asset_name: str) -> List[Dict]:
        """Recherche un actif spécifique dans les paires disponibles."""
        pairs_data = await self.get_asset_pairs()
        results = []
        
        if 'result' not in pairs_data:
            print("Erreur lors de la récupération des paires:", pairs_data)
            return results
        
        asset_name = asset_name.upper()
        for pair_name, pair_info in pairs_data['result'].items():
            if (asset_name in pair_name or 
                asset_name in pair_info.get('wsname', '') or
                asset_name in pair_info.get('base', '') or
                asset_name in pair_info.get('quote', '')):
                results.append({
                    'pair': pair_name,
                    'wsname': pair_info.get('wsname', 'N/A'),
                    'base': pair_info.get('base', 'N/A'),
                    'quote': pair_info.get('quote', 'N/A'),
                    'status': pair_info.get('status', 'N/A')
                })
        
        return results

async def main():
    assets_to_check = ['APE', 'APENFT', 'BTC', 'ETH']
    
    async with KrakenPairChecker() as checker:
        # Vérifier les paires pour chaque actif
        for asset in assets_to_check:
            print(f"\n🔍 Recherche des paires pour {asset}...")
            pairs = await checker.search_asset(asset)
            
            if pairs:
                print(f"✅ {len(pairs)} paires trouvées pour {asset}:")
                for idx, pair in enumerate(pairs, 1):
                    print(f"{idx}. {pair['pair']} (WS: {pair['wsname']})")
                    print(f"   Base: {pair['base']}, Quote: {pair['quote']}")
                    print(f"   Statut: {pair['status']}")
            else:
                print(f"❌ Aucune paire trouvée pour {asset}")
        
        # Afficher le nombre total de paires disponibles
        all_pairs = await checker.get_asset_pairs()
        if 'result' in all_pairs:
            print(f"\nℹ️ Nombre total de paires disponibles: {len(all_pairs['result'])}")
        else:
            print("\n❌ Impossible de récupérer le nombre total de paires")

if __name__ == "__main__":
    print("🔄 Vérification des paires disponibles sur Kraken...")
    asyncio.run(main())
