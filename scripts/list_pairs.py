#!/usr/bin/env python3
"""
Script pour lister toutes les paires de trading disponibles sur Kraken.
"""
import asyncio
import sys
import os
import json
from typing import Dict, List, Optional

# Ajouter le répertoire src au chemin Python
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from core.api.kraken import KrakenAPI

async def list_available_pairs() -> None:
    """Affiche toutes les paires de trading disponibles sur Kraken."""
    try:
        # Initialiser l'API Kraken
        api = KrakenAPI()
        
        # Obtenir toutes les paires d'actifs
        print("⏳ Récupération des paires de trading depuis Kraken...")
        asset_pairs = await api.get_asset_pairs()
        
        if not asset_pairs or not isinstance(asset_pairs, dict):
            print("❌ Aucune paire de trading trouvée ou format de réponse inattendu")
            return
        
        # Extraire les noms des paires et les trier
        pairs = sorted(asset_pairs.keys())
        
        # Afficher le nombre total de paires
        print(f"\n✅ {len(pairs)} paires de trading disponibles sur Kraken :\n")
        
        # Afficher les paires par groupe de 5 pour une meilleure lisibilité
        for i in range(0, len(pairs), 5):
            print("  " + "  ".join(f"{pair:12}" for pair in pairs[i:i+5]))
        
        # Optionnel : sauvegarder dans un fichier
        with open('kraken_pairs.json', 'w') as f:
            json.dump(asset_pairs, f, indent=2)
        print(f"\n📝 Liste des paires sauvegardée dans 'kraken_pairs.json'")
        
    except Exception as e:
        print(f"❌ Erreur lors de la récupération des paires : {str(e)}")
        if hasattr(e, '__traceback__'):
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(list_available_pairs())
