import os
import json
from dotenv import load_dotenv

# Charger les variables d'environnement
load_dotenv()

def count_avaa_pairs():
    # Chemin vers le cache des paires (à adapter selon votre implémentation)
    cache_dir = os.path.join(os.path.dirname(__file__), 'data', 'cache')
    pair_cache_file = os.path.join(cache_dir, 'kraken_asset_pairs.json')
    
    if not os.path.exists(pair_cache_file):
        print("Fichier de cache des paires non trouvé.")
        return
    
    # Charger le cache des paires
    with open(pair_cache_file, 'r') as f:
        pairs_data = json.load(f)
    
    # Filtrer les paires commençant par AVAA
    avaa_pairs = {}
    for pair_id, pair_info in pairs_data.items():
        if 'wsname' in pair_info and 'AVAA' in pair_info['wsname']:
            avaa_pairs[pair_id] = pair_info
    
    # Afficher les résultats
    print(f"\n{len(avaa_pairs)} paires contenant 'AVAA' trouvées dans le cache :\n")
    for pair_id, info in avaa_pairs.items():
        print(f"ID: {pair_id}")
        print(f"WS Name: {info.get('wsname', 'N/A')}")
        print(f"Alt Name: {info.get('altname', 'N/A')}")
        print(f"Base: {info.get('base', 'N/A')}")
        print(f"Quote: {info.get('quote', 'N/A')}")
        print("-" * 50)
    
    # Vérifier les paires AVAAI spécifiquement
    avaai_pairs = {k: v for k, v in pairs_data.items() 
                  if 'wsname' in v and 'AVAAI/' in v['wsname']}
    
    print(f"\n{len(avaai_pairs)} paires AVAAI trouvées dans le cache :\n")
    for pair_id, info in avaai_pairs.items():
        print(f"ID: {pair_id}")
        print(f"WS Name: {info.get('wsname', 'N/A')}")
        print(f"Status: {info.get('status', 'N/A')}")
        print("-" * 50)

if __name__ == "__main__":
    count_avaa_pairs()
