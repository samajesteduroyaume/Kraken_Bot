import requests
import yaml
from pathlib import Path
import sys

CONFIG_PATH = Path('config/config.yaml')
API_URL = 'https://api.kraken.com/0/public/AssetPairs'

def load_config():
    with open(CONFIG_PATH, 'r') as f:
        return yaml.safe_load(f)

def validate_pairs(pairs, available_pairs):
    missing = [p for p in pairs if p not in available_pairs]
    return missing

def main():
    config = load_config()
    pairs = config['trading']['pairs']
    min_score = config['trading'].get('min_score', 0)
    min_volume = config['trading'].get('min_volume', 0)
    print("Vérification des paires Kraken...")
    resp = requests.get(API_URL)
    data = resp.json()
    if 'result' not in data:
        print("Erreur lors de la récupération des paires Kraken.")
        sys.exit(1)
    available_pairs = set(data['result'].keys())
    missing = validate_pairs(pairs, available_pairs)
    if missing:
        print(f"❌ {len(missing)} paires non trouvées sur Kraken : {missing[:10]} ...")
        sys.exit(1)
    if not (0 <= min_score <= 1):
        print(f"❌ min_score incohérent : {min_score}")
        sys.exit(1)
    if min_volume < 0:
        print(f"❌ min_volume incohérent : {min_volume}")
        sys.exit(1)
    print("✅ Config valide : toutes les paires existent et les critères sont cohérents.")
    sys.exit(0)

if __name__ == "__main__":
    main() 