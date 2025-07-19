import requests
import yaml
from pathlib import Path

CONFIG_PATH = Path('config/config.yaml')
API_URL = 'https://api.kraken.com/0/public/AssetPairs'

def load_config():
    with open(CONFIG_PATH, 'r') as f:
        return yaml.safe_load(f)

def save_config(config):
    with open(CONFIG_PATH, 'w') as f:
        yaml.dump(config, f, sort_keys=False, allow_unicode=True)

def update_pairs():
    print("Récupération des paires Kraken...")
    resp = requests.get(API_URL)
    data = resp.json()
    if 'result' not in data:
        print("Erreur lors de la récupération des paires Kraken.")
        return
    all_pairs = list(data['result'].keys())
    config = load_config()
    config['trading']['pairs'] = all_pairs
    save_config(config)
    print(f"✅ {len(all_pairs)} paires ajoutées à la config.")

if __name__ == "__main__":
    update_pairs() 