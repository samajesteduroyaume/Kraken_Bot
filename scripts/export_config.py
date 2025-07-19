import yaml
import json
import csv
from pathlib import Path

CONFIG_PATH = Path('config/config.yaml')
EXPORT_JSON = Path('config/config_export.json')
EXPORT_CSV = Path('config/config_export.csv')

def load_config():
    with open(CONFIG_PATH, 'r') as f:
        return yaml.safe_load(f)

def export_json(config):
    with open(EXPORT_JSON, 'w') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    print(f"✅ Export JSON : {EXPORT_JSON}")

def export_csv(config):
    pairs = config['trading']['pairs']
    with open(EXPORT_CSV, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['pair'])
        for p in pairs:
            writer.writerow([p])
    print(f"✅ Export CSV : {EXPORT_CSV}")

def main():
    config = load_config()
    export_json(config)
    export_csv(config)

if __name__ == "__main__":
    main() 