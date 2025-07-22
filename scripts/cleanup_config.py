#!/usr/bin/env python3
"""
Script pour nettoyer le fichier de configuration et ne conserver que les 20 paires valides.
"""
import yaml
import os
from pathlib import Path

# Liste des 20 paires valides à conserver
VALID_PAIRS = [
    # Haute liquidité
    "XBT/USD", "XBT/USDC", "XBT/USDT", "ETH/USD", "ETH/USDT",
    "ETH/USDC", "XRP/USD", "SOL/USD", "ADA/USD", "DOT/USD",
    # Liquidité moyenne
    "LINK/USD", "MATIC/USD", "DOGE/USD", "AVAX/USD", "ATOM/USD",
    # Faible liquidité
    "ALGO/USD", "FIL/USD", "LTC/USD", "UNI/USD", "AAVE/USD"
]

def load_config():
    """Charge le fichier de configuration."""
    config_path = Path(__file__).parent.parent / 'config' / 'config.yaml'
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def save_config(config):
    """Sauvegarde le fichier de configuration."""
    config_path = Path(__file__).parent.parent / 'config' / 'config.yaml'
    # Sauvegarder l'ancienne version
    backup_path = config_path.with_suffix('.yaml.bak')
    if not backup_path.exists():
        with open(config_path, 'r') as src, open(backup_path, 'w') as dst:
            dst.write(src.read())
    
    # Sauvegarder la nouvelle version
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

def clean_trading_pairs(config):
    """Nettoie la liste des paires de trading pour ne conserver que les 20 valides."""
    if 'trading' in config and 'pairs' in config['trading']:
        # Remplacer complètement la liste par les 20 paires valides
        config['trading']['pairs'] = VALID_PAIRS.copy()
    return config

def main():
    """Fonction principale."""
    print("Nettoyage du fichier de configuration...")
    
    # Charger la configuration
    config = load_config()
    
    # Nettoyer les paires de trading
    config = clean_trading_pairs(config)
    
    # Sauvegarder la configuration
    save_config(config)
    
    print("Nettoyage terminé. Configuration sauvegardée.")
    print(f"Paires conservées: {len(config['trading']['pairs'])}")
    print("\n".join(f"- {p}" for p in config['trading']['pairs']))

if __name__ == "__main__":
    main()
