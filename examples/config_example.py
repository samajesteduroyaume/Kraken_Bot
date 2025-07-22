"""
Exemple d'utilisation du gestionnaire de configuration.

Ce script montre comment charger et utiliser la configuration du bot de trading
à partir du fichier YAML et des variables d'environnement.
"""
import os
import sys
from pathlib import Path

# Ajouter le répertoire racine au PYTHONPATH
root_dir = str(Path(__file__).parent.parent)
if root_dir not in sys.path:
    sys.path.append(root_dir)

from src.config_manager import config

def main():
    """Fonction principale qui affiche des exemples d'utilisation de la configuration."""
    print("=== Exemple d'utilisation du gestionnaire de configuration ===\n")
    
    # 1. Accès aux valeurs de configuration
    print("1. Valeurs de configuration de base:")
    print(f"   - Mode de trading: {config.get('trading.mode', 'non défini')}")
    print(f"   - Paires de trading: {config.get('trading.pairs', [])[:3]}...")
    print(f"   - Niveau de log: {config.get('log.level')}")
    
    # 2. Accès à une section complète
    print("\n2. Configuration de la base de données:")
    db_config = config.get_section('postgres')
    for key, value in db_config.items():
        if 'password' in key and value:
            value = '********'  # Masquer les mots de passe
        print(f"   - {key}: {value}")
    
    # 3. Configuration des stratégies
    print("\n3. Stratégies activées:")
    strategies = config.get_section('strategies')
    for strategy, params in strategies.items():
        if params.get('enabled', False):
            print(f"   - {strategy.capitalize()} (poids: {params.get('weight', 0)})")
    
    # 4. Mise à jour d'une valeur
    print("\n4. Mise à jour d'une valeur de configuration:")
    old_value = config.get('trading.max_positions', 0)
    print(f"   - Ancienne valeur de max_positions: {old_value}")
    
    # Mettre à jour la valeur (uniquement en mémoire)
    config.update('trading.max_positions', 5)
    print(f"   - Nouvelle valeur de max_positions: {config.get('trading.max_positions')}")
    
    # Remettre l'ancienne valeur
    config.update('trading.max_positions', old_value)
    
    # 5. Utilisation des valeurs de configuration pour la logique métier
    print("\n5. Utilisation dans la logique métier:")
    current_balance = 10000  # Exemple de solde
    risk_per_trade = config.get('risk_management.max_risk_per_trade', 0.02)
    max_risk_amount = current_balance * risk_per_trade
    print(f"   - Solde actuel: ${current_balance:.2f}")
    print(f"   - Risque par trade: {risk_per_trade*100:.1f}% (${max_risk_amount:.2f})")
    
    # 6. Vérification des variables d'environnement
    print("\n6. Variables d'environnement détectées:")
    env_vars = [
        'KRAKEN_API_KEY', 'KRAKEN_API_SECRET',
        'POSTGRES_USER', 'POSTGRES_PASSWORD',
        'TRADING_MODE', 'LOG_LEVEL'
    ]
    for var in env_vars:
        value = os.getenv(var)
        if value:
            display_value = value[:3] + '...' + value[-3:] if len(value) > 6 else '********'
            print(f"   - {var}: {display_value}")
        else:
            print(f"   - {var}: Non définie")
    
    print("\n=== Fin de l'exemple ===")

if __name__ == "__main__":
    main()
