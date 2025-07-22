"""
Script de migration pour la nouvelle version d'AvailablePairs.

Ce script met à jour tous les fichiers qui importent l'ancienne version d'AvailablePairs
pour utiliser la nouvelle version refactorisée.
"""

import os
import re
from pathlib import Path

# Fichiers à mettre à jour avec leurs chemins complets
FILES_TO_UPDATE = [
    "src/utils/pair_utils.py",
    "src/config/trading_pairs_config.py",
    "src/core/trading/advanced_trader.py",
    "src/core/api/kraken_api/endpoints.py"
]

def update_file(file_path: str) -> bool:
    """Met à jour un fichier pour utiliser la nouvelle version d'AvailablePairs."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Sauvegarder une copie de sécurité
        backup_path = f"{file_path}.bak"
        with open(backup_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        # Remplacer les imports et initialisations
        new_content = content
        
        # Remplacer l'import d'AvailablePairs
        new_content = re.sub(
            r'from\s+\.*\.\.core\.market\.available_pairs\s+import\s+AvailablePairs',
            'from ..core.market.available_pairs_refactored import AvailablePairs',
            new_content
        )
        
        # Remplacer l'import d'available_pairs
        new_content = re.sub(
            r'from\s+\.*\.\.core\.market\.available_pairs\s+import\s+available_pairs',
            'from ..core.market.available_pairs_refactored import available_pairs',
            new_content
        )
        
        # Remplacer l'import de KrakenAPIError
        new_content = re.sub(
            r'from\s+\.*\.\.core\.market\.available_pairs\s+import\s+KrakenAPIError',
            'from ..core.market.available_pairs_refactored import KrakenAPIError',
            new_content
        )
        
        # Supprimer l'initialisation redondante d'available_pairs
        new_content = re.sub(
            r'# Initialize the global available_pairs instance\navailable_pairs = AvailablePairs\(\)\n\n',
            '',
            new_content
        )
        
        # Si le contenu a changé, mettre à jour le fichier
        if new_content != content:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(new_content)
            print(f"✅ {file_path} mis à jour avec succès")
            return True
        else:
            print(f"ℹ️  Aucun changement nécessaire pour {file_path}")
            return False
            
    except Exception as e:
        print(f"❌ Erreur lors de la mise à jour de {file_path}: {e}")
        return False

def main():
    """Fonction principale du script de migration."""
    print("=== Migration vers la nouvelle version d'AvailablePairs ===\n")
    
    # Obtenir le répertoire racine du projet
    project_root = Path(__file__).parent.absolute()
    
    # Mettre à jour chaque fichier
    updated_files = 0
    
    for rel_path in FILES_TO_UPDATE:
        file_path = project_root / rel_path
        if file_path.exists():
            print(f"\nTraitement de {rel_path}...")
            if update_file(str(file_path)):
                updated_files += 1
        else:
            print(f"⚠️  Fichier non trouvé : {rel_path}")
    
    print(f"\n=== Migration terminée ===")
    print(f"{updated_files} fichiers mis à jour sur {len(FILES_TO_UPDATE)} fichiers traités.")
    print("\nN'oubliez pas de :")
    print("1. Tester soigneusement votre application après la migration")
    print("2. Vérifier que toutes les fonctionnalités liées aux paires fonctionnent comme prévu")
    print("3. Supprimer les fichiers de sauvegarde (*.bak) une fois la migration validée")

if __name__ == "__main__":
    main()
