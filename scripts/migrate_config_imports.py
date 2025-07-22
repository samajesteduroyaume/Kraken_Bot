#!/usr/bin/env python3
"""
Script pour migrer les imports de l'ancien module de configuration vers le nouvel adaptateur.

Ce script parcourt tous les fichiers Python du projet et remplace les imports de
`src.core.config` par `src.core.config_adapter`.
"""
import os
import re
from pathlib import Path

# Dossiers à ignorer
IGNORE_DIRS = {
    'venv', '.git', '__pycache__', '.pytest_cache', 'node_modules',
    'dist', 'build', '*.egg-info', '*.pyc', '*.pyo', '*.pyd'
}

# Fichiers à ignorer
IGNORE_FILES = {
    'config.py', 'config_adapter.py', 'config_manager.py',
    'test_config_adapter.py', 'migrate_config_imports.py'
}

def should_ignore(path: Path) -> bool:
    """Vérifie si un fichier ou un dossier doit être ignoré."""
    # Ignorer les fichiers et dossiers cachés
    if any(part.startswith('.') and part != '.' for part in path.parts):
        return True
    
    # Ignorer les dossiers spécifiés
    if path.name in IGNORE_DIRS:
        return True
    
    # Ignorer les fichiers spécifiés
    if path.name in IGNORE_FILES:
        return True
    
    # Ignorer les fichiers non Python
    if path.suffix != '.py':
        return True
    
    return False

def update_file_imports(file_path: Path) -> int:
    """Met à jour les imports dans un fichier et retourne le nombre de modifications."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Rechercher et remplacer les imports
        pattern = r'from\s+src\.core\.config\s+import\s+Config\b'
        replacement = 'from src.core.config_adapter import Config  # Migré vers le nouvel adaptateur'
        
        new_content, count = re.subn(pattern, replacement, content)
        
        # Si des modifications ont été effectuées, écrire le fichier mis à jour
        if count > 0:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(new_content)
            return count
        
        return 0
    except Exception as e:
        print(f"Erreur lors du traitement du fichier {file_path}: {e}")
        return 0

def main():
    """Fonction principale."""
    # Définir le répertoire racine du projet
    root_dir = Path(__file__).parent.parent
    print(f"Début de la migration des imports dans: {root_dir}")
    
    # Compter les fichiers modifiés
    modified_files = 0
    total_updates = 0
    
    # Parcourir tous les fichiers Python du projet
    for file_path in root_dir.rglob('*.py'):
        if should_ignore(file_path):
            continue
        
        # Mettre à jour les imports du fichier
        updates = update_file_imports(file_path)
        
        if updates > 0:
            modified_files += 1
            total_updates += updates
            print(f"  - {file_path.relative_to(root_dir)}: {updates} import(s) mis à jour")
    
    # Afficher un résumé
    print(f"\nMigration terminée:")
    print(f"- Fichiers modifiés: {modified_files}")
    print(f"- Total des imports mis à jour: {total_updates}")
    
    if modified_files > 0:
        print("\nN'oubliez pas de vérifier que tout fonctionne correctement")
        print("avant de supprimer l'ancien fichier de configuration.")
        print("Ancien fichier: src/core/config.py")
        print("Nouveau fichier: src/core/config_adapter.py")

if __name__ == "__main__":
    main()
