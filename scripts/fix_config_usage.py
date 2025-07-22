#!/usr/bin/env python3
"""
Script pour corriger les appels à Config() dans tout le code source.

Ce script remplace toutes les occurrences de `Config()` par `Config`
pour refléter le passage d'une classe instanciable à une classe statique.
"""
import os
import re
from pathlib import Path

# Dossiers à ignorer
IGNORE_DIRS = {
    'venv', '.git', '__pycache__', '.pytest_cache', 'node_modules',
    'dist', 'build', '*.egg-info', '*.pyc', '*.pyo', '*.pyd',
    'config.py'  # On ignore l'ancien fichier de configuration
}

# Fichiers à ignorer
IGNORE_FILES = {
    'config.py', 'config_adapter.py', 'config_manager.py',
    'test_config_adapter.py', 'fix_config_usage.py'
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

def update_file_config_usage(file_path: Path) -> int:
    """
    Remplace les appels à Config() par Config dans un fichier.
    Retourne le nombre de modifications effectuées.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Rechercher et remplacer les appels à Config()
        pattern = r'(?<![\w.])Config\(\)'
        replacement = 'Config'
        
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
    print(f"Début de la correction des appels à Config() dans: {root_dir}")
    
    # Compter les fichiers modifiés
    modified_files = 0
    total_updates = 0
    
    # Parcourir tous les fichiers Python du projet
    for file_path in root_dir.rglob('*.py'):
        if should_ignore(file_path):
            continue
        
        # Mettre à jour les appels à Config() dans le fichier
        updates = update_file_config_usage(file_path)
        
        if updates > 0:
            modified_files += 1
            total_updates += updates
            print(f"  - {file_path.relative_to(root_dir)}: {updates} appel(s) à Config() corrigé(s)")
    
    # Afficher un résumé
    print(f"\nCorrection terminée:")
    print(f"- Fichiers modifiés: {modified_files}")
    print(f"- Total des corrections: {total_updates}")
    
    if modified_files > 0:
        print("\nVeuillez vérifier que les modifications sont correctes")
        print("et exécuter les tests pour vous assurer que tout fonctionne comme prévu.")

if __name__ == "__main__":
    main()
