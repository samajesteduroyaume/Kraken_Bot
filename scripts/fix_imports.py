import os
import re
from pathlib import Path

# Mappage des anciens chemins vers les nouveaux
IMPORT_MAPPING = {
    # Ancien chemin -> Nouveau chemin
    'from src\.core\.': 'from kraken_bot.core.',
    'from src\.ml\.': 'from kraken_bot.ml.',
    'from src\.utils\.': 'from kraken_bot.utils.',
    'from src\.core\.types\.': 'from kraken_bot.core.types.',
    'from src\.core\.trading\.': 'from kraken_bot.core.trading.',
    'from src\.core\.api\.': 'from kraken_bot.api.',
    'from src\.core\.signals\.': 'from kraken_bot.core.signals.',
    'from src\.core\.strategies\.': 'from kraken_bot.core.strategies.',
    'from src\.core\.analysis\.': 'from kraken_bot.core.indicators.',
    'from src\.core\.risk\.': 'from kraken_bot.risk.',
}

def update_imports_in_file(file_path):
    """Met à jour les imports dans un fichier."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        original_content = content
        
        # Appliquer les remplacements
        for old, new in IMPORT_MAPPING.items():
            content = re.sub(fr'\b{old}\b', new, content)
        
        # Écrire uniquement si des modifications ont été apportées
        if content != original_content:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"Updated imports in {file_path}")
            return True
        return False
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return False

def main():
    # Parcourir tous les fichiers Python
    root_dir = Path('src/kraken_bot')
    updated_files = 0
    
    for py_file in root_dir.rglob('*.py'):
        if update_imports_in_file(py_file):
            updated_files += 1
    
    print(f"\nMise à jour terminée. {updated_files} fichiers modifiés.")

if __name__ == '__main__':
    main()
