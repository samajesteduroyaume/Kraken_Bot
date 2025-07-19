#!/bin/bash

# Script pour exécuter tous les contrôles de qualité

set -e  # Arrêter en cas d'erreur

# Couleurs pour la sortie
GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${GREEN}Démarrage des contrôles de qualité...${NC}"
echo "===================================="

# Black - Formatage du code
echo -e "\n${GREEN}1. Exécution de Black...${NC}"
if ! black --check .; then
    echo -e "${RED}Black a trouvé des problèmes de formatage. Exécutez 'black .' pour les corriger.${NC}"
    exit 1
fi

# isort - Organisation des imports
echo -e "\n${GREEN}2. Vérification de l'ordre des imports avec isort...${NC}"
if ! isort --check-only .; then
    echo -e "${RED}isort a trouvé des imports mal organisés. Exécutez 'isort .' pour les corriger.${NC}"
    exit 1
fi

# Flake8 - Vérification du style
echo -e "\n${GREEN}3. Vérification du style avec Flake8...${NC}"
if ! flake8 .; then
    echo -e "${RED}Flake8 a trouvé des problèmes de style.${NC}"
    exit 1
fi

# Pylint - Analyse statique
echo -e "\n${GREEN}4. Analyse statique avec Pylint...${NC}"
if ! pylint --rcfile=.pylintrc $(find . -name "*.py" | grep -v "venv/" | grep -v ".venv/" | grep -v "migrations/"); then
    echo -e "${YELLOW}Pylint a trouvé des problèmes à corriger.${NC}"
    # Ne pas échouer pour Pylint car il peut y avoir des avertissements
fi

# MyPy - Vérification des types
echo -e "\n${GREEN}5. Vérification des types avec MyPy...${NC}"
if ! mypy .; then
    echo -e "${RED}MyPy a trouvé des erreurs de typage.${NC}"
    exit 1
fi

echo -e "\n${GREEN}Tous les contrôles de qualité sont passés avec succès !${NC}"
