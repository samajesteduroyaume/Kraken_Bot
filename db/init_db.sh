#!/bin/bash

# Vérifier si PostgreSQL est installé
if ! command -v psql &> /dev/null; then
    echo "PostgreSQL n'est pas installé. Installation en cours..."
    sudo apt update
    sudo apt install -y postgresql postgresql-contrib
fi

# Vérifier si PostgreSQL est démarré
if ! systemctl is-active --quiet postgresql; then
    echo "Démarrage de PostgreSQL..."
    sudo systemctl start postgresql
fi

# Créer la base de données et l'utilisateur
sudo -u postgres psql -f init_db.sql

# Vérifier la création
if [ $? -eq 0 ]; then
    echo "Base de données initialisée avec succès !"
else
    echo "Erreur lors de l'initialisation de la base de données"
fi
