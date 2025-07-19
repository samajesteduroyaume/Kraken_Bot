#!/bin/bash

# Script de réinitialisation de la base de données Kraken Bot
# Ce script va :
# 1. Arrêter l'application si elle tourne
# 2. Supprimer et recréer la base de données
# 3. Exécuter le script d'initialisation

# Obtenir le répertoire du script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Charger les variables d'environnement depuis .env
if [ -f "$PROJECT_ROOT/.env" ]; then
    set -a
    source "$PROJECT_ROOT/.env"
    set +a
else
    echo -e "\033[1;31m[ERREUR] Fichier .env introuvable dans $PROJECT_ROOT\033[0m"
    exit 1
fi

# Fonction pour afficher les messages d'information
info() {
    echo -e "\033[1;34m[INFO]\033[0m $1"
}

# Fonction pour afficher les messages de succès
success() {
    echo -e "\033[1;32m[SUCCÈS]\033[0m $1"
}

# Fonction pour afficher les messages d'erreur et quitter
error() {
    echo -e "\033[1;31m[ERREUR]\033[0m $1"
    exit 1
}

# Vérifier si l'utilisateur est root
if [ "$(id -u)" -eq 0 ]; then
    error "Ce script ne doit pas être exécuté en tant que root"
fi

# Vérifier si les outils nécessaires sont installés
for cmd in psql dropdb createdb; do
    if ! command -v $cmd &> /dev/null; then
        error "La commande $cmd n'est pas installée. Veuillez installer postgresql-client"
    fi
done

# Arrêter l'application si elle tourne
info "Vérification des processus en cours d'exécution..."
if pgrep -f "python.*main.py" > /dev/null; then
    info "Arrêt de l'application en cours..."
    pkill -f "python.*main.py"
    sleep 2
    if pgrep -f "python.*main.py" > /dev/null; then
        error "Impossible d'arrêter l'application. Veuillez l'arrêter manuellement."
    fi
    success "Application arrêtée avec succès"
fi

# Vérifier la connexion à PostgreSQL
info "Vérification de la connexion à PostgreSQL..."
if ! PGPASSWORD=$POSTGRES_PASSWORD psql -h $POSTGRES_HOST -p $POSTGRES_PORT -U $POSTGRES_USER -d postgres -c "SELECT 1" > /dev/null 2>&1; then
    error "Impossible de se connecter à PostgreSQL. Vérifiez vos paramètres de connexion dans .env"
fi

# Demander confirmation avant de supprimer la base de données
read -p "Êtes-vous sûr de vouloir réinitialiser la base de données ${POSTGRES_DB} ? Cette action est irréversible. (o/N) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Oo]$ ]]; then
    info "Annulation de la réinitialisation de la base de données"
    exit 0
fi

# Supprimer la base de données existante
info "Suppression de la base de données ${POSTGRES_DB}..."
if ! PGPASSWORD=$POSTGRES_PASSWORD dropdb -h $POSTGRES_HOST -p $POSTGRES_PORT -U $POSTGRES_USER --if-exists $POSTGRES_DB; then
    error "Échec de la suppression de la base de données"
fi
success "Base de données supprimée avec succès"

# Créer une nouvelle base de données
info "Création d'une nouvelle base de données ${POSTGRES_DB}..."
if ! PGPASSWORD=$POSTGRES_PASSWORD createdb -h $POSTGRES_HOST -p $POSTGRES_PORT -U $POSTGRES_USER $POSTGRES_DB; then
    error "Échec de la création de la base de données"
fi
success "Nouvelle base de données créée avec succès"

# Exécuter le script d'initialisation
info "Exécution du script d'initialisation..."
if ! PGPASSWORD=$POSTGRES_PASSWORD psql -h $POSTGRES_HOST -p $POSTGRES_PORT -U $POSTGRES_USER -d $POSTGRES_DB -f "$PROJECT_ROOT/db/init_improved.sql"; then
    error "Échec de l'exécution du script d'initialisation"
fi

# Vérifier que les tables ont été créées
TABLE_COUNT=$(PGPASSWORD=$POSTGRES_PASSWORD psql -h $POSTGRES_HOST -p $POSTGRES_PORT -U $POSTGRES_USER -d $POSTGRES_DB -t -c "SELECT count(*) FROM information_schema.tables WHERE table_schema = 'public';" | tr -d '[:space:]')

if [ "$TABLE_COUNT" -lt 1 ]; then
    error "Aucune table n'a été créée. Vérifiez le script d'initialisation."
fi

success "Base de données réinitialisée avec succès !"
echo "Tables créées : $TABLE_COUNT"

exit 0
