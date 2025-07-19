#!/bin/bash

# Configuration
REPO_URL="https://github.com/votre-compte/Kraken_Bot-main.git"
UPDATE_INTERVAL=86400  # 24 heures
LOG_FILE="/app/logs/update.log"

# Fonction pour vérifier les mises à jour
check_updates() {
    echo "🔍 Vérification des mises à jour..." | tee -a $LOG_FILE
    
    # Sauvegarder l'état actuel
    git status > /tmp/current_status
    
    # Pull les changements
    git pull origin main
    
    # Vérifier si des changements ont été trouvés
    if ! diff /tmp/current_status <(git status) > /dev/null; then
        echo "✅ Nouvelle version trouvée !" | tee -a $LOG_FILE
        
        # Installer les nouvelles dépendances
        pip install -r requirements.txt
        
        # Redémarrer les services
        docker-compose down
        docker-compose up -d
        
        echo "✅ Mise à jour terminée" | tee -a $LOG_FILE
    else
        echo "ℹ️ Aucune mise à jour disponible" | tee -a $LOG_FILE
    fi
}

# Fonction pour vérifier l'état du système
check_system() {
    echo "🔍 Vérification du système..." | tee -a $LOG_FILE
    
    # Vérifier l'espace disque
    df -h | tee -a $LOG_FILE
    
    # Vérifier la mémoire
    free -h | tee -a $LOG_FILE
    
    # Vérifier les services
    docker-compose ps | tee -a $LOG_FILE
}

# Fonction principale
main() {
    # Initialisation
    echo "🚀 Début de la vérification" | tee -a $LOG_FILE
    
    # Vérifier les mises à jour
    check_updates
    
    # Vérifier le système
    check_system
    
    # Attendre avant la prochaine vérification
    echo "⏳ Attente avant la prochaine vérification..." | tee -a $LOG_FILE
    sleep $UPDATE_INTERVAL
    
    # Relancer le script
    exec $0
}

# Exécuter le script
main
