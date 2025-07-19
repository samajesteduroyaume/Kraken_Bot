#!/bin/bash
set -euo pipefail

# Configuration
DB_WAIT_TIME=${DB_WAIT_TIME:-30}  # Augmentation du temps d'attente par défaut
SERVICE_START_DELAY=${SERVICE_START_DELAY:-10}
MAX_RETRIES=5
RETRY_DELAY=5

# Créer le répertoire de logs s'il n'existe pas
mkdir -p logs

# Fonction pour logger les messages
log() {
    local level="INFO"
    local message="$1"
    
    # Détection du niveau de log
    if [[ "$1" == "ERREUR"* ]] || [[ "$1" == "ERROR"* ]]; then
        level="ERROR"
    elif [[ "$1" == "AVERTISSEMENT"* ]] || [[ "$1" == "WARNING"* ]]; then
        level="WARNING"
    fi
    
    local timestamp=$(date +'%Y-%m-%d %H:%M:%S')
    local log_entry="[${timestamp}] [${level}] ${message}"
    
    # Afficher sur la sortie standard
    echo "${log_entry}"
    
    # Écrire dans le fichier de log
    echo "${log_entry}" >> logs/entrypoint.log
}

# Fonction pour exécuter une commande avec des tentatives
run_with_retry() {
    local cmd="$1"
    local max_attempts=${2:-3}
    local delay=${3:-5}
    local attempt=1
    local exit_code=0

    while true; do
        log "Tentative ${attempt}/${max_attempts}: ${cmd}"
        
        # Exécuter la commande et capturer la sortie
        if output=$($cmd 2>&1); then
            log "Succès: ${cmd}"
            echo "${output}"
            return 0
        else
            exit_code=$?
            log "Échec (code ${exit_code}) de la commande: ${cmd}"
            log "Sortie d'erreur: ${output}"
            
            if [ ${attempt} -lt ${max_attempts} ]; then
                log "Nouvelle tentative dans ${delay} secondes..."
                sleep ${delay}
                attempt=$((attempt + 1))
            else
                log "Nombre maximum de tentatives atteint (${max_attempts}). Abandon."
                return ${exit_code}
            fi
        fi
    done
}

# Vérifier les dépendances Python
check_python_deps() {
    log "Vérification des dépendances Python..."
    local required_deps=("python" "pip")
    
    for dep in "${required_deps[@]}"; do
        if ! command -v ${dep} &> /dev/null; then
            log "ERREUR: ${dep} n'est pas installé"
            exit 1
        fi
    done
    
    log "Toutes les dépendances système sont installées"
}

# Fonction pour vérifier la connexion à la base de données
check_db_connection() {
    log "Vérification de la connexion à la base de données..."
    
    local max_attempts=5
    local attempt=1
    
    while [ $attempt -le $max_attempts ]; do
        if python scripts/check_db.py; then
            log "Connexion à la base de données établie avec succès"
            return 0
        fi
        
        log "Tentative $attempt/$max_attempts échouée. Nouvelle tentative dans $RETRY_DELAY secondes..."
        sleep $RETRY_DELAY
        attempt=$((attempt + 1))
    done
    
    log "ERREUR: Impossible de se connecter à la base de données après $max_attempts tentatives"
    return 1
}

# Fonction pour initialiser la base de données
initialize_database() {
    log "Initialisation de la base de données..."
    
    if ! run_with_retry "python scripts/init_db.py" 5 5; then
        log "ERREUR: Échec de l'initialisation de la base de données après 5 tentatives"
        return 1
    fi
    
    log "Base de données initialisée avec succès"
    return 0
}

# Fonction pour démarrer le service ML
start_ml_service() {
    if [ "${ENABLE_ML_SERVICE:-true}" != "true" ]; then
        log "Service ML désactivé (ENABLE_ML_SERVICE=false)"
        return 0
    fi
    
    log "Démarrage du service d'entraînement automatique..."
    
    # Vérifier si le module auto_train existe
    if ! python -c "import src.services.auto_train" &> /dev/null; then
        log "AVERTISSEMENT: Le module src.services.auto_train n'existe pas. Le service ML ne sera pas démarré."
        return 0
    fi
    
    # Démarrer le service en arrière-plan
    python -m src.services.auto_train >> logs/ml_service.log 2>&1 &
    ML_PID=$!
    
    # Vérifier si le processus est toujours en cours d'exécution
    if ! ps -p ${ML_PID} > /dev/null; then
        log "ERREUR: Le service ML n'a pas pu démarrer"
        cat logs/ml_service.log
        return 1
    fi
    
    log "Service ML démarré avec succès (PID: ${ML_PID})"
    
    # Attente supplémentaire pour s'assurer que le service ML est prêt
    log "Attente de ${SERVICE_START_DELAY} secondes pour le démarrage du service ML..."
    sleep "${SERVICE_START_DELAY}"
    
    return 0
}

# Fonction principale
main() {
    log "Démarrage de l'initialisation du conteneur..."
    
    # Vérifier les dépendances
    check_python_deps
    
    # Attente pour la base de données
    log "Attente de ${DB_WAIT_TIME} secondes pour la base de données..."
    sleep "${DB_WAIT_TIME}"
    
    # Vérifier la connexion à la base de données
    if ! check_db_connection; then
        log "ERREUR: Impossible de se connecter à la base de données après ${MAX_RETRIES} tentatives"
        exit 1
    fi
    
    # Initialiser la base de données
    if ! initialize_database; then
        exit 1
    fi
    
    # Démarrer le service ML si activé
    if ! start_ml_service; then
        log "AVERTISSEMENT: Le service ML n'a pas pu démarrer, mais l'application va continuer"
    fi
    
    # Démarrer l'application principale
    log "Démarrage de l'application principale..."
    
    # Exécuter la commande passée en argument (par défaut: python main.py)
    if [ $# -gt 0 ]; then
        log "Exécution de la commande: $*"
        exec "$@"
    else
        log "Aucune commande spécifiée, utilisation par défaut: python main.py"
        exec python main.py
    fi
}

# Démarrer le script
main "$@"
