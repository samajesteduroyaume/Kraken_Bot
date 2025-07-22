#!/bin/bash

# ============================================
# Script de démarrage du Kraken Trading Bot
# ============================================

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"  # Répertoire du script
VENV_DIR="${SCRIPT_DIR}/.venv"  # Répertoire de l'environnement virtuel
LOG_DIR="${SCRIPT_DIR}/logs"  # Répertoire des logs
LOG_FILE="${LOG_DIR}/bot_$(date +%Y%m%d_%H%M%S).log"  # Fichier de log

# Fonction pour afficher un message avec formatage
print_message() {
    echo -e "\n[$(date +'%Y-%m-%d %H:%M:%S')] $1"
}

# Création du répertoire de logs si nécessaire
mkdir -p "${LOG_DIR}"

# Vérification de Python
if ! command -v python3 &> /dev/null; then
    print_message "❌ Python 3 n'est pas installé. Veuillez l'installer avant de continuer."
    exit 1
fi

# Vérification et activation de l'environnement virtuel
if [ ! -d "${VENV_DIR}" ]; then
    print_message "🔧 Création de l'environnement virtuel..."
    python3 -m venv "${VENV_DIR}" || {
        print_message "❌ Échec de la création de l'environnement virtuel"
        exit 1
    }
    
    # Activation de l'environnement virtuel
    source "${VENV_DIR}/bin/activate"
    
    # Mise à jour de pip
    print_message "🔄 Mise à jour de pip..."
    pip install --upgrade pip || {
        print_message "❌ Échec de la mise à jour de pip"
        exit 1
    }
    
    # Installation des dépendances
    print_message "📦 Installation des dépendances..."
    pip install -r requirements.txt || {
        print_message "❌ Échec de l'installation des dépendances"
        exit 1
    }
    
    print_message "✅ Environnement virtuel configuré avec succès"
else
    # Activation de l'environnement virtuel existant
    source "${VENV_DIR}/bin/activate"
fi

# Vérification des variables d'environnement
if [ ! -f ".env" ]; then
    print_message "⚠️  Le fichier .env n'existe pas. Création d'un fichier .env exemple..."
    cat > .env << 'EOL'
# Configuration Kraken API
KRAKEN_API_KEY=votre_cle_api_ici
KRAKEN_PRIVATE_KEY=votre_cle_privee_ici

# Configuration de la base de données
DATABASE_URL=postgresql+asyncpg://user:password@localhost:5432/kraken_bot
REDIS_URL=redis://localhost:6379/0

# Configuration du trading
TRADING_PAIRS=XBT/USD,ETH/USD
TIMEFRAME=1h
MAX_OPEN_TRADES=3
RISK_PER_TRADE=1.0
DRY_RUN=true

# Configuration du logging
LOG_LEVEL=INFO

# Configuration de Sentry (optionnel)
SENTRY_DSN=
EOL
    
    print_message "ℹ️  Un fichier .env exemple a été créé. Veuillez le configurer avant de relancer le bot."
    exit 0
fi

# Vérification de la configuration
print_message "🔍 Vérification de la configuration..."
python3 -c "
import os
from dotenv import load_dotenv
load_dotenv()

required_vars = [
    'KRAKEN_API_KEY',
    'KRAKEN_PRIVATE_KEY',
    'DATABASE_URL',
    'REDIS_URL'
]

missing = [var for var in required_vars if not os.getenv(var)]
if missing:
    print(f'❌ Variables manquantes dans .env: {\", \".join(missing)}')
    exit(1)
" || exit 1

# Démarrage du bot
print_message "🚀 Démarrage du Kraken Trading Bot..."
print_message "📝 Les logs seront enregistrés dans: ${LOG_FILE}"

# Exécution du bot avec redirection des sorties vers le fichier de log
{
    echo "=========================================="
    echo "Démarrage du bot - $(date)"
    echo "=========================================="
    echo "Répertoire: ${SCRIPT_DIR}"
    echo "Environnement virtuel: ${VENV_DIR}"
    echo "Version Python: $(python3 --version)"
    echo "Version pip: $(pip --version | cut -d ' ' -f 2)"
    echo "Variables d'environnement chargées: $(grep -v '^#' .env | grep '=' | cut -d '=' -f 1 | tr '\n' ' ')"
    echo "=========================================="
    
    # Lancement du bot
    python3 -m kraken_bot
    
    EXIT_CODE=$?
    echo "\n=========================================="
    echo "Arrêt du bot - $(date)"
    echo "Code de sortie: ${EXIT_CODE}"
    echo "=========================================="
} 2>&1 | tee -a "${LOG_FILE}"

# Vérification du code de sortie
if [ ${PIPESTATUS[0]} -ne 0 ]; then
    print_message "❌ Le bot s'est arrêté avec une erreur. Consultez le fichier de log: ${LOG_FILE}"
    exit 1
else
    print_message "✅ Arrêt normal du bot. Consultez le fichier de log: ${LOG_FILE}"
    exit 0
fi
