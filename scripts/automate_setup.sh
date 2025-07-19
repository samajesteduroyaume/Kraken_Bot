#!/bin/bash

# Fonction pour installer Redis
install_redis() {
    echo "📦 Installation de Redis..."
    sudo apt-get update
    sudo apt-get install -y redis-server
    
    # Configuration Redis
    sudo sed -i 's/supervised no/supervised systemd/' /etc/redis/redis.conf
    sudo sed -i 's/protected-mode yes/protected-mode no/' /etc/redis/redis.conf
    sudo sed -i 's/appendonly no/appendonly yes/' /etc/redis/redis.conf
    
    # Démarrage Redis
    sudo systemctl start redis
    sudo systemctl enable redis
    
    echo "✅ Redis installé et configuré"
}

# Fonction pour installer les dépendances Python
install_python_deps() {
    echo "📦 Installation des dépendances Python..."
    
    # Installer les dépendances système requises
    sudo apt-get update
    sudo apt-get install -y \
        python3-dev \
        python3-pip \
        build-essential \
        libssl-dev \
        libffi-dev \
        python3-venv
    
    # Créer un environnement virtuel
    python3 -m venv venv
    source venv/bin/activate
    
    # Installer les dépendances Python
    pip install --upgrade pip
    pip install -r requirements.txt
    
    echo "✅ Dépendances Python installées"
}

# Fonction pour configurer les variables d'environnement
setup_env() {
    echo "⚙️ Configuration des variables d'environnement..."
    
    # Créer un fichier .env.local si nécessaire
    if [ ! -f ".env.local" ]; then
        # Copier le modèle
        cp .env.local.example .env.local
        
        # Générer les mots de passe
        sed -i "s/REDIS_PASSWORD=$(openssl rand -base64 32)/REDIS_PASSWORD=$(openssl rand -base64 32)/" .env.local
        sed -i "s/DB_PASSWORD=$(openssl rand -base64 32)/DB_PASSWORD=$(openssl rand -base64 32)/" .env.local
        
        echo "✅ Configuration .env.local créée"
    fi
}

# Fonction pour démarrer les services
start_services() {
    echo "🚀 Démarrage des services..."
    
    # Activer l'environnement virtuel
    source venv/bin/activate
    
    # Démarrer Redis
    echo "Starting Redis..."
    sudo systemctl restart redis
    
    # Démarrer le moniteur Redis
    echo "Starting Redis Monitor..."
    python -m src.monitoring.redis_monitor &
    
    # Démarrer le nettoyeur de cache
    echo "Starting Cache Cleaner..."
    python -m scripts.cleanup_redis_cache &
    
    # Démarrer le système de reporting
    echo "Starting Trading Reporter..."
    python -m src.reporting.trading_reporter &
    
    # Démarrer le système de backup
    echo "Starting Log Backup..."
    python -m src.backup.log_backup &
    
    echo "✅ Tous les services démarrés"
}

# Fonction pour vérifier l'état des services
check_services() {
    echo "🔍 Vérification des services..."
    
    # Vérifier Redis
    redis-cli ping && echo "✅ Redis: OK" || echo "❌ Redis: KO"
    
    # Vérifier les processus Python
    ps aux | grep -v grep | grep "redis_monitor" && echo "✅ Redis Monitor: OK" || echo "❌ Redis Monitor: KO"
    ps aux | grep -v grep | grep "cleanup_redis_cache" && echo "✅ Cache Cleaner: OK" || echo "❌ Cache Cleaner: KO"
    ps aux | grep -v grep | grep "trading_reporter" && echo "✅ Trading Reporter: OK" || echo "❌ Trading Reporter: KO"
    ps aux | grep -v grep | grep "log_backup" && echo "✅ Log Backup: OK" || echo "❌ Log Backup: KO"
}

# Fonction principale
deploy() {
    # Installation des dépendances
    install_redis
    install_python_deps
    setup_env
    
    # Démarrage des services
    start_services
    
    # Vérification
    check_services
}

# Exécuter le déploiement
deploy

# Ajouter un trap pour arrêter proprement les services
trap 'echo "🛑 Arrêt des services..."; kill $(jobs -p)' SIGINT SIGTERM

# Garder le script en vie
while true; do
    sleep 60
    check_services
    echo "🔄 Vérification périodique des services..."
done
