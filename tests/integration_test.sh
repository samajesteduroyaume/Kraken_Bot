#!/bin/bash

# Configuration Redis
REDIS_HOST="localhost"
REDIS_PORT=6379
REDIS_PASSWORD="test_password"

# Fonction pour attendre Redis
wait_for_redis() {
    echo "⏳ Attente de Redis..."
    while ! redis-cli -h $REDIS_HOST -p $REDIS_PORT -a $REDIS_PASSWORD ping &>/dev/null; do
        echo "." -n
        sleep 1
    done
    echo "\n✅ Redis prêt !"
}

# Fonction pour exécuter les tests
run_tests() {
    echo "🚀 Exécution des tests..."
    
    # Démarrer Redis en mode test
    chmod +x tests/start_redis_test.sh
    tests/start_redis_test.sh
    sleep 2

    # Attendre que Redis soit prêt
    wait_for_redis

    # Exécuter les tests unitaires
    python3 -m pytest tests/test_services.py -v
    
    # Arrêter Redis
    redis-cli -h $REDIS_HOST -p $REDIS_PORT -a $REDIS_PASSWORD shutdown
}

# Fonction principale
main() {
    # Exécuter les tests
    run_tests
    docker-compose build
    docker-compose up -d
    
    # Attendre que tous les services soient prêts
    wait_for_service "redis"
    wait_for_service "bot"
    
    # Exécuter les tests
    run_tests
    
    # Arrêter les services
    echo "🛑 Arrêt des services..."
    docker-compose down
}

# Exécuter le script
main
