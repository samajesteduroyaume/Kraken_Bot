#!/bin/bash

# Démarrer Redis en mode test
redis-server --daemonize yes --port 6379 --requirepass test_password --protected-mode no

# Attendre que Redis soit prêt
sleep 2

echo "Redis test server démarré sur le port 6379 avec mot de passe 'test_password'"
