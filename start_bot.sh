#!/bin/bash

# Activer l'environnement virtuel
source .venv/bin/activate

# Mettre à jour toutes les paires Kraken dans la config
python scripts/auto_update_pairs.py

# Lancer le bot avec redémarrage automatique en cas de crash
while true; do
    echo "[START] $(date) - Lancement du bot Kraken"
    python main.py >> logs/trading.log 2>&1
    echo "[CRASH] $(date) - Le bot s'est arrêté. Redémarrage dans 5s..."
    sleep 5
done 