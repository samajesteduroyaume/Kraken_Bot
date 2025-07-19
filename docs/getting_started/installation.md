# 🚀 Installation

Ce guide vous aidera à installer et configurer Kraken_Bot sur votre système.

## Prérequis

- Python 3.12 ou supérieur
- pip (gestionnaire de paquets Python)
- Git (pour cloner le dépôt)
- Compte API Kraken (clé API et clé secrète)
- PostgreSQL (version 12 ou supérieure)
- Redis (pour le cache et les files d'attente)

## Installation

1. **Cloner le dépôt** :
   ```bash
   git clone https://github.com/yourusername/Kraken_Bot.git
   cd Kraken_Bot
   ```

2. **Créer un environnement virtuel** :
   ```bash
   python -m venv venv
   source venv/bin/activate  # Sur Linux/Mac
   # OU
   .\venv\Scripts\activate  # Sur Windows
   ```

3. **Installer les dépendances** :
   ```bash
   pip install -r requirements.txt
   ```

4. **Configurer les variables d'environnement** :
   Créez un fichier `.env` à la racine du projet avec le contenu suivant :
   ```env
   # Configuration de la base de données
   POSTGRES_USER=postgres
   POSTGRES_PASSWORD=votre_mot_de_passe
   POSTGRES_DB=kraken_bot
   POSTGRES_HOST=localhost
   POSTGRES_PORT=5432

   # Configuration de l'API Kraken
   KRAKEN_API_KEY=votre_cle_api
   KRAKEN_SECRET=votre_cle_secrete

   # Configuration Redis
   REDIS_HOST=localhost
   REDIS_PORT=6379
   ```

5. **Initialiser la base de données** :
   ```bash
   python scripts/init_db.py
   ```

## Vérification de l'installation

Pour vérifier que tout est correctement installé, exécutez :

```bash
python -c "import sys; print(f'Python {sys.version}')"
python -c "import pandas as pd; print(f'Pandas {pd.__version__}')"
python -c "from src.core.api.kraken import KrakenAPI; print('KrakenAPI importé avec succès')"
```

## Problèmes courants

- **Erreur de dépendances manquantes** : Assurez-vous d'avoir installé toutes les dépendances avec `pip install -r requirements.txt`
- **Problèmes de connexion à la base de données** : Vérifiez que PostgreSQL est en cours d'exécution et que les identifiants dans `.env` sont corrects
- **Problèmes d'API Kraken** : Vérifiez que vos clés API sont valides et ont les bonnes permissions

## Étapes suivantes

- [Configuration initiale](configuration.md)
- [Guide d'utilisation rapide](../user_guide/overview.md)
