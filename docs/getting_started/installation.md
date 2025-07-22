# 🚀 Guide d'Installation

Ce guide vous aidera à installer et configurer Kraken_Bot sur votre système. Suivez ces étapes pour une installation réussie.

## 📋 Prérequis Système

### Configuration minimale recommandée
- **Système d'exploitation** : Linux (Ubuntu/Debian), macOS, ou Windows 10/11
- **Processeur** : 2+ cœurs (4+ recommandés)
- **Mémoire vive** : 4 Go minimum (8 Go recommandés)
- **Espace disque** : 10 Go d'espace libre

### Logiciels requis
- **Python** : 3.12 ou supérieur
- **Pip** : Gestionnaire de paquets Python
- **Git** : Pour cloner le dépôt
- **Docker** et **Docker Compose** (recommandé pour une installation simplifiée)
- **Base de données** : PostgreSQL 14+ (recommandé) ou SQLite (pour le développement)
- **Cache** : Redis 6.0+ (recommandé pour les performances)

### Compte API Kraken
- Un compte vérifié sur [Kraken](https://www.kraken.com/)
- Clé API avec les permissions nécessaires
- Clé secrète associée

## 🛠 Installation

### Option 1 : Installation avec Docker (Recommandé)

1. **Cloner le dépôt** :
   ```bash
   git clone https://github.com/samajesteduroyaume/Kraken_Bot.git
   cd Kraken_Bot
   ```

2. **Configurer les variables d'environnement** :
   ```bash
   cp .env.example .env
   # Éditer le fichier .env avec vos paramètres
   nano .env
   ```

3. **Démarrer les services** :
   ```bash
   docker-compose up -d
   ```

4. **Vérifier les logs** :
   ```bash
   docker-compose logs -f
   ```

### Option 2 : Installation Manuelle

1. **Cloner le dépôt** :
   ```bash
   git clone https://github.com/samajesteduroyaume/Kraken_Bot.git
   cd Kraken_Bot
   ```

2. **Créer et activer un environnement virtuel** :
   ```bash
   # Linux/macOS
   python -m venv venv
   source venv/bin/activate
   
   # Windows
   python -m venv venv
   .\venv\Scripts\activate
   ```

3. **Installer les dépendances** :
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

4. **Configurer la base de données** :
   ```bash
   # Installer PostgreSQL et créer une base de données
   # Puis exécuter les migrations
   python manage.py migrate
   ```

5. **Configurer Redis** :
   ```bash
   # Installer et démarrer Redis
   # Sur Ubuntu/Debian :
   sudo apt update
   sudo apt install redis-server
   sudo systemctl enable redis-server
   sudo systemctl start redis-server
   ```

## ⚙️ Configuration

### Fichier .env

Créez un fichier `.env` à la racine du projet avec le contenu suivant :

```env
# === Configuration de l'API Kraken ===
# Format recommandé : API-KEY-... pour les nouvelles clés
KRAKEN_API_KEY=votre_cle_api
KRAKEN_SECRET=votre_cle_secrete

# === Base de données ===
# PostgreSQL (recommandé pour la production)
DB_ENGINE=django.db.backends.postgresql
DB_NAME=kraken_bot
DB_USER=postgres
DB_PASSWORD=votre_mot_de_passe
DB_HOST=localhost
DB_PORT=5432

# SQLite (pour le développement uniquement)
# DB_ENGINE=django.db.backends.sqlite3
# DB_NAME=db.sqlite3

# === Redis (pour le cache et les files d'attente) ===
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_DB=0
REDIS_PASSWORD=

# === Paramètres de l'application ===
DEBUG=True
SECRET_KEY=votre_secret_key_tres_long_et_securise
ALLOWED_HOSTS=localhost,127.0.0.1

# === Paramètres de trading (optionnels) ===
# Montant maximum par trade (en USDT)
MAX_TRADE_AMOUNT=1000
# Pourcentage maximum du capital à risquer par trade
MAX_RISK_PERCENT=2
```

### Génération de clé secrète

Pour générer une clé secrète sécurisée :
```bash
python -c "from django.core.management.utils import get_random_secret_key; print(get_random_secret_key())"
```

## 🔍 Vérification de l'installation

1. **Lancer les tests** :
   ```bash
   python -m pytest
   ```

2. **Démarrer le serveur de développement** :
   ```bash
   python manage.py runserver
   ```

3. **Accéder à l'interface d'administration** :
   Ouvrez votre navigateur à l'adresse : http://localhost:8000/admin/

## 🔄 Mise à jour

Pour mettre à jour vers la dernière version :

```bash
git pull origin main
pip install -r requirements.txt
python manage.py migrate
python manage.py collectstatic --noinput
```

## 🚨 Dépannage

### Erreurs courantes

1. **Erreur de connexion à la base de données** :
   - Vérifiez que PostgreSQL est en cours d'exécution
   - Vérifiez les identifiants dans le fichier .env

2. **Problèmes avec l'API Kraken** :
   - Vérifiez que votre clé API est valide et a les bonnes permissions
   - Assurez-vous que votre compte Kraken est vérifié

3. **Erreurs Redis** :
   - Vérifiez que Redis est installé et en cours d'exécution
   - Vérifiez la configuration dans le fichier .env
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
