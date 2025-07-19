# Documentation des Variables d'Environnement

Ce document décrit toutes les variables d'environnement nécessaires pour le bon fonctionnement du bot de trading Kraken.

## Variables Requises

### Configuration de l'API Kraken
- `KRAKEN_API_KEY`: Clé API Kraken (obligatoire)
  - Type: chaîne de caractères
  - Description: Clé API pour accéder aux services Kraken
  - Exemple: `MGvQ3MJ4Ya0yp/zI0k7ett8ug0mUP/3cRscuBBwckpWTGmDEu/qhPBge`

- `KRAKEN_API_SECRET`: Secret API Kraken (obligatoire)
  - Type: chaîne de caractères
  - Description: Secret associé à la clé API
  - Exemple: `6y+hHCZ+YdwGck5Z4WtS8TJUa51ErvEx3E1k0OM24Bzr/tDHY4raO29ZhmCafcOoUUGXMEdvxjE8U5g/mbDvZw==`

### Configuration de la Base de Données
- `POSTGRES_USER`: Utilisateur PostgreSQL
  - Type: chaîne de caractères
  - Description: Nom d'utilisateur pour la connexion à la base de données
  - Valeur par défaut: `kraken_bot`

- `POSTGRES_PASSWORD`: Mot de passe PostgreSQL
  - Type: chaîne de caractères
  - Description: Mot de passe pour la connexion à la base de données
  - Valeur par défaut: `kraken_bot_password`

- `POSTGRES_DB`: Nom de la base de données
  - Type: chaîne de caractères
  - Description: Nom de la base de données à utiliser
  - Valeur par défaut: `kraken_bot`

## Variables Optionnelles

### Configuration du Logging
- `LOG_LEVEL`: Niveau de logging
  - Type: chaîne de caractères
  - Description: Niveau de détail des logs (DEBUG, INFO, WARNING, ERROR, CRITICAL)
  - Valeur par défaut: `INFO`

- `LOG_DIR`: Répertoire des logs
  - Type: chaîne de caractères
  - Description: Répertoire où seront stockés les fichiers de log
  - Valeur par défaut: `~/kraken_bot_logs`

- `LOG_MAX_SIZE`: Taille maximale d'un fichier de log
  - Type: entier
  - Description: Taille maximale en octets avant rotation du fichier
  - Valeur par défaut: `10485760` (10MB)

- `LOG_BACKUP_COUNT`: Nombre de fichiers de backup
  - Type: entier
  - Description: Nombre maximum de fichiers de log à conserver
  - Valeur par défaut: `5`

### Configuration du Trading
- `MAX_TRADE_AMOUNT`: Montant maximum par trade
  - Type: flottant
  - Description: Montant maximum en USD par trade
  - Valeur par défaut: `100.0`

- `MIN_TRADE_AMOUNT`: Montant minimum par trade
  - Type: flottant
  - Description: Montant minimum en USD par trade
  - Valeur par défaut: `10.0`

- `INITIAL_BALANCE`: Solde initial
  - Type: flottant
  - Description: Solde initial du portefeuille en USD
  - Valeur par défaut: `10000.0`

- `RISK_PERCENTAGE`: Pourcentage de risque
  - Type: flottant
  - Description: Pourcentage du capital à risquer par trade
  - Valeur par défaut: `1.0`

- `MAX_DAILY_DRAWDOWN`: Drawdown quotidien maximum
  - Type: flottant
  - Description: Pourcentage maximum de drawdown quotidien
  - Valeur par défaut: `5.0`

- `MAX_LEVERAGE`: Effet de levier maximum
  - Type: flottant
  - Description: Effet de levier maximum autorisé
  - Valeur par défaut: `3.0`

### Configuration de la Simulation
- `SIMULATION_MODE`: Mode simulation
  - Type: booléen
  - Description: Active/Désactive le mode simulation
  - Valeur par défaut: `true`

- `SIMULATION_BALANCE_BTC`: Solde initial en BTC
  - Type: flottant
  - Description: Solde initial en BTC pour la simulation
  - Valeur par défaut: `0.0`

- `SIMULATION_BALANCE_EUR`: Solde initial en EUR
  - Type: flottant
  - Description: Solde initial en EUR pour la simulation
  - Valeur par défaut: `1000.0`

## Configuration des Notifications
- `NOTIFICATIONS_ENABLED`: Notifications
  - Type: booléen
  - Description: Active/Désactive les notifications
  - Valeur par défaut: `true`

- `TELEGRAM_BOT_TOKEN`: Token Telegram
  - Type: chaîne de caractères
  - Description: Token du bot Telegram pour les notifications
  - Exemple: `123456789:ABCdefGHIjklMNopQRstUVwxyZ`

- `TELEGRAM_CHAT_ID`: ID du chat Telegram
  - Type: chaîne de caractères
  - Description: ID du chat où recevoir les notifications
  - Exemple: `123456789`

## Configuration du Rate Limiting
- `RATE_LIMIT_WINDOW`: Fenêtre de rate limiting
  - Type: entier
  - Description: Durée de la fenêtre en secondes
  - Valeur par défaut: `60`

- `RATE_LIMIT_REQUESTS`: Nombre de requêtes maximum
  - Type: entier
  - Description: Nombre maximum de requêtes par fenêtre
  - Valeur par défaut: `20`

## Sécurité
- **Attention** : Les clés API et les mots de passe doivent être stockés dans un fichier `.env.local` qui ne doit pas être commité dans le contrôle de version.
- Il est recommandé d'utiliser des variables d'environnement pour toutes les configurations sensibles.
- Les valeurs par défaut fournies sont des valeurs de sécurité minimales et peuvent être ajustées selon les besoins.

## Exemple de Configuration
```yaml
# .env.local
KRAKEN_API_KEY=your_api_key_here
KRAKEN_API_SECRET=your_api_secret_here
POSTGRES_PASSWORD=your_secure_password
TELEGRAM_BOT_TOKEN=your_telegram_bot_token
TELEGRAM_CHAT_ID=your_chat_id
```

## Validation des Valeurs
- Toutes les valeurs numériques doivent être positives
- Les pourcentages doivent être entre 0 et 100
- Les niveaux de logging doivent être parmi: DEBUG, INFO, WARNING, ERROR, CRITICAL
- Les chemins de fichier doivent être valides
- Les clés API et secrets ne doivent pas être vides
