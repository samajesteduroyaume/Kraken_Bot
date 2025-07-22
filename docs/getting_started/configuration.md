# ⚙️ Guide de Configuration

Ce guide détaille comment configurer Kraken_Bot pour répondre précisément à vos besoins de trading. Une configuration correcte est essentielle pour des performances optimales et une expérience de trading fluide.

## 📁 Structure des Fichiers de Configuration

Kraken_Bot utilise une structure de configuration hiérarchique :

1. **`.env`** - Variables d'environnement sensibles (ne jamais versionner)
2. **`config/config.yaml`** - Configuration principale de l'application
3. **`config/strategies/`** - Configurations spécifiques aux stratégies
4. **`config/indicators/`** - Paramètres des indicateurs techniques
5. **`config/risk/`** - Paramètres de gestion des risques

## 🔐 Configuration de Base (.env)

Le fichier `.env` contient des informations sensibles. Il est automatiquement ignoré par Git.

### Configuration Requise

```env
# === Configuration de l'API Kraken ===
# Format recommandé : API-KEY-... pour les nouvelles clés
KRAKEN_API_KEY=votre_cle_api
KRAKEN_SECRET=votre_cle_secrete

# === Base de données (PostgreSQL recommandé) ===
DB_ENGINE=django.db.backends.postgresql
DB_NAME=kraken_bot
DB_USER=postgres
DB_PASSWORD=votre_mot_de_passe
DB_HOST=localhost
DB_PORT=5432

# === Redis (Cache et files d'attente) ===
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_DB=0
REDIS_PASSWORD=
```

### Configuration Optionnelle

```env
# === Paramètres de l'application ===
DEBUG=True
SECRET_KEY=générer_une_clé_secrète_sécurisée
ALLOWED_HOSTS=localhost,127.0.0.1

# === Paramètres de Trading ===
TRADING_ENABLED=false  # Activer le trading réel
DRY_RUN=true          # Mode simulation activé
MAX_TRADE_AMOUNT=1000  # Montant maximum par trade (USDT)
MAX_RISK_PERCENT=2     # % maximum du capital à risquer

# === Paramètres de Journalisation ===
LOG_LEVEL=INFO
LOG_FILE=logs/kraken_bot.log
LOG_MAX_SIZE=10  # Taille max en Mo
LOG_BACKUP_COUNT=5  # Nombre de fichiers de log à conserver
```

## ⚙️ Configuration Principale (config.yaml)

Le fichier `config/config.yaml` contient la configuration détaillée de l'application.

### Section Générale

```yaml
general:
  environment: development  # development, staging, production
  log_level: INFO  # DEBUG, INFO, WARNING, ERROR, CRITICAL
  timezone: Europe/Paris
  default_currency: USDT
  data_dir: data/
  cache_ttl: 300  # Durée de vie du cache en secondes
```

### Configuration de l'API

```yaml
api:
  base_url: https://api.kraken.com
  version: 0
  public_endpoint: /public
  private_endpoint: /private
  rate_limit_per_second: 1  # Limite de requêtes par seconde
  max_retries: 3
  retry_delay: 1  # Délai entre les tentatives en secondes
```

### Configuration des Paires de Trading

```yaml
trading_pairs:
  enabled:
    - BTC/USDT
    - ETH/USDT
    - SOL/USDT
  quote_currencies:
    - USDT
    - USD
    - EUR
  blacklist:
    - XRP/USDT  # Paires à exclure
```

### Configuration des Stratégies

```yaml
strategies:
  default: mean_reversion  # Stratégie par défaut
  enabled:
    mean_reversion:
      enabled: true
      risk_per_trade: 1.0  # % du capital à risquer
      take_profit: 2.5     # % de profit cible
      stop_loss: 1.0       # % de perte maximale
    momentum:
      enabled: true
      rsi_period: 14
      rsi_overbought: 70
      rsi_oversold: 30
```

## 🔧 Configuration Avancée

### Gestion des Risques

```yaml
risk_management:
  max_portfolio_risk: 10.0  # % maximum du portefeuille à risquer
  daily_loss_limit: 5.0     # % de perte quotidienne maximale
  position_sizing: kelly    # Méthode de calcul de la taille de position
  max_leverage: 5.0         # Effet de levier maximum
  stop_loss:
    type: trailing         # trailing, fixed, atr
    default: 2.0           # % par défaut
    atr_multiplier: 2.0    # Multiplicateur ATR pour le stop dynamique
```

### Configuration des Indicateurs Techniques

```yaml
technical_indicators:
  rsi:
    period: 14
    overbought: 70
    oversold: 30
  macd:
    fast_period: 12
    slow_period: 26
    signal_period: 9
  bollinger_bands:
    period: 20
    std_dev: 2.0
```

## 🔄 Synchronisation de la Configuration

Après avoir modifié la configuration, redémarrez le service :

```bash
docker-compose restart
# Ou, pour une installation manuelle
python manage.py check_config
```

## 🔍 Validation de la Configuration

Vérifiez que votre configuration est valide avec :

```bash
python manage.py validate_config
```

## 🔒 Bonnes Pratiques de Sécurité

1. **Ne jamais** versionner le fichier `.env`
2. Utiliser des permissions restrictives : `chmod 600 .env`
3. Régénérer régulièrement les clés API
4. Utiliser des mots de passe forts et uniques
5. Activer l'authentification à deux facteurs sur votre compte Kraken
# Paramètres de trading
trading:
  base_currency: EUR
  quote_currency: USDT
  max_open_trades: 5
  stake_amount: 100  # Montant par trade en devise de base
  stop_loss: -0.05   # Stop loss à -5%
  take_profit: 0.1   # Take profit à +10%

# Stratégies
strategies:
  enabled:
    - momentum
    - mean_reversion
  default_strategy: momentum

# Configuration des stratégies
momentum:
  rsi_period: 14
  rsi_overbought: 70
  rsi_oversold: 30
  ema_fast: 12
  ema_slow: 26

mean_reversion:
  bb_period: 20
  bb_std: 2.0
  rsi_period: 14
  rsi_overbought: 75
  rsi_oversold: 25
```

## Configuration avancée

### Stratégies personnalisées

Pour ajouter une stratégie personnalisée :

1. Créez un nouveau fichier dans `src/strategies/` (par exemple `my_strategy.py`)
2. Implémentez votre logique en suivant l'interface de base
3. Ajoutez la configuration dans `config/strategies/my_strategy.yaml`
4. Activez-la dans `config/config.yaml`

### Paramètres de l'API

Pour optimiser les appels API :

```yaml
api:
  rate_limit: 3        # Nombre maximum de requêtes par seconde
  retry_attempts: 3    # Nombre de tentatives en cas d'échec
  retry_delay: 5       # Délai entre les tentatives (secondes)
  timeout: 30          # Timeout des requêtes (secondes)
```

### Journalisation

Configuration avancée des logs :

```yaml
logging:
  level: INFO
  format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
  file: logs/trading_bot.log
  max_size: 10  # Taille max du fichier en Mo
  backup_count: 5
```

## Vérification de la configuration

Pour vérifier que votre configuration est valide :

```bash
python scripts/validate_config.py
```

## Problèmes courants

- **Fichier de configuration manquant** : Copiez `config/config.example.yaml` vers `config/config.yaml`
- **Erreurs de validation** : Vérifiez les messages d'erreur et la documentation des paramètres
- **Problèmes de connexion** : Vérifiez les paramètres de base de données et d'API

## Étapes suivantes

- [Guide utilisateur](../user_guide/overview.md)
- [Création de stratégies personnalisées](../developer_guide/creating_strategies.md)
