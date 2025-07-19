# ⚙️ Configuration

Ce guide explique comment configurer Kraken_Bot pour répondre à vos besoins spécifiques de trading.

## Fichiers de configuration principaux

Kraken_Bot utilise plusieurs fichiers de configuration :

1. **`.env`** - Variables d'environnement sensibles
2. **`config/config.yaml`** - Configuration principale de l'application
3. **`config/strategies/`** - Dossier contenant les configurations des stratégies

## Configuration de base

### 1. Fichier .env

Le fichier `.env` contient des informations sensibles. Ne le partagez jamais et ne le versionnez pas (il est déjà dans `.gitignore`).

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

# Paramètres de trading (optionnels)
TRADING_ENABLED=false  # Mettre à true pour activer le trading réel
DRY_RUN=true          # Mode simulation activé par défaut
```

### 2. Fichier config.yaml

Le fichier principal de configuration se trouve dans `config/config.yaml` :

```yaml
# Configuration générale
general:
  environment: development  # ou 'production'
  log_level: INFO
  timezone: Europe/Paris

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
