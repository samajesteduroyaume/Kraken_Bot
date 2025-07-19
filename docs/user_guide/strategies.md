# 📈 Stratégies de Trading

Kraken_Bot propose plusieurs stratégies de trading intégrées que vous pouvez utiliser ou personnaliser selon vos besoins.

## Stratégies Disponibles

### 1. Stratégie Momentum

**Description** : Cette stratégie identifie les tendances fortes et prend des positions dans la direction de la tendance.

**Indicateurs utilisés** :
- RSI (Relative Strength Index)
- Moyennes Mobiles Exponentielles (EMA)
- MACD (Moving Average Convergence Divergence)

**Paramètres par défaut** :
```yaml
momentum:
  rsi_period: 14
  rsi_overbought: 70
  rsi_oversold: 30
  ema_fast: 12
  ema_slow: 26
  macd_fast: 12
  macd_slow: 26
  macd_signal: 9
```

### 2. Stratégie Mean Reversion

**Description** : Cette stratégie part du principe que les prix finissent par revenir à leur moyenne.

**Indicateurs utilisés** :
- Bandes de Bollinger
- RSI
- Support et Résistance

**Paramètres par défaut** :
```yaml
mean_reversion:
  bb_period: 20
  bb_std: 2.0
  rsi_period: 14
  rsi_overbought: 75
  rsi_oversold: 25
  support_lookback: 20
  resistance_lookback: 20
```

### 3. Stratégie Breakout

**Description** : Identifie les niveaux de support/résistance et prend des positions lors de la rupture de ces niveaux.

**Indicateurs utilisés** :
- Niveaux de Support/Résistance
- Volume
- ATR (Average True Range)

**Paramètres par défaut** :
```yaml
breakout:
  support_lookback: 20
  resistance_lookback: 20
  min_volume: 1000
  atr_period: 14
  atr_multiplier: 1.5
```

## Configuration des Stratégies

### Activation/Désactivation

Pour activer ou désactiver des stratégies, modifiez le fichier `config/config.yaml` :

```yaml
strategies:
  enabled:
    - momentum
    - mean_reversion
    - breakout
  default_strategy: momentum
```

### Personnalisation des Paramètres

Chaque stratégie peut être personnalisée via le fichier de configuration :

1. Copiez les paramètres par défaut de la stratégie
2. Collez-les dans votre `config/config.yaml`
3. Modifiez les valeurs selon vos préférences

### Exemple de Configuration Avancée

```yaml
# Dans config/config.yaml
momentum:
  rsi_period: 10
  rsi_overbought: 75
  rsi_oversold: 35
  ema_fast: 10
  ema_slow: 21

mean_reversion:
  bb_period: 21
  bb_std: 2.1
  rsi_period: 10
  rsi_overbought: 80
  rsi_oversold: 20
```

## Création d'une Stratégie Personnalisée

1. Créez un nouveau fichier dans `src/strategies/` (ex: `ma_strategie.py`)
2. Implémentez votre stratégie en héritant de `BaseStrategy`
3. Ajoutez votre stratégie à `src/strategies/__init__.py`
4. Créez un fichier de configuration dans `config/strategies/`
5. Activez votre stratégie dans `config/config.yaml`

## Backtesting

Testez vos stratégies sur des données historiques :

```bash
python -m src.core.backtesting.backtester --strategy momentum --pairs "BTC/EUR,ETH/EUR" --timerange 20230101-20231231
```

## Meilleures Pratiques

1. **Testez toujours** : Utilisez le backtesting avant de trader en réel
2. **Commencez petit** : Testez avec de petits montants
3. **Surveillez** : Vérifiez régulièrement les performances
4. **Gérez les risques** : Utilisez des stops loss et prenez des profits

## Dépannage

- **Aucun trade n'est exécuté** : Vérifiez les logs et les paramètres de la stratégie
- **Trop de faux signaux** : Ajustez les paramètres ou utilisez des filtres supplémentaires
- **Problèmes de performance** : Vérifiez les intervalles de temps et le nombre de paires tradées

## Prochaines Étapes

- [Gestion des risques](risk_management.md)
- [Création de stratégies personnalisées](../developer_guide/creating_strategies.md)
- [Configuration avancée](../getting_started/configuration.md)
