# 📊 Guide des Stratégies de Trading

Kraken_Bot offre une gamme complète de stratégies de trading algorithmique, de la plus simple à la plus sophistiquée. Ce guide détaille chaque stratégie disponible, ses paramètres et comment les personnaliser pour vos besoins spécifiques.

## 🎯 Vue d'Ensemble des Stratégies

### Stratégies Disponibles

| Stratégie | Type | Horizon | Risque | Description |
|-----------|------|---------|--------|-------------|
| Momentum | Tendance | Court/Moyen | Élevé | Suit les tendances fortes |
| Mean Reversion | Contre-tendance | Court | Moyen | Parie sur le retour à la moyenne |
| Breakout | Rupture | Tous | Variable | Capitalise sur les mouvements après consolidation |
| Grid Trading | Marché latéral | Court | Contrôlé | Profite des mouvements dans une fourchette |
| Swing Trading | Tendance | Moyen/Long | Modéré | Capture les mouvements de marché moyens |
| Arbitrage | Marché | Court | Faible | Exploite les écarts de prix entre marchés |
| Market Making | Marché | Très court | Contrôlé | Fournit de la liquidité sur le carnet d'ordres |

## 🚀 Stratégies Détail

### 1. Stratégie Momentum

**📊 Description** :
La stratégie Momentum identifie et suit les tendances fortes du marché en utilisant une combinaison d'indicateurs techniques avancés.

**📈 Indicateurs Clés** :
- **RSI (14 périodes)** : Identifie les conditions de surachat/survente
- **EMA 12/26** : Confirme la direction de la tendance
- **MACD** : Détecte les changements de momentum
- **Volume** : Confirme la force de la tendance

**⚙️ Paramètres Recommandés** :
```yaml
momentum:
  # Paramètres RSI
  rsi_period: 14
  rsi_overbought: 70
  rsi_oversold: 30
  
  # Paramètres des moyennes mobiles
  ema_fast: 12
  ema_slow: 26
  ema_signal: 9
  
  # Paramètres MACD
  macd_fast: 12
  macd_slow: 26
  macd_signal: 9
  
  # Gestion du risque
  stop_loss: 2.0  # %
  take_profit: 4.0  # %
  position_size: 5.0  # % du capital
  
  # Filtres additionnels
  min_volume_btc: 10.0  # Volume minimum en BTC
  trend_strength: 0.5  # Force minimale de la tendance (0-1)
```

**🎯 Règles d'Entrée** :
1. EMA12 > EMA26 (tendance haussière)
2. MACD > Signal Line
3. RSI > 50 (pour les positions longues)
4. Volume supérieur à la moyenne mobile du volume (20 périodes)

**⚠️ Règles de Sortie** :
1. Take-profit atteint
2. Stop-loss déclenché
3. Inversion de tendance (MACD croise en dessous de la ligne de signal)

### 2. Stratégie Mean Reversion

**📊 Description** :
Cette stratégie part du principe que les prix ont tendance à revenir vers leur moyenne après des écarts importants.

**📉 Indicateurs Clés** :
- Bandes de Bollinger (20,2)
- RSI (14)
- Support/Résistance dynamiques
- ATR pour le position sizing

**⚙️ Paramètres Recommandés** :
```yaml
mean_reversion:
  # Bandes de Bollinger
  bb_period: 20
  bb_std: 2.0
  
  # RSI
  rsi_period: 14
  rsi_overbought: 75
  rsi_oversold: 25
  
  # Support/Résistance
  support_lookback: 20
  resistance_lookback: 20
  
  # Gestion du risque
  atr_period: 14
  atr_multiplier: 2.5
  max_position_size: 10.0  # % du capital
  
  # Filtres
  min_adx: 25  # Force de la tendance minimale
  max_volatility: 5.0  # Volatilité maximale en %
```

**🎯 Règles d'Entrée** :
1. Prix touche la bande de Bollinger inférieure ET RSI < 30 (pour un achat)
2. Prix touche la bande de Bollinger supérieure ET RSI > 70 (pour une vente)
3. Volume supérieur à la moyenne
4. ATR dans une fourchette acceptable

**⚠️ Règles de Sortie** :
1. Prix atteint la moyenne mobile centrale des Bandes de Bollinger
2. Stop-loss basé sur l'ATR
3. Invalidation du signal (RSI retourne dans la zone neutre)

### 3. Stratégie Breakout Avancé

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
