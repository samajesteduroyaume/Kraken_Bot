# Guide de la Stratégie de Retour à la Moyenne (Mean Reversion)

## Table des matières
- [Vue d'ensemble](#vue-densemble)
- [Fonctionnalités clés](#fonctionnalités-clés)
- [Indicateurs techniques](#indicateurs-techniques)
- [Génération des signaux](#génération-des-signaux)
- [Gestion des risques](#gestion-des-risques)
- [Configuration](#configuration)
- [Utilisation](#utilisation)
- [Exemples](#exemples)
- [Conseils](#conseils)
- [Dépannage](#dépannage)

## Vue d'ensemble

La stratégie de retour à la moyenne (Mean Reversion) est basée sur le principe que les prix des actifs financiers ont tendance à revenir vers leur moyenne historique après des périodes d'écart important. Cette stratégie est particulièrement efficace sur les marchés en range ou faiblement tendanciels.

## Fonctionnalités clés

- **Multi-timeframe** : Analyse sur plusieurs périodes pour confirmer les signaux
- **Indicateurs avancés** : Utilisation de plusieurs indicateurs techniques pour valider les signaux
- **Gestion dynamique du risque** : Ajustement automatique de la taille des positions
- **Stop-loss et take-profit dynamiques** : Basés sur la volatilité du marché
- **Filtres avancés** : Filtrage des faux signaux par volume, tendance et volatilité

## Indicateurs techniques

### Bandes de Bollinger
- **Objectif** : Identifier les niveaux de surachat et de survente
- **Paramètres** : Période de 20, écarts-types de 2.0
- **Utilisation** : Les prix proches de la bande supérieure indiquent un surachat, proches de la bande inférieure une survente

### RSI (Indice de Force Relative)
- **Objectif** : Confirmer les conditions de surachat/survente
- **Paramètres** : Période de 14, surachat à 70, survente à 30

### Moyennes Mobiles Simples (SMA)
- **Objectifs** :
  - SMA 20 : Tendance à court terme
  - SMA 50 : Tendance à moyen terme
  - SMA 200 : Tendance à long terme

### ADX (Indice de Direction Moyenne)
- **Objectif** : Évaluer la force de la tendance
- **Seuil** : ADX > 25 indique une tendance forte

## Génération des signaux

### Signal d'achat
1. Le prix touche ou dépasse la bande de Bollinger inférieure
2. Le RSI est en territoire de survente (< 30)
3. Le volume est supérieur à la moyenne sur 20 périodes
4. L'ADX est inférieur à 25 (faible tendance)

### Signal de vente
1. Le prix touche ou dépasse la bande de Bollinger supérieure
2. Le RSI est en territoire de surachat (> 70)
3. Le volume est supérieur à la moyenne sur 20 périodes
4. L'ADX est inférieur à 25 (faible tendance)

## Gestion des risques

### Taille de position
- **Méthode** : Fraction fixe du capital (1% par défaut)
- **Levier maximum** : 5x

### Stop-loss
- **Type** : Basé sur l'ATR (2x ATR par défaut)
- **Alternatives** : Pourcentage fixe (2%) ou niveau technique

### Take-profit
- **Ratio risque/rendement** : 2:1 par défaut
- **Ajustement dynamique** : Basé sur la volatilité récente

## Configuration

### Fichier de configuration
Tous les paramètres sont définis dans `config/mean_reversion_config.py`.

### Paramètres importants
- `timeframes` : Périodes d'analyse (ex: ["15m", "1h", "4h", "1d"])
- `symbols` : Paires de trading (ex: ["BTC/USD", "ETH/USD"])
- `risk_management/risk_per_trade_pct` : Pourcentage du capital à risquer par trade
- `stop_loss_type` : Type de stop-loss (atr, fixed, percentage)

## Utilisation

### Initialisation
```python
from strategies.mean_reversion_strategy import MeanReversionStrategy
from config.mean_reversion_config import MEAN_REVERSION_STRATEGY_CONFIG

strategy = MeanReversionStrategy(config=MEAN_REVERSION_STRATEGY_CONFIG)
```

### Génération de signaux
```python
# Données OHLCV au format DataFrame
data = {'15m': df_15m, '1h': df_1h, '4h': df_4h, '1d': df_1d}
signals = strategy.generate_signals(data)
```

## Exemples

### Exemple de configuration personnalisée
```python
custom_config = {
    "indicators": {
        "bollinger_bands": {"window": 20, "window_dev": 2.0},
        "rsi": {"period": 14, "overbought": 70, "oversold": 30}
    },
    "risk_management": {
        "risk_per_trade_pct": 1.0,
        "max_leverage": 3.0
    }
}

strategy = MeanReversionStrategy(config=custom_config)
```

## Conseils

1. **Optimisation** : Ajustez les périodes des indicateurs selon la paire de trading et le timeframe
2. **Filtrage** : Activez les filtres de volume et de tendance pour réduire les faux signaux
3. **Backtesting** : Testez toujours la stratégie sur des données historiques avant de l'utiliser en temps réel
4. **Gestion du risque** : Ne risquez jamais plus de 1-2% de votre capital par trade

## Dépannage

### Problème : Trop de faux signaux
- **Solution** : Augmentez les seuils de confirmation ou activez des filtres supplémentaires

### Problème : Manque de signaux
- **Solution** : Élargissez les bandes de Bollinger ou réduisez les seuils du RSI

### Problème : Performances médiocres en tendance
- **Solution** : Désactivez la stratégie lorsque l'ADX est élevé (> 25) ou utilisez un filtre de tendance
