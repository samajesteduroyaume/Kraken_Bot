# Guide de la Stratégie de Momentum Avancée

## Vue d'ensemble
La stratégie de momentum avancée est conçue pour identifier et exploiter les mouvements directionnels forts sur les marchés financiers. Elle combine plusieurs indicateurs techniques avec une gestion avancée du risque pour générer des signaux de trading précis.

## Fonctionnalités clés

### Indicateurs techniques
- **RSI (Relative Strength Index)** - Identifie les conditions de surachat/survente
- **MACD (Moving Average Convergence Divergence)** - Détecte les changements de tendance
- **Moyennes mobiles (EMA/SMA)** - Détermine la tendance à court et long terme
- **Bandes de Bollinger** - Identifie la volatilité et les niveaux de surachat/survente
- **Ichimoku Cloud** - Fournit des signaux de tendance et des niveaux de support/résistance
- **ADX (Average Directional Index)** - Mesure la force de la tendance
- **Stochastique** - Identifie les retournements potentiels
- **ATR (Average True Range)** - Ajuste le risque en fonction de la volatilité
- **VWAP (Volume Weighted Average Price)** - Fournit des niveaux de prix pondérés par le volume

### Gestion du risque
- Taille de position dynamique basée sur la volatilité
- Stop-loss et take-profit automatiques
- Trailing stop pour verrer les bénéfices
- Limitation du nombre de trades par jour
- Contrôle du drawdown maximum

### Analyse multi-timeframe
- Analyse simultanée sur plusieurs horizons temporels
- Confirmation des signaux sur plusieurs timeframes
- Adaptation dynamique aux conditions de marché

## Configuration requise

### Dépendances
- Python 3.8+
- pandas
- numpy
- ta (Technical Analysis Library)
- scikit-learn (pour les fonctionnalités ML avancées)

### Installation
1. Clonez le dépôt
2. Installez les dépendances :
   ```bash
   pip install -r requirements.txt
   ```

## Utilisation

### Initialisation
```python
from strategies.momentum_strategy import MomentumStrategy
from config.momentum_config import MOMENTUM_STRATEGY_CONFIG

# Initialisation de la stratégie
strategy = MomentumStrategy(config=MOMENTUM_STRATEGY_CONFIG)
```

### Génération de signaux
```python
# Données OHLCV par timeframe
data = {
    '15m': df_15m,  # DataFrame avec les colonnes: open, high, low, close, volume
    '1h': df_1h,
    '4h': df_4h,
    '1d': df_1d
}

# Génération des signaux
signals = strategy.generate_signals(data)

# Affichage des signaux
for signal in signals:
    print(f"Signal: {'Achat' if signal.direction > 0 else 'Vente'}")
    print(f"Prix: {signal.price}")
    print(f"Stop-loss: {signal.stop_loss}")
    print(f"Take-profit: {signal.take_profit}")
    print(f"Force: {signal.strength.name}")
```

### Gestion du risque
La stratégie calcule automatiquement la taille de position optimale en fonction :
- Du risque par trade défini
- De la volatilité actuelle du marché
- De la distance au stop-loss
- De la taille maximale de position autorisée

## Paramètres de configuration

### Paramètres généraux
- `risk_per_trade`: Pourcentage du portefeuille à risquer par trade (par défaut: 1%)
- `max_position_size`: Taille maximale de position en pourcentage du portefeuille (par défaut: 15%)
- `timeframes`: Liste des timeframes à analyser (par défaut: ['15m', '1h', '4h', '1d'])

### Paramètres des indicateurs
- `rsi.period`: Période du RSI (par défaut: 14)
- `rsi.overbought`: Seuil de surachat (par défaut: 70)
- `rsi.oversold`: Seuil de survente (par défaut: 30)
- `macd.fast`: Période rapide du MACD (par défaut: 12)
- `macd.slow`: Période lente du MACD (par défaut: 26)
- `macd.signal`: Période de la ligne de signal (par défaut: 9)

### Paramètres de trading
- `stop_loss_pct`: Pourcentage de stop-loss (par défaut: 2%)
- `take_profit_pct`: Pourcentage de take-profit (par défaut: 4%)
- `trailing_stop`: Activer le trailing stop (par défaut: True)
- `trailing_stop_pct`: Pourcentage du trailing stop (par défaut: 1%)

## Exemple complet

```python
import pandas as pd
from strategies.momentum_strategy import MomentumStrategy
from config.momentum_config import MOMENTUM_STRATEGY_CONFIG

# Chargement des données (exemple)
def load_data(timeframe):
    # Implémentez le chargement des données depuis votre source
    pass

# Initialisation de la stratégie
config = MOMENTUM_STRATEGY_CONFIG.copy()
config['portfolio_value'] = 100000.0  # Définir la valeur du portefeuille

strategy = MomentumStrategy(config=config)

# Chargement des données pour chaque timeframe
timeframes = ['15m', '1h', '4h', '1d']
data = {tf: load_data(tf) for tf in timeframes}

# Génération des signaux
signals = strategy.generate_signals(data)

# Exécution des trades
for signal in signals:
    # Implémentez votre logique d'exécution
    print(f"Exécution d'un {'Achat' if signal.direction > 0 else 'Vente'} à {signal.price}")
```

## Conseils d'utilisation

1. **Backtest** : Toujours effectuer des backtests approfondis avant d'utiliser en production
2. **Optimisation** : Ajustez les paramètres en fonction de la paire de trading et des conditions de marché
3. **Surveillance** : Surveillez régulièrement les performances et ajustez si nécessaire
4. **Diversification** : Combinez avec d'autres stratégies pour diversifier le risque

## Dépannage

### Problèmes courants
- **Aucun signal généré** : Vérifiez les paramètres des indicateurs et les conditions de marché
- **Trop de faux signaux** : Ajustez les seuils et ajoutez des filtres supplémentaires
- **Performances médiocres** : Réoptimisez les paramètres ou envisagez d'autres paires de trading

## Support
Pour toute question ou problème, veuillez ouvrir une issue sur le dépôt GitHub.
