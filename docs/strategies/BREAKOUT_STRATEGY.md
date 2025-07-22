# Stratégie de Breakout

## Aperçu
La stratégie de Breakout est conçue pour identifier et exploiter les mouvements de prix importants qui se produisent lorsque le prix franchit un niveau de support ou de résistance clé. Cette stratégie est particulièrement efficace sur les marchés en tendance et peut générer des signaux d'achat et de vente basés sur la confirmation de la cassure.

## Indicateurs techniques utilisés

### 1. ADX (Average Directional Index)
- **Période par défaut** : 14
- **Seuil de tendance** : 25
- **Utilisation** : Mesure la force de la tendance actuelle. Une valeur élevée indique une tendance forte.

### 2. ATR (Average True Range)
- **Période par défaut** : 14
- **Multiplicateur** : 2.0
- **Utilisation** : Mesure la volatilité du marché et aide à définir les niveaux de stop-loss.

### 3. Bandes de Bollinger
- **Période** : 20
- **Écart-type** : 2.0
- **Utilisation** : Identifie les niveaux de surachat et de survente, ainsi que les périodes de faible volatilité qui précèdent souvent les mouvements importants.

### 4. RSI (Relative Strength Index)
- **Période** : 14
- **Surachat** : 70
- **Survente** : 30
- **Utilisation** : Identifie les conditions de surachat et de survente.

### 5. ROC (Rate of Change)
- **Période** : 14
- **Utilisation** : Mesure la vélocité des mouvements de prix.

## Gestion des risques

### Stop-Loss
- **Méthode** : Basé sur l'ATR (Average True Range)
- **Multiplicateur ATR** : 2.0
- **Calcul** : `Prix d'entrée ± (ATR × Multiplicateur)`

### Take-Profit
- **Ratio** : 1.5 (par rapport au stop-loss)
- **Calcul** : `Prix d'entrée ± (Distance du stop-loss × Ratio)`

### Taille de position
- **Méthode** : Fraction fixe du capital
- **Risque par trade** : 2% du capital
- **Taille maximale de position** : 10% du capital

## Filtres de signal

### Filtre de volume
- **Activé** : Oui
- **Volume minimum (en BTC)** : 1.0
- **Période de volume** : 20 périodes

### Filtre de tendance
- **Activé** : Oui
- **Force minimale de tendance** : 0.6

### Filtre de volatilité
- **Activé** : Oui
- **ATR minimum (%)** : 0.5
- **ATR maximum (%)** : 5.0

## Configuration recommandée

### Timeframes multiples
- **Court terme** : 15 minutes
- **Moyen terme** : 1 heure
- **Long terme** : 4 heures

### Paramètres avancés
- **Nombre de confirmations requises** : 2 timeframes
- **Force minimale du signal** : 0.6
- **Période de chauffe** : 200 bougies

## Utilisation

### Configuration minimale requise
```python
from src.strategies.breakout_strategy import BreakoutStrategy
from src.strategies.breakout_config import DEFAULT_BREAKOUT_CONFIG

# Créer une instance de la stratégie
strategy = BreakoutStrategy(config=DEFAULT_BREAKOUT_CONFIG)

# Ou avec une configuration personnalisée
custom_config = {
    'symbol': 'BTC/USDT',
    'timeframes': ['30m', '1h', '4h'],
    'risk_management': {
        'risk_per_trade': 0.01,  # 1% de risque par trade
        'max_position_size': 0.05  # 5% du capital maximum par position
    }
}

strategy = BreakoutStrategy(config=custom_config)
```

## Backtesting
La stratégie peut être testée sur des données historiques en utilisant la méthode `backtest` de la classe de base. Les paramètres de backtest peuvent être personnalisés dans la configuration.

## Notes importantes
1. La stratégie fonctionne mieux sur les marchés en tendance.
2. Les faux signaux sont plus probables pendant les périodes de faible volatilité.
3. Il est recommandé d'utiliser cette stratégie en conjonction avec d'autres indicateurs de confirmation.
4. Les paramètres doivent être optimisés pour chaque paire de trading et conditions de marché spécifiques.
