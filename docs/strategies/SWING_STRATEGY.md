# Stratégie de Swing Trading

## Aperçu
La stratégie de Swing Trading est conçue pour capturer les mouvements de prix à court et moyen terme en utilisant une combinaison d'indicateurs techniques avancés. Elle vise à identifier les points d'entrée et de sortie optimaux en se basant sur la convergence de plusieurs signaux techniques.

## Indicateurs techniques utilisés

### 1. RSI (Relative Strength Index)
- **Période par défaut** : 14
- **Seuil de survente** : 30
- **Seuil de surachat** : 70
- **Utilisation** : Identifie les conditions de survente et de surachat du marché.

### 2. MACD (Moving Average Convergence Divergence)
- **Période rapide** : 12
- **Période lente** : 26
- **Période du signal** : 9
- **Utilisation** : Détecte les changements de tendance et le momentum.

### 3. ATR (Average True Range)
- **Période** : 14
- **Multiplicateur** : 2.0
- **Utilisation** : Mesure la volatilité et définit les niveaux de stop-loss dynamiques.

### 4. Moyennes Mobiles (SMA)
- **Période courte** : 50
- **Période longue** : 200
- **Utilisation** : Identifie la tendance à moyen et long terme.

### 5. Bandes de Bollinger
- **Période** : 20
- **Écart-type** : 2.0
- **Utilisation** : Identifie les niveaux de support et résistance dynamiques.

### 6. MFI (Money Flow Index)
- **Période** : 14
- **Utilisation** : Mesure la pression d'achat/vente en tenant compte du volume.

### 7. ADX (Average Directional Index)
- **Période** : 14
- **Utilisation** : Mesure la force de la tendance.

## Gestion des risques

### Stop-Loss
- **Méthode** : Basé sur l'ATR
- **Multiplicateur ATR** : 2.0
- **Calcul** : `Prix d'entrée ± (ATR × Multiplicateur)`

### Take-Profit
- **Ratio** : 2.0 (par rapport au stop-loss)
- **Calcul** : `Prix d'entrée ± (Distance du stop-loss × Ratio)`

### Taille de position
- **Méthode** : Fraction fixe du capital
- **Risque par trade** : 1% du capital
- **Taille maximale de position** : 15% du capital

## Filtres de signal

### Filtre de volume
- **Activé** : Oui
- **Volume minimum (en BTC)** : 1.0
- **Période de volume** : 20 périodes

### Filtre de tendance
- **Activé** : Oui
- **Force minimale de tendance** : 0.6

### Filtre de volatilité
- **ATR minimum (%)** : 0.5
- **ATR maximum (%)** : 5.0

## Configuration recommandée

### Timeframes multiples
- **Court terme** : 15 minutes
- **Moyen terme** : 1 heure
- **Long terme** : 4 heures
- **Très long terme** : 1 jour

### Paramètres avancés
- **Nombre de confirmations requises** : 2 timeframes
- **Force minimale du signal** : 0.6
- **Période de chauffe** : 200 bougies

## Utilisation

### Configuration minimale requise
```python
from src.strategies.swing_strategy import SwingStrategy
from src.strategies.swing_config import get_swing_config

# Créer une instance de la stratégie avec la configuration par défaut
strategy = SwingStrategy(config=get_swing_config())

# Ou avec une configuration personnalisée
custom_config = {
    'symbol': 'BTC/EUR',
    'timeframes': ['30m', '1h', '4h'],
    'risk_management': {
        'risk_per_trade': 0.01,  # 1% de risque par trade
        'max_position_size': 0.10  # 10% du capital maximum par position
    }
}

strategy = SwingStrategy(config=get_swing_config(custom_config))
```

## Backtesting

### Paramètres recommandés
- **Période de test** : 1 an minimum
- **Frais de transaction** : 0.1% par trade
- **Slippage** : 0.05% par trade
- **Solde initial** : 10 000 €

### Métriques de performance
- **Rendement total** : Objectif > 30% par an
- **Maximum Drawdown** : Objectif < 15%
- **Ratio de Sharpe** : Objectif > 1.5
- **Taux de réussite** : Objectif > 55%

## Limitations et considérations

1. **Marchés en range** : La stratégie peut générer des faux signaux dans les marchés sans tendance marquée.
2. **Latence** : Une exécution rapide est nécessaire pour capturer les mouvements rapides.
3. **Optimisation** : Les paramètres doivent être optimisés régulièrement pour s'adapter aux conditions de marché changeantes.
4. **Gestion du risque** : Le respect strict des règles de gestion du risque est essentiel pour la réussite à long terme.

## Améliorations possibles

1. **Filtres supplémentaires** : Ajout de filtres basés sur les fondamentaux ou le sentiment du marché.
2.**Apprentissage automatique** : Utilisation de modèles prédictifs pour améliorer la précision des signaux.
3. **Optimisation dynamique** : Ajustement automatique des paramètres en fonction des conditions de marché.
4. **Intégration de données alternatives** : Utilisation de données de chaîne ou de sentiment pour confirmer les signaux.
