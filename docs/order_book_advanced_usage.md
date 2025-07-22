# Guide d'utilisation avancée du carnet d'ordres

Ce document explique comment utiliser les fonctionnalités avancées du module de carnet d'ordres pour améliorer votre stratégie de trading.

## Table des matières
1. [Introduction](#introduction)
2. [Configuration initiale](#configuration-initiale)
3. [Utilisation des indicateurs avancés](#utilisation-des-indicateurs-avancés)
4. [Intégration avec le générateur de signaux](#intégration-avec-le-générateur-de-signaux)
5. [Exemples pratiques](#exemples-pratiques)
6. [Bonnes pratiques](#bonnes-pratiques)

## Introduction

Le module de carnet d'ordres avancé fournit des indicateurs sophistiqués basés sur la structure et la dynamique du carnet d'ordres, permettant une analyse plus approfondie des conditions de marché.

## Configuration initiale

### Prérequis
- Python 3.8+
- Bibliothèques requises (installées via `pip install -r requirements.txt`)
- Accès à l'API Kraken ou à un flux de données de marché

### Installation

```bash
# Cloner le dépôt
git clone https://github.com/votre-utilisateur/Kraken_Bot.git
cd Kraken_Bot

# Installer les dépendances
pip install -r requirements.txt
```

## Utilisation des indicateurs avancés

### Initialisation du gestionnaire de carnet d'ordres

```python
from src.core.market.order_book import OrderBookManager
from src.core.market.order_book_indicators import OrderBookIndicators
from src.core.market.order_book_signal_adapter import OrderBookSignalAdapter

# Initialiser le gestionnaire de carnet d'ordres
order_book_manager = OrderBookManager()

# Initialiser les indicateurs avancés
order_book_indicators = OrderBookIndicators()

# Initialiser l'adaptateur de signaux
signal_adapter = OrderBookSignalAdapter(order_book_manager)
```

### Métriques disponibles

#### 1. Profondeur de marché
```python
# Obtenir la profondeur de marché
depth_profile = signal_adapter.get_market_depth_profile()
print(f"Profondeur d'achat: {depth_profile['bid_depth']}")
print(f"Profondeur de vente: {depth_profile['ask_depth']}")
```

#### 2. Niveaux de support et résistance
```python
# Obtenir les niveaux de support/résistance
levels = signal_adapter.get_support_resistance_levels(num_levels=3)
print("Supports:", levels['support'])
print("Résistances:", levels['resistance'])
```

#### 3. Zones de liquidité
```python
# Identifier les zones de liquidité
liquidity_zones = signal_adapter.get_liquidity_zones()
print("Zones d'achat:", liquidity_zones['bid_zones'])
print("Zones de vente:", liquidity_zones['ask_zones'])
```

## Intégration avec le générateur de signaux

### Configuration du générateur de signaux

```python
from src.core.signal_generator import SignalGenerator

# Initialiser le générateur de signaux avec le gestionnaire de carnet d'ordres
signal_generator = SignalGenerator(order_book_manager=order_book_manager)

# Générer des signaux complets
signals = signal_generator.generate_signals(price_history)
```

### Structure des signaux générés

```python
{
    'technical': {
        'trend': 'up',  # 'up', 'down', ou 'neutral'
        'volatility': 'medium',  # 'low', 'medium', ou 'high'
        'rsi': 65.5,
        'macd': 12.3,
        'indicators': {
            # Tous les indicateurs techniques calculés
        }
    },
    'order_book': {
        'available': True,
        'spread': 1.2,
        'mid_price': 50123.5,
        'imbalance': 0.15,
        'support_resistance': {
            'support': [...],
            'resistance': [...]
        },
        'liquidity_zones': {
            'bid_zones': [...],
            'ask_zones': [...]
        },
        'market_quality': {
            'liquidity': 'high',
            'volatility': 'medium',
            'spread': 'tight',
            'order_imbalance': 'slight_buy',
            'overall_quality': 'high',
            'suggested_strategy': 'market_making'
        }
    },
    'sentiment': {
        'price': 'bullish',
        'market': 'neutral'
    },
    'ml': {
        'prediction': 'buy',
        'confidence': 0.78
    },
    'combined': {
        'final_signal': 'buy',
        'strength': 0.75,
        'confidence': 0.82,
        'recommended_actions': ['market_making', 'trend_following'],
        'risk_level': 'medium',
        'market_conditions': {
            # Détails des conditions de marché
        }
    }
}
```

## Exemples pratiques

### Exemple 1: Stratégie de Market Making

```python
# Vérifier les conditions pour le market making
market_quality = signals['order_book']['market_quality']
if market_quality['suggested_strategy'] == 'market_making' and \
   market_quality['liquidity'] == 'high' and \
   market_quality['spread'] == 'tight':
    
    # Calculer les prix d'achat et de vente
    spread = signals['order_book']['spread']
    mid_price = signals['order_book']['mid_price']
    
    # Placer des ordres autour du prix moyen
    buy_price = mid_price - (spread * 0.4)
    sell_price = mid_price + (spread * 0.4)
    
    print(f"Ordre d'achat à {buy_price}")
    print(f"Ordre de vente à {sell_price}")
```

### Exemple 2: Détection de rupture de niveau

```python
# Vérifier les signaux de rupture
current_price = price_history.iloc[-1]
resistance_levels = signals['order_book']['support_resistance']['resistance']

# Vérifier si le prix franchit un niveau de résistance
for level in resistance_levels:
    if current_price > level['price'] and level['strength'] > 0.7:
        print(f"Rupture du niveau de résistance à {level['price']}")
        # Prendre une position longue
        take_long_position()
        break
```

## Bonnes pratiques

1. **Gestion du risque**
   - Toujours vérifier la liquidité avant de passer des ordres importants
   - Utiliser les niveaux de support/résistance pour placer des stops
   - Adapter la taille des positions en fonction de la volatilité

2. **Optimisation des performances**
   - Mettre en cache les calculs coûteux
   - Utiliser des intervalles de temps appropriés pour l'analyse
   - Éviter le sur-ajustement des paramètres

3. **Surveillance continue**
   - Surveiller la qualité du signal au fil du temps
   - Ajuster les stratégies en fonction des conditions de marché changeantes
   - Journaliser les décisions pour analyse ultérieure

4. **Gestion des erreurs**
   - Toujours gérer les erreurs potentielles lors de l'accès aux données
   - Mettre en place des valeurs par défaut pour les indicateurs manquants
   - Surveiller les logs pour détecter les problèmes potentiels

## Dépannage

### Problème: Données manquantes dans le carnet d'ordres
**Solution:** Vérifiez la connexion à l'API et assurez-vous que le flux de données est actif.

### Problème: Signaux incohérents
**Solution:** Vérifiez la cohérence des données d'entrée et ajustez les paramètres des indicateurs si nécessaire.

### Problème: Performances médiocres
**Solution:** Optimisez les calculs en réduisant la fréquence des mises à jour ou en utilisant des algorithmes plus efficaces.

## Conclusion

Le module de carnet d'ordres avancé offre des outils puissants pour analyser les conditions de marché et prendre des décisions de trading éclairées. En combinant ces indicateurs avec d'autres signaux techniques et fondamentaux, vous pouvez développer des stratégies de trading plus robustes et performantes.

Pour toute question ou problème, veuillez consulter la documentation officielle ou ouvrir une issue sur le dépôt GitHub.
