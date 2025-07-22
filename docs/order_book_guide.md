# Guide du module de carnet d'ordres

## Vue d'ensemble

Le module de carnet d'ordres offre une solution complète pour gérer et analyser les carnets d'ordres en temps réel. Il est conçu pour être efficace en mémoire et en performances, même avec des carnets volumineux.

## Fonctionnalités principales

- **Métriques avancées** : Spread, VWAP, déséquilibre, profondeur du marché
- **Mises à jour en temps réel** via WebSocket
- **Visualisations intégrées** pour l'analyse du carnet d'ordres
- **Gestion mémoire optimisée** pour les carnets volumineux
- **API simple et intuitive** pour l'intégration avec d'autres composants

## Installation

Assurez-vous d'avoir installé les dépendances requises :

```bash
pip install -r requirements.txt
```

## Utilisation de base

### Initialisation

```python
from src.core.market.order_book import OrderBookManager
from src.core.api.kraken_api import KrakenAPI

# Initialiser l'API Kraken
api = KrakenAPI()

# Créer un gestionnaire de carnet d'ordres
order_book = OrderBookManager(symbol='XBT/USD', api=api)

# Démarrer les mises à jour automatiques (toutes les secondes)
await order_book.start(update_interval=1.0)
```

### Accès aux données

```python
# Obtenir le snapshot actuel
snapshot = order_book.current_snapshot

# Accéder aux meilleurs prix
best_bid = snapshot.best_bid
best_ask = snapshot.best_ask
spread = snapshot.spread

# Obtenir les métriques avancées
metrics = snapshot.metrics
vwap_bid = metrics.vwap_bid
imbalance = metrics.imbalance
```

### Utilisation du WebSocket

Pour une mise à jour en temps réel :

```python
from src.core.market.order_book_websocket import OrderBookWebSocket

async def handle_update(update):
    print(f"Mise à jour du carnet: {update['symbol']}")
    print(f"Meilleur achat: {update['best_bid']}, Meilleure vente: {update['best_ask']}")

# Créer et démarrer le WebSocket
websocket = OrderBookWebSocket(
    symbol='XBT/USD',
    callback=handle_update,
    depth=10  # Nombre de niveaux à suivre
)

# Démarrer la connexion
await websocket.connect()
```

## Visualisation

Le module inclut un outil de visualisation intégré :

```python
from examples.order_book_visualization import OrderBookVisualizer

async def main():
    visualizer = OrderBookVisualizer(symbol='XBT/USD', depth=10)
    await visualizer.start()

# Démarrer la visualisation
import asyncio
asyncio.run(main())
```

## Gestion de la mémoire

Le module est optimisé pour une utilisation efficace de la mémoire :

- Utilisation de `__slots__` pour réduire l'empreinte mémoire
- Nettoyage périodique des données anciennes
- Gestion efficace des grands ensembles de données avec des structures optimisées

## Bonnes pratiques

1. **Gestion des erreurs** : Toujours envelopper les appels dans des blocs try/except
2. **Nettoyage** : Appeler `stop()` pour libérer les ressources
3. **Mise à l'échelle** : Pour les applications haute fréquence, ajustez les paramètres de profondeur et d'intervalle
4. **Surveillance** : Surveillez l'utilisation de la mémoire pour les carnets très actifs

## Exemples avancés

### Calcul de l'impact de prix

```python
# Calculer l'impact d'un ordre important
impact = order_book.get_price_impact(
    amount=10.0,  # 10 BTC
    side='buy'    # ou 'sell'
)
print(f"Impact de prix estimé: {impact:.4f}%")
```

### Analyse de la profondeur du marché

```python
# Obtenir la profondeur de marché
heatmap = order_book.get_liquidity_heatmap(
    price_bins=20,
    volume_bins=10
)

# Afficher la heatmap
import matplotlib.pyplot as plt
plt.imshow(heatmap['data'], cmap='viridis')
plt.colorbar()
plt.show()
```

## Dépannage

### Problèmes courants

1. **Connexion perdue** : Le module tente automatiquement de se reconnecter
2. **Données manquantes** : Vérifiez la connexion Internet et les autorisations de l'API
3. **Problèmes de performance** : Réduisez la profondeur ou augmentez l'intervalle de mise à jour

### Journalisation

Activez le logging détaillé pour le débogage :

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Contribution

Les contributions sont les bienvenues ! Assurez-vous d'écrire des tests pour les nouvelles fonctionnalités et de maintenir la couverture de code.

## Licence

Ce projet est sous licence MIT. Voir le fichier `LICENSE` pour plus de détails.
