# API du module Order Book

## Vue d'ensemble
Le module `order_book` fournit une interface pour accéder et analyser les carnets d'ordres des marchés de crypto-monnaies. Il est conçu pour être performant, précis et facile à intégrer dans des stratégies de trading avancées.

## Classes principales

### OrderBookManager
Gère la mise à jour et l'accès au carnet d'ordres pour une paire de trading spécifique.

#### Méthodes principales

##### `__init__(symbol: str, api: Any, config: Optional[Dict] = None)`
Initialise le gestionnaire de carnet d'ordres.

**Paramètres :**
- `symbol` (str): Paire de trading (ex: 'XXBTZUSD')
- `api`: Instance de l'API d'échange
- `config` (dict, optionnel): Configuration supplémentaire

##### `async start(update_interval: float = 1.0) -> None`
Démarre les mises à jour automatiques du carnet d'ordres.

**Paramètres :**
- `update_interval` (float): Intervalle de mise à jour en secondes

##### `async stop() -> None`
Arrête les mises à jour automatiques.

##### `async update() -> None`
Met à jour manuellement le carnet d'ordres.

##### `get_metrics() -> OrderBookMetrics`
Retourne les métriques actuelles du carnet d'ordres.

### OrderBookSnapshot
Représente un instantané du carnet d'ordres à un moment donné.

#### Propriétés
- `bids`: Liste des offres d'achat (prix décroissants)
- `asks`: Liste des offres de vente (prix croissants)
- `timestamp`: Horodatage de l'instantané
- `metrics`: Objet OrderBookMetrics contenant les métriques calculées

#### Méthodes
- `get_price_levels(side: str, limit: int = 10) -> List[PriceLevel]`
  - Récupère les N meilleurs niveaux de prix pour un côté donné ('bid' ou 'ask')
  
- `get_cumulative_volume(side: str, price: float) -> float`
  - Calcule le volume cumulé jusqu'à un certain niveau de prix

### OrderBookMetrics
Contient les métriques calculées à partir du carnet d'ordres.

#### Propriétés
- `spread`: Écart entre le meille prix d'achat et de vente
- `imbalance`: Déséquilibre entre l'offre et la demande
- `vwap_bid`: Prix moyen pondéré par le volume pour les offres d'achat
- `vwap_ask`: Prix moyen pondéré par le volume pour les offres de vente

## Exemples d'utilisation

### Exemple 1: Récupération des métriques de base
```python
from src.core.market.data_manager import MarketDataManager

# Initialisation
market_data = MarketDataManager(api)

# Mise à jour des données
await market_data.update_market_data('XXBTZUSD')

# Récupération du gestionnaire de carnet d'ordres
order_book_mgr = market_data.get_order_book_manager('XXBTZUSD')

# Récupération des métriques
metrics = order_book_mgr.get_metrics()
print(f"Spread: {metrics.spread}")
print(f"Imbalance: {metrics.imbalance}")
print(f"VWAP Bid/Ask: {metrics.vwap_bid}/{metrics.vwap_ask}")
```

### Exemple 2: Utilisation dans une stratégie de trading
```python
from src.core.trading.signals import SignalGenerator

class MyTradingStrategy:
    def __init__(self):
        self.signal_generator = SignalGenerator()
    
    async def on_market_data_update(self, symbol: str, ohlc_data: dict, order_book: dict):
        # Génération des signaux avec les données OHLC et le carnet d'ordres
        signals = await self.signal_generator.generate_signals(
            ohlc_data=ohlc_data,
            pair=symbol,
            timeframe='1h',
            order_book=order_book
        )
        
        # Prise de décision basée sur les signaux
        if signals.get('imbalance_signal') == 'buy' and signals.get('spread_signal') == 'normal':
            self.place_buy_order()
        elif signals.get('imbalance_signal') == 'sell' and signals.get('vwap_signal') == 'overbought':
            self.place_sell_order()
```

### Exemple 3: Analyse avancée du carnet d'ordres
```python
# Récupération d'un instantané
snapshot = order_book_mgr.current_snapshot

# Analyse des 5 meilleurs niveaux de prix
for level in snapshot.get_price_levels('bid', limit=5):
    print(f"Prix: {level['price']}, Volume: {level['amount']}")

# Calcul du volume cumulé jusqu'à un certain prix
cumulative_volume = snapshot.get_cumulative_volume(
    'ask', 
    price=float(snapshot.asks[0]['price']) * 1.01  # 1% au-dessus du meilleur prix de vente
)
print(f"Volume cumulé: {cumulative_volume}")
```

## Bonnes pratiques

1. **Gestion des erreurs** : Toujours envelopper les appels dans des blocs try/except pour gérer les erreurs réseau ou de format de données.

2. **Performance** : Pour les stratégies haute fréquence, évitez de créer de nouveaux objets dans des boucles serrées.

3. **Mémoire** : Les instantanés du carnet d'ordres peuvent être volumineux. Pensez à les nettoyer lorsqu'ils ne sont plus nécessaires.

4. **Tests** : Testez toujours vos stratégies avec des données historiques avant de les exécuter en production.

## Référence des erreurs

| Code d'erreur | Description | Solution suggérée |
|--------------|-------------|-------------------|
| ORDER_BOOK_UPDATE_FAILED | Échec de la mise à jour du carnet d'ordres | Vérifier la connexion réseau et les identifiants API |
| INVALID_ORDER_BOOK_DATA | Données du carnet d'ordres invalides | Vérifier le format des données reçues de l'API |
| INVALID_SIDE | Côté de l'ordre invalide | Utiliser 'bid' ou 'ask' |

## Notes de version

### v1.0.0 (2025-07-22)
- Première version stable
- Support des carnets d'ordres en temps réel
- Calcul des métriques avancées (spread, imbalance, VWAP)
- Intégration avec le gestionnaire de données de marché
