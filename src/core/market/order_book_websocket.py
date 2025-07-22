"""
Module WebSocket pour la mise à jour en temps réel du carnet d'ordres.

Ce module implémente une connexion WebSocket à l'API Kraken pour recevoir
les mises à jour du carnet d'ordres en temps réel.
"""

import asyncio
import json
import logging
from decimal import Decimal
from typing import Dict, List, Optional, Any, Callable, Awaitable

import websockets
from websockets.exceptions import ConnectionClosed

from .order_book import OrderBookManager, OrderBookSnapshot
from ..types.market_types import PriceLevel

# Configuration du logger
logger = logging.getLogger(__name__)

class OrderBookWebSocket:
    """Gestionnaire WebSocket pour les mises à jour en temps réel du carnet d'ordres."""
    
    # URL du WebSocket public de Kraken
    WS_PUBLIC_URL = "wss://ws.kraken.com"
    
    # Version de l'API WebSocket
    WS_VERSION = "1.4.0"
    
    def __init__(
        self,
        symbol: str,
        callback: Callable[[Dict[str, Any]], Awaitable[None]],
        depth: int = 100,
        max_retries: int = 5,
        reconnect_interval: float = 5.0
    ):
        """
        Initialise le client WebSocket pour le carnet d'ordres.
        
        Args:
            symbol: Symbole de la paire de trading (ex: 'XBT/USD')
            callback: Fonction de rappel appelée à chaque mise à jour
            depth: Nombre de niveaux de prix à récupérer (10, 25, 100, 500, 1000)
            max_retries: Nombre maximum de tentatives de reconnexion
            reconnect_interval: Délai entre les tentatives de reconnexion (secondes)
        """
        self.symbol = symbol.upper()
        self.callback = callback
        self.depth = depth
        self.max_retries = max_retries
        self.reconnect_interval = reconnect_interval
        
        # État de la connexion
        self._ws: Optional[websockets.WebSocketClientProtocol] = None
        self._running = False
        self._subscription_id: Optional[str] = None
        self._retry_count = 0
        
        # Gestionnaire de carnet d'ordres local
        self.order_book = OrderBookManager(symbol, None)  # Pas besoin d'API pour le WebSocket
        
        # Verrou pour les opérations thread-safe
        self._lock = asyncio.Lock()
        
    async def connect(self) -> None:
        """Établit la connexion WebSocket et s'abonne aux mises à jour."""
        if self._running:
            logger.warning("Déjà connecté")
            return
            
        self._running = True
        self._retry_count = 0
        
        while self._running and self._retry_count < self.max_retries:
            try:
                logger.info(f"Connexion au WebSocket pour {self.symbol}...")
                
                # Établir la connexion WebSocket
                self._ws = await websockets.connect(
                    self.WS_PUBLIC_URL,
                    ping_interval=20,
                    ping_timeout=10,
                    close_timeout=1
                )
                
                # Réinitialiser le compteur de tentatives après une connexion réussie
                self._retry_count = 0
                
                # S'abonner au carnet d'ordres
                await self._subscribe()
                
                # Écouter les messages
                await self._listen()
                
            except (ConnectionClosed, ConnectionError) as e:
                logger.warning(f"Connexion WebSocket perdue: {e}")
                await self._handle_disconnect()
                
            except Exception as e:
                logger.error(f"Erreur WebSocket: {e}", exc_info=True)
                await self._handle_disconnect()
                
            # Attendre avant de réessayer
            if self._running and self._retry_count < self.max_retries:
                wait_time = self.reconnect_interval * (self._retry_count + 1)
                logger.info(f"Nouvelle tentative dans {wait_time} secondes...")
                await asyncio.sleep(wait_time)
                
        if self._retry_count >= self.max_retries:
            logger.error("Nombre maximum de tentatives de reconnexion atteint")
            self._running = False
    
    async def disconnect(self) -> None:
        """Ferme la connexion WebSocket."""
        self._running = False
        if self._ws:
            await self._ws.close()
            self._ws = None
            
    async def _subscribe(self) -> None:
        """S'abonne aux mises à jour du carnet d'ordres."""
        if not self._ws:
            raise RuntimeError("WebSocket non connecté")
            
        # Message de souscription
        subscribe_msg = {
            "event": "subscribe",
            "pair": [self.symbol.replace('/', '')],
            "subscription": {
                "name": "book",
                "depth": self.depth
            }
        }
        
        logger.debug(f"Envoi de la souscription: {subscribe_msg}")
        await self._ws.send(json.dumps(subscribe_msg))
        
        # Attendre la confirmation de souscription
        response = await self._ws.recv()
        response_data = json.loads(response)
        
        if response_data.get("event") == "subscriptionStatus" and response_data.get("status") == "subscribed":
            self._subscription_id = response_data.get("channelID")
            logger.info(f"Abonnement réussi au canal {self._subscription_id}")
            
            # Le premier message contient l'état complet du carnet
            snapshot = await self._ws.recv()
            await self._process_message(snapshot)
        else:
            logger.error(f"Échec de l'abonnement: {response_data}")
            raise ConnectionError("Échec de l'abonnement au carnet d'ordres")
    
    async def _listen(self) -> None:
        """Écoute les messages du WebSocket."""
        if not self._ws:
            raise RuntimeError("WebSocket non connecté")
            
        logger.info("Écoute des mises à jour du carnet d'ordres...")
        
        try:
            async for message in self._ws:
                await self._process_message(message)
        except Exception as e:
            logger.error(f"Erreur lors de l'écoute des messages: {e}", exc_info=True)
            raise
    
    async def _process_message(self, message: str) -> None:
        """Traite un message reçu du WebSocket."""
        try:
            data = json.loads(message)
            
            # Ignorer les messages de statut et de cœur
            if isinstance(data, dict) and data.get("event") in ["heartbeat", "systemStatus"]:
                return
                
            # Traiter les mises à jour du carnet d'ordres
            if isinstance(data, list) and len(data) > 1 and data[1] == "book-100":
                await self._handle_order_book_update(data)
                
        except json.JSONDecodeError:
            logger.error(f"Impossible de décoder le message: {message}")
        except Exception as e:
            logger.error(f"Erreur lors du traitement du message: {e}", exc_info=True)
    
    async def _handle_order_book_update(self, data: List[Any]) -> None:
        """
        Traite une mise à jour du carnet d'ordres.
        
        Args:
            data: Données brutes reçues du WebSocket
        """
        try:
            # Le format du message est: [channelID, "book-100", {asks: [...], bids: [...], ...}]
            if len(data) < 3 or not isinstance(data[2], dict):
                logger.warning(f"Format de message inattendu: {data}")
                return
                
            update = data[2]
            
            # Mettre à jour le carnet d'ordres local
            async with self._lock:
                # Créer un nouveau snapshot si nécessaire
                if self.order_book._current_snapshot is None:
                    self.order_book._current_snapshot = OrderBookSnapshot([], [])
                
                # Appliquer les mises à jour
                if 'as' in update:  # Snapshot initial des asks
                    self.order_book._current_snapshot.asks = [
                        {'price': Decimal(price), 'amount': Decimal(amount), 'timestamp': time.time()}
                        for price, amount, _ in update['as']
                    ]
                if 'bs' in update:  # Snapshot initial des bids
                    self.order_book._current_snapshot.bids = [
                        {'price': Decimal(price), 'amount': Decimal(amount), 'timestamp': time.time()}
                        for price, amount, _ in update['bs']
                    ]
                
                # Appliquer les mises à jour incrémentielles
                if 'a' in update:  # Mise à jour des asks
                    self._apply_updates(update['a'], 'ask')
                if 'b' in update:  # Mise à jour des bids
                    self._apply_updates(update['b'], 'bid')
                
                # Trier les niveaux (les bids en ordre décroissant, les asks en ordre croissant)
                self.order_book._current_snapshot.bids.sort(key=lambda x: -float(x['price']))
                self.order_book._current_snapshot.asks.sort(key=lambda x: float(x['price']))
                
                # Mettre à jour les métriques
                self.order_book._current_snapshot.metrics.update(
                    self.order_book._current_snapshot.bids,
                    self.order_book._current_snapshot.asks
                )
                
                # Créer une copie des données pour le callback
                snapshot = self.order_book.current_snapshot
                
            # Appeler le callback avec les données mises à jour
            if self.callback and snapshot:
                await self.callback({
                    'symbol': self.symbol,
                    'bids': [
                        {'price': float(level['price']), 
                         'amount': float(level['amount']),
                         'timestamp': level.get('timestamp')}
                        for level in snapshot.bids
                    ],
                    'asks': [
                        {'price': float(level['price']), 
                         'amount': float(level['amount']),
                         'timestamp': level.get('timestamp')}
                        for level in snapshot.asks
                    ],
                    'timestamp': datetime.utcnow().isoformat(),
                    'metrics': {
                        'best_bid': float(snapshot.metrics.best_bid) if snapshot.metrics.best_bid else None,
                        'best_ask': float(snapshot.metrics.best_ask) if snapshot.metrics.best_ask else None,
                        'spread': float(snapshot.metrics.spread) if snapshot.metrics.spread else None,
                        'imbalance': float(snapshot.metrics.imbalance) if hasattr(snapshot.metrics, 'imbalance') else None,
                        'vwap_bid': float(snapshot.metrics.vwap_bid) if hasattr(snapshot.metrics, 'vwap_bid') else None,
                        'vwap_ask': float(snapshot.metrics.vwap_ask) if hasattr(snapshot.metrics, 'vwap_ask') else None
                    }
                })
                
        except Exception as e:
            logger.error(f"Erreur lors du traitement de la mise à jour: {e}", exc_info=True)
    
    async def _handle_disconnect(self) -> None:
        """Gère la déconnexion du WebSocket."""
        self._retry_count += 1
        
        if self._ws:
            try:
                await self._ws.close()
            except Exception:
                pass
            self._ws = None
            
        self._subscription_id = None
    
    def _apply_updates(self, updates: List[Tuple[str, str, str]], side: str) -> None:
        """
        Applique les mises à jour incrémentielles au carnet d'ordres.
        
        Args:
            updates: Liste des mises à jour (prix, montant, horodatage)
            side: 'bid' ou 'ask' pour indiquer le côté du carnet à mettre à jour
        """
        if self.order_book._current_snapshot is None:
            return
            
        # Sélectionner la liste à mettre à jour
        levels = (self.order_book._current_snapshot.bids if side == 'bid' 
                 else self.order_book._current_snapshot.asks)
        
        for price_str, amount_str, timestamp_str in updates:
            try:
                price = Decimal(price_str)
                amount = Decimal(amount_str)
                timestamp = float(timestamp_str) if timestamp_str else time.time()
                
                # Trouver l'index du niveau de prix existant
                index = next((i for i, level in enumerate(levels) 
                            if level['price'] == price), None)
                
                if amount <= 0:
                    # Supprimer le niveau si le montant est zéro ou négatif
                    if index is not None:
                        levels.pop(index)
                else:
                    # Mettre à jour ou ajouter le niveau
                    if index is not None:
                        levels[index]['amount'] = amount
                        levels[index]['timestamp'] = timestamp
                    else:
                        levels.append({
                            'price': price,
                            'amount': amount,
                            'timestamp': timestamp
                        })
                        
            except (ValueError, TypeError, Exception) as e:
                logger.warning(f"Erreur lors du traitement de la mise à jour {side} {price_str}: {e}")
    
    async def __aenter__(self):
        """Support du gestionnaire de contexte."""
        await self.connect()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Nettoyage du gestionnaire de contexte."""
        await self.disconnect()


async def run_websocket_example():
    """Exemple d'utilisation du WebSocket du carnet d'ordres."""
    def print_update(update):
        print(f"\n=== Mise à jour du carnet d'ordres pour {update['symbol']} ===")
        print(f"Meilleur achat: {update['metrics']['best_bid']}")
        print(f"Meilleure vente: {update['metrics']['best_ask']}")
        print(f"Spread: {update['metrics']['spread']}")
        print(f"Déséquilibre: {update['metrics']['imbalance']:.2%}")
    
    async def callback(update):
        print_update(update)
    
    # Créer et démarrer le WebSocket
    websocket = OrderBookWebSocket(
        symbol='XBT/USD',
        callback=callback,
        depth=10
    )
    
    try:
        print("Démarrage du WebSocket du carnet d'ordres (Ctrl+C pour arrêter)...")
        await websocket.connect()
    except KeyboardInterrupt:
        print("\nArrêt du WebSocket...")
    except Exception as e:
        print(f"Erreur: {e}")
    finally:
        await websocket.disconnect()


if __name__ == "__main__":
    import asyncio
    
    # Configurer le logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Désactiver les logs verbeux de websockets
    logging.getLogger('websockets').setLevel(logging.WARNING)
    
    # Exécuter l'exemple
    asyncio.run(run_websocket_example())
