"""
Module d'exécution des ordres pour le bot de trading Kraken.
Gère l'envoi, la surveillance et l'exécution des ordres via l'API Kraken.
"""

from typing import Dict, Optional, List
import logging
import asyncio
from dataclasses import dataclass
from src.core.api.kraken import KrakenAPI

logger = logging.getLogger(__name__)


@dataclass
class Order:
    """Représente un ordre de trading."""
    id: str
    pair: str
    type: str  # 'buy' ou 'sell'
    side: str  # 'long' ou 'short'
    size: float
    price: float
    status: str  # 'pending', 'executed', 'cancelled'
    timestamp: float


class OrderExecutor:
    """Gestionnaire d'exécution des ordres pour le bot de trading."""

    def __init__(self, api: 'KrakenAPI', config: Dict):
        """Initialise le gestionnaire d'ordres."""
        self.api = api
        self.config = config
        self.orders: Dict[str, Order] = {}  # {order_id: Order}
        self.order_queue = asyncio.Queue()
        self.running = False

    async def start(self) -> None:
        """Démarre le gestionnaire d'ordres."""
        self.running = True
        asyncio.create_task(self.process_orders())

    async def stop(self) -> None:
        """Arrête le gestionnaire d'ordres."""
        self.running = False
        await self.order_queue.put(None)  # Signal d'arrêt

    async def place_order(self,
                          pair: str,
                          type: str,
                          side: str,
                          size: float,
                          price: Optional[float] = None) -> str:
        """
        Place un ordre sur Kraken.

        Args:
            pair: Paire de trading
            type: Type d'ordre ('buy' ou 'sell')
            side: Côté ('long' ou 'short')
            size: Taille de l'ordre
            price: Prix (None pour un ordre de marché)

        Returns:
            ID de l'ordre
        """
        try:
            # Préparer les paramètres
            params = {
                'pair': pair,
                'type': type,
                'ordertype': 'limit' if price else 'market',
                'volume': str(size),
                'leverage': str(self.config.get('default_leverage', 1)),
                'validate': False  # Désactiver la validation pour les ordres réels
            }

            if price:
                params['price'] = str(price)

            # Envoyer l'ordre
            result = await self.api.create_order(params)

            # Créer l'objet Order
            order_id = result['txid'][0]
            order = Order(
                id=order_id,
                pair=pair,
                type=type,
                side=side,
                size=size,
                price=price or 0.0,
                status='pending',
                timestamp=result['timestamp']
            )

            # Ajouter à la file d'attente
            await self.order_queue.put(order)

            return order_id

        except Exception as e:
            logger.error(f"Erreur lors de la création de l'ordre: {str(e)}")
            raise

    async def cancel_order(self, order_id: str) -> bool:
        """
        Annule un ordre.

        Args:
            order_id: ID de l'ordre à annuler

        Returns:
            True si l'annulation a réussi, False sinon
        """
        try:
            if order_id not in self.orders:
                logger.error(f"Ordre {order_id} non trouvé")
                return False

            result = await self.api.cancel_order(order_id)
            if result.get('status') == 'ok':
                self.orders[order_id].status = 'cancelled'
                return True

            return False

        except Exception as e:
            logger.error(f"Erreur lors de l'annulation de l'ordre: {str(e)}")
            raise

    async def process_orders(self) -> None:
        """Traite les ordres dans la file d'attente."""
        while self.running:
            order = await self.order_queue.get()
            if order is None:  # Signal d'arrêt
                break

            try:
                # Vérifier le statut de l'ordre
                status = await self.api.get_order_status(order.id)
                if status.get('status') == 'filled':
                    order.status = 'executed'
                    logger.info(f"Ordre {order.id} exécuté")

                elif status.get('status') == 'canceled':
                    order.status = 'cancelled'
                    logger.info(f"Ordre {order.id} annulé")

                # Mettre à jour le dictionnaire des ordres
                self.orders[order.id] = order

            except Exception as e:
                logger.error(
                    f"Erreur lors du traitement de l'ordre {order.id}: {str(e)}")

            finally:
                self.order_queue.task_done()

    async def get_open_orders(self) -> List[Order]:
        """Récupère la liste des ordres ouverts."""
        try:
            orders = await self.api.get_open_orders()
            return [self.orders[order_id]
                    for order_id in orders if order_id in self.orders]

        except Exception as e:
            logger.error(
                f"Erreur lors de la récupération des ordres ouverts: {str(e)}")
            raise
