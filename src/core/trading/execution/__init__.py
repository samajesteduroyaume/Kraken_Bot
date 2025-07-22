"""
Gestion avancée de l'exécution des ordres de trading.

Ce module fournit des fonctionnalités complètes pour :
- L'exécution des ordres (market, limit, stop-loss, take-profit)
- La gestion des positions (ouverture, fermeture, suivi)
- La surveillance des ordres actifs
- Le suivi des performances
- La gestion des erreurs et reprises
"""
import asyncio
import logging
import json
import os
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum, auto

from src.core.config_adapter import Config  # Migré vers le nouvel adaptateur
from src.core.api.kraken import KrakenAPI
from src.utils.helpers import generate_unique_id


def ensure_data_dir() -> str:
    """Crée et retourne le répertoire de données pour les trades."""
    data_dir = os.path.join(os.path.expanduser('~'), '.kraken_bot', 'data')
    os.makedirs(data_dir, exist_ok=True)
    return data_dir


logger = logging.getLogger(__name__)


class OrderStatus(Enum):
    """États possibles d'un ordre."""
    PENDING = auto()
    OPEN = auto()
    CLOSED = auto()
    CANCELED = auto()
    EXPIRED = auto()
    REJECTED = auto()
    PARTIALLY_FILLED = auto()


class OrderType(Enum):
    """Types d'ordres supportés."""
    MARKET = 'market'
    LIMIT = 'limit'
    STOP_LOSS = 'stop-loss'
    TAKE_PROFIT = 'take-profit'
    STOP_LOSS_LIMIT = 'stop-loss-limit'
    TAKE_PROFIT_LIMIT = 'take-profit-limit'


@dataclass
class Trade:
    """Représente une transaction de trading complète."""
    id: str
    pair: str
    side: str  # 'buy' or 'sell'
    order_type: OrderType
    amount: float
    filled: float = 0.0
    price: Optional[float] = None
    average_price: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    leverage: float = 1.0
    status: OrderStatus = OrderStatus.PENDING
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    closed_at: Optional[datetime] = None
    fee: float = 0.0
    pnl: float = 0.0
    pnl_pct: float = 0.0
    exit_price: Optional[float] = None
    exit_reason: Optional[str] = None
    tags: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convertit l'objet en dictionnaire pour la sérialisation."""
        data = asdict(self)
        data['status'] = self.status.name
        data['order_type'] = self.order_type.value
        data['created_at'] = self.created_at.isoformat()
        data['updated_at'] = self.updated_at.isoformat()
        if self.closed_at:
            data['closed_at'] = self.closed_at.isoformat()
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Trade':
        """Crée un objet Trade à partir d'un dictionnaire."""
        if 'created_at' in data and isinstance(data['created_at'], str):
            data['created_at'] = datetime.fromisoformat(data['created_at'])
        if 'updated_at' in data and isinstance(data['updated_at'], str):
            data['updated_at'] = datetime.fromisoformat(data['updated_at'])
        if 'closed_at' in data and data['closed_at'] and isinstance(
                data['closed_at'], str):
            data['closed_at'] = datetime.fromisoformat(data['closed_at'])

        if 'status' in data and isinstance(data['status'], str):
            data['status'] = OrderStatus[data['status']]
        if 'order_type' in data and isinstance(data['order_type'], str):
            data['order_type'] = OrderType(data['order_type'])

        return cls(**data)


class OrderSlippageError(Exception):
    """Exception levée lorsqu'un ordre subit un glissement de prix excessif."""


class OrderSizeError(Exception):
    """Exception levée lorsque la taille de l'ordre est invalide."""


class OrderExecutor:
    """
    Gestionnaire avancé d'exécution des ordres avec suivi des performances.

    Cette classe gère :
    - L'exécution des ordres (market, limit, stop-loss, take-profit)
    - Le suivi des positions et des ordres actifs
    - La gestion des erreurs et reprises
    - Les métriques de performance
    - La journalisation et la persistance des trades
    """

    def __init__(self,
                 api: KrakenAPI,
                 config: Optional[Dict] = None,
                 data_dir: Optional[str] = None):
        """
        Initialise le gestionnaire d'exécution.

        Args:
            api: Instance de l'API Kraken
            config: Configuration du trading (optionnel)
            data_dir: Répertoire pour la sauvegarde des données (optionnel)
        """
        self.api = api
        self.config = config or Config
        self.data_dir = data_dir or ensure_data_dir()

        # Suivi des états
        self.active_orders: Dict[str, Trade] = {}
        self.positions: Dict[str, Dict] = {}
        self.trade_history: List[Trade] = []
        self.closed_trades: List[Trade] = []

        # Métriques de performance
        self.metrics = {
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'total_pnl': 0.0,
            'win_rate': 0.0,
            'profit_factor': 0.0,
            'max_drawdown': 0.0,
            'sharpe_ratio': 0.0,
            'last_updated': datetime.utcnow()
        }

        # Configuration
        self.max_slippage_pct = getattr(self.config, 'max_slippage_pct', 1.0)
        self.max_position_size_pct = getattr(
            self.config, 'max_position_size_pct', 10.0)
        self.max_retries = getattr(self.config, 'max_retries', 3)
        self.retry_delay = getattr(self.config, 'retry_delay', 1.0)

        # Initialisation
        self._load_state()
        logger.info(
            f"OrderExecutor initialisé. {len(self.trade_history)} trades chargés.")

    def _get_state_filepath(self) -> str:
        """Retourne le chemin du fichier de sauvegarde de l'état."""
        return os.path.join(self.data_dir, 'order_executor_state.json')

    def _save_state(self) -> bool:
        """Sauvegarde l'état actuel dans un fichier."""
        try:
            state = {
                'active_orders': [
                    order.to_dict() for order in self.active_orders.values()],
                'trade_history': [
                    trade.to_dict() for trade in self.trade_history],
                'closed_trades': [
                    trade.to_dict() for trade in self.closed_trades],
                'metrics': self.metrics,
                'last_saved': datetime.utcnow().isoformat()}

            filepath = self._get_state_filepath()
            with open(filepath, 'w') as f:
                json.dump(state, f, indent=2)

            logger.debug(f"État sauvegardé dans {filepath}")
            return True

        except Exception as e:
            logger.error(f"Erreur lors de la sauvegarde de l'état: {e}")
            return False

    def _load_state(self) -> bool:
        """Charge l'état depuis un fichier."""
        try:
            filepath = self._get_state_filepath()
            if not os.path.exists(filepath):
                logger.info(
                    "Aucun fichier d'état trouvé, démarrage avec un état vide.")
                return False

            with open(filepath, 'r') as f:
                state = json.load(f)

            # Charger les trades
            self.active_orders = {
                order['id']: Trade.from_dict(order)
                for order in state.get('active_orders', [])
            }

            self.trade_history = [
                Trade.from_dict(trade)
                for trade in state.get('trade_history', [])
            ]

            self.closed_trades = [
                Trade.from_dict(trade)
                for trade in state.get('closed_trades', [])
            ]

            self.metrics = state.get('metrics', self.metrics)

            logger.info(f"État chargé depuis {filepath}")
            return True

        except Exception as e:
            logger.error(f"Erreur lors du chargement de l'état: {e}")
            return False

    def _update_metrics(self, trade: Trade) -> None:
        """Met à jour les métriques de performance."""
        if trade.status != OrderStatus.CLOSED:
            return

        self.metrics['total_trades'] += 1

        if trade.pnl > 0:
            self.metrics['winning_trades'] += 1
        elif trade.pnl < 0:
            self.metrics['losing_trades'] += 1

        self.metrics['total_pnl'] += trade.pnl

        # Mettre à jour le taux de réussite
        if self.metrics['total_trades'] > 0:
            self.metrics['win_rate'] = (
                self.metrics['winning_trades'] / self.metrics['total_trades']
            )

        # Mettre à jour le profit factor
        winning_amount = sum(
            t.pnl for t in self.closed_trades if t.pnl > 0
        )
        losing_amount = abs(sum(
            t.pnl for t in self.closed_trades if t.pnl < 0
        ))

        if losing_amount > 0:
            self.metrics['profit_factor'] = winning_amount / losing_amount

        self.metrics['last_updated'] = datetime.utcnow()

    def _validate_order_parameters(self,
                                   pair: str,
                                   order_type: str,
                                   side: str,
                                   amount: float,
                                   price: Optional[float] = None) -> bool:
        """Valide les paramètres d'un ordre."""
        if amount <= 0:
            raise OrderSizeError(
                f"Le montant doit être supérieur à zéro: {amount}")

        if order_type != 'market' and price is None:
            raise ValueError(
                "Un prix doit être spécifié pour les ordres non-market")

        if order_type == 'market' and price is not None:
            logger.warning("Le prix est ignoré pour les ordres de marché")

        # Vérifier la taille maximale de la position
        if self._calculate_position_size(
                pair, amount) > self.max_position_size_pct:
            raise OrderSizeError(
                f"La taille de la position dépasse {self.max_position_size_pct}% du capital"
            )

        return True

    def _calculate_position_size(self, pair: str, amount: float) -> float:
        """Calcule la taille de la position en pourcentage du capital."""
        # Cette méthode devrait être implémentée pour calculer la taille de la position
        # en fonction du capital disponible et de la gestion des risques
        return 0.0  # À implémenter

    def _check_slippage(self,
                        expected_price: float,
                        actual_price: float) -> bool:
        """Vérifie si le glissement de prix est acceptable."""
        if expected_price == 0:
            return False

        slippage = abs(actual_price - expected_price) / expected_price * 100
        if slippage > self.max_slippage_pct:
            logger.warning(
                f"Glissement de prix excessif: {slippage:.2f}% "
                f"(max: {self.max_slippage_pct}%)"
            )
            return False

        return True

    async def _execute_with_retry(self,
                                  func,
                                  *args,
                                  max_retries: Optional[int] = None,
                                  **kwargs) -> Any:
        """Exécute une fonction avec des tentatives de reprise en cas d'échec."""
        max_retries = max_retries or self.max_retries
        last_exception = None

        for attempt in range(max_retries + 1):
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                last_exception = e
                if attempt < max_retries:
                    wait_time = self.retry_delay * \
                        (2 ** attempt)  # Backoff exponentiel
                    logger.warning(
                        f"Tentative {attempt + 1}/{max_retries} échouée. "
                        f"Nouvelle tentative dans {wait_time:.1f}s. Erreur: {e}")
                    await asyncio.sleep(wait_time)

        raise last_exception or Exception(
            "Erreur inconnue lors de l'exécution avec reprise")

    async def execute_order(self,
                            pair: str,
                            order_type: Union[str, OrderType],
                            side: str,
                            amount: float,
                            price: Optional[float] = None,
                            stop_loss: Optional[float] = None,
                            take_profit: Optional[float] = None,
                            leverage: float = 1.0,
                            reduce_only: bool = False,
                            post_only: bool = False,
                            time_in_force: str = 'GTC',
                            client_order_id: Optional[str] = None,
                            tags: Optional[Dict[str, Any]] = None) -> Trade:
        """
        Exécute un ordre sur le marché avec gestion avancée.

        Args:
            pair: Paire de trading (ex: 'XBTUSD')
            order_type: Type d'ordre (market, limit, stop-loss, etc.)
            side: Direction de l'ordre ('buy' ou 'sell')
            amount: Montant de la devise de base à trader
            price: Prix pour les ordres limités (obligatoire sauf pour les ordres market)
            stop_loss: Prix de stop-loss (optionnel)
            take_profit: Prix de take-profit (optionnel)
            leverage: Effet de levier à utiliser (1.0 pour pas de levier)
            reduce_only: Si True, l'ordre ne peut qu'augmenter la position
            post_only: Si True, garantit que l'ordre sera ajouté au carnet d'ordres
            time_in_force: Durée de validité de l'ordre (GTC, IOC, FOK, etc.)
            client_order_id: ID personnalisé pour l'ordre (optionnel)
            tags: Métadonnées supplémentaires à associer au trade

        Returns:
            Objet Trade représentant l'ordre exécuté

        Raises:
            OrderSizeError: Si la taille de l'ordre est invalide
            OrderSlippageError: Si le glissement de prix est trop important
            ValueError: Si les paramètres sont invalides
            Exception: Pour les autres erreurs d'exécution
        """
        try:
            # Valider et formater les paramètres
            order_type_enum = OrderType(order_type) if isinstance(
                order_type, str) else order_type
            side = side.lower()

            if side not in ['buy', 'sell']:
                raise ValueError(
                    "Le côté de l'ordre doit être 'buy' ou 'sell'")

            # Valider les paramètres de l'ordre
            self._validate_order_parameters(
                pair, order_type_enum.value, side, amount, price)

            # Obtenir le prix actuel pour les ordres de marché
            current_price = None
            if order_type_enum == OrderType.MARKET:
                ticker = await self._execute_with_retry(
                    self.api.get_ticker,
                    pair
                )
                current_price = float(ticker.get('c', [0])[0])  # Dernier prix

                # Vérifier le glissement de prix si un prix de référence est
                # fourni
                if price is not None and not self._check_slippage(
                        price, current_price):
                    raise OrderSlippageError(
                        f"Glissement de prix trop important: {current_price} vs {price}"
                    )

                # Utiliser le prix actuel pour les ordres de marché
                price = current_price

            # Créer un nouvel objet Trade
            trade_id = client_order_id or generate_unique_id()
            trade = Trade(
                id=trade_id,
                pair=pair,
                side=side,
                order_type=order_type_enum,
                amount=amount,
                price=price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                leverage=leverage,
                tags=tags or {}
            )

            # Préparer les paramètres de l'ordre pour l'API
            order_params = {
                'pair': pair,
                'type': side,
                'ordertype': order_type_enum.value,
                'volume': str(amount),
                'leverage': str(leverage),
                'validate': self.config.get('validate_orders', False),
                'userref': trade_id,  # Référence utilisateur pour le suivi
                'oflags': []
            }

            # Ajouter les flags d'ordre
            if post_only:
                order_params['oflags'].append('post')

            if reduce_only:
                order_params['oflags'].append('reduce-only')

            # Ajouter le prix pour les ordres limités
            if order_type_enum != OrderType.MARKET and price is not None:
                order_params['price'] = str(price)

                # Ajouter le temps de validité
                order_params['timeinforce'] = time_in_force

            # Ajouter les ordres conditionnels
            if stop_loss is not None:
                order_params['stopprice'] = str(stop_loss)
                # ou 'mark' selon la stratégie
                order_params['stoptrigger'] = 'last'

            if take_profit is not None:
                order_params['takeprofit'] = str(take_profit)

            # Exécuter l'ordre avec reprise en cas d'échec
            logger.info(f"Exécution de l'ordre: {order_params}")
            result = await self._execute_with_retry(
                self.api.add_order,
                **order_params
            )

            if not result or 'txid' not in result:
                raise ValueError(
                    "Échec de l'exécution de l'ordre: aucune transaction ID reçue")

            # Mettre à jour le trade avec les informations de l'échange
            txid = result['txid'][0] if isinstance(
                result['txid'], list) else result['txid']
            trade.tags.update({
                'txid': txid,
                'status_url': f"https://www.kraken.com/orders/{txid}",
                'exchange_timestamp': datetime.utcnow().isoformat()
            })

            # Ajouter le trade à l'historique
            self.active_orders[trade_id] = trade
            self.trade_history.append(trade)

            # Sauvegarder l'état
            self._save_state()

            logger.info(
                f"Ordre exécuté avec succès: {trade_id} (txid: {txid})")
            return trade

        except Exception as e:
            error_msg = f"Erreur lors de l'exécution de l'ordre {pair} {side} {amount}@{price}: {str(e)}"
            logger.error(error_msg, exc_info=True)

            # Mettre à jour le statut du trade en cas d'erreur
            if 'trade' in locals():
                trade.status = OrderStatus.REJECTED
                trade.tags['error'] = str(e)
                self._update_metrics(trade)
                self._save_state()

            raise

    async def cancel_order(self, order_id: str) -> bool:
        """
        Annule un ordre en attente.

        Args:
            order_id: ID de l'ordre à annuler

        Returns:
            True si l'annulation a réussi, False sinon
        """
        try:
            result = await self.api.cancel_order(txid=order_id)
            if result and 'count' in result and result['count'] > 0:
                if order_id in self.active_orders:
                    self.active_orders[order_id]['status'] = 'cancelled'
                return True
            return False
        except Exception as e:
            logger.error(
                f"Erreur lors de l'annulation de l'ordre {order_id}: {e}")
            return False

    async def _update_active_order(
            self,
            trade: Trade,
            order_info: Dict) -> None:
        """Met à jour les informations d'un ordre actif."""
        # Mettre à jour les champs de base
        trade.updated_at = datetime.utcnow()

        # Mettre à jour le statut
        if 'status' in order_info:
            try:
                trade.status = OrderStatus[order_info['status'].upper()]
            except (KeyError, AttributeError):
                logger.warning(
                    f"Statut d'ordre inconnu: {order_info.get('status')}")

        # Mettre à jour le montant exécuté
        if 'vol_exec' in order_info:
            trade.filled = float(order_info['vol_exec'])

        # Mettre à jour le prix moyen d'exécution
        if 'avg_price' in order_info and order_info['avg_price'] is not None:
            trade.average_price = float(order_info['avg_price'])

        # Mettre à jour les métadonnées
        trade.tags.update({
            'last_updated': trade.updated_at.isoformat(),
            'order_info': order_info
        })

        # Si l'ordre est partiellement ou complètement exécuté
        if trade.filled > 0:
            trade.status = OrderStatus.PARTIALLY_FILLED

            # Si l'ordre est complètement exécuté
            if trade.filled >= trade.amount * 0.999:  # Tolérance de 0.1% pour les arrondis
                trade.status = OrderStatus.CLOSED
                trade.closed_at = trade.updated_at
                trade.exit_price = trade.average_price

                # Calculer le PnL
                self._calculate_trade_pnl(trade)

                # Déplacer vers les trades fermés
                if trade.id in self.active_orders:
                    self.closed_trades.append(self.active_orders.pop(trade.id))

    async def _check_closed_order(self, trade: Trade, result: Dict) -> None:
        """Vérifie et traite un ordre qui n'est plus dans les ordres ouverts."""
        try:
            # Récupérer les détails de l'ordre fermé
            order_details = await self._execute_with_retry(
                self.api.get_order,
                trade.tags.get('txid', trade.id),
                trades=True
            )

            if not order_details:
                logger.warning(
                    f"Impossible de récupérer les détails de l'ordre {trade.id}")
                trade.status = OrderStatus.REJECTED
                trade.tags['error'] = 'Détails de l\'ordre introuvables'
                result['errors'].append(trade.to_dict())
                return

            # Mettre à jour les informations du trade
            trade.updated_at = datetime.utcnow()
            trade.closed_at = trade.updated_at

            # Mettre à jour le statut et les métriques
            await self._update_active_order(trade, order_details)

            # Ajouter aux résultats
            if trade.status == OrderStatus.CLOSED:
                result['closed'].append(trade.to_dict())
            else:
                result['updated'].append(trade.to_dict())

        except Exception as e:
            error_msg = f"Erreur lors de la vérification de l'ordre {trade.id}: {str(e)}"
            logger.error(error_msg, exc_info=True)
            trade.tags['error'] = error_msg
            result['errors'].append(trade.to_dict())

    async def close_position(
            self,
            position_id: str,
            amount: Optional[float] = None) -> bool:
        """
        Ferme une position ouverte.

        Args:
            position_id: ID de la position à fermer
            amount: Montant à fermer (optionnel, ferme tout si non spécifié)

        Returns:
            True si la fermeture a réussi, False sinon
        """
        try:
            if position_id not in self.positions:
                logger.warning(f"Position {position_id} non trouvée")
                return False

            position = self.positions[position_id]
            close_side = 'sell' if position['side'] == 'buy' else 'buy'
            close_amount = amount or position['amount']

            # Placer un ordre de clôture
            order_result = await self.execute_order(
                pair=position['pair'],
                order_type='market',
                side=close_side,
                amount=close_amount,
                leverage=position['leverage']
            )

            if order_result:
                # Mettre à jour la position
                if amount and amount < position['amount']:
                    # Fermeture partielle
                    self.positions[position_id]['amount'] -= amount
                else:
                    # Fermeture complète
                    del self.positions[position_id]

                return True

            return False

        except Exception as e:
            logger.error(
                f"Erreur lors de la fermeture de la position {position_id}: {e}")
            return False

    def get_open_positions(self) -> Dict[str, Dict]:
        """
        Récupère les positions ouvertes.

        Returns:
            Dictionnaire des positions ouvertes
        """
        return self.positions

    def get_active_orders(self) -> Dict[str, Trade]:
        """
        Récupère les ordres actifs.

        Returns:
            Dictionnaire des ordres actifs
        """
        return self.active_orders

    async def update_orders_status(self) -> None:
        """
        Met à jour le statut de tous les ordres actifs.

        Cette méthode vérifie l'état de tous les ordres actifs et met à jour
        leur statut en fonction de la réponse de l'API.
        """
        try:
            # Récupérer les ordres ouverts depuis l'API
            open_orders = await self.api.get_open_orders()

            # Mettre à jour chaque ordre actif
            for order_id, trade in self.active_orders.items():
                if order_id not in open_orders:
                    # L'ordre n'est plus dans les ordres ouverts
                    self._check_closed_order(trade, open_orders)
                else:
                    # Mettre à jour l'ordre actif
                    self._update_active_order(trade, open_orders[order_id])

        except Exception as e:
            logger.error(f"Erreur lors de la mise à jour des ordres: {e}")

    def get_trade_history(self) -> List[Dict]:
        """
        Récupère l'historique des trades.

        Returns:
            Liste des trades effectués
        """
        return self.trade_history
