"""
Classe de base pour les traders de trading.

Cette classe implémente l'interface ITrader et fournit les fonctionnalités
communes à tous les traders.
"""
from typing import Dict, List, Optional, Any
from decimal import Decimal
from datetime import datetime
import logging
from src.core.types.market_data import MarketData
from ..types import Portfolio, Position, Trade, Order, TradingContext, TradingMetrics
from ..types.types import TradeSignal, TradingConfig
from ..analysis.predictor import MLPredictor
from ..api.kraken import KrakenAPI
from ..config import Config
from .interfaces import ITrader
from src.core.trading.risk import RiskManager

logger = logging.getLogger(__name__)


class BaseTrader(ITrader):
    """Classe de base pour les fonctionnalités de trading communes."""

    def __init__(self,
                 api: KrakenAPI,
                 predictor: Optional[MLPredictor] = None,
                 config: Optional[Dict[str, Any]] = None):
        """
        Initialise le trader de base.

        Args:
            api: Instance de l'API Kraken
            predictor: Prédicteur ML (optionnel)
            config: Configuration du trading (optionnel)
        """
        self.api = api
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        self.running = False
        self.market_data: Dict[str, MarketData] = {}
        self.portfolio: Portfolio = {
            'total_value': Decimal('0'),
            'cash': Decimal('0'),
            'positions': [],
            'realized_pnl': Decimal('0'),
            'unrealized_pnl': Decimal('0'),
            'timestamp': datetime.now()
        }
        self.metrics: TradingMetrics = {
            'total_trades': 0,
            'win_rate': 0.0,
            'avg_profit': Decimal('0'),
            'avg_loss': Decimal('0'),
            'max_drawdown': Decimal('0'),
            'sharpe_ratio': 0.0,
            'sortino_ratio': 0.0,
            'calmar_ratio': 0.0,
            'profit_factor': 0.0,
            'roi': Decimal('0')
        }

    def initialize(self, config: Dict[str, Any]) -> None:
        """Initialise le trader avec la configuration donnée.

        Args:
            config: Configuration de trading
        """
        self.config = config
        self.running = True
        self.logger.info("Trader initialisé avec configuration:")
        self.logger.info(f"- Stratégie: {config.get('strategy', 'N/A')}")
        self.logger.info(f"- Profil de risque: {config.get('risk_profile', 'N/A')}")
        self.logger.info(f"- Levier maximum: {config.get('max_leverage', 'N/A')}")

    async def update_market_data(self, market_data: MarketData) -> None:
        """Met à jour les données de marché.

        Args:
            market_data: Données de marché pour la paire
        """
        try:
            symbol = market_data.get('symbol', None)
            if symbol:
                self.market_data[symbol] = market_data
                self.logger.info(
                    f"Données de marché mises à jour pour {symbol}")
            else:
                self.logger.warning("Aucune clé 'symbol' trouvée dans market_data")
        except Exception as e:
            self.logger.error(
                f"Erreur lors de la mise à jour des données de marché: {str(e)}")

    async def update_balance(self, portfolio: Portfolio) -> None:
        """Met à jour le portefeuille.

        Args:
            portfolio: Portefeuille mis à jour
        """
        try:
            self.portfolio = portfolio
            self.logger.debug(
                f"Portefeuille mis à jour: {portfolio['total_value']} USD")
        except Exception as e:
            self.logger.error(
                f"Erreur lors de la mise à jour du portefeuille: {str(e)}")

    async def generate_signals(self) -> List[TradeSignal]:
        """Génère les signaux de trading.

        Returns:
            Liste des signaux de trading générés
        """
        signals = []
        for pair, data in self.market_data.items():
            # Implémentation spécifique à la stratégie
            signal = self._generate_signal_for_pair(pair, data)
            if signal:
                signals.append(signal)
        return signals

    def _generate_signal_for_pair(
            self,
            pair: str,
            data: MarketData) -> Optional[TradeSignal]:
        """Génère un signal pour une paire spécifique.

        Cette méthode doit être implémentée par les classes filles.

        Args:
            pair: Paire de trading
            data: Données de marché

        Returns:
            Signal de trading ou None
        """
        raise NotImplementedError(
            "Cette méthode doit être implémentée par les classes filles")

    async def execute_trades(self, signals: List[TradeSignal]) -> List[Trade]:
        """Exécute les trades basés sur les signaux.

        Args:
            signals: Liste des signaux de trading

        Returns:
            Liste des trades exécutés
        """
        trades = []
        for signal in signals:
            trade = await self._execute_trade(signal)
            if trade:
                trades.append(trade)
        return trades

    async def _execute_trade(self, signal: TradeSignal) -> Optional[Trade]:
        """Exécute un trade basé sur un signal.

        Cette méthode doit être implémentée par les classes filles.

        Args:
            signal: Signal de trading

        Returns:
            Détails du trade exécuté ou None
        """
        raise NotImplementedError(
            "Cette méthode doit être implémentée par les classes filles")

    async def manage_positions(self) -> List[Order]:
        """Gère les positions ouvertes.

        Returns:
            Liste des ordres de gestion des positions
        """
        orders = []
        for position in self.portfolio['positions']:
            order = await self._manage_position(position)
            if order:
                orders.append(order)
        return orders

    async def _manage_position(self, position: Position) -> Optional[Order]:
        """Gère une position spécifique.

        Cette méthode doit être implémentée par les classes filles.

        Args:
            position: Position à gérer

        Returns:
            Ordre de gestion de la position ou None
        """
        raise NotImplementedError(
            "Cette méthode doit être implémentée par les classes filles")

    def get_trading_status(self) -> TradingContext:
        """Retourne l'état actuel du trading.

        Returns:
            Contexte de trading complet
        """
        return {
            'portfolio': self.portfolio,
            'positions': self.portfolio['positions'],
            'signals': [],  # À implémenter par les classes filles
            'orders': [],   # À implémenter par les classes filles
            'trades': [],   # À implémenter par les classes filles
            'metrics': self.metrics,
            'timestamp': datetime.now()
        }

    async def stop(self) -> None:
        """Arrête le trader proprement.

        Cette méthode doit :
        - Fermer toutes les positions ouvertes
        - Annuler tous les ordres en attente
        - Nettoyer les ressources
        """
        self.running = False

        # Fermer les positions ouvertes
        await self._close_all_positions()

        # Annuler les ordres en attente
        await self._cancel_all_orders()

        self.logger.info("Trader arrêté proprement")

    async def _close_all_positions(self) -> None:
        """Ferme toutes les positions ouvertes."""
        for position in self.portfolio['positions']:
            await self._close_position(position)

    async def _close_position(self, position: Position) -> None:
        """Ferme une position spécifique.

        Cette méthode doit être implémentée par les classes filles.

        Args:
            position: Position à fermer
        """
        raise NotImplementedError(
            "Cette méthode doit être implémentée par les classes filles")

    async def _cancel_all_orders(self) -> None:
        """Annule tous les ordres en attente."""
        # À implémenter par les classes filles

    async def get_metrics(self) -> TradingMetrics:
        """Retourne les métriques de performance.

        Returns:
            Métriques de performance du trading
        """
        return self.metrics

    async def get_positions(self) -> List[Position]:
        """Retourne la liste des positions actuelles.

        Returns:
            Liste des positions ouvertes
        """
        return self.portfolio['positions']

    async def get_orders(self) -> List[Order]:
        """Retourne la liste des ordres en cours.

        Returns:
            Liste des ordres en attente
        """
        # À implémenter par les classes filles
        return []

    async def get_trades(self) -> List[Trade]:
        """Retourne l'historique des trades.

        Returns:
            Liste des trades exécutés
        """
        # À implémenter par les classes filles
        return []

    async def get_status(self) -> Dict:
        """Récupère l'état actuel du trading."""
        try:
            status = {
                'timestamp': datetime.now().isoformat(),
                'balances': self.portfolio,
                'active_orders': self.get_orders(),
                'positions': self.get_positions(),
                'market_data': self.market_data,
                'trade_history': self.get_trades(),
                'metrics': self.get_metrics()
            }
            return status
        except Exception as e:
            self.logger.error(f"Erreur lors de la récupération du statut: {e}")
            raise
