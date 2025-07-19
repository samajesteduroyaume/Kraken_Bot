from abc import ABC, abstractmethod
from typing import List
from src.core.types import Portfolio
from typing import TYPE_CHECKING
from src.core.types.types import TradingConfig

# Utilisation d'annotations forward pour MarketData
MarketData = 'MarketData'

if TYPE_CHECKING:
    from src.core.types import TradeSignal, Position, Order, Trade, TradingMetrics, TradingContext


class ITrader(ABC):
    """Interface pour les traders.

    Cette interface définit les méthodes communes que tous les traders doivent implémenter.
    Elle garantit une interface cohérente pour l'interaction avec le système de trading.
    """

    @abstractmethod
    def initialize(self, config: 'TradingConfig') -> None:
        """Initialise le trader avec la configuration donnée.

        Args:
            config: Configuration de trading
        """

    @abstractmethod
    async def update_market_data(self, market_data: MarketData) -> None:
        """Met à jour les données de marché.

        Args:
            market_data: Données de marché pour la paire
        """

    @abstractmethod
    async def update_balance(self, portfolio: Portfolio) -> None:
        """Met à jour le portefeuille.

        Args:
            portfolio: Portefeuille mis à jour
        """

    @abstractmethod
    async def generate_signals(self) -> List['TradeSignal']:
        """Génère les signaux de trading.

        Returns:
            Liste des signaux de trading générés
        """

    @abstractmethod
    async def execute_trades(
            self,
            signals: List['TradeSignal']) -> List['Trade']:
        """Exécute les trades basés sur les signaux.

        Args:
            signals: Liste des signaux de trading

        Returns:
            Liste des trades exécutés
        """

    @abstractmethod
    async def manage_positions(self) -> List['Order']:
        """Gère les positions ouvertes.

        Returns:
            Liste des ordres de gestion des positions
        """

    @abstractmethod
    def get_trading_status(self) -> 'TradingContext':
        """Retourne l'état actuel du trading.

        Returns:
            Contexte de trading complet
        """

    @abstractmethod
    async def stop(self) -> None:
        """Arrête le trader proprement.

        Cette méthode doit :
        - Fermer toutes les positions ouvertes
        - Annuler tous les ordres en attente
        - Nettoyer les resources
        """

    @abstractmethod
    async def get_metrics(self) -> 'TradingMetrics':
        """Retourne les métriques de performance.

        Returns:
            Métriques de performance du trading
        """

    @abstractmethod
    async def get_positions(self) -> List['Position']:
        """Retourne la liste des positions actuelles.

        Returns:
            Liste des positions ouvertes
        """

    @abstractmethod
    async def get_orders(self) -> List['Order']:
        """Retourne la liste des ordres en cours.

        Returns:
            Liste des ordres en attente
        """

    @abstractmethod
    async def get_trades(self) -> List['Trade']:
        """Retourne l'historique des trades.

        Returns:
            Liste des trades exécutés
        """
