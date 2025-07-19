"""
Gestionnaire de levier pour le trading.
"""

from typing import Dict, Any
import logging
from decimal import Decimal
from src.core.types import Position

logger = logging.getLogger(__name__)


class LeverageStrategy:
    """
    Stratégie de gestion de levier.
    """

    def __init__(self,
                 initial_leverage: Decimal = Decimal('1.0'),
                 max_leverage: Decimal = Decimal('10.0'),
                 adjustment_threshold: Decimal = Decimal('0.05')):
        """
        Initialise la stratégie de levier.

        Args:
            initial_leverage: Levier initial
            max_leverage: Levier maximum autorisé
            adjustment_threshold: Seuil de variation pour ajuster le levier
        """
        self.initial_leverage = initial_leverage
        self.max_leverage = max_leverage
        self.adjustment_threshold = adjustment_threshold
        self.current_leverage = initial_leverage

    def adjust_leverage(self, position: Position,
                        market_data: Dict[str, Any]) -> Decimal:
        """
        Ajuste le levier en fonction de la position et des données de marché.

        Args:
            position: Position actuelle
            market_data: Données de marché

        Returns:
            Decimal: Nouveau levier ajusté
        """
        volatility = Decimal(str(market_data.get('volatility', 0)))

        # Ajustement basique basé sur la volatilité
        if volatility > Decimal('0.02'):
            new_leverage = self.current_leverage * (1 - volatility)
        else:
            new_leverage = self.current_leverage

        # Appliquer les limites
        new_leverage = min(new_leverage, self.max_leverage)
        new_leverage = max(new_leverage, self.initial_leverage)

        # Appliquer le seuil d'ajustement
        if abs(new_leverage - self.current_leverage) > self.adjustment_threshold:
            self.current_leverage = new_leverage
            logger.info(f"Levier ajusté à: {self.current_leverage}")

        return self.current_leverage


class LeverageManager:
    """
    Gestionnaire de levier pour le trading.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialise le gestionnaire de levier.

        Args:
            config: Configuration de trading
        """
        self.config = config
        self.strategies = {}

    def get_strategy(self, symbol: str) -> LeverageStrategy:
        """
        Obtient ou crée une stratégie de levier pour un symbole.

        Args:
            symbol: Symbole de trading

        Returns:
            LeverageStrategy: Stratégie de levier
        """
        if symbol not in self.strategies:
            self.strategies[symbol] = LeverageStrategy(
                initial_leverage=Decimal('1.0'),
                max_leverage=self.config['max_leverage'],
                adjustment_threshold=Decimal('0.05')
            )
        return self.strategies[symbol]

    def adjust_leverage(self, position: Position,
                        market_data: Dict[str, Any]) -> Decimal:
        """
        Ajuste le levier pour une position donnée.

        Args:
            position: Position actuelle
            market_data: Données de marché

        Returns:
            Decimal: Levier ajusté
        """
        strategy = self.get_strategy(position['symbol'])
        return strategy.adjust_leverage(position, market_data)

    def get_current_leverage(self) -> float:
        """Retourne le levier courant moyen sur toutes les stratégies."""
        if not self.strategies:
            return float(self.config.get('max_leverage', 1.0))
        return float(sum(s.current_leverage for s in self.strategies.values()) / len(self.strategies))

    def get_leverage_history(self) -> list:
        """Retourne l'historique des leviers (ici, juste les leviers courants pour chaque stratégie)."""
        return [float(s.current_leverage) for s in self.strategies.values()]

    def update_leverage_based_on_market_conditions(self, market_volatility: float, signal_confidence: float, recent_performance: float, max_drawdown: float, position_size: float) -> float:
        """Met à jour le levier en fonction des conditions de marché réelles."""
        # Exemple : ajustement simple basé sur la volatilité et la confiance
        base_leverage = float(self.config.get('max_leverage', 1.0))
        if market_volatility > 0.05:
            base_leverage *= 0.5
        if signal_confidence < 0.5:
            base_leverage *= 0.7
        if max_drawdown > 0.1:
            base_leverage *= 0.5
        return max(1.0, min(base_leverage, float(self.config.get('max_leverage', 1.0))))
