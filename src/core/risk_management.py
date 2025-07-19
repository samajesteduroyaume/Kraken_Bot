"""
Module de gestion des risques pour le bot de trading Kraken.
Gère la gestion des risques, les seuils de stop-loss et take-profit,
et la taille des positions.
"""

from typing import Dict
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class RiskParameters:
    """Paramètres de gestion des risques."""
    max_position_size: float = 0.1  # Pourcentage du portefeuille maximum par trade
    stop_loss_percentage: float = 2.0  # Pourcentage de stop-loss
    take_profit_percentage: float = 4.0  # Pourcentage de take-profit
    max_drawdown: float = 10.0  # Pourcentage de drawdown maximum
    max_leverage: float = 5.0  # Levier maximum
    trailing_stop: bool = True  # Utiliser un stop-loss suiveur

    def validate(self) -> bool:
        """Valide les paramètres de risque."""
        if not (0 < self.max_position_size <= 1):
            logger.error("max_position_size doit être entre 0 et 1")
            return False
        if not (0 < self.stop_loss_percentage <= 100):
            logger.error("stop_loss_percentage doit être entre 0 et 100")
            return False
        if not (0 < self.take_profit_percentage <= 100):
            logger.error("take_profit_percentage doit être entre 0 et 100")
            return False
        if not (0 < self.max_drawdown <= 100):
            logger.error("max_drawdown doit être entre 0 et 100")
            return False
        if not (1 <= self.max_leverage <= 10):
            logger.error("max_leverage doit être entre 1 et 10")
            return False
        return True


class RiskManager:
    """Gestionnaire de risques pour le bot de trading."""

    def __init__(self, config: Dict):
        """Initialise le gestionnaire de risques."""
        self.config = config
        self.risk_params = RiskParameters(**config.get('risk_parameters', {}))
        self.current_positions: Dict[str, Dict] = {}
        self.portfolio_value = 0

    def calculate_position_size(self,
                                pair: str,
                                current_price: float,
                                portfolio_value: float) -> float:
        """
        Calcule la taille de la position en fonction du risque.

        Args:
            pair: Paire de trading
            current_price: Prix actuel
            portfolio_value: Valeur du portefeuille

        Returns:
            Taille de la position en unités
        """
        try:
            # Calculer la taille maximale autorisée
            max_position = portfolio_value * self.risk_params.max_position_size

            # Convertir en unités
            position_size = max_position / current_price

            # Appliquer le levier
            position_size *= self.risk_params.max_leverage

            return position_size

        except Exception as e:
            logger.error(
                f"Erreur lors du calcul de la taille de position: {str(e)}")
            raise

    def calculate_stop_loss(self,
                            entry_price: float,
                            is_long: bool) -> float:
        """
        Calcule le niveau de stop-loss.

        Args:
            entry_price: Prix d'entrée
            is_long: Position longue (True) ou courte (False)

        Returns:
            Prix de stop-loss
        """
        try:
            percentage = self.risk_params.stop_loss_percentage / 100
            if is_long:
                return entry_price * (1 - percentage)
            else:
                return entry_price * (1 + percentage)

        except Exception as e:
            logger.error(f"Erreur lors du calcul du stop-loss: {str(e)}")
            raise

    def calculate_take_profit(self,
                              entry_price: float,
                              is_long: bool) -> float:
        """
        Calcule le niveau de take-profit.

        Args:
            entry_price: Prix d'entrée
            is_long: Position longue (True) ou courte (False)

        Returns:
            Prix de take-profit
        """
        try:
            percentage = self.risk_params.take_profit_percentage / 100
            if is_long:
                return entry_price * (1 + percentage)
            else:
                return entry_price * (1 - percentage)

        except Exception as e:
            logger.error(f"Erreur lors du calcul du take-profit: {str(e)}")
            raise

    def validate_trade(self,
                       pair: str,
                       position_size: float,
                       entry_price: float,
                       is_long: bool) -> bool:
        """
        Valide une nouvelle position.

        Args:
            pair: Paire de trading
            position_size: Taille de la position
            entry_price: Prix d'entrée
            is_long: Position longue (True) ou courte (False)

        Returns:
            True si la trade est valide, False sinon
        """
        try:
            if position_size <= 0:
                logger.error("Taille de position invalide: doit être positive")
                return False

            # Vérifier la taille de position
            if position_size > self.risk_params.max_position_size:
                logger.error(
                    f"Taille de position trop grande: {position_size} > {self.risk_params.max_position_size}")
                return False

            # Vérifier le risque par trade
            stop_loss_percentage = self.risk_params.stop_loss_percentage
            risk_per_trade = (
                entry_price *
                position_size *
                stop_loss_percentage /
                100)

            # Calculer le risque maximal autorisé
            portfolio_value = self.config.portfolio_value
            max_risk = portfolio_value * \
                self.config.trading_config['risk_per_trade']

            logger.debug(f"Paramètres de validation:")
            logger.debug(f"  Position size: {position_size}")
            logger.debug(f"  Entry price: {entry_price}")
            logger.debug(f"  Stop loss %: {stop_loss_percentage}")
            logger.debug(f"  Portfolio value: {portfolio_value}")
            logger.debug(f"  Risk per trade: {risk_per_trade}")
            logger.debug(f"  Max risk: {max_risk}")

            if risk_per_trade > max_risk:
                logger.error(
                    f"Risque par trade trop élevé: {risk_per_trade} > {max_risk}")
                return False

            return True

        except Exception as e:
            logger.error(f"Erreur lors de la validation de la trade: {str(e)}")
            raise
