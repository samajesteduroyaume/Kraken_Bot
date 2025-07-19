from typing import Dict, Optional
from datetime import datetime, timedelta
from src.core.logging.logging import LoggerManager


class RiskManager:
    """Gère le risque global et par paire."""

    def __init__(
        self,
        max_global_risk: float = 0.05,  # 5% du capital
        max_pair_risk: float = 0.01,    # 1% du capital par paire
        stop_loss_threshold: float = 0.02,  # 2% de stop loss
        take_profit_threshold: float = 0.03,  # 3% de take profit
        risk_adjustment_frequency: timedelta = timedelta(hours=1),
        logger: Optional[LoggerManager] = None
    ):
        self.max_global_risk = max_global_risk
        self.max_pair_risk = max_pair_risk
        self.stop_loss_threshold = stop_loss_threshold
        self.take_profit_threshold = take_profit_threshold
        self.risk_adjustment_frequency = risk_adjustment_frequency
        self.last_adjustment = datetime.min
        self.logger = logger or LoggerManager()
        self.logger = self.logger.get_logger('risk_manager')
        self.pair_risks: Dict[str, float] = {}
        self.global_risk = 0.0

    def calculate_risk_allocation(
        self,
        pair_metrics: Dict[str, Dict],
        account_balance: float,
        market_trend: str
    ) -> Dict[str, float]:
        """Calcule l'allocation du risque pour chaque paire."""
        try:
            # Calculer le risque global maximum
            global_risk_limit = account_balance * self.max_global_risk

            # Initialiser les allocations
            allocations = {}
            total_risk = 0.0

            # Calculer le score de risque pour chaque paire
            for pair, metrics in pair_metrics.items():
                base_risk = self.max_pair_risk * account_balance

                # Ajuster selon la tendance du marché
                trend_adjustment = 1.0
                if market_trend == 'bullish':
                    trend_adjustment = 1.2  # 20% plus de risque
                elif market_trend == 'bearish':
                    trend_adjustment = 0.8  # 20% moins de risque

                # Ajuster selon la volatilité
                volatility_adjustment = 1.0 - metrics.get('volatility', 0)

                # Calculer le risque ajusté
                adjusted_risk = base_risk * trend_adjustment * volatility_adjustment

                # Appliquer les limites
                adjusted_risk = min(adjusted_risk, global_risk_limit)

                allocations[pair] = adjusted_risk
                total_risk += adjusted_risk

            # Normaliser les allocations
            if total_risk > 0:
                for pair in allocations:
                    allocations[pair] = (
                        allocations[pair] / total_risk) * global_risk_limit

            self.logger.info(
                f"Allocation du risque calculée. Total: {total_risk:.2f} USD")
            return allocations

        except Exception as e:
            self.logger.error(
                f"Erreur lors du calcul de l'allocation du risque: {str(e)}")
            return {pair: self.max_pair_risk *
                    account_balance for pair in pair_metrics.keys()}

    def adjust_position_size(
        self,
        current_price: float,
        pair_risk: float,
        market_trend: str,
        pair_metrics: Dict[str, Dict]
    ) -> float:
        """Ajuste la taille de position selon le risque et la tendance."""
        try:
            # Calculer la taille de position de base
            base_size = pair_risk / current_price

            # Ajuster selon la tendance
            trend_adjustment = 1.0
            if market_trend == 'bullish':
                trend_adjustment = 1.2
            elif market_trend == 'bearish':
                trend_adjustment = 0.8

            # Ajuster selon la volatilité
            volatility_adjustment = 1.0 - pair_metrics.get('volatility', 0)

            # Calculer la taille ajustée
            adjusted_size = base_size * trend_adjustment * volatility_adjustment

            self.logger.info(
                f"Taille ajustée: {adjusted_size:.2f} (base: {base_size:.2f}, "
                f"tendance: {market_trend}, ajustement: {trend_adjustment:.2f}, "
                f"volatilité: {volatility_adjustment:.2f})")

            return adjusted_size

        except Exception as e:
            self.logger.error(
                f"Erreur lors de l'ajustement de la taille de position: {str(e)}")
            return pair_risk / current_price

    def calculate_stop_loss(
        self,
        entry_price: float,
        pair_metrics: Dict[str, Dict]
    ) -> float:
        """Calcule le niveau de stop loss."""
        try:
            # Ajuster le seuil de stop loss selon la volatilité
            volatility = pair_metrics.get('volatility', 0)
            adjusted_threshold = self.stop_loss_threshold * (1 + volatility)

            # Calculer le stop loss
            stop_loss = entry_price * (1 - adjusted_threshold)

            self.logger.info(
                f"Stop loss calculé: {stop_loss:.2f} (entrée: {entry_price:.2f}, "
                f"volatilité: {volatility:.2f}, seuil: {adjusted_threshold:.2%})")

            return stop_loss

        except Exception as e:
            self.logger.error(f"Erreur lors du calcul du stop loss: {str(e)}")
            return entry_price * (1 - self.stop_loss_threshold)

    def calculate_take_profit(
        self,
        entry_price: float,
        pair_metrics: Dict[str, Dict]
    ) -> float:
        """Calcule le niveau de take profit."""
        try:
            # Ajuster le seuil de take profit selon la tendance
            momentum = pair_metrics.get('momentum', 0)
            adjusted_threshold = self.take_profit_threshold * (1 + momentum)

            # Calculer le take profit
            take_profit = entry_price * (1 + adjusted_threshold)

            self.logger.info(
                f"Take profit calculé: {take_profit:.2f} (entrée: {entry_price:.2f}, "
                f"momentum: {momentum:.2f}, seuil: {adjusted_threshold:.2%})")

            return take_profit

        except Exception as e:
            self.logger.error(
                f"Erreur lors du calcul du take profit: {str(e)}")
            return entry_price * (1 + self.take_profit_threshold)

    def should_reduce_position(
        self,
        current_price: float,
        entry_price: float,
        pair_metrics: Dict[str, Dict]
    ) -> bool:
        """Décide si une position doit être réduite."""
        try:
            # Calculer le drawdown
            drawdown = (entry_price - current_price) / entry_price

            # Ajuster le seuil de réduction selon la volatilité
            volatility = pair_metrics.get('volatility', 0)
            reduction_threshold = self.stop_loss_threshold * (1 + volatility)

            # Décider de la réduction
            should_reduce = drawdown > reduction_threshold

            self.logger.info(
                f"Réduction de position: {should_reduce} (drawdown: {drawdown:.2%}, "
                f"seuil: {reduction_threshold:.2%})")

            return should_reduce

        except Exception as e:
            self.logger.error(
                f"Erreur lors de la décision de réduction: {str(e)}")
            return False
