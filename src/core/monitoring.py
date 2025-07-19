"""
Module de surveillance pour le bot de trading Kraken.
Gère la surveillance des performances, la détection d'erreurs et les alertes.
"""

from typing import Dict
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class MonitoringManager:
    """Gestionnaire de surveillance pour le bot de trading."""

    def __init__(self, config: Dict):
        """Initialise le gestionnaire de surveillance."""
        self.config = config
        self.alert_thresholds = config.get('alert_thresholds', {
            'max_drawdown': 10.0,  # Pourcentage
            'max_position_size': 0.1,  # Pourcentage du portefeuille
            'max_trade_frequency': 10,  # Trades par heure
            'min_balance': 100.0  # USD
        })
        self.last_alerts = {}
        self.trade_count = 0
        self.last_trade_time = datetime.now()

    async def check_performance(self, portfolio: Dict) -> None:
        """
        Vérifie les performances et déclenche les alertes si nécessaire.

        Args:
            portfolio: Dictionnaire du portefeuille
        """
        try:
            # Vérifier le drawdown
            current_drawdown = portfolio.get('current_drawdown', 0)
            if current_drawdown > self.alert_thresholds['max_drawdown']:
                await self.send_alert(
                    'drawdown',
                    f"Drawdown élevé: {current_drawdown:.2f}%"
                )

            # Vérifier le solde minimum
            balance = portfolio.get('balance', {}).get('USD', 0)
            if balance < self.alert_thresholds['min_balance']:
                await self.send_alert(
                    'balance',
                    f"Solde bas: {balance:.2f} USD"
                )

        except Exception as e:
            logger.error(
                f"Erreur lors de la vérification des performances: {str(e)}")

    async def check_trade_frequency(self) -> None:
        """
        Vérifie la fréquence des trades.
        """
        try:
            current_time = datetime.now()
            time_diff = current_time - self.last_trade_time

            if time_diff.total_seconds() < 3600:  # 1 heure
                self.trade_count += 1

                if self.trade_count > self.alert_thresholds['max_trade_frequency']:
                    await self.send_alert(
                        'trade_frequency',
                        f"Fréquence de trades élevée: {self.trade_count} trades en {time_diff.total_seconds()} secondes"
                    )

            else:
                self.trade_count = 0
                self.last_trade_time = current_time

        except Exception as e:
            logger.error(
                f"Erreur lors de la vérification de la fréquence des trades: {str(e)}")

    async def send_alert(self, alert_type: str, message: str) -> None:
        """
        Envoie une alerte.

        Args:
            alert_type: Type d'alerte
            message: Message d'alerte
        """
        try:
            # Vérifier si une alerte similaire a déjà été envoyée
            last_alert = self.last_alerts.get(alert_type)
            if last_alert and (datetime.now() -
                               last_alert).total_seconds() < 3600:  # 1 heure
                return

            # Enregistrer le timestamp de l'alerte
            self.last_alerts[alert_type] = datetime.now()

            # Enregistrer dans les logs
            logger.warning(f"ALERT: {message}")

            # Envoyer une notification (à implémenter)
            # notification_manager.send_notification(message)

        except Exception as e:
            logger.error(f"Erreur lors de l'envoi de l'alerte: {str(e)}")
