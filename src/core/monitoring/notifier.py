import logging
from typing import Dict, List, Optional
from datetime import datetime
from datetime import timedelta
from ..types.market_data import MarketData
from ..types.types import TradeSignal, Position
from ..types.trading import TradingMetrics


class Notifier:
    """
    Système de notification pour le trading.

    Cette classe gère :
    - Les alertes de trading
    - Les notifications de risque
    - Les rapports de performance
    - Les alertes techniques
    """

    def __init__(self,
                 config: Optional[Dict] = None):
        """
        Initialise le système de notification.

        Args:
            config: Configuration des notifications
        """
        self.config = config or {}
        self.logger = logging.getLogger(__name__)

        # Configuration des notifications
        self.alert_thresholds = {
            'price_change': self.config.get('price_change_threshold', 0.05),
            'volume_change': self.config.get('volume_change_threshold', 0.5),
            'risk_level': self.config.get('risk_level_threshold', 0.8),
            'position_size': self.config.get('position_size_threshold', 0.1)
        }

        # Historique des notifications
        self.notification_history: List[Dict] = []

    def send_alert(self,
                   alert_type: str,
                   message: str,
                   data: Optional[Dict] = None) -> None:
        """
        Envoie une alerte.

        Args:
            alert_type: Type d'alerte
            message: Message de l'alerte
            data: Données supplémentaires
        """
        notification = {
            'type': alert_type,
            'message': message,
            'data': data or {},
            'timestamp': datetime.now()
        }

        # Ajouter à l'historique
        self.notification_history.append(notification)

        # Enregistrer dans les logs
        self.logger.info(f"Alerte {alert_type}: {message}")

        # Envoyer la notification selon la configuration
        self._send_notification(notification)

    def _send_notification(self, notification: Dict) -> None:
        """Envoie la notification selon la configuration."""
        # À implémenter selon les canaux de notification configurés

    def check_market_alerts(self, market_data: MarketData) -> None:
        """
        Vérifie les alertes de marché.

        Args:
            market_data: Données de marché
        """
        try:
            # Vérifier la volatilité
            if self._check_volatility(market_data):
                self.send_alert(
                    'volatility',
                    f"Volatilité élevée détectée pour {market_data['symbol']}",
                    {'data': market_data}
                )

            # Vérifier le volume
            if self._check_volume(market_data):
                self.send_alert(
                    'volume',
                    f"Volume anormal détecté pour {market_data['symbol']}",
                    {'data': market_data}
                )

            # Vérifier les patterns
            if self._check_patterns(market_data):
                self.send_alert(
                    'pattern',
                    f"Pattern technique détecté pour {market_data['symbol']}",
                    {'data': market_data}
                )

        except Exception as e:
            self.logger.error(
                f"Erreur lors de la vérification des alertes de marché: {e}")

    def check_risk_alerts(self, metrics: TradingMetrics) -> None:
        """
        Vérifie les alertes de risque.

        Args:
            metrics: Métriques de trading
        """
        try:
            # Vérifier le drawdown
            if metrics['max_drawdown'] > self.alert_thresholds['risk_level']:
                self.send_alert(
                    'risk',
                    f"Drawdown maximum atteint: {metrics['max_drawdown']}",
                    {'metrics': metrics}
                )

            # Vérifier le ratio risque/reward
            if metrics['profit_factor'] < 1.0:
                self.send_alert(
                    'risk',
                    f"Ratio risque/reward négatif: {metrics['profit_factor']}",
                    {'metrics': metrics}
                )

        except Exception as e:
            self.logger.error(
                f"Erreur lors de la vérification des alertes de risque: {e}")

    def check_trade_alerts(self, signal: TradeSignal) -> None:
        """
        Vérifie les alertes de trading.

        Args:
            signal: Signal de trading
        """
        try:
            # Vérifier la confiance
            if signal['confidence'] < 0.5:
                self.send_alert(
                    'trade', f"Signal faible pour {signal['symbol']}: {signal['confidence']}", {
                        'signal': signal})

            # Vérifier le risque
            if signal.get(
                'risk_percentage',
                    0) > self.alert_thresholds['risk_level']:
                self.send_alert(
                    'risk',
                    f"Risque élevé pour {signal['symbol']}: {signal['risk_percentage']}%",
                    {'signal': signal}
                )

        except Exception as e:
            self.logger.error(
                f"Erreur lors de la vérification des alertes de trading: {e}")

    def check_position_alerts(self, position: Position) -> None:
        """
        Vérifie les alertes de position.

        Args:
            position: Position en cours
        """
        try:
            # Vérifier la taille de position
            if position['amount'] > self.alert_thresholds['position_size']:
                self.send_alert(
                    'position',
                    f"Position trop grande pour {position['symbol']}: {position['amount']}",
                    {'position': position}
                )

            # Vérifier la durée de position
            if self._check_position_duration(position):
                self.send_alert(
                    'position',
                    f"Position longue durée pour {position['symbol']}",
                    {'position': position}
                )

        except Exception as e:
            self.logger.error(
                f"Erreur lors de la vérification des alertes de position: {e}")

    def _check_volatility(self, market_data: MarketData) -> bool:
        """Vérifie si la volatilité est élevée."""
        if not market_data.get('analysis'):
            return False

        volatility = market_data['analysis'].get('volatility', 0)
        return volatility > self.alert_thresholds['price_change']

    def _check_volume(self, market_data: MarketData) -> bool:
        """Vérifie si le volume est anormal."""
        if not market_data.get('analysis'):
            return False

        volume = market_data['analysis'].get('trade_volume', 0)
        return volume > self.alert_thresholds['volume_change']

    def _check_patterns(self, market_data: MarketData) -> bool:
        """Vérifie si des patterns techniques sont détectés."""
        if not market_data.get('analysis'):
            return False

        patterns = market_data['analysis'].get('patterns', [])
        return any(patterns)

    def _check_position_duration(self, position: Position) -> bool:
        """Vérifie si la position est trop longue."""
        if not position.get('timestamp'):
            return False

        duration = datetime.now() - position['timestamp']
        max_duration = timedelta(
            hours=self.config.get(
                'max_position_duration', 24))
        return duration > max_duration

    def generate_report(self, metrics: TradingMetrics) -> str:
        """
        Génère un rapport de performance.

        Args:
            metrics: Métriques de trading

        Returns:
            Rapport formaté
        """
        report = f"""Rapport de Performance - {datetime.now()}

Métriques de Performance:
- ROI: {metrics['roi']}%
- Sharpe Ratio: {metrics['sharpe_ratio']}
- Sortino Ratio: {metrics['sortino_ratio']}
- Calmar Ratio: {metrics['calmar_ratio']}
- Profit Factor: {metrics['profit_factor']}

Statistiques de Trading:
- Nombre de trades: {metrics['total_trades']}
- Taux de réussite: {metrics['win_rate']}%
- Gain moyen: {metrics['avg_profit']}
- Perte moyenne: {metrics['avg_loss']}
- Drawdown maximum: {metrics['max_drawdown']}%

Alertes:
- Nombre d'alertes: {len(self.notification_history)}
- Alertes critiques: {sum(1 for n in self.notification_history
                         if n['type'] in ['risk', 'position'])}
"""
        return report

    def save_report(self, report: str, filename: Optional[str] = None) -> None:
        """
        Sauvegarde un rapport.

        Args:
            report: Contenu du rapport
            filename: Nom du fichier (optionnel)
        """
        try:
            if not filename:
                filename = f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"

            with open(filename, 'w') as f:
                f.write(report)

        except Exception as e:
            self.logger.error(f"Erreur lors de la sauvegarde du rapport: {e}")

    def get_notification_history(
            self,
            limit: Optional[int] = None,
            alert_type: Optional[str] = None) -> List[Dict]:
        """
        Récupère l'historique des notifications.

        Args:
            limit: Nombre maximum de notifications
            alert_type: Type d'alerte à filtrer

        Returns:
            Liste des notifications filtrées
        """
        history = self.notification_history

        if alert_type:
            history = [n for n in history if n['type'] == alert_type]

        if limit:
            history = history[-limit:]

        return history
