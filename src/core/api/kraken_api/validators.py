"""
Module de validation des données pour l'API Kraken.
"""

from typing import Dict, Any, Optional
from datetime import datetime
import logging


class Validator:
    def __init__(self):
        self.logger = logging.getLogger(__name__ + '.Validator')

    def validate_pair(self, pair: str) -> None:
        """Valide une paire de trading."""
        if not isinstance(pair, str):
            self.logger.error(
                f"Type invalide pour pair: {type(pair).__name__}")
            raise ValueError("La paire doit être une chaîne")

        if not pair.strip():
            self.logger.error(
                "La paire ne peut pas être vide ou contenir uniquement des espaces")
            raise ValueError(
                "La paire ne peut pas être vide ou contenir uniquement des espaces")

        if '-' not in pair or len(pair) > 20:
            self.logger.error(f"Format invalide pour la paire: {pair}")
            raise ValueError(
                "La paire doit contenir un tiret et avoir une longueur maximale de 20 caractères")

    def validate_timestamp(self, timestamp: Optional[int]) -> None:
        """Valide un timestamp."""
        if timestamp is not None:
            if not isinstance(timestamp, int):
                self.logger.error(
                    f"Type invalide pour timestamp: {type(timestamp).__name__}")
                raise ValueError("Le timestamp doit être un entier")

            if timestamp < 0:
                self.logger.error(f"Timestamp négatif: {timestamp}")
                raise ValueError("Le timestamp ne peut pas être négatif")

            if timestamp > int(datetime.now().timestamp()):
                self.logger.error(f"Timestamp dans le futur: {timestamp}")
                raise ValueError("Le timestamp ne peut pas être dans le futur")

    def validate_interval(self, interval: int) -> None:
        """Valide un intervalle de temps."""
        allowed_intervals = [1, 5, 15, 30, 60, 240, 1440, 10080, 21600]
        if interval not in allowed_intervals:
            self.logger.error(f"Intervalle invalide: {interval}")
            raise ValueError(
                f"L'intervalle doit être l'un des suivants: {allowed_intervals}")

    def validate_order_params(self, params: Dict[str, Any]) -> None:
        """Valide les paramètres d'un ordre."""
        required_fields = ['pair', 'type', 'ordertype', 'volume']

        for field in required_fields:
            if field not in params:
                self.logger.error(
                    f"Champ manquant dans les paramètres d'ordre: {field}")
                raise ValueError(f"Le champ {field} est requis")

        # Validation spécifique pour chaque champ
        self.validate_pair(params['pair'])

        if not isinstance(params['volume'], (int, float)
                          ) or params['volume'] <= 0:
            self.logger.error(f"Volume invalide: {params['volume']}")
            raise ValueError("Le volume doit être un nombre positif")

    def validate_txid(self, txid: str) -> None:
        """Valide un ID de transaction."""
        if not isinstance(txid, str):
            self.logger.error(
                f"Type invalide pour txid: {type(txid).__name__}")
            raise ValueError("Le txid doit être une chaîne")

        if not txid.strip():
            self.logger.error(
                "Le txid ne peut pas être vide ou contenir uniquement des espaces")
            raise ValueError(
                "Le txid ne peut pas être vide ou contenir uniquement des espaces")

        if len(txid) < 10 or len(txid) > 50:
            self.logger.error(f"Longueur invalide pour txid: {len(txid)}")
            raise ValueError("Le txid doit avoir entre 10 et 50 caractères")


class KrakenValidator(Validator):
    """Validateur spécifique Kraken héritant du validateur générique."""
    pass
