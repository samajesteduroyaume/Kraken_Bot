"""
Module de gestion de portefeuille pour le bot de trading Kraken.
Gère les positions, les valeurs et les performances du portefeuille.
"""

from typing import Dict, Tuple
import logging
import pandas as pd

logger = logging.getLogger(__name__)


class PortfolioManager:
    """Gestionnaire de portefeuille pour le bot de trading."""

    def __init__(self, config: Dict):
        """Initialise le gestionnaire de portefeuille."""
        self.config = config
        # {pair: {size, entry_price, side, etc.}}
        self.positions: Dict[str, Dict] = {}
        self.balance: Dict[str, float] = {}   # {currency: amount}
        self.performance: Dict = {
            'total_return': 0.0,
            'daily_return': 0.0,
            'max_drawdown': 0.0,
            'current_drawdown': 0.0,
            'peak_value': 0.0
        }
        self.history = pd.DataFrame(
            columns=[
                'timestamp',
                'value',
                'return',
                'drawdown'])

    def update_balance(self, balances: Dict[str, float]) -> None:
        """
        Met à jour le solde du portefeuille.

        Args:
            balances: Dictionnaire des soldes par devise
        """
        self.balance = balances

    def get_total_value(self, current_prices: Dict[str, float]) -> float:
        """
        Calcule la valeur totale du portefeuille.

        Args:
            current_prices: Prix actuels des paires

        Returns:
            Valeur totale du portefeuille en USD
        """
        try:
            total_value = 0.0

            # Ajouter la valeur des positions
            for pair, position in self.positions.items():
                base_currency = pair[:4]  # Ex: XXBTZUSD -> XXBT
                quote_currency = pair[-4:]  # Ex: XXBTZUSD -> ZUSD

                if quote_currency == 'ZUSD':
                    total_value += position['size'] * current_prices[pair]
                else:
                    # Convertir en USD si nécessaire
                    usd_pair = f"{base_currency}ZUSD"
                    if usd_pair in current_prices:
                        total_value += position['size'] * \
                            current_prices[usd_pair]

            # Ajouter la valeur des soldes
            for currency, amount in self.balance.items():
                if currency == 'ZUSD':
                    total_value += amount
                else:
                    usd_pair = f"{currency}ZUSD"
                    if usd_pair in current_prices:
                        total_value += amount * current_prices[usd_pair]

            return total_value

        except Exception as e:
            logger.error(
                f"Erreur lors du calcul de la valeur totale: {str(e)}")
            raise

    def add_position(self,
                     pair: str,
                     size: float,
                     entry_price: float,
                     side: str) -> None:
        """
        Ajoute une nouvelle position.

        Args:
            pair: Paire de trading
            size: Taille de la position
            entry_price: Prix d'entrée
            side: 'long' ou 'short'
        """
        try:
            if pair not in self.positions:
                self.positions[pair] = {
                    'size': size,
                    'entry_price': entry_price,
                    'side': side,
                    'timestamp': pd.Timestamp.now()
                }
                logger.info(
                    f"Position ouverte: {pair} {side} {size} à {entry_price}")

        except Exception as e:
            logger.error(f"Erreur lors de l'ajout de position: {str(e)}")
            raise

    def close_position(self, pair: str) -> Tuple[float, float]:
        """
        Ferme une position.

        Args:
            pair: Paire de trading

        Returns:
            Tuple (profit/loss, pourcentage de rendement)
        """
        try:
            if pair not in self.positions:
                logger.error(f"Position {pair} non trouvée")
                return 0.0, 0.0

            position = self.positions[pair]
            entry_price = position['entry_price']
            size = position['size']
            side = position['side']

            # Calculer le P/L
            current_price = self.get_current_price(pair)
            if side == 'long':
                pl = (current_price - entry_price) * size
            else:
                pl = (entry_price - current_price) * size

            # Calculer le pourcentage de rendement
            percentage = (pl / (entry_price * size)) * 100

            # Supprimer la position
            del self.positions[pair]
            logger.info(
                f"Position fermée: {pair} {side} {size} à {current_price}")
            logger.info(f"Profit/Loss: {pl:.2f} USD ({percentage:.2f}%)")

            return pl, percentage

        except Exception as e:
            logger.error(f"Erreur lors de la fermeture de position: {str(e)}")
            raise

    def get_current_price(self, pair: str) -> float:
        """Récupère le prix actuel d'une paire."""
        # À implémenter avec l'API Kraken
        return 0.0

    def calculate_performance(self, current_prices: Dict[str, float]) -> None:
        """
        Calcule les métriques de performance du portefeuille.

        Args:
            current_prices: Prix actuels des paires
        """
        try:
            current_value = self.get_total_value(current_prices)

            # Calculer le rendement quotidien
            if len(self.history) > 0:
                last_value = self.history.iloc[-1]['value']
                daily_return = (current_value - last_value) / last_value
                self.performance['daily_return'] = daily_return

            # Calculer le drawdown
            peak = max(self.performance['peak_value'], current_value)
            drawdown = (peak - current_value) / peak

            self.performance['peak_value'] = peak
            self.performance['current_drawdown'] = drawdown
            self.performance['max_drawdown'] = max(
                self.performance['max_drawdown'], drawdown)

            # Ajouter à l'historique
            self.history = self.history.append({
                'timestamp': pd.Timestamp.now(),
                'value': current_value,
                'return': self.performance['daily_return'],
                'drawdown': self.performance['current_drawdown']
            }, ignore_index=True)

        except Exception as e:
            logger.error(f"Erreur lors du calcul de la performance: {str(e)}")
            raise
