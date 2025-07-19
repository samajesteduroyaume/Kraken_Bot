from typing import Dict
import pandas as pd
from ta import momentum, volatility, trend, volume
import logging


class TechnicalAnalyzer:
    """Analyse technique des données de marché."""

    def __init__(self):
        """Initialise l'analyseur technique."""
        self.logger = logging.getLogger('technical_analyzer')

    def calculate_indicators(
            self, price_history: pd.Series) -> Dict[str, pd.Series]:
        """
        Calcule les indicateurs techniques.

        Args:
            price_history: Historique des prix

        Returns:
            Dictionnaire d'indicateurs
        """
        try:
            df = pd.DataFrame()
            df['price'] = price_history

            # Indicateurs de momentum
            df['rsi'] = momentum.RSIIndicator(df['price'], window=14).rsi()
            df['stoch'] = momentum.StochasticOscillator(
                df['price'], df['price'], df['price'], window=14, smooth_window=3).stoch()

            # Indicateurs de volatilité
            df['bb_upper'] = volatility.BollingerBands(
                df['price'], window=20, window_dev=2
            ).bollinger_hband()
            df['bb_lower'] = volatility.BollingerBands(
                df['price'], window=20, window_dev=2
            ).bollinger_lband()
            df['atr'] = volatility.AverageTrueRange(
                df['price'], df['price'], df['price'], window=14
            ).average_true_range()

            # Indicateurs de tendance
            df['ema_short'] = trend.EMAIndicator(
                df['price'], window=12).ema_indicator()
            df['ema_long'] = trend.EMAIndicator(
                df['price'], window=26).ema_indicator()
            df['macd'] = trend.MACD(df['price']).macd()
            df['macd_signal'] = trend.MACD(df['price']).macd_signal()

            # Indicateurs de volume
            df['obv'] = volume.OnBalanceVolumeIndicator(
                df['price'], df['price']
            ).on_balance_volume()

            # Nettoyer les NaN
            indicators = df.dropna().to_dict('series')

            return indicators

        except Exception as e:
            self.logger.error(
                f"Erreur lors du calcul des indicateurs: {str(e)}")
            return {}

    def analyze_trend(self, price_history: pd.Series) -> str:
        """
        Analyse la tendance du marché.

        Args:
            price_history: Historique des prix

        Returns:
            'bullish', 'bearish' ou 'neutral'
        """
        try:
            # Calculer les EMAs
            ema_short = trend.EMAIndicator(
                price_history, window=12).ema_indicator()
            ema_long = trend.EMAIndicator(
                price_history, window=26).ema_indicator()

            # Analyser la tendance
            if ema_short.iloc[-1] > ema_long.iloc[-1]:
                return 'bullish'
            elif ema_short.iloc[-1] < ema_long.iloc[-1]:
                return 'bearish'
            else:
                return 'neutral'

        except Exception as e:
            self.logger.error(
                f"Erreur lors de l'analyse de la tendance: {str(e)}")
            return 'neutral'

    def analyze_volatility(self, price_history: pd.Series) -> float:
        """
        Analyse la volatilité du marché.

        Args:
            price_history: Historique des prix

        Returns:
            Score de volatilité (0-1)
        """
        try:
            # Calculer le RSI
            rsi = momentum.RSIIndicator(price_history, window=14).rsi()

            # Calculer la volatilité
            volatility = price_history.pct_change().rolling(window=20).std()

            # Normaliser la volatilité
            normalized_volatility = volatility.iloc[-1] / volatility.mean()

            return float(normalized_volatility)

        except Exception as e:
            self.logger.error(
                f"Erreur lors de l'analyse de la volatilité: {str(e)}")
            return 0.0
