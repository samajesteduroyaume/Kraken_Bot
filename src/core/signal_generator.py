from typing import Dict, Any
import pandas as pd
from .technical_analyzer import TechnicalAnalyzer
from .sentiment_analyzer import SentimentAnalyzer
from .ml_predictor import MLPredictor
import logging


class SignalGenerator:
    """Génération de signaux de trading basée sur différents indicateurs."""

    def __init__(self):
        """Initialise le générateur de signaux."""
        self.technical_analyzer = TechnicalAnalyzer()
        self.sentiment_analyzer = SentimentAnalyzer()
        self.ml_predictor = MLPredictor()
        self.logger = logging.getLogger('signal_generator')

    def generate_signals(self, price_history: pd.Series) -> Dict[str, Any]:
        """
        Génère les signaux de trading.

        Args:
            price_history: Historique des prix

        Returns:
            Dictionnaire de signaux
        """
        try:
            signals = {}

            # Analyse technique
            indicators = self.technical_analyzer.calculate_indicators(
                price_history)
            signals['technical'] = {
                'trend': self.technical_analyzer.analyze_trend(price_history),
                'volatility': self.technical_analyzer.analyze_volatility(price_history),
                'rsi': indicators['rsi'].iloc[-1],
                'macd': indicators['macd'].iloc[-1]
            }

            # Analyse du sentiment
            signals['sentiment'] = {
                'price': self.sentiment_analyzer.get_price_sentiment(price_history),
                'market': 'neutral'  # À implémenter avec des données de news
            }

            # Prédiction ML
            self.ml_predictor.train(price_history)
            signals['ml'] = {
                'probability': self.ml_predictor.predict(price_history)
            }

            # Combinaison des signaux
            signals['combined'] = self.combine_signals(signals)

            return signals

        except Exception as e:
            self.logger.error(
                f"Erreur lors de la génération des signaux: {str(e)}")
            return {}

    def combine_signals(self, signals: Dict[str, Any]) -> Dict[str, float]:
        """
        Combine les différents signaux en un seul signal final.

        Args:
            signals: Dictionnaire des signaux

        Returns:
            Dictionnaire avec le signal combiné
        """
        try:
            # Score technique
            technical_score = 0.0
            if signals['technical']['trend'] == 'bullish':
                technical_score += 0.4
            elif signals['technical']['trend'] == 'bearish':
                technical_score -= 0.4

            # Score sentiment
            sentiment_score = 0.0
            if signals['sentiment']['price'] == 'bullish':
                sentiment_score += 0.3
            elif signals['sentiment']['price'] == 'bearish':
                sentiment_score -= 0.3

            # Score ML
            ml_score = signals['ml']['probability'] * 0.3

            # Score final
            final_score = technical_score + sentiment_score + ml_score

            # Normalisation
            normalized_score = (final_score + 1) / 2

            return {
                'score': normalized_score,
                'strength': abs(final_score),
                'direction': 'long' if final_score > 0 else 'short'
            }

        except Exception as e:
            self.logger.error(
                f"Erreur lors de la combinaison des signaux: {str(e)}")
            return {'score': 0.5, 'strength': 0.0, 'direction': 'neutral'}
