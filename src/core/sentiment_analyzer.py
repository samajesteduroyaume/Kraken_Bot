from typing import Dict, Any, List
import pandas as pd
import numpy as np
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import logging


class SentimentAnalyzer:
    """Analyse du sentiment du marché."""

    def __init__(self):
        """Initialise l'analyseur de sentiment."""
        self.analyzer = SentimentIntensityAnalyzer()
        self.logger = logging.getLogger('sentiment_analyzer')

    def analyze_sentiment(self, text: str) -> Dict[str, float]:
        """
        Analyse le sentiment d'un texte.

        Args:
            text: Texte à analyser

        Returns:
            Dictionnaire avec les scores de sentiment
        """
        try:
            sentiment = self.analyzer.polarity_scores(text)
            return sentiment

        except Exception as e:
            self.logger.error(
                f"Erreur lors de l'analyse du sentiment: {str(e)}")
            return {'compound': 0.0, 'neg': 0.0, 'neu': 1.0, 'pos': 0.0}

    def get_market_sentiment(self, news: List[Dict[str, Any]]) -> str:
        """
        Analyse le sentiment du marché basé sur les nouvelles.

        Args:
            news: Liste de nouvelles avec leur texte

        Returns:
            'bullish', 'bearish' ou 'neutral'
        """
        try:
            if not news:
                return 'neutral'

            # Analyser chaque nouvelle
            sentiments = []
            for item in news:
                if 'text' in item:
                    sentiment = self.analyze_sentiment(item['text'])
                    sentiments.append(sentiment['compound'])

            # Calculer le sentiment moyen
            avg_sentiment = np.mean(sentiments)

            # Déterminer le sentiment du marché
            if avg_sentiment > 0.2:
                return 'bullish'
            elif avg_sentiment < -0.2:
                return 'bearish'
            else:
                return 'neutral'

        except Exception as e:
            self.logger.error(
                f"Erreur lors de l'analyse du sentiment du marché: {str(e)}")
            return 'neutral'

    def get_price_sentiment(self, price_history: pd.Series) -> str:
        """
        Analyse le sentiment basé sur les mouvements de prix.

        Args:
            price_history: Historique des prix

        Returns:
            'bullish', 'bearish' ou 'neutral'
        """
        try:
            # Calculer les indicateurs
            rsi = momentum.RSIIndicator(price_history, window=14).rsi()
            macd = trend.MACD(price_history).macd()

            # Analyser le sentiment
            if rsi.iloc[-1] > 70 and macd.iloc[-1] > 0:
                return 'bullish'
            elif rsi.iloc[-1] < 30 and macd.iloc[-1] < 0:
                return 'bearish'
            else:
                return 'neutral'

        except Exception as e:
            self.logger.error(
                f"Erreur lors de l'analyse du sentiment des prix: {str(e)}")
            return 'neutral'
