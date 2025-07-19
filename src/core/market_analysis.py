"""
Module d'analyse du marché pour le bot de trading Kraken.
Analyse les tendances, les volumes et les indicateurs techniques.
"""

from typing import Dict, List
import logging
import pandas as pd
from datetime import datetime

logger = logging.getLogger(__name__)


class MarketAnalyzer:
    """Analyseur de marché pour le bot de trading."""

    def __init__(self, config: Dict):
        """Initialise l'analyseur de marché."""
        self.config = config
        self.indicators = config.get('indicators', [])
        self.timeframes = config.get('timeframes', [1, 5, 15, 60])  # Minutes
        self.trend_thresholds = config.get('trend_thresholds', {
            'rsi': {'overbought': 70, 'oversold': 30},
            'macd': {'bullish': 0, 'bearish': 0},
            'ema': {'fast': 20, 'slow': 50}
        })

    def analyze_trend(self, df: pd.DataFrame) -> str:
        """
        Analyse la tendance du marché.

        Args:
            df: DataFrame avec les données OHLCV

        Returns:
            'bullish', 'bearish' ou 'neutral'
        """
        try:
            # Analyse RSI
            rsi = df['rsi'].iloc[-1]
            if rsi > self.trend_thresholds['rsi']['overbought']:
                return 'bearish'
            elif rsi < self.trend_thresholds['rsi']['oversold']:
                return 'bullish'

            # Analyse EMA croisées
            ema_fast = df[f'ema_{self.trend_thresholds["ema"]["fast"]}'].iloc[-1]
            ema_slow = df[f'ema_{self.trend_thresholds["ema"]["slow"]}'].iloc[-1]

            if ema_fast > ema_slow:
                return 'bullish'
            elif ema_fast < ema_slow:
                return 'bearish'

            return 'neutral'

        except Exception as e:
            logger.error(f"Erreur lors de l'analyse de la tendance: {str(e)}")
            return 'neutral'

    def analyze_volume(self, df: pd.DataFrame) -> Dict:
        """
        Analyse le volume de trading.

        Args:
            df: DataFrame avec les données OHLCV

        Returns:
            Dictionnaire avec les métriques de volume
        """
        try:
            volume = df['volume'].iloc[-1]
            avg_volume = df['volume'].rolling(window=20).mean().iloc[-1]
            volume_ratio = volume / avg_volume

            return {
                'current_volume': volume,
                'average_volume': avg_volume,
                'volume_ratio': volume_ratio,
                'is_high_volume': volume_ratio > 1.5
            }

        except Exception as e:
            logger.error(f"Erreur lors de l'analyse du volume: {str(e)}")
            return {
                'current_volume': 0,
                'average_volume': 0,
                'volume_ratio': 0,
                'is_high_volume': False
            }

    def detect_patterns(self, df: pd.DataFrame) -> List[str]:
        """
        Détecte les patterns techniques.

        Args:
            df: DataFrame avec les données OHLCV

        Returns:
            Liste des patterns détectés
        """
        try:
            patterns = []

            # Hammer pattern
            last_candle = df.iloc[-1]
            body = abs(last_candle['close'] - last_candle['open'])
            lower_wick = last_candle['low'] - \
                min(last_candle['open'], last_candle['close'])

            if body > 0 and lower_wick > 2 * body:
                patterns.append('hammer')

            # Doji pattern
            if abs(last_candle['close'] - last_candle['open']) < 0.01:
                patterns.append('doji')

            return patterns

        except Exception as e:
            logger.error(f"Erreur lors de la détection des patterns: {str(e)}")
            return []

    def analyze_market_conditions(self, df: pd.DataFrame) -> Dict:
        """
        Analyse les conditions du marché.

        Args:
            df: DataFrame avec les données OHLCV

        Returns:
            Dictionnaire avec l'analyse complète du marché
        """
        try:
            trend = self.analyze_trend(df)
            volume_metrics = self.analyze_volume(df)
            patterns = self.detect_patterns(df)

            return {
                'trend': trend,
                'volume': volume_metrics,
                'patterns': patterns,
                'last_price': df['close'].iloc[-1],
                'timestamp': df.index[-1]
            }

        except Exception as e:
            logger.error(f"Erreur lors de l'analyse du marché: {str(e)}")
            return {
                'trend': 'neutral',
                'volume': {},
                'patterns': [],
                'last_price': 0,
                'timestamp': datetime.now()
            }
