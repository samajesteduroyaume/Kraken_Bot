"""
Module d'analyse technique pour le bot de trading Kraken.
Calcule et analyse les indicateurs techniques.
"""

from typing import Dict, List
import logging
import pandas as pd
from ta import add_all_ta_features
from ta.utils import dropna

logger = logging.getLogger(__name__)


class TechnicalAnalyzer:
    """Analyseur technique pour le bot de trading."""

    def __init__(self, config: Dict):
        """Initialise l'analyseur technique."""
        self.config = config
        self.indicators = config.get('indicators', {
            'rsi': {'window': 14, 'overbought': 70, 'oversold': 30},
            'macd': {'fast': 12, 'slow': 26, 'signal': 9},
            'bb': {'window': 20, 'std': 2},
            'ema': {'fast': 20, 'slow': 50}
        })

    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calcule tous les indicateurs techniques.

        Args:
            df: DataFrame avec les données OHLCV

        Returns:
            DataFrame avec les indicateurs calculés
        """
        try:
            # Nettoyer les données
            df = dropna(df)

            # Ajouter tous les indicateurs avec ta-lib
            df = add_all_ta_features(
                df,
                open="open",
                high="high",
                low="low",
                close="close",
                volume="volume",
                fillna=True
            )

            # Calculer les EMA spécifiques
            df[f'ema_{self.indicators["ema"]["fast"]}'] = df['close'].ewm(
                span=self.indicators['ema']['fast'], adjust=False
            ).mean()

            df[f'ema_{self.indicators["ema"]["slow"]}'] = df['close'].ewm(
                span=self.indicators['ema']['slow'], adjust=False
            ).mean()

            return df

        except Exception as e:
            logger.error(f"Erreur lors du calcul des indicateurs: {str(e)}")
            raise

    def analyze_trend(self, df: pd.DataFrame) -> Dict:
        """
        Analyse la tendance du marché.

        Args:
            df: DataFrame avec les données et indicateurs

        Returns:
            Dictionnaire avec l'analyse de la tendance
        """
        try:
            # Analyse RSI
            rsi = df['momentum_rsi'].iloc[-1]
            rsi_trend = 'neutral'
            if rsi > self.indicators['rsi']['overbought']:
                rsi_trend = 'overbought'
            elif rsi < self.indicators['rsi']['oversold']:
                rsi_trend = 'oversold'

            # Analyse MACD
            macd = df['trend_macd'].iloc[-1]
            macd_signal = df['trend_macd_signal'].iloc[-1]
            macd_trend = 'neutral'
            if macd > macd_signal:
                macd_trend = 'bullish'
            elif macd < macd_signal:
                macd_trend = 'bearish'

            # Analyse EMA croisées
            ema_fast = df[f'ema_{self.indicators["ema"]["fast"]}'].iloc[-1]
            ema_slow = df[f'ema_{self.indicators["ema"]["slow"]}'].iloc[-1]
            ema_trend = 'neutral'
            if ema_fast > ema_slow:
                ema_trend = 'bullish'
            elif ema_fast < ema_slow:
                ema_trend = 'bearish'

            return {
                'rsi': rsi_trend,
                'macd': macd_trend,
                'ema': ema_trend,
                'overall': self.get_overall_trend(
                    rsi_trend,
                    macd_trend,
                    ema_trend)}

        except Exception as e:
            logger.error(f"Erreur lors de l'analyse de la tendance: {str(e)}")
            return {'overall': 'neutral'}

    def get_overall_trend(self, *trends: str) -> str:
        """
        Calcule la tendance globale à partir des différents indicateurs.

        Args:
            trends: Tendances des différents indicateurs

        Returns:
            Tendance globale
        """
        try:
            bullish_count = trends.count('bullish')
            bearish_count = trends.count('bearish')

            if bullish_count > bearish_count:
                return 'bullish'
            elif bearish_count > bullish_count:
                return 'bearish'
            return 'neutral'

        except Exception as e:
            logger.error(
                f"Erreur lors du calcul de la tendance globale: {str(e)}")
            return 'neutral'

    def detect_patterns(self, df: pd.DataFrame) -> List[str]:
        """
        Détecte les patterns techniques.

        Args:
            df: DataFrame avec les données et indicateurs

        Returns:
            Liste des patterns détectés
        """
        try:
            patterns = []

            # Détecter Hammer
            last_candle = df.iloc[-1]
            body = abs(last_candle['close'] - last_candle['open'])
            lower_wick = last_candle['low'] - \
                min(last_candle['open'], last_candle['close'])

            if body > 0 and lower_wick > 2 * body:
                patterns.append('hammer')

            # Détecter Doji
            if abs(last_candle['close'] - last_candle['open']) < 0.01:
                patterns.append('doji')

            return patterns

        except Exception as e:
            logger.error(f"Erreur lors de la détection des patterns: {str(e)}")
            return []
