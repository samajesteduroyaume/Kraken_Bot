from typing import Dict, Optional, Tuple, Any
import pandas as pd
from datetime import datetime
from .technical_analyzer import TechnicalAnalyzer
from src.core.logging.logging import LoggerManager


class MarketTrendDetector:
    """Détecte les tendances du marché."""

    def __init__(
        self,
        technical_analyzer: Optional[TechnicalAnalyzer] = None,
        rsi_window: int = 14,
        macd_fast: int = 12,
        macd_slow: int = 26,
        macd_signal: int = 9,
        logger: Optional[LoggerManager] = None
    ):
        self.technical_analyzer = technical_analyzer or TechnicalAnalyzer()
        self.rsi_window = rsi_window
        self.macd_fast = macd_fast
        self.macd_slow = macd_slow
        self.macd_signal = macd_signal
        self.logger = logger or LoggerManager()
        self.logger = self.logger.get_logger()
        self.trends: Dict[str, str] = {}
        self.last_update: Dict[str, datetime] = {}

    def _calculate_rsi(self, data: pd.Series) -> float:
        """Calcule le RSI."""
        delta = data.diff()
        gain = (
            delta.where(
                delta > 0,
                0)).rolling(
            window=self.rsi_window).mean()
        loss = (-delta.where(delta < 0, 0)
                ).rolling(window=self.rsi_window).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    def _calculate_macd(
            self, data: pd.Series) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calcule le MACD."""
        exp1 = data.ewm(span=self.macd_fast, adjust=False).mean()
        exp2 = data.ewm(span=self.macd_slow, adjust=False).mean()
        macd = exp1 - exp2
        signal = macd.ewm(span=self.macd_signal, adjust=False).mean()
        histogram = macd - signal
        return macd, signal, histogram

    def detect_trend(self, pair_data: pd.DataFrame) -> str:
        """Détermine la tendance du marché pour une paire."""
        try:
            # Calculer les indicateurs
            rsi = self._calculate_rsi(pair_data['close'])
            macd, signal, _ = self._calculate_macd(pair_data['close'])

            # Extraire les valeurs actuelles
            current_rsi = rsi.iloc[-1]
            current_macd = macd.iloc[-1]
            current_signal = signal.iloc[-1]

            # Déterminer la tendance
            if current_rsi > 70 and current_macd > current_signal:
                trend = 'bullish'
            elif current_rsi < 30 and current_macd < current_signal:
                trend = 'bearish'
            else:
                trend = 'neutral'

            # Mettre à jour les tendances
            self.trends[pair_data.name] = trend
            self.last_update[pair_data.name] = datetime.now()

            self.logger.info(
                f"Tendance détectée pour {pair_data.name}: {trend}")
            return trend

        except Exception as e:
            self.logger.error(
                f"Erreur lors de la détection de la tendance: {str(e)}")
            return self.trends.get(pair_data.name, 'neutral')

    def get_trend_metrics(self, pair: str) -> Dict[str, Any]:
        """Retourne les métriques de tendance pour une paire."""
        return {
            'trend': self.trends.get(
                pair,
                'neutral'),
            'last_update': self.last_update.get(
                pair,
                datetime.min).isoformat(),
            'rsi_window': self.rsi_window,
            'macd_fast': self.macd_fast,
            'macd_slow': self.macd_slow,
            'macd_signal': self.macd_signal}
