"""
Module de gestion des stratégies de trading pour le bot Kraken.
Gère les différents types de stratégies et leur exécution.
"""

from typing import Dict, Optional, Tuple
import logging
import pandas as pd
from enum import Enum

logger = logging.getLogger(__name__)


class TradingSignal(Enum):
    """Signaux de trading possibles."""
    BUY = 'buy'
    SELL = 'sell'
    HOLD = 'hold'
    UNKNOWN = 'unknown'


class StrategyManager:
    """Gestionnaire de stratégies de trading."""

    def __init__(self, config: Dict):
        """Initialise le gestionnaire de stratégies."""
        self.config = config
        self.strategies = config.get('strategies', {
            'momentum': {
                'enabled': True,
                'parameters': {
                    'rsi_threshold': 70,
                    'macd_threshold': 0,
                    'ema_fast': 20,
                    'ema_slow': 50
                }
            },
            'mean_reversion': {
                'enabled': True,
                'parameters': {
                    'bb_threshold': 2.0,
                    'volume_threshold': 1.5
                }
            },
            'ml': {
                'enabled': True,
                'parameters': {
                    'confidence_threshold': 0.6
                }
            }
        })

    def analyze_momentum(self, df: pd.DataFrame) -> TradingSignal:
        """
        Analyse la stratégie de momentum.

        Args:
            df: DataFrame avec les données et indicateurs

        Returns:
            Signal de trading
        """
        try:
            rsi = df['rsi'].iloc[-1]
            macd = df['macd'].iloc[-1]

            # Conditions d'achat
            if rsi < self.strategies['momentum']['parameters'][
                    'rsi_threshold'] and macd > self.strategies['momentum']['parameters']['macd_threshold']:
                return TradingSignal.BUY

            # Conditions de vente
            elif rsi > self.strategies['momentum']['parameters']['rsi_threshold'] and macd < self.strategies['momentum']['parameters']['macd_threshold']:
                return TradingSignal.SELL

            return TradingSignal.HOLD

        except Exception as e:
            logger.error(f"Erreur lors de l'analyse du momentum: {str(e)}")
            return TradingSignal.UNKNOWN

    def analyze_mean_reversion(self, df: pd.DataFrame) -> TradingSignal:
        """
        Analyse la stratégie de mean reversion.

        Args:
            df: DataFrame avec les données et indicateurs

        Returns:
            Signal de trading
        """
        try:
            price = df['close'].iloc[-1]
            bb_upper = df['bb_upper'].iloc[-1]
            bb_lower = df['bb_lower'].iloc[-1]
            volume_ratio = df['volume_ratio'].iloc[-1]

            # Conditions d'achat
            if price < bb_lower and volume_ratio > self.strategies[
                    'mean_reversion']['parameters']['volume_threshold']:
                return TradingSignal.BUY

            # Conditions de vente
            elif price > bb_upper and volume_ratio > self.strategies['mean_reversion']['parameters']['volume_threshold']:
                return TradingSignal.SELL

            return TradingSignal.HOLD

        except Exception as e:
            logger.error(
                f"Erreur lors de l'analyse de mean reversion: {str(e)}")
            return TradingSignal.UNKNOWN

    def analyze_ml(self, ml_prediction: Tuple[float, float]) -> TradingSignal:
        """
        Analyse la prédiction ML.

        Args:
            ml_prediction: Tuple (probabilité de hausse, probabilité de baisse)

        Returns:
            Signal de trading
        """
        try:
            up_prob, down_prob = ml_prediction
            threshold = self.strategies['ml']['parameters']['confidence_threshold']

            if up_prob > threshold and up_prob > down_prob:
                return TradingSignal.BUY

            elif down_prob > threshold and down_prob > up_prob:
                return TradingSignal.SELL

            return TradingSignal.HOLD

        except Exception as e:
            logger.error(f"Erreur lors de l'analyse ML: {str(e)}")
            return TradingSignal.UNKNOWN

    def get_trading_signal(self,
                           df: pd.DataFrame,
                           ml_prediction: Optional[Tuple[float,
                                                         float]] = None) -> TradingSignal:
        """
        Obtient le signal de trading final en combinant les stratégies.

        Args:
            df: DataFrame avec les données et indicateurs
            ml_prediction: Prédiction ML (optionnelle)

        Returns:
            Signal de trading final
        """
        try:
            # Analyser chaque stratégie
            momentum_signal = self.analyze_momentum(df)
            mean_reversion_signal = self.analyze_mean_reversion(df)
            ml_signal = self.analyze_ml(
                ml_prediction) if ml_prediction else TradingSignal.UNKNOWN

            # Combiner les signaux
            signals = [momentum_signal, mean_reversion_signal, ml_signal]

            # Compter les votes
            buy_votes = signals.count(TradingSignal.BUY)
            sell_votes = signals.count(TradingSignal.SELL)

            # Décider du signal final
            if buy_votes > sell_votes:
                return TradingSignal.BUY

            elif sell_votes > buy_votes:
                return TradingSignal.SELL

            return TradingSignal.HOLD

        except Exception as e:
            logger.error(
                f"Erreur lors de la génération du signal de trading: {str(e)}")
            return TradingSignal.UNKNOWN
