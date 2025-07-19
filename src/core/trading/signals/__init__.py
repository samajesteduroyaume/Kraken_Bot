"""
Gestion des signaux de trading.

Ce module est responsable de la génération et de la gestion des signaux de trading
basés sur l'analyse technique et les prédictions du modèle ML.
"""
from typing import Dict, Optional, Any
import pandas as pd
from ta.trend import MACD, ADXIndicator
from ta.momentum import RSIIndicator
from ta.volatility import AverageTrueRange, BollingerBands

from src.ml.predictor import MLPredictor


class SignalGenerator:
    """Génère des signaux de trading basés sur des indicateurs techniques et du ML."""

    def __init__(self, predictor: Optional[MLPredictor] = None):
        """
        Initialise le générateur de signaux.

        Args:
            predictor: Prédicteur ML (optionnel)
        """
        self.predictor = predictor or MLPredictor()

    async def generate_signals(self,
                               ohlc_data: Dict[str, pd.DataFrame],
                               pair: str,
                               timeframe: str = '1h') -> Dict[str, Any]:
        """
        Génère des signaux de trading pour une paire donnée.

        Args:
            ohlc_data: Données OHLCV par timeframe
            pair: Paire de trading (ex: 'BTC/USD')
            timeframe: Timeframe pour l'analyse

        Returns:
            Dictionnaire contenant les signaux générés
        """
        if timeframe not in ohlc_data or ohlc_data[timeframe].empty:
            return {}

        df = ohlc_data[timeframe].copy()
        signals = {}

        # 1. Signaux basés sur les indicateurs techniques
        signals.update(self._generate_technical_signals(df))

        # 2. Signaux basés sur le ML
        ml_signals = await self._generate_ml_signals(df, pair)
        signals.update(ml_signals)

        # 3. Combinaison des signaux
        signals['final_signal'] = self._combine_signals(signals)

        return signals

    def _generate_technical_signals(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Génère des signaux basés sur des indicateurs techniques."""
        signals = {}

        # RSI
        rsi = RSIIndicator(close=df['close'], window=14)
        signals['rsi'] = rsi.rsi().iloc[-1]
        signals['rsi_signal'] = 'buy' if signals['rsi'] < 30 else 'sell' if signals['rsi'] > 70 else 'neutral'

        # MACD
        macd = MACD(close=df['close'])
        signals['macd'] = macd.macd().iloc[-1]
        signals['macd_signal'] = macd.macd_signal().iloc[-1]
        signals['macd_hist'] = signals['macd'] - signals['macd_signal']
        signals['macd_signal'] = 'buy' if signals['macd_hist'] > 0 else 'sell'

        # Bollinger Bands
        bb = BollingerBands(close=df['close'], window=20, window_dev=2)
        signals['bb_upper'] = bb.bollinger_hband().iloc[-1]
        signals['bb_lower'] = bb.bollinger_lband().iloc[-1]
        signals['bb_middle'] = bb.bollinger_mavg().iloc[-1]
        signals['bb_signal'] = 'buy' if df['close'].iloc[-1] < signals['bb_lower'] else \
            'sell' if df['close'].iloc[-1] > signals['bb_upper'] else 'neutral'

        # ADX (Force de la tendance)
        adx = ADXIndicator(
            high=df['high'],
            low=df['low'],
            close=df['close'],
            window=14)
        signals['adx'] = adx.adx().iloc[-1]
        signals['adx_signal'] = 'strong_trend' if signals['adx'] > 25 else 'weak_trend'

        return signals

    async def _generate_ml_signals(self,
                                   df: pd.DataFrame,
                                   pair: str) -> Dict[str, Any]:
        """Génère des signaux basés sur des prédictions ML."""
        try:
            # Préparer les caractéristiques pour le modèle ML
            features = {
                'rsi': df['close'].rolling(window=14).apply(
                    lambda x: 100 -
                    (100 / (1 + (x[x > 0].pct_change().dropna().mean() * 100)))
                ).iloc[-1],
                'macd': MACD(close=df['close']).macd().iloc[-1],
                'macd_signal': MACD(close=df['close']).macd_signal().iloc[-1],
                'bb_upper': BollingerBands(close=df['close']).bollinger_hband().iloc[-1],
                'bb_lower': BollingerBands(close=df['close']).bollinger_lband().iloc[-1],
                'atr': AverageTrueRange(high=df['high'], low=df['low'], close=df['close']).average_true_range().iloc[-1],
                'adx': ADXIndicator(high=df['high'], low=df['low'], close=df['close']).adx().iloc[-1],
                'ema_20': df['close'].ewm(span=20, adjust=False).mean().iloc[-1],
                'ema_50': df['close'].ewm(span=50, adjust=False).mean().iloc[-1],
                'ema_200': df['close'].ewm(span=200, adjust=False).mean().iloc[-1],
                'volume_ma': df['volume'].rolling(window=20).mean().iloc[-1],
                # Spread moyen en %
                'spread': (df['high'] - df['low']).mean() / df['close'].mean() * 100
            }

            # Obtenir la prédiction du modèle ML
            prediction = await self.predictor.predict(features)

            return {
                'ml_prediction': prediction.get(
                    'prediction', 'hold'), 'ml_confidence': prediction.get(
                    'confidence', 0.0), 'risk_score': prediction.get(
                    'risk_score', 0.5), 'recommended_leverage': prediction.get(
                    'recommended_leverage', 1.0)}

        except Exception as e:
            logger.error(f"Erreur lors de la génération des signaux ML: {e}")
            return {
                'ml_prediction': 'hold',
                'ml_confidence': 0.0,
                'risk_score': 0.5,
                'recommended_leverage': 1.0
            }

    def _combine_signals(self, signals: Dict[str, Any]) -> str:
        """Combine plusieurs signaux en un signal final."""
        # Logique de combinaison des signaux
        buy_signals = 0
        sell_signals = 0

        # Poids des différents signaux
        weights = {
            'rsi_signal': 1.0,
            'macd_signal': 1.5,
            'bb_signal': 1.0,
            'ml_prediction': 2.0
        }

        # Compter les signaux d'achat/vente
        for signal_name, weight in weights.items():
            if signal_name in signals:
                if signals[signal_name] == 'buy':
                    buy_signals += weight
                elif signals[signal_name] == 'sell':
                    sell_signals += weight

        # Décision finale basée sur la somme pondérée
        if buy_signals > sell_signals + 1.5:  # Seuil pour éviter les faux signaux
            return 'buy'
        elif sell_signals > buy_signals + 1.5:
            return 'sell'
        else:
            return 'hold'
