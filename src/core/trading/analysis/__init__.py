"""
Analyse de marché et indicateurs techniques.

Ce module fournit des outils pour l'analyse technique des marchés
et le calcul d'indicateurs avancés.
"""
from typing import Dict, Optional, Tuple
import pandas as pd
import numpy as np
import ta
from ta.trend import MACD, ADXIndicator, PSARIndicator, EMAIndicator, SMAIndicator
from ta.momentum import RSIIndicator, StochasticOscillator, ROCIndicator
from ta.volatility import AverageTrueRange, BollingerBands
from ta.volume import VolumeWeightedAveragePrice, OnBalanceVolumeIndicator


class MarketAnalyzer:
    """Effectue des analyses de marché et calcule des indicateurs techniques."""

    def __init__(self, ohlc_data: Optional[Dict[str, pd.DataFrame]] = None):
        """
        Initialise l'analyseur de marché.

        Args:
            ohlc_data: Données OHLCV par timeframe (optionnel)
        """
        self.ohlc_data = ohlc_data or {}
        self.indicators = {}

    def update_data(self, ohlc_data: Dict[str, pd.DataFrame]) -> None:
        """
        Met à jour les données OHLCV.

        Args:
            ohlc_data: Nouvelles données OHLCV par timeframe
        """
        self.ohlc_data = ohlc_data
        self.indicators.clear()

    def calculate_all_indicators(self, timeframe: str) -> Dict[str, float]:
        """
        Calcule tous les indicateurs pour un timeframe donné.

        Args:
            timeframe: Le timeframe pour lequel calculer les indicateurs

        Returns:
            Dictionnaire contenant tous les indicateurs calculés
        """
        if timeframe not in self.ohlc_data:
            return {}

        df = self.ohlc_data[timeframe]
        indicators = {}

        # 1. Tendance
        indicators.update(self._calculate_trend_indicators(df))

        # 2. Momentum
        indicators.update(self._calculate_momentum_indicators(df))

        # 3. Volatilité
        indicators.update(self._calculate_volatility_indicators(df))

        # 4. Volume
        indicators.update(self._calculate_volume_indicators(df))

        # 5. Autres indicateurs personnalisés
        indicators.update(self._calculate_custom_indicators(df))

        self.indicators[timeframe] = indicators
        return indicators

    def _calculate_trend_indicators(
            self, df: pd.DataFrame) -> Dict[str, float]:
        """Calcule les indicateurs de tendance."""
        indicators = {}

        # Moyennes mobiles
        for period in [5, 10, 20, 50, 100, 200]:
            sma = SMAIndicator(close=df['close'], window=period)
            ema = EMAIndicator(close=df['close'], window=period)

            indicators[f'sma_{period}'] = sma.sma_indicator().iloc[-1]
            indicators[f'ema_{period}'] = ema.ema_indicator().iloc[-1]

        # MACD
        macd = MACD(close=df['close'])
        indicators['macd'] = macd.macd().iloc[-1]
        indicators['macd_signal'] = macd.macd_signal().iloc[-1]
        indicators['macd_hist'] = indicators['macd'] - \
            indicators['macd_signal']

        # ADX
        adx = ADXIndicator(
            high=df['high'],
            low=df['low'],
            close=df['close'],
            window=14)
        indicators['adx'] = adx.adx().iloc[-1]
        indicators['adx_pos'] = adx.adx_pos().iloc[-1]
        indicators['adx_neg'] = adx.adx_neg().iloc[-1]

        # PSAR
        psar = PSARIndicator(high=df['high'], low=df['low'], close=df['close'])
        indicators['psar'] = psar.psar().iloc[-1]
        indicators['psar_up'] = psar.psar_up().iloc[-1]
        indicators['psar_down'] = psar.psar_down().iloc[-1]

        return indicators

    def _calculate_momentum_indicators(
            self, df: pd.DataFrame) -> Dict[str, float]:
        """Calcule les indicateurs de momentum."""
        indicators = {}

        # RSI
        for period in [7, 14, 21]:
            rsi = RSIIndicator(close=df['close'], window=period)
            indicators[f'rsi_{period}'] = rsi.rsi().iloc[-1]

        # Stochastique
        stoch = StochasticOscillator(
            high=df['high'],
            low=df['low'],
            close=df['close'],
            window=14,
            smooth_window=3
        )
        indicators['stoch_k'] = stoch.stoch().iloc[-1]
        indicators['stoch_d'] = stoch.stoch_signal().iloc[-1]

        # ROC
        roc = ROCIndicator(close=df['close'], window=14)
        indicators['roc'] = roc.roc().iloc[-1]

        # CCI
        cci = ta.trend.CCIIndicator(
            high=df['high'],
            low=df['low'],
            close=df['close'],
            window=20
        )
        indicators['cci'] = cci.cci().iloc[-1]

        return indicators

    def _calculate_volatility_indicators(
            self, df: pd.DataFrame) -> Dict[str, float]:
        """Calcule les indicateurs de volatilité."""
        indicators = {}

        # Bandes de Bollinger
        for period in [10, 20, 50]:
            bb = BollingerBands(close=df['close'], window=period, window_dev=2)
            indicators[f'bb_upper_{period}'] = bb.bollinger_hband().iloc[-1]
            indicators[f'bb_middle_{period}'] = bb.bollinger_mavg().iloc[-1]
            indicators[f'bb_lower_{period}'] = bb.bollinger_lband().iloc[-1]
            indicators[f'bb_width_{period}'] = bb.bollinger_wband().iloc[-1]

        # ATR
        atr = AverageTrueRange(
            high=df['high'],
            low=df['low'],
            close=df['close'],
            window=14
        )
        indicators['atr'] = atr.average_true_range().iloc[-1]

        # Keltner Channels
        kc = ta.volatility.KeltnerChannel(
            high=df['high'],
            low=df['low'],
            close=df['close'],
            window=20
        )
        indicators['kc_upper'] = kc.keltner_channel_hband().iloc[-1]
        indicators['kc_middle'] = kc.keltner_channel_mband().iloc[-1]
        indicators['kc_lower'] = kc.keltner_channel_lband().iloc[-1]

        return indicators

    def _calculate_volume_indicators(
            self, df: pd.DataFrame) -> Dict[str, float]:
        """Calcule les indicateurs de volume."""
        indicators = {}

        # VWAP
        vwap = VolumeWeightedAveragePrice(
            high=df['high'],
            low=df['low'],
            close=df['close'],
            volume=df['volume'],
            window=20
        )
        indicators['vwap'] = vwap.volume_weighted_average_price().iloc[-1]

        # OBV
        obv = OnBalanceVolumeIndicator(
            close=df['close'],
            volume=df['volume']
        )
        indicators['obv'] = obv.on_balance_volume().iloc[-1]

        # Volume MA
        for period in [5, 10, 20, 50]:
            indicators[f'volume_ma_{period}'] = df['volume'].rolling(
                window=period).mean().iloc[-1]

        # Volume ROC
        volume_roc = ROCIndicator(close=df['volume'], window=14)
        indicators['volume_roc'] = volume_roc.roc().iloc[-1]

        return indicators

    def _calculate_custom_indicators(
            self, df: pd.DataFrame) -> Dict[str, float]:
        """Calcule des indicateurs personnalisés."""
        indicators = {}

        # Ratio de Sharpe (simplifié)
        returns = df['close'].pct_change().dropna()
        if len(returns) > 1:
            sharpe_ratio = np.sqrt(252) * (returns.mean() / returns.std())
            indicators['sharpe_ratio'] = sharpe_ratio if not np.isnan(
                sharpe_ratio) else 0.0

        # Ratio de Sortino
        if len(returns) > 1:
            downside_returns = returns[returns < 0]
            if len(downside_returns) > 0:
                sortino_ratio = np.sqrt(
                    252) * (returns.mean() / downside_returns.std())
                indicators['sortino_ratio'] = sortino_ratio if not np.isnan(
                    sortino_ratio) else 0.0

        # Volatilité historique
        if len(returns) > 1:
            indicators['historical_volatility'] = returns.std() * np.sqrt(252)

        # Spread moyen
        if 'spread' in df.columns:
            indicators['avg_spread'] = df['spread'].mean()

        return indicators

    def get_market_regime(self, timeframe: str = '1h') -> str:
        """
        Détermine le régime de marché actuel.

        Args:
            timeframe: Le timeframe à analyser

        Returns:
            'bullish', 'bearish' ou 'ranging'
        """
        if timeframe not in self.indicators:
            self.calculate_all_indicators(timeframe)

        indicators = self.indicators.get(timeframe, {})

        # Règles simples pour la démonstration
        if 'adx' in indicators and 'ema_50' in indicators and 'ema_200' in indicators:
            ema_50 = indicators['ema_50']
            ema_200 = indicators['ema_200']
            adx = indicators['adx']

            if adx > 25:  # Tendance forte
                if ema_50 > ema_200:
                    return 'bullish'
                else:
                    return 'bearish'

        return 'ranging'

    def get_support_resistance_levels(self,
                                      df: pd.DataFrame,
                                      window: int = 20) -> Tuple[float, float]:
        """
        Identifie les niveaux de support et résistance.

        Args:
            df: DataFrame avec les données de prix
            window: Taille de la fenêtre pour l'identification des niveaux

        Returns:
            Tuple (support, resistance)
        """
        # Utilise les points pivots comme niveaux de support/résistance
        pivot = (df['high'].iloc[-1] + df['low'].iloc[-1] +
                 df['close'].iloc[-1]) / 3

        # Niveaux de support et résistance basés sur la volatilité
        atr = AverageTrueRange(
            high=df['high'],
            low=df['low'],
            close=df['close'],
            window=14
        ).average_true_range().iloc[-1]

        support = pivot - atr
        resistance = pivot + atr

        return support, resistance

    def analyze_market(self, data: dict) -> dict:
        """Analyse le marché pour une paire donnée et retourne les indicateurs principaux."""
        # On suppose que data est un dict contenant un DataFrame OHLCV sous la clé '1h' ou autre timeframe
        timeframe = '1h' if '1h' in data else next(iter(data.keys()), None)
        if not timeframe:
            return {}
        df = data[timeframe]
        return self.calculate_all_indicators(timeframe)

    def get_market_metrics(self) -> dict:
        """Retourne un résumé des métriques de marché pour toutes les paires."""
        metrics = {}
        for pair, ohlc_dict in self.ohlc_data.items():
            timeframe = '1h' if '1h' in ohlc_dict else next(iter(ohlc_dict.keys()), None)
            if not timeframe:
                continue
            df = ohlc_dict[timeframe]
            indicators = self.calculate_all_indicators(timeframe)
            # Exemple de métriques principales
            metrics[pair] = {
                'score': indicators.get('sharpe_ratio', 0),
                'volatility': indicators.get('historical_volatility', 0),
                'momentum': indicators.get('roc', 0),
                'volume': indicators.get('vwap', 0)
            }
        return metrics
