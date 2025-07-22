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
                               timeframe: str = '1h',
                               order_book: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Génère des signaux de trading pour une paire donnée.

        Args:
            ohlc_data: Données OHLCV par timeframe
            pair: Paire de trading (ex: 'BTC/USD')
            timeframe: Timeframe pour l'analyse
            order_book: Snapshot du carnet d'ordres (optionnel)

        Returns:
            Dictionnaire contenant les signaux générés
        """
        if timeframe not in ohlc_data or ohlc_data[timeframe].empty:
            return {}

        df = ohlc_data[timeframe].copy()
        signals = {}

        # 1. Signaux basés sur les indicateurs techniques
        signals.update(self._generate_technical_signals(df, order_book))

        # 2. Signaux basés sur le ML
        ml_signals = await self._generate_ml_signals(df, pair)
        signals.update(ml_signals)

        # 3. Combinaison des signaux
        signals['final_signal'] = self._combine_signals(signals)

        return signals

    def _generate_technical_signals(self, df: pd.DataFrame, order_book: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Génère des signaux basés sur des indicateurs techniques et les données du carnet d'ordres.
        
        Args:
            df: DataFrame contenant les données OHLCV
            order_book: Snapshot du carnet d'ordres (optionnel) contenant 'bids', 'asks' et 'metrics'
            
        Returns:
            Dictionnaire des signaux techniques enrichis des métriques du carnet d'ordres
        """
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
        
        # Métriques du carnet d'ordres (si disponible)
        if order_book and 'metrics' in order_book:
            metrics = order_book['metrics']
            signals.update({
                'order_book_imbalance': metrics.get('imbalance', 0.0),
                'order_book_spread': metrics.get('spread', 0.0),
                'vwap_bid': metrics.get('vwap_bid', 0.0),
                'vwap_ask': metrics.get('vwap_ask', 0.0),
                'liquidity_imbalance': self._calculate_liquidity_imbalance(order_book)
            })
            
            # Signaux basés sur le carnet d'ordres
            signals.update(self._generate_order_book_signals(order_book, df['close'].iloc[-1]))

        return signals

    def _calculate_liquidity_imbalance(self, order_book: Dict[str, Any], depth_levels: int = 5) -> float:
        """
        Calcule le déséquilibre de liquidité sur les N premiers niveaux du carnet d'ordres.
        
        Args:
            order_book: Carnet d'ordres avec 'bids' et 'asks'
            depth_levels: Nombre de niveaux à considérer
            
        Returns:
            Ratio de déséquilibre entre l'offre et la demande (-1 à 1)
        """
        try:
            bids = order_book.get('bids', [])[:depth_levels]
            asks = order_book.get('asks', [])[:depth_levels]
            
            if not bids or not asks:
                return 0.0
                
            total_bid_volume = sum(float(bid['amount']) for bid in bids)
            total_ask_volume = sum(float(ask['amount']) for ask in asks)
            
            if total_bid_volume + total_ask_volume == 0:
                return 0.0
                
            return (total_bid_volume - total_ask_volume) / (total_bid_volume + total_ask_volume)
            
        except Exception as e:
            logging.warning(f"Erreur dans le calcul du déséquilibre de liquidité: {e}")
            return 0.0
    
    def _generate_order_book_signals(self, order_book: Dict[str, Any], current_price: float) -> Dict[str, Any]:
        """
        Génère des signaux basés sur l'analyse du carnet d'ordres.
        
        Args:
            order_book: Carnet d'ordres avec 'bids', 'asks' et 'metrics'
            current_price: Prix actuel du marché
            
        Returns:
            Dictionnaire de signaux basés sur le carnet d'ordres
        """
        signals = {}
        
        try:
            metrics = order_book.get('metrics', {})
            spread = metrics.get('spread', 0.0)
            imbalance = metrics.get('imbalance', 0.0)
            
            # Signal basé sur le spread
            signals['spread_ratio'] = spread / current_price if current_price > 0 else 0.0
            signals['spread_signal'] = 'high' if signals['spread_ratio'] > 0.0015 else 'normal'
            
            # Signal basé sur le déséquilibre
            signals['imbalance_ratio'] = imbalance
            signals['imbalance_signal'] = 'buy' if imbalance > 0.2 else 'sell' if imbalance < -0.2 else 'neutral'
            
            # Signal basé sur le VWAP
            if 'vwap_bid' in metrics and 'vwap_ask' in metrics and current_price > 0:
                mid_vwap = (metrics['vwap_bid'] + metrics['vwap_ask']) / 2
                signals['vwap_premium'] = (current_price - mid_vwap) / mid_vwap
                signals['vwap_signal'] = 'overbought' if signals['vwap_premium'] > 0.001 else \
                                       'oversold' if signals['vwap_premium'] < -0.001 else 'neutral'
            
        except Exception as e:
            logging.warning(f"Erreur dans la génération des signaux du carnet d'ordres: {e}")
        
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
