"""
Implémentation de la stratégie de Swing Trading.

Cette stratégie vise à capturer les mouvements de marché à moyen terme en utilisant
une combinaison d'indicateurs techniques et d'analyse de tendance.
"""
from typing import Dict, List, Any, Optional, Tuple
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from decimal import Decimal, ROUND_HALF_UP
import logging

from .base_strategy import BaseStrategy, SignalStrength, TradeSignal
from .swing_config import get_swing_config

logger = logging.getLogger(__name__)

class SwingStrategy(BaseStrategy):
    """
    Stratégie de Swing Trading qui combine plusieurs indicateurs techniques
    pour identifier des opportunités d'entrée et de sortie sur des mouvements
    de marché à moyen terme.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialise la stratégie de Swing Trading.
        
        Args:
            config: Configuration de la stratégie. Si None, utilise la configuration par défaut.
        """
        # Charger la configuration par défaut si aucune n'est fournie
        if config is None:
            config = get_swing_config()
            
        # Appeler le constructeur de la classe parente
        super().__init__(config)
        
        # Initialiser les indicateurs
        self.indicators = {}
        self._initialize_indicators()
    
    def _initialize_indicators(self):
        """Initialise les indicateurs techniques basés sur la configuration."""
        self.indicators = {}
        for tf in self.config['timeframes']:
            self.indicators[tf] = {}
    
    def calculate_indicators(self, data: Dict[str, pd.DataFrame]) -> Dict[str, Dict[str, pd.Series]]:
        """
        Calcule les indicateurs techniques pour chaque timeframe.
        
        Args:
            data: Dictionnaire de DataFrames OHLCV par timeframe
            
        Returns:
            Dictionnaire contenant les indicateurs calculés pour chaque timeframe
        """
        results = {}
        
        for tf, df in data.items():
            if tf not in self.config['timeframes']:
                continue
                
            indicators = {}
            
            # RSI
            if 'rsi' in self.config['indicators']:
                indicators['rsi'] = self._calculate_rsi(df, **self.config['indicators']['rsi'])
            
            # MACD
            if 'macd' in self.config['indicators']:
                macd, signal, hist = self._calculate_macd(df, **self.config['indicators']['macd'])
                indicators['macd'] = macd
                indicators['macd_signal'] = signal
                indicators['macd_hist'] = hist
            
            # ATR
            if 'atr' in self.config['indicators']:
                indicators['atr'] = self._calculate_atr(df, **self.config['indicators']['atr'])
            
            # Moyennes mobiles
            if 'sma' in self.config['indicators']:
                indicators['sma_short'] = self._calculate_sma(df, self.config['indicators']['sma']['short_period'])
                indicators['sma_long'] = self._calculate_sma(df, self.config['indicators']['sma']['long_period'])
            
            # Bandes de Bollinger
            if 'bollinger_bands' in self.config['indicators']:
                bb_high, bb_mid, bb_low = self._calculate_bollinger_bands(
                    df, **self.config['indicators']['bollinger_bands']
                )
                indicators['bb_high'] = bb_high
                indicators['bb_mid'] = bb_mid
                indicators['bb_low'] = bb_low
            
            # MFI (Money Flow Index)
            if 'mfi' in self.config['indicators']:
                indicators['mfi'] = self._calculate_mfi(df, **self.config['indicators']['mfi'])
            
            # ADX (Average Directional Index)
            if 'adx' in self.config['indicators']:
                indicators['adx'] = self._calculate_adx(df, **self.config['indicators']['adx'])
            
            # ROC (Rate of Change)
            if 'roc' in self.config['indicators']:
                indicators['roc'] = self._calculate_roc(df, **self.config['indicators']['roc'])
            
            # Force de la tendance (combinaison de plusieurs indicateurs)
            indicators['trend_strength'] = self._calculate_trend_strength(indicators)
            
            results[tf] = indicators
            
        return results
        
    def generate_signals(self, data: Dict[str, pd.DataFrame]) -> List[TradeSignal]:
        """
        Génère des signaux de trading basés sur les indicateurs calculés.
        
        Args:
            data: Dictionnaire de DataFrames OHLCV par timeframe
            
        Returns:
            Liste de signaux de trading
        """
        # Calculer les indicateurs
        indicators = self.calculate_indicators(data)
        
        # Générer les signaux pour chaque timeframe
        signals = []
        for tf in self.config['timeframes']:
            if tf not in indicators or tf not in data:
                continue
                
            df = data[tf]
            ind = indicators[tf]
            
            # Vérifier si nous avons suffisamment de données
            if len(df) < 50:  # Nombre arbitraire pour s'assurer d'avoir assez de données
                continue
            
            # Dernier point de données
            last_idx = df.index[-1]
            
            # Vérifier les conditions d'entrée
            entry_signal = self._check_entry_conditions(df, ind)
            
            if entry_signal:
                # Créer un signal de trading
                signal = TradeSignal(
                    symbol=self.config['symbol'],
                    direction=1 if entry_signal['direction'] == 'long' else -1,
                    strength=entry_signal['strength'],
                    price=Decimal(str(df['close'].iloc[-1])).quantize(Decimal('0.01')),
                    stop_loss=entry_signal.get('stop_loss'),
                    take_profit=entry_signal.get('take_profit'),
                    timestamp=last_idx.to_pydatetime(),
                    metadata={
                        'timeframe': tf,
                        'strategy': 'swing',
                        'indicators': {k: float(v.iloc[-1]) for k, v in ind.items() if isinstance(v, pd.Series)}
                    }
                )
                signals.append(signal)
        
        return signals
    
    def _check_entry_conditions(self, df: pd.DataFrame, indicators: Dict[str, pd.Series]) -> Optional[Dict]:
        """
        Vérifie les conditions d'entrée pour un timeframe donné.
        
        Args:
            df: DataFrame OHLCV
            indicators: Dictionnaire d'indicateurs techniques
            
        Returns:
            Dictionnaire avec les informations du signal ou None si aucune condition n'est remplie
        """
        # Dernier point de données
        last_idx = -1
        
        # Vérifier si nous avons tous les indicateurs nécessaires
        required_indicators = ['rsi', 'macd', 'atr', 'sma_short', 'sma_long', 'trend_strength']
        if not all(ind in indicators for ind in required_indicators):
            return None
        
        # Récupérer les valeurs des indicateurs
        rsi = indicators['rsi'].iloc[last_idx]
        macd = indicators['macd'].iloc[last_idx]
        macd_signal = indicators.get('macd_signal', pd.Series([0])).iloc[last_idx]
        atr = indicators['atr'].iloc[last_idx]
        sma_short = indicators['sma_short'].iloc[last_idx]
        sma_long = indicators['sma_long'].iloc[last_idx]
        trend_strength = indicators['trend_strength'].iloc[last_idx]
        close = df['close'].iloc[last_idx]
        
        # Vérifier la force de la tendance
        min_trend_strength = self.config['signal_generation']['trend_filter']['min_trend_strength']
        if trend_strength < min_trend_strength:
            return None
        
        # Vérifier le volume si le filtre est activé
        if self.config['signal_generation']['volume_filter']['enabled']:
            min_volume_ratio = self.config['signal_generation']['volume_filter']['min_volume_ratio']
            volume_ma = df['volume'].rolling(window=20).mean().iloc[last_idx]
            if df['volume'].iloc[last_idx] < volume_ma * min_volume_ratio:
                return None
        
        # Vérifier la volatilité si le filtre est activé
        if self.config['signal_generation']['volatility_filter']['enabled']:
            max_atr_ratio = self.config['signal_generation']['volatility_filter']['max_atr_ratio']
            atr_ratio = atr / close
            if atr_ratio > max_atr_ratio:
                return None
        
        # Conditions pour un signal d'achat
        if (rsi < 30 and  # RSI en zone de surachat
            macd > macd_signal and  # Croisement haussier du MACD
            close > sma_short > sma_long):  # Tendance haussière
            
            # Calculer le stop loss et le take profit
            stop_loss = close - (2 * atr)
            take_profit = close + (4 * atr)  # Ratio risque/rendement de 1:2
            
            return {
                'direction': 'long',
                'strength': SignalStrength.STRONG if rsi < 25 else SignalStrength.MODERATE,
                'stop_loss': Decimal(str(stop_loss)).quantize(Decimal('0.01')),
                'take_profit': Decimal(str(take_profit)).quantize(Decimal('0.01'))
            }
        
        # Conditions pour un signal de vente
        elif (rsi > 70 and  # RSI en zone de survente
              macd < macd_signal and  # Croisement baissier du MACD
              close < sma_short < sma_long):  # Tendance baissière
            
            # Calculer le stop loss et le take profit
            stop_loss = close + (2 * atr)
            take_profit = close - (4 * atr)  # Ratio risque/rendement de 1:2
            
            return {
                'direction': 'short',
                'strength': SignalStrength.STRONG_SELL if rsi > 75 else SignalStrength.MODERATE_SELL,
                'stop_loss': Decimal(str(stop_loss)).quantize(Decimal('0.01')),
                'take_profit': Decimal(str(take_profit)).quantize(Decimal('0.01'))
            }
        
        return None
    
    def _calculate_trend_strength(self, indicators: Dict[str, pd.Series]) -> pd.Series:
        """
        Calcule une mesure de la force de la tendance basée sur plusieurs indicateurs.
        
        Args:
            indicators: Dictionnaire d'indicateurs techniques
            
        Returns:
            Série pandas représentant la force de la tendance (0 à 1)
        """
        # Moyenne pondérée de plusieurs indicateurs de tendance
        trend_indicators = []
        
        # ADX (si disponible)
        if 'adx' in indicators:
            adx = indicators['adx']
            # Normaliser entre 0 et 1 (25 = tendance faible, 50+ = forte)
            adx_strength = (adx - 25) / 50
            trend_indicators.append(adx_strength.clip(0, 1) * 0.4)
        
        # Pente des moyennes mobiles (si disponibles)
        if 'sma_short' in indicators and 'sma_long' in indicators:
            sma_short = indicators['sma_short']
            sma_long = indicators['sma_long']
            
            # Écart relatif entre les moyennes mobiles
            sma_ratio = (sma_short / sma_long - 1).abs()
            sma_strength = sma_ratio / (sma_ratio + 0.01)  # Normaliser entre 0 et 1
            trend_indicators.append(sma_strength * 0.3)
        
        # Force du signal MACD (si disponible)
        if 'macd_hist' in indicators:
            macd_hist = indicators['macd_hist']
            macd_strength = macd_hist.rolling(window=14).std() / (macd_hist.abs().rolling(window=14).mean() + 1e-6)
            macd_strength = macd_strength / (macd_strength + 0.1)  # Normaliser entre 0 et 1
            trend_indicators.append(macd_strength * 0.3)
        
        # Si aucun indicateur de tendance n'est disponible, retourner une série de zéros
        if not trend_indicators:
            return pd.Series(0, index=next(iter(indicators.values())).index)
        
        # Moyenne pondérée des indicateurs de tendance
        trend_strength = sum(trend_indicators) / len(trend_indicators)
        
        return trend_strength.clip(0, 1)
    
    # Méthodes utilitaires pour le calcul des indicateurs
    def _calculate_rsi(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calcule le RSI (Relative Strength Index)."""
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    def _calculate_macd(self, df: pd.DataFrame, fast_period: int = 12, 
                        slow_period: int = 26, signal_period: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calcule le MACD (Moving Average Convergence Divergence)."""
        exp1 = df['close'].ewm(span=fast_period, adjust=False).mean()
        exp2 = df['close'].ewm(span=slow_period, adjust=False).mean()
        macd = exp1 - exp2
        signal = macd.ewm(span=signal_period, adjust=False).mean()
        histogram = macd - signal
        return macd, signal, histogram
    
    def _calculate_atr(self, df: pd.DataFrame, period: int = 14, multiplier: float = 2.0) -> pd.Series:
        """Calcule l'ATR (Average True Range)."""
        high = df['high']
        low = df['low']
        close = df['close']
        
        tr1 = high - low
        tr2 = (high - close.shift()).abs()
        tr3 = (low - close.shift()).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        return tr.rolling(window=period).mean() * multiplier
    
    def _calculate_sma(self, df: pd.DataFrame, period: int) -> pd.Series:
        """Calcule une moyenne mobile simple (SMA)."""
        return df['close'].rolling(window=period).mean()
    
    def _calculate_bollinger_bands(self, df: pd.DataFrame, window: int = 20, 
                                 window_dev: float = 2.0) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calcule les bandes de Bollinger."""
        middle_band = df['close'].rolling(window=window).mean()
        std = df['close'].rolling(window=window).std()
        upper_band = middle_band + (std * window_dev)
        lower_band = middle_band - (std * window_dev)
        return upper_band, middle_band, lower_band
    
    def _calculate_mfi(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calcule le MFI (Money Flow Index)."""
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        money_flow = typical_price * df['volume']
        
        # Flux monétaire positif/négatif
        positive_flow = money_flow.where(df['close'] > df['close'].shift(1), 0)
        negative_flow = money_flow.where(df['close'] < df['close'].shift(1), 0)
        
        # Somme mobile sur la période
        positive_mf = positive_flow.rolling(window=period).sum()
        negative_mf = negative_flow.rolling(window=period).sum()
        
        # Calcul du ratio de flux monétaire
        mfr = positive_mf / negative_mf
        return 100 - (100 / (1 + mfr))
    
    def _calculate_adx(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calcule l'ADX (Average Directional Index)."""
        high, low, close = df['high'], df['low'], df['close']
        
        # True Range et Directional Movement
        tr1 = high - low
        tr2 = (high - close.shift()).abs()
        tr3 = (low - close.shift()).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        up = high.diff()
        down = low.diff() * -1
        
        plus_dm = up.where((up > down) & (up > 0), 0)
        minus_dm = down.where((down > up) & (down > 0), 0)
        
        # Lissage exponentiel
        tr_smooth = tr.ewm(alpha=1/period, adjust=False).mean()
        plus_dm_smooth = plus_dm.ewm(alpha=1/period, adjust=False).mean()
        minus_dm_smooth = minus_dm.ewm(alpha=1/period, adjust=False).mean()
        
        # Indicateurs directionnels
        plus_di = 100 * plus_dm_smooth / tr_smooth
        minus_di = 100 * minus_dm_smooth / tr_smooth
        
        # ADX
        dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di)
        adx = dx.ewm(alpha=1/period, adjust=False).mean()
        
        return adx
    
    def _calculate_roc(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calcule le ROC (Rate of Change)."""
        return (df['close'] / df['close'].shift(period) - 1) * 100
