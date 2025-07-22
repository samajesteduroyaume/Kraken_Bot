"""
Stratégie de retour à la moyenne unifiée.

Cette stratégie combine les fonctionnalités des anciennes stratégies de retour à la moyenne
et de swing trading dans une implémentation plus robuste et maintenable.
"""
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
import logging
from datetime import datetime
from decimal import Decimal

from ta.momentum import RSIIndicator, StochasticOscillator, ROCIndicator
from ta.volatility import BollingerBands, AverageTrueRange
from ta.trend import EMAIndicator, SMAIndicator

from ..base_strategy import BaseStrategy, SignalStrength, TradeSignal
from ...utils.helpers import calculate_support_resistance

logger = logging.getLogger(__name__)

class MeanReversionStrategy(BaseStrategy):
    """
    Stratégie de retour à la moyenne avancée.
    
    Combine les signaux de suracheté/survendu avec une gestion du risque sophistiquée.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialise la stratégie de retour à la moyenne.
        
        Args:
            config: Configuration de la stratégie (optionnel)
        """
        from ...config.strategy_config import get_config
        
        # Charge la configuration par défaut et la fusionne avec celle fournie
        default_config = get_config('mean_reversion')
        if config:
            default_config.update(config)
            
        super().__init__(default_config)
        self.name = "MeanReversionStrategy"
        
        # Paramètres des indicateurs
        self.rsi_period = self.config.get('rsi_period', 14)
        self.stoch_period = self.config.get('stoch_period', 14)
        self.bb_period = self.config.get('bb_period', 20)
        self.bb_std = self.config.get('bb_std', 2.0)
        self.atr_period = self.config.get('atr_period', 14)
        
        # Seuils de décision
        self.rsi_overbought = self.config.get('rsi_overbought', 70)
        self.rsi_oversold = self.config.get('rsi_oversold', 30)
        self.stoch_overbought = self.config.get('stoch_overbought', 80)
        self.stoch_oversold = self.config.get('stoch_oversold', 20)
        
        logger.info(f"Initialisation de {self.name} avec la configuration : {self.config}")
    
    def calculate_indicators(self, data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """
        Calcule les indicateurs techniques pour chaque timeframe.
        
        Args:
            data: Dictionnaire de DataFrames OHLCV par timeframe
            
        Returns:
            Dictionnaire de DataFrames avec les indicateurs ajoutés
        """
        result = {}
        
        for tf, df in data.items():
            # Copie pour éviter de modifier les données d'origine
            df = df.copy()
            
            # RSI
            rsi = RSIIndicator(close=df['close'], window=self.rsi_period)
            df['rsi'] = rsi.rsi()
            
            # Stochastique
            stoch = StochasticOscillator(
                high=df['high'],
                low=df['low'],
                close=df['close'],
                window=self.stoch_period,
                smooth_window=3
            )
            df['stoch_k'] = stoch.stoch()
            df['stoch_d'] = stoch.stoch_signal()
            
            # Bandes de Bollinger
            bb = BollingerBands(close=df['close'], window=self.bb_period, window_dev=self.bb_std)
            df['bb_upper'] = bb.bollinger_hband()
            df['bb_middle'] = bb.bollinger_mavg()
            df['bb_lower'] = bb.bollinger_lband()
            
            # ATR pour le position sizing
            atr = AverageTrueRange(
                high=df['high'],
                low=df['low'],
                close=df['close'],
                window=self.atr_period
            )
            df['atr'] = atr.average_true_range()
            
            # Support et résistance
            supports, resistances = calculate_support_resistance(df)
            df['support'] = pd.Series(supports, index=df.index)
            df['resistance'] = pd.Series(resistances, index=df.index)
            
            # Moyennes mobiles pour confirmation de tendance
            df['ema20'] = EMAIndicator(close=df['close'], window=20).ema_indicator()
            df['ema50'] = EMAIndicator(close=df['close'], window=50).ema_indicator()
            
            result[tf] = df
            
        return result
    
    def generate_signals(self, data: Dict[str, pd.DataFrame]) -> List[TradeSignal]:
        """
        Génère les signaux de trading basés sur la stratégie de retour à la moyenne.
        
        Args:
            data: Dictionnaire de DataFrames avec indicateurs par timeframe
            
        Returns:
            Liste de signaux de trading
        """
        signals = []
        primary_tf = self.timeframes[0]
        
        if primary_tf not in data:
            logger.warning(f"Données manquantes pour le timeframe primaire: {primary_tf}")
            return signals
            
        df = data[primary_tf]
        
        # Dernière bougie
        current = df.iloc[-1]
        prev = df.iloc[-2] if len(df) > 1 else current
        
        # Conditions d'achat (suracheté)
        buy_conditions = [
            current['close'] < current['bb_lower'],  # Prix en dessous de la bande inférieure
            current['rsi'] < self.rsi_oversold,  # RSI en zone de survente
            current['stoch_k'] < self.stoch_oversold,  # Stochastique en zone de survente
            current['close'] > current['support'],  # Au-dessus du support
            current['ema20'] > current['ema50']  # Tendance haussière à plus long terme
        ]
        
        # Conditions de vente (suracheté)
        sell_conditions = [
            current['close'] > current['bb_upper'],  # Prix au-dessus de la bande supérieure
            current['rsi'] > self.rsi_overbought,  # RSI en zone de surachat
            current['stoch_k'] > self.stoch_overbought,  # Stochastique en zone de surachat
            current['close'] < current['resistance'],  # En dessous de la résistance
            current['ema20'] < current['ema50']  # Tendance baissière à plus long terme
        ]
        
        # Génération du signal
        if all(buy_conditions):
            signals.append(TradeSignal(
                symbol=self.config.get('symbol', 'UNKNOWN'),
                direction=1,  # Achat
                strength=SignalStrength.STRONG,
                price=float(current['close']),
                timestamp=datetime.now(),
                metadata={
                    'strategy': self.name,
                    'timeframe': primary_tf,
                    'rsi': float(current['rsi']),
                    'stoch_k': float(current['stoch_k']),
                    'stoch_d': float(current['stoch_d']),
                    'bb_lower': float(current['bb_lower']),
                    'bb_upper': float(current['bb_upper'])
                }
            ))
        elif all(sell_conditions):
            signals.append(TradeSignal(
                symbol=self.config.get('symbol', 'UNKNOWN'),
                direction=-1,  # Vente
                strength=SignalStrength.STRONG,
                price=float(current['close']),
                timestamp=datetime.now(),
                metadata={
                    'strategy': self.name,
                    'timeframe': primary_tf,
                    'rsi': float(current['rsi']),
                    'stoch_k': float(current['stoch_k']),
                    'stoch_d': float(current['stoch_d']),
                    'bb_lower': float(current['bb_lower']),
                    'bb_upper': float(current['bb_upper'])
                }
            ))
            
        return signals
    
    def calculate_confidence(self, data: Dict[str, pd.DataFrame], signals: List[TradeSignal]) -> List[float]:
        """
        Calcule un score de confiance pour chaque signal.
        
        Args:
            data: Données avec indicateurs
            signals: Liste des signaux à évaluer
            
        Returns:
            Liste des scores de confiance (0.0 à 1.0)
        """
        confidences = []
        
        for signal in signals:
            # Base confidence
            confidence = 0.5
            
            # Get the relevant timeframe data
            tf = signal.metadata.get('timeframe', self.timeframes[0])
            if tf not in data:
                confidences.append(0.0)
                continue
                
            df = data[tf]
            current = df.iloc[-1]
            
            # Adjust confidence based on distance from mean
            bb_middle = current['bb_middle']
            price = current['close']
            bb_width = current['bb_upper'] - current['bb_lower']
            
            if bb_width > 0:  # Éviter la division par zéro
                # Plus le prix est loin de la moyenne, plus la confiance est élevée
                distance_from_mean = abs(price - bb_middle) / bb_width
                confidence = min(distance_from_mean * 2, 1.0)  # Normalisé entre 0 et 1
            
            # Ajustement basé sur la convergence des indicateurs
            rsi_extreme = (
                (current['rsi'] < self.rsi_oversold) or 
                (current['rsi'] > self.rsi_overbought)
            )
            
            stoch_extreme = (
                (current['stoch_k'] < self.stoch_oversold) or 
                (current['stoch_k'] > self.stoch_overbought)
            )
            
            if rsi_extreme and stoch_extreme:
                confidence *= 1.3  # Boost de 30% si les deux indicateurs sont extrêmes
                
            # Normalize to 0-1 range
            confidence = max(0.0, min(1.0, confidence))
            confidences.append(confidence)
            
        return confidences
