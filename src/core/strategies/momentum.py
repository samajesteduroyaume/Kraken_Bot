"""
Stratégie de momentum basée sur le RSI et d'autres indicateurs de momentum.
"""
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
from datetime import datetime

from .base_strategy import BaseStrategy
from .types import TradingSignal, SignalAction, StrategyType, MarketData, Indicators, StrategyConfig


class MomentumStrategy(BaseStrategy):
    """
    Stratégie de momentum qui utilise le RSI, le MACD et d'autres indicateurs
    pour identifier les mouvements de prix forts et les retournements potentiels.
    """
    
    def __init__(
        self, 
        rsi_period: int = 14,
        rsi_overbought: float = 70.0,
        rsi_oversold: float = 30.0,
        macd_fast: int = 12,
        macd_slow: int = 26,
        macd_signal: int = 9,
        volatility_filter: float = 1.5,
        risk_multiplier: float = 1.3,
        **kwargs
    ):
        """
        Initialise la stratégie de momentum.
        
        Args:
            rsi_period: Période pour le calcul du RSI
            rsi_overbought: Seuil de surachat pour le RSI
            rsi_oversold: Seuil de survente pour le RSI
            macd_fast: Période rapide pour le MACD
            macd_slow: Période lente pour le MACD
            macd_signal: Période du signal pour le MACD
            volatility_filter: Seuil de volatilité pour filtrer les signaux faibles
            risk_multiplier: Multiplicateur de risque pour le position sizing
            **kwargs: Arguments additionnels pour la configuration
        """
        config = StrategyConfig(
            risk_multiplier=risk_multiplier,
            parameters={
                'rsi_period': rsi_period,
                'rsi_overbought': rsi_overbought,
                'rsi_oversold': rsi_oversold,
                'macd_fast': macd_fast,
                'macd_slow': macd_slow,
                'macd_signal': macd_signal,
                'volatility_filter': volatility_filter
            }
        )
        
        super().__init__(
            name="Momentum",
            description="Stratégie de momentum utilisant RSI, MACD et filtres de volatilité",
            strategy_type=StrategyType.MOMENTUM,
            config=config
        )
        
        # Validation des paramètres
        if rsi_period < 2:
            raise ValueError("La période du RSI doit être d'au moins 2")
        if not (0 <= rsi_oversold < rsi_overbought <= 100):
            raise ValueError("Les seuils RSI doivent être entre 0 et 100 avec rsi_oversold < rsi_overbought")
        if macd_fast >= macd_slow:
            raise ValueError("La période rapide du MACD doit être inférieure à la période lente")
    
    async def calculate_indicators(self, market_data: MarketData) -> Indicators:
        """
        Calcule les indicateurs nécessaires pour la stratégie.
        
        Args:
            market_data: Données de marché (doit contenir 'close' et 'volume')
            
        Returns:
            Dictionnaire contenant les indicateurs calculés
        """
        if 'close' not in market_data:
            raise ValueError("Les données de marché doivent contenir les prix de clôture ('close')")
        
        close_prices = pd.Series(market_data['close'])
        volume = pd.Series(market_data.get('volume', np.ones_like(close_prices)))
        
        # Récupérer les paramètres
        rsi_period = self.config.parameters.get('rsi_period', 14)
        macd_fast = self.config.parameters.get('macd_fast', 12)
        macd_slow = self.config.parameters.get('macd_slow', 26)
        macd_signal = self.config.parameters.get('macd_signal', 9)
        
        # Calcul du RSI
        rsi = self._calculate_rsi(close_prices, rsi_period)
        
        # Calcul du MACD
        ema_fast = close_prices.ewm(span=macd_fast, adjust=False).mean()
        ema_slow = close_prices.ewm(span=macd_slow, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=macd_signal, adjust=False).mean()
        macd_hist = macd_line - signal_line
        
        # Calcul de la volatilité (écart-type des rendements sur 20 périodes)
        returns = close_prices.pct_change()
        volatility = returns.rolling(window=20).std() * np.sqrt(252)  # Annualisé
        
        # Volume moyen sur 20 périodes
        avg_volume = volume.rolling(window=20).mean()
        
        indicators = {
            'rsi': rsi,
            'macd_line': macd_line,
            'macd_signal': signal_line,
            'macd_hist': macd_hist,
            'volatility': volatility,
            'volume': volume,
            'avg_volume': avg_volume,
            'close': close_prices
        }
        
        return indicators
    
    async def analyze(
        self, 
        market_data: MarketData,
        indicators: Optional[Indicators] = None
    ) -> List[TradingSignal]:
        """
        Analyse les données de marché et génère des signaux de trading basés sur le momentum.
        
        Args:
            market_data: Données de marché brutes
            indicators: Indicateurs précalculés (optionnel)
            
        Returns:
            Liste de signaux de trading
        """
        # Calculer les indicateurs si non fournis
        if indicators is None:
            indicators = await self.calculate_indicators(market_data)
        
        # Vérifier que nous avons suffisamment de données
        if len(indicators['close']) < 2:
            return []
        
        # Récupérer les paramètres
        rsi_overbought = self.config.parameters.get('rsi_overbought', 70.0)
        rsi_oversold = self.config.parameters.get('rsi_oversold', 30.0)
        volatility_filter = self.config.parameters.get('volatility_filter', 1.5)
        
        # Dernières valeurs des indicateurs
        rsi = indicators['rsi']
        macd_line = indicators['macd_line']
        macd_signal = indicators['macd_signal']
        macd_hist = indicators['macd_hist']
        volatility = indicators['volatility']
        volume = indicators['volume']
        avg_volume = indicators['avg_volume']
        close_prices = indicators['close']
        
        # Dernières valeurs
        current_rsi = rsi.iloc[-1]
        prev_rsi = rsi.iloc[-2] if len(rsi) > 1 else 50.0
        current_macd = macd_line.iloc[-1]
        current_signal = macd_signal.iloc[-1]
        prev_macd = macd_line.iloc[-2] if len(macd_line) > 1 else 0
        prev_signal = macd_signal.iloc[-2] if len(macd_signal) > 1 else 0
        current_volatility = volatility.iloc[-1] if not volatility.empty else 0
        current_volume = volume.iloc[-1]
        current_avg_volume = avg_volume.iloc[-1] if not avg_volume.empty else 0
        current_price = close_prices.iloc[-1]
        
        # Vérifier les conditions de volume et de volatilité
        volume_ok = current_volume > current_avg_volume * 0.8  # Volume au moins 80% de la moyenne
        volatility_ok = current_volatility < volatility_filter  # Volatilité sous le seuil
        
        signals = []
        
        # Signal d'achat: RSI sort de la zone de survente et croise à la hausse
        # OU croisement haussier du MACD avec confirmation de volume
        if (
            (prev_rsi < rsi_oversold and current_rsi > rsi_oversold and current_rsi > prev_rsi) or
            (prev_macd <= prev_signal and current_macd > current_signal and volume_ok)
        ) and volatility_ok:
            # Calculer la force du signal basée sur la convergence des indicateurs
            rsi_strength = max(0, (rsi_oversold - current_rsi) / rsi_oversold) if current_rsi < rsi_oversold else 0
            macd_strength = abs(macd_hist.iloc[-1]) / current_price * 1000  # Normalisation approximative
            strength = min(1.0, 0.5 + (rsi_strength * 0.3) + (macd_strength * 0.2))
            
            signal = self._create_signal(
                symbol=market_data.get('symbol', 'UNKNOWN'),
                action=SignalAction.BUY,
                price=current_price,
                confidence=strength,
                metadata={
                    'strategy': 'momentum',
                    'rsi': float(current_rsi),
                    'macd': float(current_macd),
                    'signal': float(current_signal),
                    'volatility': float(current_volatility),
                    'volume_ratio': float(current_volume / current_avg_volume) if current_avg_volume > 0 else 1.0
                }
            )
            signals.append(signal)
            self.logger.info(f"Signal ACHAT (momentum) généré: {signal}")
        
        # Signal de vente: RSI sort de la zone de surachat et croise à la baisse
        # OU croisement baissier du MACD avec confirmation de volume
        elif (
            (prev_rsi > rsi_overbought and current_rsi < rsi_overbought and current_rsi < prev_rsi) or
            (prev_macd >= prev_signal and current_macd < current_signal and volume_ok)
        ) and volatility_ok:
            # Calculer la force du signal
            rsi_strength = max(0, (current_rsi - rsi_overbought) / (100 - rsi_overbought)) if current_rsi > rsi_overbought else 0
            macd_strength = abs(macd_hist.iloc[-1]) / current_price * 1000  # Normalisation approximative
            strength = min(1.0, 0.5 + (rsi_strength * 0.3) + (macd_strength * 0.2))
            
            signal = self._create_signal(
                symbol=market_data.get('symbol', 'UNKNOWN'),
                action=SignalAction.SELL,
                price=current_price,
                confidence=strength,
                metadata={
                    'strategy': 'momentum',
                    'rsi': float(current_rsi),
                    'macd': float(current_macd),
                    'signal': float(current_signal),
                    'volatility': float(current_volatility),
                    'volume_ratio': float(current_volume / current_avg_volume) if current_avg_volume > 0 else 1.0
                }
            )
            signals.append(signal)
            self.logger.info(f"Signal VENTE (momentum) généré: {signal}")
        
        return signals
