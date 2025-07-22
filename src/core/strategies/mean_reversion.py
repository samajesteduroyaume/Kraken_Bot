"""
Stratégie de réversion à la moyenne basée sur les écarts par rapport à la moyenne mobile.
"""
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
from datetime import datetime

from .base_strategy import BaseStrategy
from .types import TradingSignal, SignalAction, StrategyType, MarketData, Indicators, StrategyConfig


class MeanReversionStrategy(BaseStrategy):
    """
    Stratégie de réversion à la moyenne qui identifie les écarts extrêmes
    par rapport à une moyenne mobile pour générer des signaux de trading.
    """
    
    def __init__(
        self, 
        lookback: int = 20,
        std_multiplier: float = 2.0,
        entry_threshold: float = 2.0,
        ma_type: str = 'sma',
        risk_multiplier: float = 1.2,
        **kwargs
    ):
        """
        Initialise la stratégie de réversion à la moyenne.
        
        Args:
            lookback: Période de lookback pour le calcul de la moyenne et l'écart-type
            std_multiplier: Multiplicateur de l'écart-type pour les bandes de Bollinger
            entry_threshold: Seuil d'entrée en écarts-types
            ma_type: Type de moyenne mobile ('sma' ou 'ema')
            risk_multiplier: Multiplicateur de risque pour le position sizing
            **kwargs: Arguments additionnels pour la configuration
        """
        config = StrategyConfig(
            risk_multiplier=risk_multiplier,
            parameters={
                'lookback': lookback,
                'std_multiplier': std_multiplier,
                'entry_threshold': entry_threshold,
                'ma_type': ma_type
            }
        )
        
        super().__init__(
            name="MeanReversion",
            description="Stratégie de réversion à la moyenne utilisant les bandes de Bollinger",
            strategy_type=StrategyType.MEAN_REVERSION,
            config=config
        )
        
        # Validation des paramètres
        if lookback < 5:
            raise ValueError("La période de lookback doit être d'au moins 5")
        if std_multiplier <= 0:
            raise ValueError("Le multiplicateur d'écart-type doit être positif")
        if entry_threshold <= 0:
            raise ValueError("Le seuil d'entrée doit être positif")
    
    async def calculate_indicators(self, market_data: MarketData) -> Indicators:
        """
        Calcule les indicateurs nécessaires pour la stratégie.
        
        Args:
            market_data: Données de marché (doit contenir 'close')
            
        Returns:
            Dictionnaire contenant les indicateurs calculés
        """
        if 'close' not in market_data:
            raise ValueError("Les données de marché doivent contenir les prix de clôture ('close')")
        
        close_prices = pd.Series(market_data['close'])
        lookback = self.config.parameters.get('lookback', 20)
        std_multiplier = self.config.parameters.get('std_multiplier', 2.0)
        ma_type = self.config.parameters.get('ma_type', 'sma')
        
        # Calcul de la moyenne mobile et de l'écart-type
        ma = self._calculate_ma(close_prices, lookback, ma_type)
        std = close_prices.rolling(window=lookback).std()
        
        # Calcul des bandes de Bollinger
        upper_band = ma + (std * std_multiplier)
        lower_band = ma - (std * std_multiplier)
        
        # Distance par rapport à la moyenne en écarts-types
        z_score = (close_prices - ma) / std
        
        indicators = {
            'ma': ma,
            'upper_band': upper_band,
            'lower_band': lower_band,
            'z_score': z_score,
            'close': close_prices
        }
        
        return indicators
    
    async def analyze(
        self, 
        market_data: MarketData,
        indicators: Optional[Indicators] = None
    ) -> List[TradingSignal]:
        """
        Analyse les données de marché et génère des signaux de trading.
        
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
        z_score = indicators['z_score']
        close_prices = indicators['close']
        upper_band = indicators['upper_band']
        lower_band = indicators['lower_band']
        
        # Dernières valeurs
        current_z = z_score.iloc[-1]
        prev_z = z_score.iloc[-2] if len(z_score) > 1 else 0
        current_price = close_prices.iloc[-1]
        current_upper = upper_band.iloc[-1]
        current_lower = lower_band.iloc[-1]
        
        # Seuil d'entrée en écarts-types
        entry_threshold = self.config.parameters.get('entry_threshold', 2.0)
        
        signals = []
        
        # Signal d'achat: le prix est sous la bande inférieure et commence à remonter
        if (current_z <= -entry_threshold and 
            current_z > prev_z and 
            current_price < current_lower):
            
            # Calculer la force du signal basée sur l'écart par rapport à la bande
            strength = min(1.0, abs(current_z) / entry_threshold * 0.5)  # Normalisé entre 0.5 et 1.0
            
            signal = self._create_signal(
                symbol=market_data.get('symbol', 'UNKNOWN'),
                action=SignalAction.BUY,
                price=current_price,
                confidence=strength,
                metadata={
                    'strategy': 'mean_reversion',
                    'z_score': float(current_z),
                    'price': float(current_price),
                    'lower_band': float(current_lower),
                    'upper_band': float(current_upper)
                }
            )
            signals.append(signal)
            self.logger.info(f"Signal ACHAT (réversion) généré: {signal}")
        
        # Signal de vente: le prix est au-dessus de la bande supérieure et commence à baisser
        elif (current_z >= entry_threshold and 
              current_z < prev_z and 
              current_price > current_upper):
            
            # Calculer la force du signal
            strength = min(1.0, abs(current_z) / entry_threshold * 0.5)  # Normalisé entre 0.5 et 1.0
            
            signal = self._create_signal(
                symbol=market_data.get('symbol', 'UNKNOWN'),
                action=SignalAction.SELL,
                price=current_price,
                confidence=strength,
                metadata={
                    'strategy': 'mean_reversion',
                    'z_score': float(current_z),
                    'price': float(current_price),
                    'lower_band': float(current_lower),
                    'upper_band': float(current_upper)
                }
            )
            signals.append(signal)
            self.logger.info(f"Signal VENTE (réversion) généré: {signal}")
        
        return signals
