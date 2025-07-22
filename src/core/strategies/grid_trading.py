"""
Stratégie de Grid Trading qui place des ordres d'achat et de vente à intervalles réguliers
au-dessus et en dessous du prix actuel, créant ainsi une grille de prix.
"""
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging

from .base_strategy import BaseStrategy
from .types import TradingSignal, SignalAction, StrategyType, MarketData, Indicators, StrategyConfig
from ..utils.technical_indicators import calculate_atr


class GridTradingStrategy(BaseStrategy):
    """
    Stratégie de Grid Trading qui place des ordres d'achat et de vente à intervalles réguliers
    autour du prix actuel, créant une grille de prix pour profiter des mouvements de marché.
    """
    
    def __init__(
        self,
        grid_levels: int = 5,
        grid_spacing_pct: float = 1.5,
        take_profit_pct: float = 1.0,
        stop_loss_pct: float = 2.0,
        position_size_pct: float = 10.0,
        max_drawdown_pct: float = 10.0,
        volatility_lookback: int = 20,
        volatility_threshold: float = 0.5,
        risk_multiplier: float = 1.0,
        **kwargs
    ):
        """
        Initialise la stratégie de Grid Trading.
        
        Args:
            grid_levels: Nombre de niveaux de grille (ordres d'achat/vente)
            grid_spacing_pct: Espacement entre les niveaux en pourcentage
            take_profit_pct: Take profit en pourcentage
            stop_loss_pct: Stop loss en pourcentage
            position_size_pct: Taille de position en pourcentage du capital par niveau
            max_drawdown_pct: Drawdown maximum autorisé en pourcentage
            volatility_lookback: Période de lookback pour le calcul de la volatilité
            volatility_threshold: Seuil minimum de volatilité pour activer la stratégie
            risk_multiplier: Multiplicateur de risque pour le position sizing
            **kwargs: Arguments additionnels pour la configuration
        """
        config = StrategyConfig(
            risk_multiplier=risk_multiplier,
            parameters={
                'grid_levels': grid_levels,
                'grid_spacing_pct': grid_spacing_pct,
                'take_profit_pct': take_profit_pct,
                'stop_loss_pct': stop_loss_pct,
                'position_size_pct': position_size_pct,
                'max_drawdown_pct': max_drawdown_pct,
                'volatility_lookback': volatility_lookback,
                'volatility_threshold': volatility_threshold
            }
        )
        
        super().__init__(
            name="GridTrading",
            description="Stratégie de Grid Trading avec gestion dynamique des niveaux",
            strategy_type=StrategyType.GRID,
            config=config
        )
        
        # État de la grille
        self.grid_levels = grid_levels
        self.grid_spacing_pct = grid_spacing_pct / 100  # Conversion en décimal
        self.take_profit_pct = take_profit_pct / 100
        self.stop_loss_pct = stop_loss_pct / 100
        self.position_size_pct = position_size_pct / 100
        self.max_drawdown_pct = max_drawdown_pct / 100
        self.volatility_lookback = volatility_lookback
        self.volatility_threshold = volatility_threshold / 100
        
        # Suivi des positions et des niveaux de grille
        self.current_grid = []
        self.last_grid_update = None
        self.current_price = 0.0
        self.volatility = 0.0
        
        self.logger = logging.getLogger(__name__)
    
    async def calculate_indicators(self, market_data: MarketData) -> Indicators:
        """
        Calcule les indicateurs techniques nécessaires pour la stratégie.
        
        Args:
            market_data: Données de marché (OHLCV)
            
        Returns:
            Dictionnaire d'indicateurs techniques
        """
        close_prices = market_data.close
        
        # Calcul de la volatilité (ATR)
        atr = calculate_atr(
            high=market_data.high,
            low=market_data.low,
            close=close_prices,
            period=self.volatility_lookback
        )
        
        # Dernier prix de clôture
        self.current_price = close_prices.iloc[-1] if len(close_prices) > 0 else 0.0
        
        # Volatilité normalisée (ATR / Prix)
        self.volatility = (atr.iloc[-1] / self.current_price) if self.current_price > 0 else 0.0
        
        return {
            'atr': atr,
            'volatility': self.volatility,
            'current_price': self.current_price
        }
    
    def _generate_grid_levels(self, current_price: float) -> List[Dict]:
        """
        Génère les niveaux de la grille autour du prix actuel.
        
        Args:
            current_price: Prix actuel du marché
            
        Returns:
            Liste des niveaux de grille avec leurs paramètres
        """
        if current_price <= 0:
            return []
            
        grid = []
        
        # Niveaux d'achat (en dessous du prix actuel)
        for i in range(1, self.grid_levels + 1):
            level_price = current_price * (1 - i * self.grid_spacing_pct)
            if level_price <= 0:
                continue
                
            grid.append({
                'type': 'buy',
                'level': i,
                'price': level_price,
                'take_profit': level_price * (1 + self.take_profit_pct),
                'stop_loss': level_price * (1 - self.stop_loss_pct),
                'position_size': self.position_size_pct,
                'active': True
            })
        
        # Niveaux de vente (au-dessus du prix actuel)
        for i in range(1, self.grid_levels + 1):
            level_price = current_price * (1 + i * self.grid_spacing_pct)
            
            grid.append({
                'type': 'sell',
                'level': i,
                'price': level_price,
                'take_profit': level_price * (1 - self.take_profit_pct),
                'stop_loss': level_price * (1 + self.stop_loss_pct),
                'position_size': self.position_size_pct,
                'active': True
            })
        
        # Trier la grille par prix (du plus bas au plus haut)
        grid.sort(key=lambda x: x['price'])
        return grid
    
    async def analyze(
        self, 
        market_data: MarketData,
        indicators: Optional[Indicators] = None
    ) -> List[TradingSignal]:
        """
        Analyse les données de marché et génère des signaux de trading basés sur la grille.
        
        Args:
            market_data: Données de marché (OHLCV)
            indicators: Indicateurs techniques précalculés (optionnel)
            
        Returns:
            Liste de signaux de trading
        """
        if indicators is None:
            indicators = await self.calculate_indicators(market_data)
        
        signals = []
        
        # Vérifier si la volatilité est suffisante pour trader
        if self.volatility < self.volatility_threshold:
            self.logger.info(f"Volatilité trop faible ({self.volatility:.2%} < {self.volatility_threshold:.2%}) - Stratégie inactive")
            return signals
        
        # Mettre à jour la grille si nécessaire (premier appel ou si le prix a significativement changé)
        if not self.current_grid or not self.last_grid_update or \
           (datetime.utcnow() - self.last_grid_update) > timedelta(hours=1):
            
            self.current_grid = self._generate_grid_levels(self.current_price)
            self.last_grid_update = datetime.utcnow()
            self.logger.info(f"Grille mise à jour autour du prix {self.current_price:.2f}")
        
        # Vérifier les niveaux de grille déclenchés
        for level in self.current_grid:
            if not level['active']:
                continue
                
            # Vérifier si le prix a atteint un niveau d'achat
            if level['type'] == 'buy' and self.current_price <= level['price']:
                signals.append(TradingSignal(
                    symbol=market_data.symbol,
                    action=SignalAction.BUY,
                    price=level['price'],
                    take_profit=level['take_profit'],
                    stop_loss=level['stop_loss'],
                    size=level['position_size'],
                    strategy=self.name,
                    timestamp=datetime.utcnow(),
                    metadata={
                        'grid_level': level['level'],
                        'grid_type': level['type'],
                        'volatility': self.volatility
                    }
                ))
                
                # Désactiver ce niveau jusqu'à la prochaine mise à jour de la grille
                level['active'] = False
                
            # Vérifier si le prix a atteint un niveau de vente
            elif level['type'] == 'sell' and self.current_price >= level['price']:
                signals.append(TradingSignal(
                    symbol=market_data.symbol,
                    action=SignalAction.SELL,
                    price=level['price'],
                    take_profit=level['take_profit'],
                    stop_loss=level['stop_loss'],
                    size=level['position_size'],
                    strategy=self.name,
                    timestamp=datetime.utcnow(),
                    metadata={
                        'grid_level': level['level'],
                        'grid_type': level['type'],
                        'volatility': self.volatility
                    }
                ))
                
                # Désactiver ce niveau jusqu'à la prochaine mise à jour de la grille
                level['active'] = False
        
        return signals
