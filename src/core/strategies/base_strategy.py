"""
Classe de base abstraite pour toutes les stratégies de trading.
"""
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, AsyncContextManager
import asyncio
import pandas as pd
import numpy as np
import logging
import time
from datetime import datetime
from contextlib import asynccontextmanager

from ..config import (
    BaseStrategyConfig,
    StrategyType,
    StrategyConfig
)

from .types import (
    TradingSignal, 
    SignalAction,
    MarketData,
    Indicators
)

class BaseStrategy(ABC):
    """
    Classe de base abstraite pour toutes les stratégies de trading.
    
    Attributs:
        name: Nom de la stratégie
        description: Description courte de la stratégie
        strategy_type: Type de stratégie (TREND_FOLLOWING, MEAN_REVERSION, etc.)
        config: Configuration de la stratégie
        logger: Logger pour la journalisation
    """
    
    def __init__(
        self, 
        name: str, 
        description: str, 
        strategy_type: StrategyType,
        config: Optional[BaseStrategyConfig] = None
    ):
        """
        Initialise la stratégie de base.
        
        Args:
            name: Nom de la stratégie
            description: Description de la stratégie
            strategy_type: Type de stratégie (TREND_FOLLOWING, MEAN_REVERSION, etc.)
            config: Configuration de la stratégie (optionnel)
        """
        self.name = name
        self.description = description
        self.strategy_type = strategy_type
        self.config = config or BaseStrategyConfig()
        self.logger = logging.getLogger(f"strategy.{name.lower()}")
        self._last_signal: Optional[TradingSignal] = None
        self._performance_metrics: Dict[str, float] = {}
        self._lock = asyncio.Lock()  # Verrou pour les opérations critiques
        self._state: Dict[str, Any] = {
            'is_active': True,
            'last_updated': datetime.utcnow(),
            'error_count': 0,
            'last_error': None,
            'metrics': {}
        }
    
    @abstractmethod
    async def analyze(
        self, 
        market_data: MarketData,
        indicators: Optional[Indicators] = None
    ) -> List[TradingSignal]:
        """
        Analyse les données de marché et génère des signaux de trading.
        
        Args:
            market_data: Données de marché (OHLCV, etc.)
            indicators: Indicateurs techniques précalculés (optionnel)
            
        Returns:
            Liste de signaux de trading générés par la stratégie
        """
        pass
    
    async def calculate_indicators(
        self, 
        market_data: MarketData
    ) -> Indicators:
        """
        Calcule les indicateurs techniques nécessaires à la stratégie.
        
        Args:
            market_data: Données de marché brutes
            
        Returns:
            Dictionnaire d'indicateurs techniques
        """
        return {}
    
    @asynccontextmanager
    async def safe_execution(self) -> AsyncContextManager[None]:
        """
        Contexte pour exécuter des opérations critiques de manière thread-safe.
        
        Exemple d'utilisation:
            async with strategy.safe_execution():
                # Code critique
                await strategy.update_state(...)
        """
        async with self._lock:
            try:
                yield
            except Exception as e:
                self._handle_error(e)
                raise
    
    def _handle_error(self, error: Exception) -> None:
        """Gère les erreurs de manière centralisée."""
        self._state['error_count'] += 1
        self._state['last_error'] = {
            'timestamp': datetime.utcnow(),
            'type': type(error).__name__,
            'message': str(error)
        }
        self.logger.error(
            f"Erreur dans la stratégie {self.name}: {error}", 
            exc_info=True
        )
    
    async def update_state(self, updates: Dict[str, Any]) -> None:
        """
        Met à jour l'état de la stratégie de manière thread-safe.
        
        Args:
            updates: Dictionnaire des mises à jour d'état
        """
        async with self._lock:
            self._state.update(updates)
            self._state['last_updated'] = datetime.utcnow()
    
    def get_state(self) -> Dict[str, Any]:
        """
        Récupère une copie de l'état actuel de la stratégie.
        
        Returns:
            Dictionnaire contenant l'état de la stratégie
        """
        return self._state.copy()
    
    async def update_config(self, config_updates: Dict[str, Any]) -> None:
        """
        Met à jour la configuration de la stratégie de manière thread-safe.
        
        Args:
            config_updates: Dictionnaire des mises à jour de configuration
        """
        async with self._lock:
            # Créer une nouvelle instance de configuration avec les mises à jour
            updated_config = self.config.copy(update=config_updates)
            
            # Valider la configuration mise à jour
            try:
                self.config = self.config.__class__.parse_obj(updated_config.dict())
            except Exception as e:
                self.logger.error(f"Erreur lors de la mise à jour de la configuration: {e}")
                raise
    
    def get_performance_metrics(self) -> Dict[str, float]:
        """
        Retourne les métriques de performance de la stratégie.
        
        Returns:
            Dictionnaire des métriques de performance
        """
        return self._performance_metrics.copy()
    
    def _create_signal(
        self,
        symbol: str,
        action: SignalAction,
        price: float,
        confidence: float,
        metadata: Optional[Dict[str, Any]] = None
    ) -> TradingSignal:
        """
        Crée un signal de trading standardisé.
        
        Args:
            symbol: Symbole de l'actif
            action: Action à entreprendre (BUY/SELL/HOLD)
            price: Prix actuel
            confidence: Niveau de confiance (0.0 à 1.0)
            metadata: Métadonnées supplémentaires (optionnel)
            
        Returns:
            Un objet TradingSignal
        """
        signal = TradingSignal(
            symbol=symbol,
            action=action,
            price=price,
            confidence=confidence,
            strategy=self.name,
            metadata=metadata or {}
        )
        self._last_signal = signal
        return signal
    
    def _calculate_rsi(
        self, 
        prices: pd.Series, 
        period: int = 14
    ) -> pd.Series:
        """
        Calcule le Relative Strength Index (RSI).
        
        Args:
            prices: Série des prix de clôture
            period: Période du RSI (par défaut: 14)
            
        Returns:
            Série des valeurs du RSI
        """
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _calculate_ma(
        self, 
        prices: pd.Series, 
        window: int, 
        ma_type: str = 'sma'
    ) -> pd.Series:
        """
        Calcule une moyenne mobile.
        
        Args:
            prices: Série de prix
            window: Taille de la fenêtre
            ma_type: Type de moyenne ('sma' ou 'ema')
            
        Returns:
            Série des valeurs de la moyenne mobile
        """
        if ma_type.lower() == 'sma':
            return prices.rolling(window=window).mean()
        elif ma_type.lower() == 'ema':
            return prices.ewm(span=window, adjust=False).mean()
        else:
            raise ValueError(f"Type de moyenne mobile non supporté: {ma_type}")
    
    def _calculate_atr(
        self, 
        high: pd.Series, 
        low: pd.Series, 
        close: pd.Series, 
        period: int = 14
    ) -> pd.Series:
        """
        Calcule l'Average True Range (ATR).
        
        Args:
            high: Série des prix hauts
            low: Série des prix bas
            close: Série des prix de clôture
            period: Période de l'ATR (par défaut: 14)
            
        Returns:
            Série des valeurs de l'ATR
        """
        tr1 = high - low
        tr2 = (high - close.shift()).abs()
        tr3 = (low - close.shift()).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.ewm(alpha=1/period, adjust=False).mean()
        return atr
