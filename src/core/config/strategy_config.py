"""
Configuration des stratégies de trading.

Ce module définit les modèles de configuration pour les différentes stratégies
de trading, avec validation des paramètres et valeurs par défaut.
"""
from typing import Dict, Any, Optional, List, Union, ClassVar
from enum import Enum
from pydantic import BaseModel, Field, field_validator, model_validator, ConfigDict, field_validator
from typing import Tuple
import numpy as np


class StrategyType(str, Enum):
    """Types de stratégies de trading supportées."""
    TREND_FOLLOWING = "trend_following"
    MEAN_REVERSION = "mean_reversion"
    MOMENTUM = "momentum"
    BREAKOUT = "breakout"
    ML = "machine_learning"
    GRID = "grid_trading"


class BaseStrategyConfig(BaseModel):
    """Configuration de base pour toutes les stratégies."""
    enabled: bool = True
    weight: float = Field(1.0, ge=0.0, le=1.0, description="Poids de la stratégie dans la combinaison")
    risk_multiplier: float = Field(1.0, gt=0.0, description="Multiplicateur de risque pour le position sizing")
    symbols: List[str] = Field(default_factory=list, description="Liste des symboles à trader")
    timeframe: str = Field("1d", description="Intervalle de temps pour les données (ex: 1m, 5m, 1h, 1d)")
    
    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True
    )
    
    @field_validator('timeframe')
    @classmethod
    def validate_timeframe(cls, v: str) -> str:
        """Valide le format de l'intervalle de temps."""
        if not v:
            raise ValueError("L'intervalle de temps ne peut pas être vide")
        if not v[-1] in ['m', 'h', 'd', 'w', 'M']:
            raise ValueError("Format d'intervalle invalide. Utilisez m (minutes), h (heures), d (jours), w (semaines) ou M (mois)")
        try:
            int(v[:-1])
        except ValueError:
            raise ValueError("Le préfixe de l'intervalle doit être un nombre")
        return v


class TrendFollowingConfig(BaseStrategyConfig):
    """Configuration pour la stratégie de suivi de tendance."""
    fast_ma: int = Field(10, gt=1, description="Période de la moyenne mobile rapide")
    slow_ma: int = Field(50, gt=1, description="Période de la moyenne mobile lente")
    ma_type: str = Field("sma", description="Type de moyenne mobile (sma ou ema)")
    min_trend_strength: float = Field(0.3, ge=0.0, le=1.0, description="Force minimale de la tendance pour générer un signal (0-1)")
    
    @model_validator(mode='after')
    def validate_moving_averages(self):
        """Valide que la période rapide est inférieure à la période lente."""
        if self.fast_ma >= self.slow_ma:
            raise ValueError("La période de la moyenne mobile rapide doit être inférieure à celle de la lente")
        return self
    
    @field_validator('ma_type')
    @classmethod
    def validate_ma_type(cls, v: str) -> str:
        """Valide le type de moyenne mobile."""
        if v.lower() not in ['sma', 'ema']:
            raise ValueError("Le type de moyenne mobile doit être 'sma' ou 'ema'")
        return v.lower()


class MeanReversionConfig(BaseStrategyConfig):
    """Configuration pour la stratégie de réversion à la moyenne."""
    lookback: int = Field(20, gt=1, description="Période de lookback pour le calcul de la moyenne et écart-type")
    std_multiplier: float = Field(2.0, gt=0.0, description="Multiplicateur de l'écart-type pour les bandes")
    entry_threshold: float = Field(2.0, gt=0.0, description="Seuil d'entrée en écarts-types")
    ma_type: str = Field("sma", description="Type de moyenne mobile (sma ou ema)")
    
    @field_validator('ma_type')
    @classmethod
    def validate_ma_type(cls, v: str) -> str:
        """Valide le type de moyenne mobile."""
        if v.lower() not in ['sma', 'ema']:
            raise ValueError("Le type de moyenne mobile doit être 'sma' ou 'ema'")
        return v.lower()


class MomentumConfig(BaseStrategyConfig):
    """Configuration pour la stratégie de momentum."""
    rsi_period: int = Field(14, gt=1, description="Période pour le calcul du RSI")
    rsi_overbought: float = Field(70.0, gt=0, lt=100, description="Seuil de surachat pour le RSI")
    rsi_oversold: float = Field(30.0, gt=0, lt=100, description="Seuil de survente pour le RSI")
    macd_fast: int = Field(12, gt=1, description="Période rapide pour le MACD")
    macd_slow: int = Field(26, gt=1, description="Période lente pour le MACD")
    macd_signal: int = Field(9, gt=1, description="Période du signal pour le MACD")
    volatility_filter: float = Field(1.5, gt=0.0, description="Seuil de volatilité pour filtrer les signaux faibles")
    
    @model_validator(mode='after')
    def validate_rsi_thresholds(self):
        """Valide que le seuil de survente est inférieur au seuil de surachat."""
        if self.rsi_oversold >= self.rsi_overbought:
            raise ValueError("Le seuil de survente doit être inférieur au seuil de surachat")
        return self
    
    @model_validator(mode='after')
    def validate_macd_periods(self):
        """Valide que les périodes MACD sont cohérentes."""
        if self.macd_fast >= self.macd_slow:
            raise ValueError("La période rapide doit être inférieure à la période lente pour le MACD")
            
        if self.macd_signal >= self.macd_slow:
            raise ValueError("La période du signal doit être inférieure à la période lente")
            
        return self


class GridTradingConfig(BaseStrategyConfig):
    """Configuration pour la stratégie de Grid Trading."""
    grid_levels: int = Field(5, gt=1, le=20, description="Nombre de niveaux de grille (ordres d'achat/vente)")
    grid_spacing_pct: float = Field(1.5, gt=0.1, le=10.0, description="Espacement entre les niveaux en pourcentage")
    take_profit_pct: float = Field(1.0, gt=0.1, le=10.0, description="Take profit en pourcentage")
    stop_loss_pct: float = Field(2.0, gt=0.1, le=20.0, description="Stop loss en pourcentage")
    position_size_pct: float = Field(10.0, gt=0.1, le=50.0, description="Taille de position en pourcentage du capital par niveau")
    max_drawdown_pct: float = Field(10.0, gt=0.1, le=50.0, description="Drawdown maximum autorisé en pourcentage")
    volatility_lookback: int = Field(20, gt=5, description="Période de lookback pour le calcul de la volatilité")
    volatility_threshold: float = Field(0.5, gt=0.0, description="Seuil minimum de volatilité pour activer la stratégie")
    
    @model_validator(mode='after')
    def validate_grid_parameters(self):
        """Valide la cohérence des paramètres de la grille."""
        if self.take_profit_pct >= self.grid_spacing_pct:
            raise ValueError("Le take profit doit être inférieur à l'espacement de la grille")
            
        if self.stop_loss_pct <= self.grid_spacing_pct * 1.5:
            raise ValueError("Le stop loss doit être au moins 1,5 fois l'espacement de la grille")
            
        return self
