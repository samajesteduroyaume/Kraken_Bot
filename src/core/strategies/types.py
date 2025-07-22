"""
Types et constantes pour les stratégies de trading.
"""
from enum import Enum, auto
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal


class SignalAction(Enum):
    """Actions possibles pour un signal de trading."""
    BUY = auto()
    SELL = auto()
    HOLD = auto()


@dataclass
class TradingSignal:
    """Représente un signal de trading standardisé."""
    symbol: str
    action: SignalAction
    price: Decimal
    confidence: float  # Entre 0.0 et 1.0
    strategy: str
    timestamp: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convertit le signal en dictionnaire pour la sérialisation."""
        return {
            'symbol': self.symbol,
            'action': self.action.name,
            'price': str(self.price),
            'confidence': self.confidence,
            'strategy': self.strategy,
            'timestamp': self.timestamp.isoformat(),
            'metadata': self.metadata
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TradingSignal':
        """Crée un signal à partir d'un dictionnaire."""
        return cls(
            symbol=data['symbol'],
            action=SignalAction[data['action']],
            price=Decimal(data['price']),
            confidence=data['confidence'],
            strategy=data['strategy'],
            timestamp=datetime.fromisoformat(data['timestamp']),
            metadata=data.get('metadata', {})
        )


class StrategyType(Enum):
    """Types de stratégies disponibles."""
    TREND_FOLLOWING = "trend_following"
    MEAN_REVERSION = "mean_reversion"
    MOMENTUM = "momentum"
    VOLATILITY = "volatility"
    MACHINE_LEARNING = "machine_learning"
    GRID = "grid"
    SWING = "swing"
    BREAKOUT = "breakout"


@dataclass
class StrategyConfig:
    """Configuration de base pour une stratégie."""
    enabled: bool = True
    risk_multiplier: float = 1.0
    max_position_size: float = 0.1  # % du capital
    parameters: Dict[str, Any] = field(default_factory=dict)


# Types pour les données de marché
MarketData = Dict[str, Any]
Indicators = Dict[str, Any]
