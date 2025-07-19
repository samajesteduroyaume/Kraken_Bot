from typing import Dict, List, Optional, TypedDict, Literal, Union, Tuple
from decimal import Decimal
from datetime import datetime
from typing import TYPE_CHECKING

__all__ = [
    'Portfolio',
    'Position',
    'Order',
    'Trade',
    'TradeSignal',
    'TradingMetrics',
    'TradingContext',
    'TradingConfig',
    'StrategyConfig',
    'OrderParams',
    'RiskProfile']

if TYPE_CHECKING:
    pass

# Types de trading


class OrderParams(TypedDict):
    """Paramètres d'ordre."""
    symbol: str
    side: Literal['buy', 'sell']
    type: Literal['market', 'limit', 'stop']
    amount: Decimal
    price: Optional[Decimal]
    leverage: Decimal
    reduce_only: bool
    post_only: bool
    time_in_force: Literal['GTC', 'IOC', 'FOK']


class Order(TypedDict):
    """Ordre de trading."""
    id: str
    symbol: str
    side: Literal['buy', 'sell']
    type: Literal['market', 'limit', 'stop']
    status: Literal['open', 'filled', 'canceled', 'rejected']
    price: Decimal
    amount: Decimal
    filled: Decimal
    remaining: Decimal
    timestamp: datetime
    fee: Decimal
    reduce_only: bool
    post_only: bool
    time_in_force: Literal['GTC', 'IOC', 'FOK']


class Trade(TypedDict):
    """Trade exécuté."""
    id: str
    order_id: str
    symbol: str
    side: Literal['buy', 'sell']
    price: Decimal
    amount: Decimal
    fee: Decimal
    timestamp: datetime
    position_side: Literal['long', 'short']


class Position(TypedDict):
    """Position de trading."""
    symbol: str
    side: Literal['long', 'short']
    entry_price: Decimal
    current_price: Decimal
    amount: Decimal
    unrealized_pnl: Decimal
    leverage: Decimal
    liquidation_price: Decimal
    timestamp: datetime
    stop_loss: Decimal
    take_profit: Decimal


class Portfolio(TypedDict):
    """Portefeuille de trading."""
    total_value: Decimal
    cash: Decimal
    # Utilisation d'une annotation forward pour Position
    positions: List['Position']
    realized_pnl: Decimal
    unrealized_pnl: Decimal
    timestamp: datetime


class TradeSignal(TypedDict, total=False):
    """Signal de trading."""
    symbol: str
    action: Literal['buy', 'sell', 'hold']
    price: Decimal
    amount: Decimal
    confidence: float
    reason: str
    timestamp: datetime
    risk_percentage: float  # Ajouté, facultatif


class Prediction(TypedDict):
    """Prédiction sur les mouvements futurs du prix."""
    price_prediction: Decimal
    confidence_score: float
    prediction_interval: Tuple[Decimal, Decimal]


class TradingMetrics(TypedDict):
    """Métriques de performance de trading."""
    total_trades: int
    win_rate: float
    avg_profit: Decimal
    avg_loss: Decimal
    max_drawdown: Decimal
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    profit_factor: float
    roi: Decimal


class TradingContext(TypedDict):
    """Contexte de trading."""
    portfolio: 'Portfolio'  # Utilisation d'une annotation forward pour Portfolio
    # Utilisation d'une annotation forward pour Position
    positions: List['Position']
    signals: List[TradeSignal]
    orders: List[Order]
    trades: List[Trade]
    metrics: TradingMetrics
    timestamp: datetime


class TradingConfig(TypedDict):
    """Configuration de trading."""
    enabled: bool
    strategy: str
    risk_profile: str
    max_leverage: Decimal
    max_position_size: Decimal
    max_open_positions: int
    position_adjustment: Literal['static', 'dynamic']
    trailing_stop: bool
    trailing_stop_distance: Decimal
    trailing_stop_activation: Decimal
    max_order_retry: int
    order_retry_delay: float
    max_slippage: Decimal
    max_latency: float


class RiskProfile(TypedDict):
    """Profil de risque pour le trading."""
    risk_level: Literal['conservative', 'moderate', 'aggressive']
    max_position_size: Decimal
    max_leverage: Decimal
    stop_loss: Decimal
    take_profit: Decimal
    max_drawdown: Decimal
    position_adjustment: Literal['static', 'dynamic']
    trailing_stop: bool
    trailing_stop_distance: Decimal
    max_trade_risk: Decimal
    max_open_positions: int
    risk_per_trade: Decimal
    min_position_size: Decimal
    risk_reward_ratio: Decimal
    max_correlation: Decimal


class StrategyConfig(TypedDict):
    """Configuration spécifique à la stratégie."""
    name: str
    parameters: Dict[str, Union[float, int, str]]
    timeframe: str
    risk_per_trade: Decimal
    max_drawdown: Decimal
    stop_loss: Decimal
    take_profit: Decimal
    trailing_stop: bool
    trailing_distance: Decimal
    max_positions: int
    cooldown_period: int
    use_ml: bool
    model_path: Optional[str]
