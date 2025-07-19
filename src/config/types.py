from typing import Dict, List, Any, TypedDict, Literal


class RSIConfig(TypedDict):
    """Configuration du RSI."""
    period: int
    overbought: int
    oversold: int


class MACDConfig(TypedDict):
    """Configuration du MACD."""
    fast: int
    slow: int
    signal: int


class BollingerConfig(TypedDict):
    """Configuration des Bandes de Bollinger."""
    period: int
    std_dev: float


class RiskProfile(TypedDict):
    """Profil de risque."""
    max_position_size: float
    stop_loss_percentage: float
    take_profit_percentage: float
    max_drawdown: float


class TradingPair(TypedDict):
    """Configuration d'une paire de trading."""
    symbol: str
    base_currency: str
    quote_currency: str
    min_size: float
    max_size: float
    step_size: float
    price_precision: int
    min_volume_btc: float
    max_spread: float
    risk_level: Literal['low', 'medium', 'high']
    risk_profile: RiskProfile
    enabled: bool


class Timeframe(TypedDict):
    """Configuration du timeframe."""
    interval: Literal['1m', '5m', '15m', '30m', '1h', '4h', '1d']
    history_days: int
    update_interval: float


class TradingConfig(TypedDict):
    """Configuration générale du trading."""
    enabled: bool
    profile: RiskProfile
    max_trade_amount: float
    strategy: str
    pairs: List[TradingPair]
    initial_balance: float
    timeframe: Timeframe
    rsi: RSIConfig
    macd: MACDConfig
    bollinger: BollingerConfig
    # Configuration spécifique à la gestion des risques
    risk_management: Dict[str, Any]
    # Configuration spécifique à l'exécution des trades
    execution: Dict[str, Any]
