"""
Configuration des profils de risque pour le trading.

Chaque profil contient des paramètres spécifiques adaptés à différents niveaux de tolérance au risque.
"""
from typing import Dict, List, Literal, TypedDict


class RiskParameters(TypedDict):
    """Paramètres de base du profil de risque."""
    risk_per_trade: float
    max_portfolio_risk: float
    max_drawdown: float
    take_profit: float
    stop_loss: float
    max_leverage: float
    leverage_strategy: Literal['conservative', 'moderate', 'aggressive']
    max_position_size: float
    max_open_positions: int
    min_confidence: float
    max_volatility: float
    max_trades_per_day: int
    trailing_stop: float
    enable_hedging: bool
    enable_short_selling: bool
    enable_margin: bool


class RiskProfileConfig(TypedDict):
    """Configuration complète d'un profil de risque."""
    name: str
    description: str
    risk_level: Literal['low', 'medium', 'high']
    parameters: RiskParameters
    allowed_strategies: List[str]
    timeframes: List[str]


# Profil Conservateur - Faible risque, rendements modérés
CONSERVATIVE: RiskProfileConfig = {
    "name": "conservative",
    "description": "Faible risque, rendements modérés",
    "risk_level": "low",
    "parameters": {
        "risk_per_trade": 0.01,
        "max_portfolio_risk": 0.1,
        "max_drawdown": 0.05,
        "take_profit": 0.03,
        "stop_loss": 0.01,
        "max_leverage": 1,
        "leverage_strategy": "conservative",
        "max_position_size": 0.1,
        "max_open_positions": 3,
        "min_confidence": 0.7,
        "max_volatility": 0.3,
        "max_trades_per_day": 2,
        "trailing_stop": 0.005,
        "enable_hedging": False,
        "enable_short_selling": False,
        "enable_margin": False
    },
    "allowed_strategies": ["mean_reversion", "swing"],
    "timeframes": ["4h", "1d"]
}

# Profil Modéré - Équilibre entre risque et rendement
MODERATE: RiskProfileConfig = {
    "name": "moderate",
    "description": "Équilibre entre risque et rendement",
    "risk_level": "medium",
    "parameters": {
        "risk_per_trade": 0.02,
        "max_portfolio_risk": 0.2,
        "max_drawdown": 0.1,
        "take_profit": 0.05,
        "stop_loss": 0.02,
        "max_leverage": 3,
        "leverage_strategy": "moderate",
        "max_position_size": 0.15,
        "max_open_positions": 5,
        "min_confidence": 0.6,
        "max_volatility": 0.4,
        "max_trades_per_day": 5,
        "trailing_stop": 0.01,
        "enable_hedging": True,
        "enable_short_selling": True,
        "enable_margin": True
    },
    "allowed_strategies": ["momentum", "mean_reversion", "breakout"],
    "timeframes": ["1h", "4h", "1d"]
}

# Profil Agressif - Haut risque, potentiel de rendement élevé
AGGRESSIVE: RiskProfileConfig = {
    "name": "aggressive",
    "description": "Haut risque, potentiel de rendement élevé",
    "risk_level": "high",
    "parameters": {
        "risk_per_trade": 0.05,
        "max_portfolio_risk": 0.4,
        "max_drawdown": 0.2,
        "take_profit": 0.1,
        "stop_loss": 0.03,
        "max_leverage": 5,
        "leverage_strategy": "aggressive",
        "max_position_size": 0.25,
        "max_open_positions": 10,
        "min_confidence": 0.5,
        "max_volatility": 0.7,
        "max_trades_per_day": 20,
        "trailing_stop": 0.02,
        "enable_hedging": True,
        "enable_short_selling": True,
        "enable_margin": True
    },
    "allowed_strategies": ["momentum", "breakout", "grid", "scalping"],
    "timeframes": ["5m", "15m", "1h"]
}

# Dictionnaire des profils disponibles
RISK_PROFILES: Dict[str, RiskProfileConfig] = {
    "conservative": CONSERVATIVE,
    "moderate": MODERATE,
    "aggressive": AGGRESSIVE
}


def get_risk_profile(profile_name: str = "moderate") -> RiskProfileConfig:
    """
    Récupère la configuration du profil de risque spécifié.

    Args:
        profile_name: Nom du profil ('conservative', 'moderate', 'aggressive')

    Returns:
        RiskProfileConfig: Configuration du profil de risque
    """
    return RISK_PROFILES.get(profile_name.lower(), MODERATE)
