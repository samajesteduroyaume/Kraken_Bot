from typing import TypedDict, Literal, Tuple
from decimal import Decimal

__all__ = ['Prediction', 'RiskProfile']

# Utilisation d'annotations forward pour éviter les imports circulaires


class Prediction(TypedDict):
    """Prédiction sur les mouvements futurs du prix."""
    price_prediction: Decimal
    confidence_score: float
    prediction_interval: Tuple[Decimal, Decimal]


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
