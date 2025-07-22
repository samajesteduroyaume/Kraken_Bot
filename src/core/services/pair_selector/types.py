"""
Définition des types pour la gestion des paires de trading.
"""
import warnings
from typing import Dict, List, Literal, Optional, TypedDict, Union, cast
from typing_extensions import NotRequired

# Import de la version robuste de normalize_pair_input
from src.utils.pair_utils import normalize_pair_input as robust_normalize_pair_input

class TradingPairBase(TypedDict):
    """Structure de base pour une paire de trading."""
    pair: str
    base_currency: str
    quote_currency: str
    min_size: float
    max_size: float
    step_size: float
    price_precision: int
    min_volume_btc: float
    max_spread: float
    risk_level: Literal['low', 'medium', 'high']
    enabled: bool

class TradingPairMetrics(TypedDict):
    """Métriques pour une paire de trading."""
    volume_24h: float
    liquidity: float
    spread: float
    volatility: float
    score: float
    last_updated: float

class TradingPair(TradingPairBase, TradingPairMetrics):
    """Paire de trading complète avec métriques."""
    # Hérite de tous les champs de TradingPairBase et TradingPairMetrics
    pass

PairInput = Union[str, Dict[str, any], TradingPair]
"""Type pour les entrées de paires qui peuvent être un string, un dict ou un TradingPair"""

def normalize_pair_input(pair_input: PairInput) -> str:
    """Déprécié: Utilisez plutôt src.utils.pair_utils.normalize_pair_input.
    
    Cette fonction est conservée pour la rétrocompatibilité mais utilise en interne
    la version robuste de pair_utils.normalize_pair_input.
    
    Args:
        pair_input: Entrée à normaliser (str, dict ou TradingPair)
        
    Returns:
        Nom de la paire normalisé (ex: "XBT/USD")
        
    Raises:
        ValueError: Si l'entrée ne peut pas être convertie en paire valide
    """
    warnings.warn(
        "pair_selector.types.normalize_pair_input est dépréciée. "
        "Utilisez plutôt src.utils.pair_utils.normalize_pair_input",
        DeprecationWarning,
        stacklevel=2
    )
    try:
        return robust_normalize_pair_input(pair_input)
    except Exception as e:
        raise ValueError(f"Format de paire non supporté: {pair_input}") from e
