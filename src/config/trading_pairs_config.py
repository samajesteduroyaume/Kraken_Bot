"""
Configuration des paires de trading recommandées pour le bot Kraken.

Ce module définit les paires de trading recommandées avec leurs paramètres spécifiques.
Les paires sont organisées par catégories basées sur leur liquidité et leur volatilité.

Note: Les paires sont maintenant validées par rapport à la liste des paires disponibles sur Kraken
via le module AvailablePairs.
"""
from typing import Dict, List, Optional
import logging

# Import relatif pour éviter les problèmes de chemin
from ..core.market.available_pairs_refactored import available_pairs

logger = logging.getLogger(__name__)

# Paires avec haute liquidité (format: "BASE/QUOTE" ou "BASEEUR")
HIGH_LIQUIDITY_PAIRS = [
    "XBT/USD",
    "XBT/USDC",
    "XBT/USDT",
    "ETH/USD",
    "ETH/USDT",
    "ETH/USDC",
    "XRP/USD",
    "SOL/USD",
    "ADA/USD",
    "DOT/USD"
]

# Paires avec liquidité moyenne
MEDIUM_LIQUIDITY_PAIRS = [
    "LINK/USD",
    "MATIC/USD",
    "DOGE/USD",
    "AVAX/USD",
    "ATOM/USD"
]

# Paires avec faible liquidité (exemples, à adapter selon les besoins)
LOW_LIQUIDITY_PAIRS = [
    "ALGO/USD",
    "FIL/USD",
    "LTC/USD",
    "UNI/USD",
    "AAVE/USD"
]

def validate_pairs(pair_list: List[str]) -> List[str]:
    """Valide une liste de paires par rapport à celles disponibles sur Kraken.
    
    Args:
        pair_list: Liste des paires à valider
        
    Returns:
        List[str]: Liste des paires valides
        
    Raises:
        RuntimeError: Si les paires disponibles ne sont pas initialisées
    """
    if not hasattr(available_pairs, '_initialized') or not available_pairs._initialized:
        error_msg = "Les données des paires ne sont pas encore initialisées. Appelez d'abord available_pairs.initialize()"
        logger.error(error_msg)
        raise RuntimeError(error_msg)
        
    valid_pairs = []
    for pair in pair_list:
        try:
            normalized_pair = available_pairs.normalize_pair(pair)
            if normalized_pair and available_pairs.is_pair_supported(normalized_pair):
                valid_pairs.append(normalized_pair)
            else:
                logger.warning(f"Paire non supportée par Kraken: {pair}")
        except Exception as e:
            logger.warning(f"Erreur lors de la validation de la paire {pair}: {str(e)}")
    
    if not valid_pairs and pair_list:
        logger.warning("Aucune paire valide n'a pu être validée dans la liste fournie")
    
    return valid_pairs

def get_trading_pairs(validate: bool = True) -> Dict[str, List[str]]:
    """Retourne la configuration des paires de trading.
    
    Args:
        validate: Si True, filtre les paires non supportées par Kraken
        
    Returns:
        Dict[str, List[str]]: Dictionnaire des paires par catégorie de liquidité
    """
    pairs = {
        'high_liquidity': HIGH_LIQUIDITY_PAIRS.copy(),
        'medium_liquidity': MEDIUM_LIQUIDITY_PAIRS.copy(),
        'low_liquidity': LOW_LIQUIDITY_PAIRS.copy()
    }
    
    if validate:
        for category in pairs:
            pairs[category] = validate_pairs(pairs[category])
    
    return pairs

def get_all_trading_pairs(validate: bool = True) -> List[str]:
    """Retourne toutes les paires de trading dans une seule liste.
    
    Args:
        validate: Si True, filtre les paires non supportées par Kraken
        
    Returns:
        List[str]: Liste de toutes les paires de trading
    """
    pairs_dict = get_trading_pairs(validate=validate)
    all_pairs = []
    for category in pairs_dict.values():
        all_pairs.extend(category)
    return all_pairs

async def initialize() -> None:
    """Initialise le module et charge les paires disponibles depuis Kraken."""
    from core.api.kraken import KrakenAPI
    
    if not available_pairs._initialized:
        api = KrakenAPI()
        await available_pairs.initialize(api=api)
        
        # Valider les paires après initialisation
        get_trading_pairs(validate=True)
