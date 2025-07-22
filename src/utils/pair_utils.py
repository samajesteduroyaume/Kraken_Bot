"""
Utility functions for handling trading pairs.

This module provides utility functions for working with trading pairs, including
extraction, validation, and normalization using the AvailablePairs class.
"""
from typing import Union, List, Dict, Any, Optional
import logging
import re
from functools import lru_cache

# Import local modules
from src.core.market.available_pairs_refactored import (
    AvailablePairs, 
    UnsupportedTradingPairError
)
from src.core.market.pair_initializer import (
    initialize_available_pairs,
    get_available_pairs,
    KrakenAPIError
)

# Logger
logger = logging.getLogger(__name__)

# Alias for backward compatibility
available_pairs = None

async def initialize_pair_utils(use_cache: bool = True) -> None:
    """
    Initialize the pair utilities by ensuring the available_pairs is initialized.
    
    This function should be called at application startup.
    
    Args:
        use_cache: If True, uses local cache if available
    """
    global available_pairs
    try:
        # Initialize the singleton instance
        available_pairs = await initialize_available_pairs(use_cache=use_cache)
        logger.info("✅ Pair utilities initialized with %d available pairs", 
                   len(available_pairs.get_available_pairs()))
    except Exception as e:
        logger.error("❌ Failed to initialize pair utilities: %s", str(e), exc_info=True)
        raise

def is_valid_pair(pair: Union[str, Dict[str, Any]]) -> bool:
    """
    Check if a trading pair is valid and supported by Kraken.
    
    This function checks if the provided pair exists in the available pairs
    and returns True if it's valid, False otherwise.
    
    Args:
        pair: Either a pair name string or a dictionary with a 'pair' key
        
    Returns:
        bool: True if the pair is valid and supported, False otherwise
    """
    global available_pairs
    
    try:
        # Try to get the instance if not already available
        if available_pairs is None:
            available_pairs = get_available_pairs()
            
        if isinstance(pair, dict):
            pair = pair.get('pair', '')
        
        if not isinstance(pair, str) or not pair.strip():
            return False
            
        return bool(available_pairs.is_pair_supported(pair))
    except Exception as e:
        logger.warning(f"Error validating pair {pair}: {str(e)}")
        return False


def extract_pair_name(pair: Union[str, Dict[str, Any]]) -> str:
    """
    Extract and normalize the pair name from either a string or a dictionary.
    
    This function uses the AvailablePairs class to ensure the pair is valid and
    returns it in the standard format (e.g., 'XBT/USD').
    
    Args:
        pair: Either a pair name string or a dictionary with a 'pair' key
        
    Returns:
        The normalized pair name in 'BASE/QUOTE' format
        
    Raises:
        ValueError: If the input is not a valid pair or the pair is not supported
        RuntimeError: If available_pairs is not initialized
    """
    global available_pairs
    
    try:
        # Try to get the instance if not already available
        if available_pairs is None:
            available_pairs = get_available_pairs()
            
        if isinstance(pair, str):
            return normalize_pair_input(pair)
        elif isinstance(pair, dict) and 'pair' in pair:
            return normalize_pair_input(str(pair['pair']))
        else:
            raise ValueError(f"Invalid pair format: {pair}")
    except Exception as e:
        logger.error(f"Error extracting pair name: {str(e)}", exc_info=True)
        raise

def extract_pair_names(pairs: List[Union[str, Dict[str, Any]]]) -> List[str]:
    """
    Extract and normalize pair names from a list of pairs.
    
    This function processes a list of pairs (either strings or dictionaries)
    and returns a list of normalized pair names. Invalid or unsupported pairs are skipped.
    
    Args:
        pairs: List of pairs (either strings or dictionaries with 'pair' key)
        
    Returns:
        List of normalized pair names in 'BASE/QUOTE' format
    """
    result = []
    for pair in pairs:
        try:
            normalized = extract_pair_name(pair)
            result.append(normalized)
            # Ajouter un avertissement si la paire n'est pas reconnue par Kraken
            if not available_pairs.is_pair_supported(normalized):
                logger.warning(
                    f"Paire non reconnue par Kraken: {normalized}. "
                    "La paire sera utilisée telle quelle, mais des erreurs peuvent survenir lors de l'exécution."
                )
        except (ValueError, UnsupportedTradingPairError) as e:
            logger.debug("Skipping invalid or unsupported pair %s: %s", pair, str(e))
            continue
    return result

def normalize_pair_input(pair: str, raise_on_error: bool = True) -> str:
    """
    Normalize a trading pair string to the standard format.
    
    This function uses the AvailablePairs class to normalize and validate
    the input pair string. It handles various input formats and converts
    them to the standard 'BASE/QUOTE' format (e.g., 'XBT/USD').
    
    Supported formats:
    - With separators: 'btc-usd', 'XBT_USD', 'XBT/USD', 'xbtusd', 'XXBTZUSD'
    - Case-insensitive: 'Btc-Usd' -> 'XBT/USD'
    - Without separator: 'btcusd' -> 'XBT/USD' (if unambiguous)
    
    Args:
        pair: The trading pair to normalize (e.g., 'btc-usd', 'XBT_USD', 'XXBTZUSD')
        raise_on_error: If True, raises an exception if the pair is not found.
                       If False, returns None instead of raising an exception.
        
    Returns:
        Normalized pair string in 'BASE/QUOTE' format (e.g., 'XBT/USD') or None if pair is invalid and raise_on_error is False
        
    Raises:
        ValueError: If the pair format is invalid or the pair is not supported (only if raise_on_error=True)
        UnsupportedTradingPairError: If the pair is not supported and raise_on_error=True
    """
    global available_pairs
    
    if not isinstance(pair, str):
        if raise_on_error:
            raise ValueError(f"Pair must be a string, got {type(pair)}")
        return None
    
    # Clean up the input
    original_pair = pair
    pair = pair.strip().upper()
    if not pair:
        if raise_on_error:
            raise ValueError("Empty pair string")
        return None
    
    # Try to get the instance if not already available
    if available_pairs is None:
        available_pairs = get_available_pairs()
    
    try:
        # Use the instance's normalize_pair method with error handling
        normalized = available_pairs.normalize_pair(pair, raise_on_error=raise_on_error)
        
        if normalized:
            return normalized
            
        # If we get here and raise_on_error is False, just return None
        if not raise_on_error:
            return None
            
        # If we get here, raise_on_error is True but we couldn't normalize the pair
        # Try to get suggestions for the pair
        try:
            # This will raise UnsupportedTradingPairError with suggestions
            available_pairs.normalize_pair(pair, raise_on_error=True)
        except UnsupportedTradingPairError as e:
            # Re-raise with the original error which contains suggestions
            raise e
            
        # Fallback to generic error if no specific suggestions
        raise ValueError(f"Unsupported or invalid trading pair format: {original_pair}")
        
    except UnsupportedTradingPairError:
        # Re-raise UnsupportedTradingPairError as is (it already contains suggestions)
        raise
        
    except Exception as e:
        logger.error(f"Error normalizing pair '{original_pair}': {str(e)}", exc_info=True)
        if raise_on_error:
            # For unsupported pairs, raise with the expected error message
            if "n'est pas reconnue" in str(e):
                raise ValueError("Unsupported or invalid trading pair")
            # For other errors, raise with the original message
            raise ValueError(f"Failed to normalize pair '{original_pair}': {str(e)}")
        return None
    
    # This point should never be reached due to the return/raise statements above
    assert False, "This code should be unreachable"
    
    # Vérifier si la paire est supportée, mais ne pas lever d'exception si ce n'est pas le cas
    if not available_pairs.is_pair_supported(normalized_pair):
        logger.warning(
            f"Paire non reconnue par Kraken: {original_pair} (normalisée: {normalized_pair}). "
            "La paire sera utilisée telle quelle, mais des erreurs peuvent survenir lors de l'exécution."
        )
    
    logger.debug(f"Paire normalisée: {original_pair} -> {normalized_pair}")
    return normalized_pair
