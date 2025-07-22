"""
Module pour initialiser et gérer l'instance unique de AvailablePairs.

Ce module assure qu'une seule instance de AvailablePairs est utilisée dans toute l'application
et qu'elle est correctement initialisée avant utilisation.
"""
import logging
from typing import Optional

from .available_pairs_refactored import AvailablePairs, KrakenAPIError

logger = logging.getLogger(__name__)

# Instance globale de AvailablePairs
_available_pairs_instance: Optional[AvailablePairs] = None
_initialized = False

async def initialize_available_pairs(use_cache: bool = True) -> AvailablePairs:
    """
    Initialise et retourne l'instance unique de AvailablePairs.
    
    Args:
        use_cache: Si True, utilise le cache local si disponible
        
    Returns:
        AvailablePairs: L'instance initialisée de AvailablePairs
        
    Raises:
        KrakenAPIError: Si l'initialisation échoue
    """
    global _available_pairs_instance, _initialized
    
    if _initialized and _available_pairs_instance is not None:
        return _available_pairs_instance
        
    try:
        logger.info("🔄 Initialisation de l'instance AvailablePairs...")
        
        # Créer une nouvelle instance
        _available_pairs_instance = AvailablePairs()
        
        # Initialiser avec le cache
        await _available_pairs_instance.initialize(use_cache=use_cache)
        
        # Vérifier que l'initialisation a réussi
        pairs = _available_pairs_instance.get_available_pairs()
        if not pairs:
            raise KrakenAPIError("Aucune paire n'a pu être chargée")
            
        logger.info(f"✅ {len(pairs)} paires de trading chargées avec succès")
        _initialized = True
        
        return _available_pairs_instance
        
    except Exception as e:
        error_msg = f"❌ Échec de l'initialisation de AvailablePairs: {str(e)}"
        logger.error(error_msg, exc_info=True)
        _initialized = False
        _available_pairs_instance = None
        raise KrakenAPIError(error_msg) from e

def get_available_pairs() -> AvailablePairs:
    """
    Récupère l'instance de AvailablePairs.
    
    Returns:
        AvailablePairs: L'instance de AvailablePairs
        
    Raises:
        RuntimeError: Si AvailablePairs n'a pas été initialisé
    """
    if _available_pairs_instance is None or not _initialized:
        raise RuntimeError(
            "AvailablePairs n'a pas été initialisé. "
            "Appelez d'abord initialize_available_pairs()."
        )
    return _available_pairs_instance

async def refresh_available_pairs() -> None:
    """
    Force le rafraîchissement des paires disponibles depuis l'API Kraken.
    
    Raises:
        RuntimeError: Si AvailablePairs n'a pas été initialisé
    """
    global _available_pairs_instance
    
    if _available_pairs_instance is None or not _initialized:
        raise RuntimeError(
            "Impossible de rafraîchir les paires: "
            "AvailablePairs n'a pas été initialisé."
        )
        
    logger.info("🔄 Mise à jour des paires disponibles depuis l'API Kraken...")
    try:
        await _available_pairs_instance.refresh_cache()
        pairs = _available_pairs_instance.get_available_pairs()
        logger.info(f"✅ {len(pairs)} paires mises à jour avec succès")
    except Exception as e:
        logger.error(f"❌ Échec de la mise à jour des paires: {str(e)}")
        raise
