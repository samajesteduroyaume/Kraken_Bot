"""
Package pour la gestion du marché et des paires de trading.

Ce package contient les modules pour gérer les paires de trading disponibles,
les données de marché, et les opérations liées au marché.
"""
from .available_pairs import AvailablePairs, KrakenAPIError, initialize_available_pairs, get_available_pairs

# Rendre les éléments importants disponibles au niveau du package
__all__ = [
    'AvailablePairs',
    'KrakenAPIError',
    'initialize_available_pairs',
    'get_available_pairs',
    'available_pairs'
]

# Variable pour stocker l'instance initialisée de AvailablePairs
_available_pairs_instance = None

# Fonction d'initialisation asynchrone
async def initialize_market(api=None):
    """
    Initialise les composants du marché.
    
    Args:
        api: Instance de l'API Kraken (optionnel)
    """
    global _available_pairs_instance
    _available_pairs_instance = await initialize_available_pairs(api)

# Fonction pour obtenir l'instance initialisée
def get_available_pairs_instance():
    """
    Récupère l'instance de AvailablePairs si elle est initialisée.
    
    Returns:
        AvailablePairs: Instance initialisée
        
    Raises:
        RuntimeError: Si le marché n'a pas été initialisé
    """
    if _available_pairs_instance is None:
        raise RuntimeError("Le marché n'a pas été initialisé. Appelez d'abord initialize_market().")
    return _available_pairs_instance

# Alias pour la compatibilité avec le code existant
def available_pairs():
    """
    Fonction pour récupérer l'instance de AvailablePairs.
    
    Returns:
        AvailablePairs: Instance initialisée
        
    Raises:
        RuntimeError: Si le marché n'a pas été initialisé
    """
    return get_available_pairs_instance()
