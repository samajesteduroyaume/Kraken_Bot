"""
Utilitaires de cache et de mémoization pour les calculs d'indicateurs.
"""
from typing import Any, Callable, Dict, List, Optional, Tuple, TypeVar, Union
import functools
import hashlib
import inspect
import json
import pickle
from datetime import datetime
import numpy as np
import pandas as pd

# Type générique pour les fonctions décorées
T = TypeVar('T')

# Cache par défaut (peut être remplacé par un autre implémentation de IndicatorCache)
_default_cache = None

class IndicatorCache:
    """
    Classe de base pour le cache des indicateurs.
    """
    def get(self, key: str) -> Any:
        """Récupère une valeur du cache."""
        raise NotImplementedError
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Stocke une valeur dans le cache."""
        raise NotImplementedError
    
    def delete(self, key: str) -> None:
        """Supprime une valeur du cache."""
        raise NotImplementedError
    
    def clear(self) -> None:
        """Vide tout le cache."""
        raise NotImplementedError


def set_default_cache(cache: Optional[IndicatorCache] = None) -> None:
    """
    Définit le cache par défaut à utiliser pour la mémoization.
    
    Args:
        cache: Instance de IndicatorCache à utiliser comme cache par défaut.
              Si None, désactive le cache par défaut.
    """
    global _default_cache
    _default_cache = cache
