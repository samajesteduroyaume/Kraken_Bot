"""
Implémentation d'un cache en mémoire pour les indicateurs.
"""
from typing import Any, Dict, Optional, Tuple
from datetime import datetime
from .cache import IndicatorCache


class MemoryCache(IndicatorCache):
    """
    Implémentation d'un cache en mémoire.
    
    Ce cache est stocké en mémoire et est perdu lorsque le programme se termine.
    Il est principalement utile pour les tests et le développement.
    """
    
    def __init__(self):
        self._cache: Dict[str, Tuple[Any, Optional[float]]] = {}
    
    def get(self, key: str) -> Any:
        """
        Récupère une valeur du cache.
        
        Args:
            key: Clé de cache
            
        Returns:
            La valeur mise en cache ou None si non trouvée ou expirée
        """
        if key not in self._cache:
            return None
            
        value, expiry = self._cache[key]
        
        # Vérifier si l'entrée a expiré
        if expiry is not None and datetime.now().timestamp() > expiry:
            del self._cache[key]
            return None
            
        return value
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """
        Stocke une valeur dans le cache.
        
        Args:
            key: Clé de cache
            value: Valeur à stocker
            ttl: Durée de vie en secondes (None pour pas d'expiration)
        """
        expiry = None
        if ttl is not None:
            expiry = datetime.now().timestamp() + ttl
        self._cache[key] = (value, expiry)
    
    def delete(self, key: str) -> None:
        """
        Supprime une valeur du cache.
        
        Args:
            key: Clé de cache à supprimer
        """
        if key in self._cache:
            del self._cache[key]
    
    def clear(self) -> None:
        """Vide tout le cache."""
        self._cache.clear()
    
    def __len__(self) -> int:
        """Retourne le nombre d'éléments dans le cache."""
        return len(self._cache)
    
    def __contains__(self, key: str) -> bool:
        """Vérifie si une clé est présente dans le cache."""
        return key in self._cache
