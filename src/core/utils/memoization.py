"""
Décorateurs de mémoization pour les calculs d'indicateurs.
"""
from typing import Any, Callable, Dict, List, Optional, TypeVar, Union
import functools
import hashlib
import inspect
import json
import pickle

from .cache import IndicatorCache, _default_cache

# Type générique pour les fonctions décorées
T = TypeVar('T')

def generate_cache_key(
    func: Callable, 
    *args: Any, 
    **kwargs: Any
) -> str:
    """
    Génère une clé de cache unique pour un appel de fonction donné.
    
    Args:
        func: La fonction à appeler
        *args: Arguments positionnels
        **kwargs: Arguments nommés
        
    Returns:
        Une chaîne de caractères représentant une clé de cache unique
    """
    # Créer une représentation sérialisable des arguments
    def _serialize(obj: Any) -> Any:
        if isinstance(obj, (int, float, str, bool, type(None))):
            return obj
        elif isinstance(obj, (list, tuple)):
            return [_serialize(x) for x in obj]
        elif isinstance(obj, dict):
            return {k: _serialize(v) for k, v in obj.items()}
        elif hasattr(obj, '__dict__'):
            return _serialize(obj.__dict__)
        else:
            return str(obj)
    
    # Obtenir la signature de la fonction pour les arguments par défaut
    sig = inspect.signature(func)
    bound_args = sig.bind(*args, **kwargs)
    bound_args.apply_defaults()
    
    # Créer un dictionnaire sérialisable des arguments
    serialized_args = _serialize({
        'module': func.__module__,
        'name': func.__name__,
        'args': bound_args.args,
        'kwargs': bound_args.kwargs
    })
    
    # Générer une clé unique à partir des arguments sérialisés
    key = hashlib.md5(json.dumps(serialized_args, sort_keys=True).encode('utf-8')).hexdigest()
    return f"{func.__module__}:{func.__name__}:{key}"

def memoize(
    func: Optional[Callable[..., T]] = None, 
    *, 
    cache: Optional[IndicatorCache] = None,
    ttl: Optional[int] = 3600,
    ignore_args: Optional[List[str]] = None
) -> Callable[..., T]:
    """
    Décorateur pour mémoizer les résultats d'une fonction.
    
    Args:
        func: La fonction à décorer (passée automatiquement par Python)
        cache: Instance de cache à utiliser (par défaut: cache mémoire)
        ttl: Durée de vie du cache en secondes (None pour pas d'expiration)
        ignore_args: Liste des noms d'arguments à ignorer pour la génération de la clé
        
    Returns:
        La fonction décorée avec mise en cache
    """
    if ignore_args is None:
        ignore_args = []
    
    def decorator(f: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(f)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            # Utiliser le cache fourni ou le cache par défaut
            current_cache = cache or _default_cache
            if current_cache is None:
                return f(*args, **kwargs)
            
            # Filtrer les arguments à ignorer
            filtered_kwargs = {k: v for k, v in kwargs.items() if k not in ignore_args}
            
            # Générer une clé de cache unique
            key = generate_cache_key(f, *args, **filtered_kwargs)
            
            # Essayer de récupérer le résultat depuis le cache
            cached_result = current_cache.get(key)
            if cached_result is not None:
                return cached_result
                
            # Si pas dans le cache, exécuter la fonction
            result = f(*args, **kwargs)
            
            # Mettre en cache le résultat
            current_cache.set(key, result, ttl=ttl)
            
            return result
            
        return wrapper
    
    # Permet d'utiliser le décorateur avec ou sans parenthèses
    if func is None:
        return decorator
    else:
        return decorator(func)
