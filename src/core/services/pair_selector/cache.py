"""
Gestion du cache pour le PairSelector.

Ce module fournit une interface pour mettre en cache les résultats d'analyse
des paires de trading, à la fois en mémoire et sur disque.
"""
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from .models import CacheStats


class CacheManager:
    """Gestionnaire de cache pour les analyses de paires.
    
    Ce gestionnaire combine un cache LRU en mémoire avec un cache disque persistant,
    avec une politique d'expiration basée sur le temps (TTL).
    """
    
    def __init__(self, cache_dir: Path, ttl: int = 3600, max_size: int = 100):
        """Initialise le gestionnaire de cache.
        
        Args:
            cache_dir: Répertoire pour le stockage des fichiers de cache
            ttl: Durée de vie des entrées du cache en secondes (par défaut: 1h)
            max_size: Nombre maximum d'entrées dans le cache mémoire
        """
        self.cache_dir = cache_dir
        self.ttl = ttl
        self.max_size = max_size
        
        # Initialiser le cache mémoire (simplifié pour l'exemple)
        self._memory_cache = {}
        self._cache_stats = CacheStats(maxsize=max_size)
        
        # Créer le répertoire de cache s'il n'existe pas
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    async def get(self, key: str) -> Optional[Dict[str, Any]]:
        """Récupère une entrée du cache.
        
        Args:
            key: Clé d'identification de l'entrée
            
        Returns:
            Les données en cache ou None si non trouvées ou expirées
        """
        # Essayer d'abord le cache mémoire
        if key in self._memory_cache:
            self._cache_stats.hits += 1
            return self._memory_cache[key]
        
        # Sinon, essayer le cache disque
        cache_file = self.cache_dir / f"{key}.json"
        
        try:
            if not cache_file.exists():
                self._cache_stats.misses += 1
                return None
                
            # Vérifier si le cache est expiré
            mtime = cache_file.stat().st_mtime
            if time.time() - mtime > self.ttl:
                self._cache_stats.misses += 1
                return None
                
            # Lire les données du cache
            with open(cache_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            # Vérifier les métadonnées du cache
            if not isinstance(data, dict) or 'cached_at' not in data:
                self._cache_stats.misses += 1
                return None
                
            # Vérifier la date de cache
            try:
                cached_at = datetime.fromisoformat(data['cached_at'])
                if (datetime.now(timezone.utc) - cached_at).total_seconds() > self.ttl:
                    self._cache_stats.misses += 1
                    return None
            except (ValueError, TypeError):
                self._cache_stats.misses += 1
                return None
            
            # Mettre à jour le cache mémoire
            self._update_memory_cache(key, data)
            self._cache_stats.hits += 1
            return data
            
        except (json.JSONDecodeError, OSError) as e:
            # En cas d'erreur, supprimer le fichier corrompu
            try:
                if cache_file.exists():
                    cache_file.unlink()
            except OSError:
                pass
                
            self._cache_stats.misses += 1
            return None
    
    async def set(self, key: str, value: Dict[str, Any]) -> bool:
        """Stocke une entrée dans le cache.
        
        Args:
            key: Clé d'identification de l'entrée
            value: Données à stocker dans le cache
            
        Returns:
            True si la mise en cache a réussi, False sinon
        """
        if not key or not isinstance(value, dict):
            return False
            
        # Préparer les données pour le cache
        cache_data = value.copy()
        cache_data.update({
            'cached_at': datetime.now(timezone.utc).isoformat(),
            'cache_version': '1.0'
        })
        
        # Mettre à jour le cache mémoire
        self._update_memory_cache(key, cache_data)
        
        # Écrire sur disque de manière asynchrone
        try:
            cache_file = self.cache_dir / f"{key}.json"
            
            # Créer un fichier temporaire d'abord
            temp_file = cache_file.with_suffix('.tmp')
            
            with open(temp_file, 'w', encoding='utf-8') as f:
                json.dump(cache_data, f, indent=2, ensure_ascii=False)
                
            # Remplacer le fichier existant de manière atomique
            temp_file.replace(cache_file)
            return True
            
        except (OSError, TypeError, ValueError) as e:
            # En cas d'erreur, essayer de supprimer le fichier temporaire
            try:
                if temp_file.exists():
                    temp_file.unlink()
            except OSError:
                pass
                
            return False
    
    def clear(self) -> Dict[str, Any]:
        """Vide complètement le cache.
        
        Returns:
            Statistiques du cache avant la réinitialisation
        """
        stats = self.get_stats()
        
        # Vider le cache mémoire
        self._memory_cache.clear()
        
        # Supprimer les fichiers de cache
        try:
            for cache_file in self.cache_dir.glob('*.json'):
                try:
                    cache_file.unlink()
                except OSError:
                    pass
        except OSError:
            pass
        
        # Réinitialiser les statistiques
        self._cache_stats = CacheStats(maxsize=self.max_size)
        
        return stats
    
    def get_stats(self) -> Dict[str, Any]:
        """Retourne les statistiques d'utilisation du cache.
        
        Returns:
            Dictionnaire contenant les statistiques du cache
        """
        return {
            'hits': self._cache_stats.hits,
            'misses': self._cache_stats.misses,
            'currsize': len(self._memory_cache),
            'maxsize': self._cache_stats.maxsize,
            'last_reset': self._cache_stats.last_reset,
            'ttl': self.ttl
        }
    
    def _update_memory_cache(self, key: str, value: Dict[str, Any]) -> None:
        """Met à jour le cache mémoire avec une nouvelle entrée.
        
        Args:
            key: Clé de l'entrée
            value: Valeur à stocker
        """
        # Si la clé existe déjà, mettre à jour la valeur
        if key in self._memory_cache:
            self._memory_cache[key] = value
            return
            
        # Si le cache est plein, supprimer l'entrée la plus ancienne
        if len(self._memory_cache) >= self.max_size:
            oldest_key = next(iter(self._memory_cache))
            del self._memory_cache[oldest_key]
            
        # Ajouter la nouvelle entrée
        self._memory_cache[key] = value
        self._cache_stats.currsize = len(self._memory_cache)
