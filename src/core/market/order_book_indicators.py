"""
Indicateurs techniques basés sur le carnet d'ordres.

Ce module fournit des indicateurs avancés qui analysent la structure et la dynamique
du carnet d'ordres pour générer des signaux de trading.
"""

from decimal import Decimal, ROUND_HALF_UP
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging
from functools import lru_cache, wraps
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import time
import hashlib
import pickle

from .order_book import OrderBookSnapshot, PriceLevel, OrderBookMetrics

logger = logging.getLogger(__name__)

@dataclass
class SupportResistanceLevel:
    """Représente un niveau de support ou de résistance."""
    price: Decimal
    strength: float  # 0.0 à 1.0
    type: str  # 'support' ou 'resistance'
    last_tested: datetime
    times_tested: int = 1

def cache_with_ttl(ttl_seconds: int = 60, maxsize: int = 128):
    """
    Décoreur pour mettre en cache le résultat d'une méthode avec un temps de vie (TTL).
    
    Args:
        ttl_seconds: Durée de vie du cache en secondes
        maxsize: Taille maximale du cache
    """
    def decorator(func):
        cache = {}
        
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Générer une clé de cache unique basée sur les arguments
            key = (func.__name__, args[1:], frozenset(kwargs.items()))
            
            # Vérifier si le résultat est en cache et toujours valide
            current_time = time.time()
            if key in cache:
                result, timestamp = cache[key]
                if current_time - timestamp < ttl_seconds:
                    return result
            
            # Calculer et mettre en cache le résultat
            result = func(*args, **kwargs)
            cache[key] = (result, current_time)
            
            # Nettoyer le cache si nécessaire
            if len(cache) > maxsize:
                # Supprimer les entrées les plus anciennes
                oldest = min(cache.items(), key=lambda x: x[1][1])
                cache.pop(oldest[0])
            
            return result
            
        return wrapper
    return decorator


class OrderBookIndicators:
    """Classe pour calculer des indicateurs avancés basés sur le carnet d'ordres."""
    
    def __init__(self, lookback_period: int = 50):
        """
        Initialise le calculateur d'indicateurs.
        
        Args:
            lookback_period: Nombre de snapshots à considérer pour l'analyse historique
        """
        self.lookback_period = lookback_period
        self.support_levels: List[SupportResistanceLevel] = []
        self.resistance_levels: List[SupportResistanceLevel] = []
        self.price_levels_history = []
        self.volume_profile = {}
        
        # Cache pour les calculs coûteux
        self._cache: Dict[str, Tuple[Any, float]] = {}
        self._cache_ttl = 5.0  # 5 secondes par défaut
        
    def update(self, snapshot: OrderBookSnapshot) -> Dict[str, float]:
        """
        Met à jour les indicateurs avec un nouveau snapshot du carnet d'ordres.
        
        Args:
            snapshot: Snapshot actuel du carnet d'ordres
            
        Returns:
            Dictionnaire des indicateurs mis à jour
        """
        if not snapshot.bids or not snapshot.asks:
            logger.warning("Snapshot vide reçu, mise à jour ignorée")
            return {}
            
        indicators = {}
        
        # 1. Métriques de base
        indicators['spread'] = float(snapshot.spread)
        indicators['mid_price'] = float(snapshot.mid_price)
        indicators['imbalance'] = float(snapshot.imbalance)
        
        # 2. Profondeur du marché
        depth_metrics = self._calculate_market_depth(snapshot)
        indicators.update(depth_metrics)
        
        # 3. Détection des niveaux de support/résistance
        self._update_support_resistance(snapshot)
        
        # 4. Analyse du flux d'ordres
        order_flow = self._analyze_order_flow(snapshot)
        indicators.update(order_flow)
        
        # 5. Métriques avancées de liquidité
        liquidity_metrics = self._calculate_liquidity_metrics(snapshot)
        indicators.update(liquidity_metrics)
        
        # 6. Mise à jour du profil de volume
        self._update_volume_profile(snapshot)
        
        return indicators
    
    @cache_with_ttl(ttl_seconds=1)
    def _calculate_market_depth(self, snapshot: OrderBookSnapshot) -> Dict[str, float]:
        """
        Calcule les métriques de profondeur du marché.
        
        Cette méthode est mise en cache avec un TTL de 1 seconde car elle est appelée fréquemment
        et ses calculs sont coûteux.
        """
        # Générer une clé de cache basée sur le contenu du snapshot
        cache_key = self._generate_snapshot_key(snapshot, prefix='market_depth')
        current_time = time.time()
        
        # Vérifier le cache
        if cache_key in self._cache:
            result, timestamp = self._cache[cache_key]
            if current_time - timestamp < self._cache_ttl:
                return result
        
        # Calculer les métriques
        metrics = {}
        
        # Profondeur sur les 10 premiers niveaux
        depth_levels = 10
        bids = snapshot.bids[:depth_levels] if len(snapshot.bids) >= depth_levels else snapshot.bids
        asks = snapshot.asks[:depth_levels] if len(snapshot.asks) >= depth_levels else snapshot.asks
        
        # Volume total sur les N premiers niveaux
        total_bid_volume = sum(float(level.volume) for level in bids)
        total_ask_volume = sum(float(level.volume) for level in asks)
        
        metrics['depth_bid_volume'] = total_bid_volume
        metrics['depth_ask_volume'] = total_ask_volume
        metrics['depth_imbalance'] = (total_bid_volume - total_ask_volume) / (total_bid_volume + total_ask_volume + 1e-10)
        
        # Prix moyen pondéré par le volume sur les N niveaux
        if bids:
            weighted_bid = sum(float(level.price) * float(level.volume) for level in bids)
            metrics['vwap_bid_depth'] = weighted_bid / (total_bid_volume + 1e-10)
            
        if asks:
            weighted_ask = sum(float(level.price) * float(level.volume) for level in asks)
            metrics['vwap_ask_depth'] = weighted_ask / (total_ask_volume + 1e-10)
        
        # Mettre en cache le résultat
        self._cache[cache_key] = (metrics, current_time)
        
        return metrics
    
    def _update_support_resistance(self, snapshot: OrderBookSnapshot):
        """Met à jour les niveaux de support et résistance."""
        # Cette méthode est simplifiée - une implémentation complète nécessiterait
        # une analyse historique plus sophistiquée
        
        # Prix de référence (prix moyen pondéré)
        mid_price = snapshot.mid_price
        
        # Niveaux de prix importants dans le carnet d'ordres
        significant_levels = []
        
        # Considérer les 5 premiers niveaux de chaque côté
        for level in snapshot.bids[:5] + snapshot.asks[:5]:
            price = level.price
            volume = level.volume
            
            # Un niveau est significatif s'il a un volume important
            if volume > sum(l.volume for l in snapshot.bids[:10]) / 10:
                significant_levels.append((price, volume))
        
        # Mettre à jour les niveaux de support/résistance
        for price, volume in significant_levels:
            if price < mid_price:
                self._update_level(self.support_levels, price, 'support', volume)
            else:
                self._update_level(self.resistance_levels, price, 'resistance', volume)
        
        # Nettoyer les anciens niveaux
        self._clean_old_levels()
    
    def _update_level(self, levels: List[SupportResistanceLevel], 
                     price: Decimal, level_type: str, volume: Decimal):
        """Met à jour un niveau de support/résistance."""
        # Tolérance pour considérer qu'un niveau est le même (0.1%)
        tolerance = float(price) * 0.001
        now = datetime.utcnow()
        
        # Vérifier si un niveau similaire existe déjà
        for level in levels:
            if abs(float(level.price) - float(price)) <= tolerance:
                # Mettre à jour le niveau existant
                level.strength = min(1.0, level.strength + 0.1)
                level.last_tested = now
                level.times_tested += 1
                return
        
        # Ajouter un nouveau niveau
        new_level = SupportResistanceLevel(
            price=price,
            strength=0.5,  # Force initiale
            type=level_type,
            last_tested=now
        )
        levels.append(new_level)
    
    def _clean_old_levels(self, max_age_hours: int = 24):
        """Supprime les niveaux trop anciens."""
        now = datetime.utcnow()
        max_age = timedelta(hours=max_age_hours)
        
        self.support_levels = [
            level for level in self.support_levels
            if (now - level.last_tested) <= max_age
        ]
        
        self.resistance_levels = [
            level for level in self.resistance_levels
            if (now - level.last_tested) <= max_age
        ]
    
    def _analyze_order_flow(self, snapshot: OrderBookSnapshot) -> Dict[str, float]:
        """Analyse le flux d'ordres pour détecter des signaux d'achat/vente."""
        metrics = {}
        
        # 1. Déséquilibre d'ordres sur les premiers niveaux
        top_bid_volume = float(snapshot.bids[0].volume) if snapshot.bids else 0
        top_ask_volume = float(snapshot.asks[0].volume) if snapshot.asks else 0
        
        metrics['top_level_imbalance'] = (top_bid_volume - top_ask_volume) / (top_bid_volume + top_ask_volume + 1e-10)
        
        # 2. Profondeur relative
        depth_levels = 5
        if len(snapshot.bids) >= depth_levels and len(snapshot.asks) >= depth_levels:
            bid_depth = sum(float(level.volume) for level in snapshot.bids[:depth_levels])
            ask_depth = sum(float(level.volume) for level in snapshot.asks[:depth_levels])
            metrics['depth_imbalance_5'] = (bid_depth - ask_depth) / (bid_depth + ask_depth + 1e-10)
        
        # 3. Pression d'achat/vente basée sur la forme du carnet
        if len(snapshot.bids) > 1 and len(snapshot.asks) > 1:
            # Pente moyenne des prix (indicateur de la forme du carnet)
            bid_prices = [float(level.price) for level in snapshot.bids[:5]]
            ask_prices = [float(level.price) for level in snapshot.asks[:5]]
            
            # Calcul des pentes (différences de prix entre niveaux)
            bid_slopes = [bid_prices[i] - bid_prices[i+1] for i in range(len(bid_prices)-1)]
            ask_slopes = [ask_prices[i+1] - ask_prices[i] for i in range(len(ask_prices)-1)]
            
            # Pente moyenne normalisée par le spread
            spread = float(snapshot.spread)
            if spread > 0:
                metrics['bid_slope'] = sum(bid_slopes) / (len(bid_slopes) * spread + 1e-10)
                metrics['ask_slope'] = sum(ask_slopes) / (len(ask_slopes) * spread + 1e-10)
        
        return metrics
    
    def _calculate_liquidity_metrics(self, snapshot: OrderBookSnapshot) -> Dict[str, float]:
        """Calcule des métriques avancées de liquidité."""
        metrics = {}
        
        # 1. Liquidité dans le spread (volume disponible à ±0.5% du prix moyen)
        mid_price = float(snapshot.mid_price)
        price_band = mid_price * 0.005  # ±0.5%
        
        # Volume total dans la bande de prix
        bid_liquidity = sum(
            float(level.volume) 
            for level in snapshot.bids 
            if mid_price - float(level.price) <= price_band
        )
        
        ask_liquidity = sum(
            float(level.volume) 
            for level in snapshot.asks 
            if float(level.price) - mid_price <= price_band
        )
        
        metrics['liquidity_bid_0.5%'] = bid_liquidity
        metrics['liquidity_ask_0.5%'] = ask_liquidity
        metrics['liquidity_imbalance_0.5%'] = (bid_liquidity - ask_liquidity) / (bid_liquidity + ask_liquidity + 1e-10)
        
        # 2. Profondeur de marché normalisée
        if snapshot.bids and snapshot.asks:
            best_bid = float(snapshot.bids[0].price)
            best_ask = float(snapshot.asks[0].price)
            spread = best_ask - best_bid
            
            if spread > 0:
                # Nombre de niveaux de prix pour atteindre un certain écart
                levels_to_1pct = 0
                cumulative_impact = 0.0
                target_impact = mid_price * 0.01  # 1%
                
                # Pour les bids (prix décroissants)
                for i, level in enumerate(snapshot.bids):
                    price = float(level.price)
                    impact = best_bid - price
                    cumulative_impact += impact * float(level.volume)
                    if cumulative_impact >= target_impact:
                        levels_to_1pct = i + 1
                        break
                
                metrics['levels_to_1pct_bid'] = levels_to_1pct
                
                # Réinitialisation pour les asks
                levels_to_1pct = 0
                cumulative_impact = 0.0
                
                # Pour les asks (prix croissants)
                for i, level in enumerate(snapshot.asks):
                    price = float(level.price)
                    impact = price - best_ask
                    cumulative_impact += impact * float(level.volume)
                    if cumulative_impact >= target_impact:
                        levels_to_1pct = i + 1
                        break
                
                metrics['levels_to_1pct_ask'] = levels_to_1pct
        
        return metrics
    
    def _generate_snapshot_key(self, snapshot: OrderBookSnapshot, prefix: str = '') -> str:
        """
        Génère une clé de cache unique pour un snapshot donné.
        
        Args:
            snapshot: Le snapshot du carnet d'ordres
            prefix: Préfixe à ajouter à la clé
            
        Returns:
            Une chaîne de caractères unique pour ce snapshot
        """
        # Utiliser les premiers niveaux de prix pour générer une empreinte unique
        key_data = {
            'prefix': prefix,
            'timestamp': snapshot.timestamp,
            'bids': [(float(level.price), float(level.volume)) for level in snapshot.bids[:5]],
            'asks': [(float(level.price), float(level.volume)) for level in snapshot.asks[:5]],
            'mid_price': float(snapshot.mid_price)
        }
        
        # Sérialiser et hacher pour obtenir une clé unique
        serialized = pickle.dumps(key_data, protocol=pickle.HIGHEST_PROTOCOL)
        return hashlib.md5(serialized).hexdigest()
    
    def _clear_expired_cache(self):
        """Nettoie le cache des entrées expirées."""
        current_time = time.time()
        expired_keys = [
            key for key, (_, timestamp) in self._cache.items()
            if current_time - timestamp > self._cache_ttl
        ]
        for key in expired_keys:
            self._cache.pop(key, None)
    
    def _update_volume_profile(self, snapshot: OrderBookSnapshot):
        """
        Met à jour le profil de volume basé sur l'historique.
        
        Cette méthode est optimisée pour les performances en utilisant des opérations vectorisées
        et en limitant la taille du profil.
        """
        # Nettoyer le cache des entrées expirées
        if len(self._cache) > 100:  # Seulement si le cache devient grand
            self._clear_expired_cache()
        
        # Prix de référence (arrondi à 0.1% près pour regrouper les niveaux)
        mid_price = float(snapshot.mid_price)
        price_step = mid_price * 0.001
        
        # Mettre à jour le profil avec les données actuelles (approche optimisée)
        for side in [snapshot.bids, snapshot.asks]:
            for level in side:
                price = float(level.price)
                volume = float(level.volume)
                
                # Arrondir le prix pour regrouper les niveaux proches
                price_key = round(price / price_step) * price_step
                
                # Mettre à jour le volume de manière atomique
                if price_key in self.volume_profile:
                    self.volume_profile[price_key] += volume
                else:
                    self.volume_profile[price_key] = volume
        
        # Limiter la taille du profil pour des raisons de performance
        max_profile_size = 1000
        if len(self.volume_profile) > max_profile_size:
            # Garder les niveaux avec le plus de volume (plus efficace avec heapq pour les grands ensembles)
            import heapq
            top_levels = heapq.nlargest(
                max_profile_size // 2,
                self.volume_profile.items(),
                key=lambda x: x[1]
            )
            self.volume_profile = dict(top_levels)
    
    def get_support_resistance_levels(self, num_levels: int = 3) -> Dict[str, List[Dict]]:
        """
        Retourne les niveaux de support et résistance les plus forts.
        
        Args:
            num_levels: Nombre de niveaux à retourner pour chaque type
            
        Returns:
            Dictionnaire avec 'support' et 'resistance' contenant les niveaux
        """
        # Trier par force (strength) et limiter au nombre demandé
        supports = sorted(
            [l for l in self.support_levels],
            key=lambda x: x.strength,
            reverse=True
        )[:num_levels]
        
        resistances = sorted(
            [l for l in self.resistance_levels],
            key=lambda x: x.strength,
            reverse=True
        )[:num_levels]
        
        # Convertir en format de dictionnaire pour la sérialisation
        def level_to_dict(level):
            return {
                'price': float(level.price),
                'strength': level.strength,
                'times_tested': level.times_tested,
                'last_tested': level.last_tested.isoformat()
            }
        
        return {
            'support': [level_to_dict(l) for l in supports],
            'resistance': [level_to_dict(l) for l in resistances]
        }
