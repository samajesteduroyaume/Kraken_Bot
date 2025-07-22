"""
Module de gestion avancée du carnet d'ordres.

Ce module fournit des fonctionnalités avancées pour l'analyse et la manipulation
du carnet d'ordres, y compris le suivi en temps réel, les métriques avancées
et l'optimisation des performances.
"""

"""
Module de gestion avancée du carnet d'ordres.

Ce module fournit des fonctionnalités avancées pour l'analyse et la manipulation
du carnet d'ordres, y compris le suivi en temps réel, les métriques avancées
et l'optimisation des performances.
"""

import decimal
from decimal import Decimal, ROUND_DOWN, ROUND_UP, ROUND_HALF_UP, getcontext
from typing import Dict, List, Optional, Tuple, TypedDict, Deque, Any, Union, Callable
from datetime import datetime, timedelta, timezone
import asyncio
import logging
import time
from collections import deque, defaultdict
import numpy as np
from dataclasses import dataclass, field

# Configurer la précision décimale pour les calculs
getcontext().prec = 12

logger = logging.getLogger(__name__)

class OrderBookLevel:
    """
    Représente un niveau de prix dans le carnet d'ordres.
    
    Optimisé pour les performances avec __slots__ et le stockage interne efficace.
    """
    __slots__ = ['_price', '_amount', '_timestamp']
    
    def __init__(self, price: Union[str, float, Decimal], 
                 amount: Union[str, float, Decimal], 
                 timestamp: Optional[datetime] = None):
        self._price = Decimal(str(price)).quantize(Decimal('0.00000001'), rounding=ROUND_HALF_UP)
        self._amount = Decimal(str(amount)).quantize(Decimal('0.00000001'), rounding=ROUND_HALF_UP)
        self._timestamp = timestamp or datetime.now(timezone.utc)
    
    @property
    def price(self) -> Decimal:
        return self._price
        
    @property
    def amount(self) -> Decimal:
        return self._amount
        
    @property
    def timestamp(self) -> datetime:
        return self._timestamp
    
    def to_dict(self) -> Dict:
        """Convertit le niveau en dictionnaire."""
        return {
            'price': float(self.price),
            'amount': float(self.amount),
            'timestamp': self.timestamp.isoformat()
        }

class PriceLevel:
    """
    Classe optimisée pour stocker un niveau de prix et son volume.
    
    Utilise __slots__ pour une meilleure performance et une empreinte mémoire réduite.
    Les valeurs sont stockées sous forme de chaînes pour éviter les conversions répétées.
    """
    __slots__ = ['_price_str', '_volume_str', '_timestamp', '_price', '_volume']
    
    # Cache pour les conversions fréquentes
    _decimal_cache = {}
    _decimal_cache_size = 1000
    
    def __init__(self, price: Union[str, float, int, Decimal], 
                 volume: Union[str, float, int, Decimal], 
                 timestamp: Optional[float] = None):
        # Initialiser les attributs
        self._price_str = None
        self._volume_str = None
        self._price = None
        self._volume = None
        
        # Convertir et valider le prix
        try:
            # Nettoyer et normaliser la chaîne de prix
            price_str = str(price).strip().replace(',', '.')
            
            # Utiliser le cache pour les conversions fréquentes
            if price_str in self._decimal_cache:
                self._price = self._decimal_cache[price_str]
            else:
                # Créer un nouvel objet Decimal avec la précision appropriée
                try:
                    self._price = Decimal(price_str).quantize(
                        Decimal('0.00000001'), 
                        rounding=ROUND_HALF_UP
                    )
                    # Mettre en cache la conversion
                    if len(self._decimal_cache) < self._decimal_cache_size:
                        self._decimal_cache[price_str] = self._price
                except decimal.InvalidOperation:
                    # En cas d'erreur, utiliser une précision moindre
                    self._price = Decimal(price_str).quantize(
                        Decimal('0.01'), 
                        rounding=ROUND_HALF_UP
                    )
            
            # Convertir et valider le volume
            volume_str = str(volume).strip().replace(',', '.')
            
            # Utiliser le cache pour les conversions fréquentes
            if volume_str in self._decimal_cache:
                self._volume = self._decimal_cache[volume_str]
            else:
                try:
                    self._volume = Decimal(volume_str).quantize(
                        Decimal('0.00000001'), 
                        rounding=ROUND_HALF_UP
                    )
                    # Mettre en cache la conversion
                    if len(self._decimal_cache) < self._decimal_cache_size:
                        self._decimal_cache[volume_str] = self._volume
                except decimal.InvalidOperation:
                    # En cas d'erreur, utiliser une précision moindre
                    self._volume = Decimal(volume_str).quantize(
                        Decimal('0.01'), 
                        rounding=ROUND_HALF_UP
                    )
            
            # Stocker les chaînes pour une conversion rapide en chaîne
            self._price_str = str(self._price)
            self._volume_str = str(self._volume)
            
            # Gérer le timestamp
            if timestamp is None:
                self._timestamp = time.time()
            elif isinstance(timestamp, (int, float)):
                self._timestamp = float(timestamp)
            else:
                self._timestamp = time.time()
                
        except Exception as e:
            logger.error(
                f"Erreur lors de la création de PriceLevel: {e}",
                exc_info=True
            )
            raise
    
    @property
    def price(self) -> Decimal:
        """Retourne le prix sous forme de Decimal."""
        return self._price
    
    @property
    def volume(self) -> Decimal:
        """Retourne le volume sous forme de Decimal."""
        return self._volume
    
    @property
    def timestamp(self) -> float:
        """Retourne le timestamp."""
        return self._timestamp
    
    @property
    def price_str(self) -> str:
        """Retourne le prix sous forme de chaîne formatée."""
        return self._price_str or str(self._price)
    
    @property
    def volume_str(self) -> str:
        """Retourne le volume sous forme de chaîne formatée."""
        return self._volume_str or str(self._volume)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convertit l'objet en dictionnaire."""
        return {
            'price': float(self._price),
            'volume': float(self._volume),
            'timestamp': self._timestamp
        }
    
    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, PriceLevel):
            return False
            
        # Comparaison rapide avec les chaînes si disponibles
        if (hasattr(self, '_price_str') and hasattr(other, '_price_str') and 
            hasattr(self, '_volume_str') and hasattr(other, '_volume_str')):
            return (self._price_str == other._price_str and 
                    self._volume_str == other._volume_str and 
                    abs(self._timestamp - other._timestamp) < 1.0)
                    
        # Fallback sur la comparaison des Decimal
        return (self._price == other.price and 
                self._volume == other.volume and 
                abs(self._timestamp - other.timestamp) < 1.0)
    
    def __repr__(self):
        return f"PriceLevel(price={self.price}, volume={self.volume}, ts={self.timestamp})"

class OrderBookMetrics:
    """
    Classe optimisée pour calculer et stocker les métriques du carnet d'ordres.
    
    Utilise des structures de données optimisées pour les calculs fréquents
    et la mise en cache des résultats intermédiaires.
    """
    __slots__ = [
        'price_precision', 'volume_precision', 'best_bid', 'best_ask', 
        'spread', 'mid_price', 'vwap_bid', 'vwap_ask', 'imbalance',
        '_cumulative_bid', '_cumulative_ask', '_last_update',
        '_bids_cache', '_asks_cache', '_cache_timestamp'
    ]
    
    # Cache partagé pour les calculs fréquents
    _shared_cache = {}
    _cache_max_size = 1000
    _cache_ttl = 5.0  # 5 secondes
    
    def __init__(self, price_precision: int = 8, volume_precision: int = 8):
        self.price_precision = price_precision
        self.volume_precision = volume_precision
        
        # Initialiser les caches
        self._bids_cache = {}
        self._asks_cache = {}
        self._cache_timestamp = 0.0
        
        # Réinitialiser les métriques
        self.clear()
    
    def _get_cache_key(self, levels: List[Dict], side: str) -> str:
        """Génère une clé de cache unique pour un ensemble de niveaux."""
        if not levels:
            return f"{side}_empty"
            
        # Utiliser les premiers niveaux pour générer une empreinte unique
        key_data = [
            (str(level.get('price', '')), str(level.get('amount', level.get('volume', ''))))
            for level in levels[:5]  # Seulement les 5 premiers niveaux pour la clé
        ]
        return f"{side}_{hash(frozenset(key_data))}"
    
    def _cleanup_old_cache(self):
        """Nettoie les entrées de cache expirées."""
        current_time = time.time()
        expired_keys = [
            key for key, (_, timestamp) in self._shared_cache.items()
            if current_time - timestamp > self._cache_ttl
        ]
        for key in expired_keys:
            self._shared_cache.pop(key, None)
    
    @property
    def cumulative_bid(self):
        """Retourne le dictionnaire des volumes cumulés pour les bids."""
        return self._cumulative_bid

    @property
    def cumulative_ask(self):
        """Retourne le dictionnaire des volumes cumulés pour les asks."""
        return self._cumulative_ask

    def clear(self):
        """Réinitialise toutes les métriques."""
        self.best_bid = Decimal('0')
        self.best_ask = Decimal('0')
        self.spread = Decimal('0')
        self.mid_price = Decimal('0')
        self.vwap_bid = Decimal('0')
        self.vwap_ask = Decimal('0')
        self.imbalance = Decimal('0')
        self._cumulative_bid = {}
        self._cumulative_ask = {}
        self._last_update = None
        self._bids_cache = {}
        self._asks_cache = {}
        self._cache_timestamp = 0.0
    
    def _ensure_dict(self, level: Any) -> Dict[str, Any]:
        """Convertit un niveau en dictionnaire s'il ne l'est pas déjà."""
        if isinstance(level, dict):
            return level
        elif isinstance(level, (list, tuple)) and len(level) >= 2:
            # Format: [price, amount, timestamp?]
            return {'price': level[0], 'amount': level[1], 'timestamp': level[2] if len(level) > 2 else None}
        elif hasattr(level, 'price') and hasattr(level, 'volume'):
            # Objet avec attributs price et volume
            return {'price': str(level.price), 'amount': str(level.volume), 'timestamp': getattr(level, 'timestamp', None)}
        else:
            raise ValueError(f"Format de niveau non supporté: {level}")
    
    def update(self, bids: List[Any], asks: List[Any]) -> None:
        """
        Met à jour les métriques avec de nouvelles données de carnet d'ordres.
        
        Args:
            bids: Liste de niveaux d'achat (peut être des dict, list, ou objets avec attributs price/volume)
            asks: Liste de niveaux de vente (même format que bids)
        """
        try:
            self._last_update = time.time()
            
            # Convertir les niveaux en dictionnaires
            bids_dicts = [self._ensure_dict(level) for level in bids]
            asks_dicts = [self._ensure_dict(level) for level in asks]
            
            # Vérifier si nous avons des données valides
            if not bids_dicts or not asks_dicts:
                logger.warning("Données de carnet d'ordres vides")
                return
            
            # Trier les offres (bids: prix décroissant, asks: prix croissant)
            bids_sorted = sorted(bids_dicts, key=lambda x: -float(x['price']))
            asks_sorted = sorted(asks_dicts, key=lambda x: float(x['price']))
                
            # Meilleures offres
            self.best_bid = Decimal(str(bids_sorted[0]['price']))
            self.best_ask = Decimal(str(asks_sorted[0]['price']))
            
            # Spread
            self.spread = self.best_ask - self.best_bid
            self.mid_price = (self.best_bid + self.best_ask) / 2
            
            # Calcul du VWAP pour les bids et asks
            self.vwap_bid = self._calculate_vwap(bids_sorted)
            self.vwap_ask = self._calculate_vwap(asks_sorted)
            
            # Calcul de l'imbalance
            total_bid = sum(Decimal(str(level['amount'])) for level in bids_sorted)
            total_ask = sum(Decimal(str(level['amount'])) for level in asks_sorted)
            
            if (total_bid + total_ask) > 0:
                self.imbalance = (total_bid - total_ask) / (total_bid + total_ask)
            else:
                self.imbalance = Decimal('0')
            
            # Mise à jour des profondeurs cumulatives
            self._update_cumulative(bids_sorted, 'bid')
            self._update_cumulative(asks_sorted, 'ask')
            
            # Nettoyage périodique des données anciennes
            if time.time() - self._last_update > 60:  # Toutes les minutes
                self._cleanup_old_data()
                
        except Exception as e:
            logger.error(f"Erreur lors de la mise à jour des métriques: {e}", exc_info=True)
    
    def _calculate_vwap(self, levels: List[Union[Dict[str, Any], PriceLevel]]) -> Decimal:
        """Calcule le Volume Weighted Average Price."""
        try:
            total_volume = Decimal('0')
            total_value = Decimal('0')
            
            for level in levels:
                if isinstance(level, dict):
                    price = Decimal(str(level.get('price', '0')))
                    amount = Decimal(str(level.get('amount', '0')))
                elif hasattr(level, 'price') and hasattr(level, 'volume'):
                    price = level.price if isinstance(level.price, Decimal) else Decimal(str(level.price))
                    amount = level.volume if isinstance(level.volume, Decimal) else Decimal(str(level.volume))
                else:
                    continue
                    
                total_volume += amount
                total_value += price * amount
            
            if total_volume <= 0:
                return Decimal('0')
                
            vwap = total_value / total_volume
            # Utilisation de ROUND_HALF_UP pour un arrondi mathématique standard
            return vwap.quantize(Decimal(f"1e-{self.price_precision}"), rounding=ROUND_HALF_UP)
            
        except Exception as e:
            logger.error(f"Erreur dans le calcul du VWAP: {e}", exc_info=True)
            return Decimal('0')
    
    def _update_cumulative(self, levels: List[Dict[str, Any]], side: str) -> None:
        """Met à jour les volumes cumulés pour un côté du carnet."""
        try:
            if not levels:
                return
                
            # Trier les niveaux par prix (décroissant pour les bids, croissant pour les asks)
            is_bid = (side.lower() == 'bid')
            sorted_levels = sorted(
                levels,
                key=lambda x: Decimal(str(x['price'])),
                reverse=is_bid
            )
            
            # Mettre à jour les volumes cumulés
            cumulative = {}
            total_volume = Decimal('0')
            
            for level in sorted_levels:
                price = Decimal(str(level['price']))
                amount = Decimal(str(level['amount']))
                total_volume += amount
                cumulative[price] = total_volume
            
            # Mettre à jour le dictionnaire approprié
            if is_bid:
                self._cumulative_bid = cumulative
            else:
                self._cumulative_ask = cumulative
                
        except Exception as e:
            logger.error(f"Erreur dans la mise à jour des volumes cumulés: {e}", exc_info=True)
    
    def _cleanup_old_data(self) -> None:
        """Nettoie les anciennes données pour économiser de la mémoire."""
        # Pour l'instant, nous gardons toutes les données, mais cette méthode
        # peut être étendue pour supprimer les niveaux de prix trop anciens
        pass
    
    def get_liquidity_at_price(self, price: Decimal, side: str) -> Decimal:
        """
        Retourne la liquidité disponible à un prix donné.
        
        Args:
            price: Prix auquel calculer la liquidité
            side: 'bid' ou 'ask'
            
        Returns:
            Montant total disponible à ce prix
            
        Raises:
            ValueError: Si le paramètre 'side' n'est ni 'bid' ni 'ask'
        """
        if side.lower() == 'bid':
            levels = self._cumulative_bid
        elif side.lower() == 'ask':
            levels = self._cumulative_ask
        else:
            raise ValueError("Le paramètre 'side' doit être 'bid' ou 'ask'")
        
        if not levels:
            return Decimal('0')
            
        try:
            # Trouver le niveau le plus proche
            closest_price = min(levels.keys(), key=lambda x: abs(x - price))
            return levels[closest_price]
        except Exception as e:
            logger.error(f"Erreur lors du calcul de la liquidité: {e}", exc_info=True)
            return Decimal('0')
    
    def get_price_for_volume(self, volume: Decimal, side: str) -> Decimal:
        """
        Trouve le prix auquel un certain volume peut être exécuté.
        
        Args:
            volume: Volume à exécuter
            side: 'bid' ou 'ask'
            
        Returns:
            Prix moyen pondéré pour exécuter le volume
        """
        if side.lower() == 'bid':
            levels = sorted(self.cumulative_bid.items(), key=lambda x: x[0], reverse=True)
        elif side.lower() == 'ask':
            levels = sorted(self.cumulative_ask.items())
        else:
            raise ValueError("Le paramètre 'side' doit être 'bid' ou 'ask'")
        
        remaining_volume = volume
        total_value = Decimal('0')
        
        for price, cum_vol in levels:
            if remaining_volume <= 0:
                break
                
            level_vol = cum_vol - (self.cumulative_bid.get(price, Decimal('0')) 
                                 if side.lower() == 'bid' 
                                 else self.cumulative_ask.get(price, Decimal('0')))
            
            vol = min(remaining_volume, level_vol)
            total_value += price * vol
            remaining_volume -= vol
        
        if volume == 0:
            return Decimal('0')
            
        return (total_value / volume).quantize(
            Decimal(f"1e-{self.price_precision}"), 
            rounding=ROUND_DOWN
        )

class OrderBookSnapshot:
    """Capture instantanée du carnet d'ordres avec métriques."""
    
    def __init__(self, bids: List[Union[PriceLevel, Dict[str, Any]]], 
                 asks: List[Union[PriceLevel, Dict[str, Any]]], 
                 timestamp: Optional[datetime] = None):
        """
        Initialise un snapshot du carnet d'ordres.
        
        Args:
            bids: Liste de PriceLevel ou de dictionnaires avec 'price' et 'amount'
            asks: Liste de PriceLevel ou de dictionnaires avec 'price' et 'amount'
            timestamp: Horodatage du snapshot (par défaut: maintenant)
        """
        self.timestamp = timestamp or datetime.now(timezone.utc)
        
        # Convertir différents formats en objets PriceLevel
        def ensure_price_level(level):
            if isinstance(level, PriceLevel):
                return level
                
            # Si c'est une liste/tuple [price, amount, timestamp?]
            if isinstance(level, (list, tuple)) and len(level) >= 2:
                return PriceLevel(
                    price=Decimal(str(level[0])),
                    volume=Decimal(str(level[1])),
                    timestamp=level[2] if len(level) > 2 else None
                )
                
            # Si c'est un dictionnaire
            if isinstance(level, dict):
                return PriceLevel(
                    price=Decimal(str(level.get('price', '0'))),
                    volume=Decimal(str(level.get('amount', level.get('volume', '0')))),
                    timestamp=level.get('timestamp')
                )
                
            raise ValueError(f"Format de niveau non supporté: {level}")
            
        self.bids = sorted([ensure_price_level(level) for level in bids], 
                          key=lambda x: -float(x.price))
        self.asks = sorted([ensure_price_level(level) for level in asks], 
                          key=lambda x: float(x.price))
                          
        self.metrics = OrderBookMetrics()
        # Convertir les bids et asks en dictionnaires
        bids_dicts = [
            {'price': str(level.price), 'amount': str(level.volume)}
            for level in self.bids
        ]
        asks_dicts = [
            {'price': str(level.price), 'amount': str(level.volume)}
            for level in self.asks
        ]
        self.metrics.update(bids_dicts, asks_dicts)
    
    @property
    def spread(self) -> Decimal:
        return self.metrics.spread
    
    @property
    def mid_price(self) -> Decimal:
        return self.metrics.mid_price
    
    @property
    def imbalance(self) -> Decimal:
        return self.metrics.imbalance
    
    def get_price_levels(self, num_levels: int = 10) -> Dict[str, List[Dict]]:
        """
        Retourne les N meilleurs niveaux de prix pour les bids et asks.
        
        Args:
            num_levels: Nombre de niveaux à retourner
            
        Returns:
            Dictionnaire avec 'bids' et 'asks' contenant les niveaux de prix
        """
        return {
            'bids': [
                {'price': float(level.price), 'amount': float(level.volume)}
                for level in self.bids[:num_levels]
            ],
            'asks': [
                {'price': float(level.price), 'amount': float(level.volume)}
                for level in self.asks[:num_levels]
            ]
        }
    
    def get_cumulative_volume(self, price: Decimal, side: str) -> Decimal:
        """
        Retourne le volume cumulé jusqu'au prix spécifié.
        
        Args:
            price: Prix limite
            side: 'bid' ou 'ask'
            
        Returns:
            Volume cumulé
        """
        if side.lower() == 'bid':
            levels = sorted(
                [(level.price, level.volume) for level in self.bids],
                key=lambda x: -x[0]  # Tri décroissant pour les bids
            )
        elif side.lower() == 'ask':
            levels = sorted(
                [(level.price, level.volume) for level in self.asks],
                key=lambda x: x[0]  # Tri croissant pour les asks
            )
        else:
            raise ValueError("Le paramètre 'side' doit être 'bid' ou 'ask'")
        
        cum_vol = Decimal('0')
        for p, a in levels:
            if (side == 'bid' and p < price) or (side == 'ask' and p > price):
                break
            cum_vol += a
            
        return cum_vol

class OrderBookManager:
    """
    Gestionnaire avancé du carnet d'ordres avec suivi en temps réel,
    analyse des métriques et historique.
    """
    
    def __init__(self, symbol: str, api=None, max_history: int = 1000):
        """
        Initialise le gestionnaire de carnet d'ordres.
        
        Args:
            symbol: Symbole de trading (ex: 'XBT/USD')
            api: Instance de l'API pour les mises à jour
            max_history: Nombre maximum de snapshots à conserver
        """
        self.symbol = symbol
        self.api = api
        self.max_history = max_history
        self._snapshots: Deque[OrderBookSnapshot] = deque(maxlen=max_history)
        self._current_snapshot: Optional[OrderBookSnapshot] = None
        self._lock = asyncio.Lock()
        self._update_task: Optional[asyncio.Task] = None
        self._running = False
        self._price_precision = 2  # Valeur par défaut, mise à jour à la première mise à jour
        self._volume_precision = 8
    
    @property
    def current_snapshot(self) -> Optional[OrderBookSnapshot]:
        """Retourne la dernière capture instantanée du carnet d'ordres."""
        return self._current_snapshot
    
    @property
    def history(self) -> List[OrderBookSnapshot]:
        """Retourne l'historique des captures instantanées."""
        return list(self._snapshots)
    
    async def start(self, update_interval: float = 1.0) -> None:
        """
        Démarre la mise à jour automatique du carnet d'ordres.
        
        Args:
            update_interval: Intervalle de mise à jour en secondes
        """
        if self._running:
            logger.warning("Le gestionnaire de carnet d'ordres est déjà en cours d'exécution")
            return
            
        self._running = True
        self._update_task = asyncio.create_task(self._update_loop(update_interval))
        logger.info(f"Gestionnaire de carnet d'ordres démarré pour {self.symbol}")
    
    async def stop(self) -> None:
        """Arrête la mise à jour automatique du carnet d'ordres."""
        if not self._running:
            return
            
        self._running = False
        if self._update_task:
            self._update_task.cancel()
            try:
                await self._update_task
            except asyncio.CancelledError:
                pass
        logger.info(f"Gestionnaire de carnet d'ordres arrêté pour {self.symbol}")
    
    async def _update_loop(self, interval: float) -> None:
        """Boucle de mise à jour automatique."""
        while self._running:
            try:
                await self.update()
                await asyncio.sleep(interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Erreur lors de la mise à jour du carnet d'ordres: {e}")
                await asyncio.sleep(min(interval, 5.0))  # Attendre avant de réessayer
    
    async def update(self) -> None:
        """Met à jour le carnet d'ordres à partir de l'API."""
        if not self.api:
            logger.error("Aucune API configurée pour la mise à jour du carnet d'ordres")
            raise ValueError("Aucune API configurée")

        try:
            # Récupérer les données brutes de l'API
            orderbook_data = await self.api.get_order_book(
                pair=self.symbol,
                count=500  # Nombre maximum de niveaux
            )

            # Vérifier si les données sont vides
            if not orderbook_data:
                logger.warning("Aucune donnée reçue de l'API, utilisation d'un carnet vide")
                snapshot = OrderBookSnapshot(bids=[], asks=[])
            else:
                # Extraire les données de la réponse
                try:
                    pair_key = next(iter(orderbook_data.keys()))
                    data = orderbook_data[pair_key]
                except (KeyError, StopIteration):
                    logger.warning("Format de données inattendu, utilisation d'un carnet vide")
                    snapshot = OrderBookSnapshot(bids=[], asks=[])
                else:
                    # Créer un nouveau snapshot
                    snapshot = OrderBookSnapshot(
                        bids=data.get('bids', []),
                        asks=data.get('asks', []),
                        timestamp=datetime.now(timezone.utc)
                    )

            # Mettre à jour l'état actuel
            async with self._lock:
                self._current_snapshot = snapshot
                self._snapshots.append(snapshot)

                # Mettre à jour la précision si nécessaire
                if snapshot.bids and snapshot.asks:
                    best_bid = snapshot.bids[0].price
                    best_ask = snapshot.asks[0].price
                    spread = float(best_ask - best_bid)

                    # Ajuster la précision en fonction du spread
                    spread_str = f"{spread:.10f}".rstrip('0').rstrip('.')
                    if '.' in spread_str:
                        self._price_precision = len(spread_str.split('.')[1])
                    else:
                        self._price_precision = 0

                    # Mettre à jour la précision des métriques
                    if hasattr(snapshot.metrics, 'price_precision'):
                        snapshot.metrics.price_precision = self._price_precision

                    # Journaliser le spread uniquement s'il est disponible
                    logger.debug(f"Carnet d'ordres mis à jour pour {self.symbol} - Spread: {spread:.8f}")
                else:
                    logger.debug(f"Carnet d'ordres mis à jour pour {self.symbol} - Pas de cotation")

        except Exception as e:
            logger.error(f"Erreur lors de la mise à jour du carnet d'ordres: {e}", exc_info=True)
            raise  # Propager l'exception pour les tests
    
    def get_order_imbalance(self, levels: int = 5) -> float:
        """
        Calcule le déséquilibre d'ordres sur N niveaux.
        
        Args:
            levels: Nombre de niveaux à considérer
            
        Returns:
            Ratio de déséquilibre entre -1 et 1
        """
        if not self._current_snapshot:
            return 0.0
            
        bids = self._current_snapshot.bids[:levels]
        asks = self._current_snapshot.asks[:levels]
        
        if not bids or not asks:
            return 0.0
            
        total_bid = sum(float(level.volume) for level in bids)
        total_ask = sum(float(level.volume) for level in asks)
        
        if total_bid + total_ask == 0:
            return 0.0
            
        return (total_bid - total_ask) / (total_bid + total_ask)
    
    def get_price_impact(self, amount: float, side: str) -> float:
        """
        Calcule l'impact de prix pour une taille d'ordre donnée.
        
        Args:
            amount: Montant à échanger
            side: 'buy' ou 'sell'
            
        Returns:
            Impact de prix en pourcentage
            
        Raises:
            ValueError: Si le paramètre 'side' n'est ni 'buy' ni 'sell'
        """
        # Vérifier le paramètre side en premier
        if side.lower() not in ('buy', 'sell'):
            raise ValueError("Le paramètre 'side' doit être 'buy' ou 'sell'")
            
        # Retourner 0 immédiatement pour un montant nul ou négatif
        if amount <= 0:
            return 0.0
            
        if not self._current_snapshot or not self._current_snapshot.bids or not self._current_snapshot.asks:
            return 0.0
            
        try:
            if side.lower() == 'buy':
                levels = self._current_snapshot.asks
                if not levels:
                    return 0.0
                best_price = float(levels[0].price)
            else:  # side == 'sell'
                levels = self._current_snapshot.bids
                if not levels:
                    return 0.0
                best_price = float(levels[0].price)
            
            if best_price <= 0:
                return 0.0
            
            remaining = amount
            total_cost = 0.0
            
            # Parcourir les niveaux jusqu'à ce que le montant soit épuisé
            for level in levels:
                if remaining <= 0:
                    break
                    
                price = float(level.price)
                volume = float(level.volume)
                available = min(volume, remaining)
                
                total_cost += price * available
                remaining -= available
            
            # Si on n'a pas pu exécuter tout le volume, 
            # utiliser le prix du dernier niveau pour la partie restante
            if remaining > 0 and levels:
                last_price = float(levels[-1].price)
                total_cost += remaining * last_price
            
            # Calculer le VWAP et l'impact de prix
            vwap = total_cost / amount
            price_impact = ((vwap / best_price) - 1) * 100
            
            # Pour les très petits montants, l'impact peut être nul
            # Dans ce cas, retourner une très petite valeur positive pour un achat
            # et négative pour une vente
            if abs(price_impact) < 1e-10:
                return 1e-10 if side.lower() == 'buy' else -1e-10
                
            # Retourner l'impact avec le bon signe selon le côté
            return price_impact if side.lower() == 'buy' else -abs(price_impact)
            
        except Exception as e:
            logger.error(f"Erreur lors du calcul de l'impact de prix: {e}", exc_info=True)
            return 0.0
    
    def get_liquidity_heatmap(self, price_bins: int = 20, 
                            volume_bins: int = 10) -> Dict[str, List[float]]:
        """
        Génère une heatmap de liquidité.
        
        Args:
            price_bins: Nombre de niveaux de prix
            volume_bins: Nombre de niveaux de volume
            
        Returns:
            Dictionnaire avec les données de la heatmap
        """
        if not self._current_snapshot:
            return {'bids': [], 'asks': []}
        
        bids = self._current_snapshot.bids
        asks = self._current_snapshot.asks
        
        if not bids or not asks:
            return {'bids': [], 'asks': []}
        
        # Prix min/max basés sur les meilleurs prix
        best_bid = float(bids[0].price)
        best_ask = float(asks[0].price)
        
        # Étendue des prix à afficher (2x le spread de chaque côté)
        spread = best_ask - best_bid
        min_price = best_bid - spread * 2
        max_price = best_ask + spread * 2
        
        # Créer les bins de prix
        price_step = (max_price - min_price) / price_bins
        price_levels = [min_price + i * price_step for i in range(price_bins + 1)]
        
        # Regrouper les volumes par niveau de prix
        def aggregate_volumes(levels, is_bid=True):
            volumes = [0.0] * price_bins
            
            for level in levels:
                price = float(level.price)
                amount = float(level.volume)
                
                # Ignorer les prix en dehors de la plage
                if price < min_price or price > max_price:
                    continue
                
                # Trouver le bon bin de prix
                bin_idx = min(int((price - min_price) / price_step), price_bins - 1)
                volumes[bin_idx] += amount
            
            return volumes
        
        # Calculer les volumes agrégés
        bid_volumes = aggregate_volumes(bids, is_bid=True)
        ask_volumes = aggregate_volumes(asks, is_bid=False)
        
        # Normaliser les volumes pour la heatmap
        max_volume = max(max(bid_volumes), max(ask_volumes), 1e-9)
        
        def normalize_volumes(volumes):
            return [min(v / max_volume * volume_bins, volume_bins) for v in volumes]
        
        return {
            'prices': price_levels,
            'bids': normalize_volumes(bid_volumes),
            'asks': normalize_volumes(ask_volumes)
        }
