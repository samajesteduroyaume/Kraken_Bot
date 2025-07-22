"""
Adaptateur pour intégrer les indicateurs du carnet d'ordres dans le système de signaux.

Ce module fait le pont entre le gestionnaire de carnet d'ordres et le générateur de signaux,
en fournissant une interface unifiée pour accéder aux métriques avancées.
"""

from typing import Dict, Any, Optional, List
import logging
from datetime import datetime
from decimal import Decimal

from .order_book import OrderBookManager, OrderBookSnapshot
from .order_book_indicators import OrderBookIndicators, SupportResistanceLevel

logger = logging.getLogger(__name__)

class OrderBookSignalAdapter:
    """
    Adaptateur qui transforme les données du carnet d'ordres en signaux de trading.
    
    Cette classe s'intègre avec OrderBookManager pour fournir des indicateurs
    avancés basés sur la structure et la dynamique du carnet d'ordres.
    """
    
    def __init__(self, order_book_manager: OrderBookManager):
        """
        Initialise l'adaptateur avec un gestionnaire de carnet d'ordres.
        
        Args:
            order_book_manager: Instance de OrderBookManager
        """
        self.ob_manager = order_book_manager
        self.indicators = OrderBookIndicators()
        self._last_update: Optional[datetime] = None
        self._cached_signals: Dict[str, Any] = {}
    
    async def update(self) -> bool:
        """
        Met à jour les indicateurs avec les dernières données du carnet d'ordres.
        
        Returns:
            True si la mise à jour a réussi, False sinon
        """
        try:
            snapshot = self.ob_manager.current_snapshot
            if not snapshot:
                logger.warning("Aucun snapshot disponible dans le gestionnaire de carnet d'ordres")
                return False
                
            # Mettre à jour les indicateurs
            self._cached_signals = self.indicators.update(snapshot)
            self._last_update = datetime.utcnow()
            return True
            
        except Exception as e:
            logger.error(f"Erreur lors de la mise à jour des indicateurs du carnet d'ordres: {e}", 
                        exc_info=True)
            return False
    
    def get_signals(self) -> Dict[str, Any]:
        """
        Retourne les signaux générés à partir des données du carnet d'ordres.
        
        Returns:
            Dictionnaire des signaux avec leurs valeurs
        """
        if not self._cached_signals:
            logger.warning("Aucun signal disponible dans le cache, mise à jour nécessaire")
            return {}
            
        return self._cached_signals
    
    def get_support_resistance_levels(self, num_levels: int = 3) -> Dict[str, List[Dict]]:
        """
        Récupère les niveaux de support et résistance détectés.
        
        Args:
            num_levels: Nombre de niveaux à retourner pour chaque type
            
        Returns:
            Dictionnaire avec 'support' et 'resistance' contenant les niveaux
        """
        return self.indicators.get_support_resistance_levels(num_levels)
    
    def get_market_depth_profile(self, price_steps: int = 20, 
                               depth_levels: int = 10) -> Dict[str, Any]:
        """
        Génère un profil de profondeur de marché.
        
        Args:
            price_steps: Nombre de niveaux de prix à inclure
            depth_levels: Nombre de niveaux de profondeur à analyser
            
        Returns:
            Dictionnaire contenant le profil de profondeur
        """
        snapshot = self.ob_manager.current_snapshot
        if not snapshot or not snapshot.bids or not snapshot.asks:
            return {}
        
        best_bid = float(snapshot.bids[0].price)
        best_ask = float(snapshot.asks[0].price)
        spread = best_ask - best_bid
        
        # Définir la plage de prix à analyser
        price_range = spread * 2  # ± le spread autour du spread
        min_price = best_bid - spread
        max_price = best_ask + spread
        price_step = price_range / price_steps
        
        profile = {
            'prices': [],
            'bid_volumes': [],
            'ask_volumes': [],
            'total_volume': 0,
            'max_volume': 0
        }
        
        # Calculer les volumes à chaque niveau de prix
        for i in range(price_steps + 1):
            price = min_price + (i * price_step)
            bid_volume = sum(
                float(level.volume) 
                for level in snapshot.bids[:depth_levels]
                if abs(float(level.price) - price) <= price_step/2
            )
            
            ask_volume = sum(
                float(level.volume)
                for level in snapshot.asks[:depth_levels]
                if abs(float(level.price) - price) <= price_step/2
            )
            
            profile['prices'].append(price)
            profile['bid_volumes'].append(bid_volume)
            profile['ask_volumes'].append(ask_volume)
            profile['total_volume'] += bid_volume + ask_volume
            profile['max_volume'] = max(profile['max_volume'], bid_volume, ask_volume)
        
        return profile
    
    def get_liquidity_zones(self, num_zones: int = 3) -> Dict[str, List[Dict]]:
        """
        Identifie les zones de liquidité importantes.
        
        Args:
            num_zones: Nombre de zones à identifier de chaque côté
            
        Returns:
            Dictionnaire avec 'bid_zones' et 'ask_zones' contenant les zones de liquidité
        """
        snapshot = self.ob_manager.current_snapshot
        if not snapshot:
            return {'bid_zones': [], 'ask_zones': []}
        
        def get_zones(levels, is_bid=True):
            """Identifie les zones de liquidité pour un côté du carnet."""
            if not levels:
                return []
                
            # Trier par volume décroissant
            sorted_levels = sorted(
                [(float(level.price), float(level.volume)) for level in levels],
                key=lambda x: x[1],
                reverse=True
            )
            
            zones = []
            added_prices = set()
            
            for price, volume in sorted_levels[:num_zones * 2]:  # Prendre plus que nécessaire pour le regroupement
                # Vérifier si ce prix est proche d'une zone existante
                merged = False
                for zone in zones:
                    zone_price = zone['price']
                    # Si le prix est à moins de 0.1% d'une zone existante, on fusionne
                    if abs(price - zone_price) / zone_price < 0.001:
                        # Moyenne pondérée des prix
                        total_volume = zone['volume'] + volume
                        zone['price'] = ((zone['price'] * zone['volume']) + (price * volume)) / total_volume
                        zone['volume'] = total_volume
                        zone['count'] += 1
                        merged = True
                        break
                
                if not merged and price not in added_prices:
                    zones.append({
                        'price': price,
                        'volume': volume,
                        'count': 1
                    })
                    added_prices.add(price)
            
            # Trier les zones par prix et limiter au nombre demandé
            zones.sort(key=lambda x: x['price'], reverse=is_bid)
            return zones[:num_zones]
        
        return {
            'bid_zones': get_zones(snapshot.bids[:20], is_bid=True),
            'ask_zones': get_zones(snapshot.asks[:20], is_bid=False)
        }
    
    def get_order_flow_signals(self) -> Dict[str, Any]:
        """
        Génère des signaux basés sur le flux d'ordres.
        
        Returns:
            Dictionnaire des signaux de flux d'ordres
        """
        snapshot = self.ob_manager.current_snapshot
        if not snapshot or not snapshot.bids or not snapshot.asks:
            return {}
        
        signals = {}
        
        # 1. Déséquilibre au carnet
        top_bid_volume = float(snapshot.bids[0].volume)
        top_ask_volume = float(snapshot.asks[0].volume)
        total_volume = top_bid_volume + top_ask_volume
        
        if total_volume > 0:
            imbalance = (top_bid_volume - top_ask_volume) / total_volume
            signals['order_flow_imbalance'] = imbalance
            
            # Signal basé sur le déséquilibre
            if imbalance > 0.7:
                signals['order_flow_signal'] = 'strong_buy'
            elif imbalance > 0.3:
                signals['order_flow_signal'] = 'buy'
            elif imbalance < -0.7:
                signals['order_flow_signal'] = 'strong_sell'
            elif imbalance < -0.3:
                signals['order_flow_signal'] = 'sell'
            else:
                signals['order_flow_signal'] = 'neutral'
        
        # 2. Profondeur relative
        depth_levels = 5
        if len(snapshot.bids) >= depth_levels and len(snapshot.asks) >= depth_levels:
            bid_depth = sum(float(level.volume) for level in snapshot.bids[:depth_levels])
            ask_depth = sum(float(level.volume) for level in snapshot.asks[:depth_levels])
            total_depth = bid_depth + ask_depth
            
            if total_depth > 0:
                depth_imbalance = (bid_depth - ask_depth) / total_depth
                signals['depth_imbalance'] = depth_imbalance
        
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
                avg_bid_slope = sum(bid_slopes) / (len(bid_slopes) * spread + 1e-10)
                avg_ask_slope = sum(ask_slopes) / (len(ask_slopes) * spread + 1e-10)
                
                signals['bid_slope'] = avg_bid_slope
                signals['ask_slope'] = avg_ask_slope
                
                # Signal basé sur la forme du carnet
                if avg_bid_slope > 0.5 and avg_ask_slope > 0.5:
                    signals['order_book_shape'] = 'consolidation'
                elif avg_bid_slope > 0.7:
                    signals['order_book_shape'] = 'bidding_war'
                elif avg_ask_slope > 0.7:
                    signals['order_book_shape'] = 'offer_wall'
                else:
                    signals['order_book_shape'] = 'normal'
        
        return signals
