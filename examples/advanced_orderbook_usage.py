"""
Exemples avancés d'utilisation des métriques du carnet d'ordres

Ce script montre comment utiliser les fonctionnalités avancées du module de carnet d'ordres
pour implémenter des stratégies de trading sophistiquées.
"""

import asyncio
import logging
from decimal import Decimal
from typing import Dict, List, Optional

# Configuration du logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import des composants du bot
from src.core.market.order_book import OrderBookManager, OrderBookSnapshot
from src.core.market.order_book_indicators import OrderBookIndicators
from src.core.market.order_book_signal_adapter import OrderBookSignalAdapter
from src.core.signal_generator import SignalGenerator

class AdvancedOrderBookTrader:
    """Classe démontrant des stratégies avancées utilisant les métriques du carnet d'ordres."""
    
    def __init__(self, pair: str = "XBT/USD"):
        """Initialise le trader avec une paire de trading."""
        self.pair = pair
        self.order_book_manager = OrderBookManager()
        self.signal_adapter = OrderBookSignalAdapter(self.order_book_manager)
        self.signal_generator = SignalGenerator(order_book_manager=self.order_book_manager)
        self.running = False
        
    async def start(self):
        """Démarre le trader."""
        self.running = True
        logger.info(f"Démarrage du trader pour la paire {self.pair}")
        
        # S'abonner aux mises à jour du carnet d'ordres
        await self.order_book_manager.subscribe(self.pair)
        
        # Boucle de trading principale
        while self.running:
            try:
                # Générer des signaux
                signals = await self.generate_signals()
                
                # Prendre des décisions de trading
                await self.make_trading_decisions(signals)
                
                # Attendre avant la prochaine itération
                await asyncio.sleep(1)
                
            except Exception as e:
                logger.error(f"Erreur dans la boucle de trading: {e}", exc_info=True)
                await asyncio.sleep(5)  # Attendre avant de réessayer
    
    async def generate_signals(self) -> Dict:
        """Génère des signaux de trading à partir du carnet d'ordres."""
        try:
            # Mettre à jour les indicateurs
            await self.signal_adapter.update()
            
            # Obtenir les signaux bruts
            signals = self.signal_adapter.get_signals()
            
            # Ajouter des signaux personnalisés
            signals["custom_signals"] = self._generate_custom_signals()
            
            return signals
            
        except Exception as e:
            logger.error(f"Erreur lors de la génération des signaux: {e}")
            return {}
    
    def _generate_custom_signals(self) -> Dict:
        """Génère des signaux personnalisés basés sur des stratégies avancées."""
        signals = {}
        
        # 1. Détection de parois d'ordres (order walls)
        signals["order_walls"] = self._detect_order_walls()
        
        # 2. Analyse de la profondeur du marché
        signals["depth_analysis"] = self._analyze_market_depth()
        
        # 3. Détection de manipulation de marché
        signals["spoofing_detected"] = self._detect_spoofing()
        
        return signals
    
    def _detect_order_walls(self, threshold: float = 0.3) -> Dict:
        """
        Détecte les parois d'ordres (grosses quantités à un niveau de prix).
        
        Args:
            threshold: Seuil relatif par rapport à la profondeur moyenne
            
        Returns:
            Dictionnaire avec les parois détectées
        """
        snapshot = self.order_book_manager.current_snapshot
        if not snapshot or not snapshot.bids or not snapshot.asks:
            return {}
        
        walls = {"bids": [], "asks": []}
        
        # Analyser les 10 premiers niveaux de chaque côté
        for side, levels in [("bids", snapshot.bids[:10]), ("asks", snapshot.asks[:10])]:
            if not levels:
                continue
                
            # Calculer le volume moyen
            volumes = [float(level.volume) for level in levels]
            avg_volume = sum(volumes) / len(volumes)
            
            # Détecter les parois (volumes > seuil * moyenne)
            for level in levels:
                volume = float(level.volume)
                if volume > avg_volume * (1 + threshold):
                    walls[side].append({
                        "price": float(level.price),
                        "volume": volume,
                        "ratio_to_avg": volume / avg_volume
                    })
        
        return walls
    
    def _analyze_market_depth(self, depth_levels: int = 5) -> Dict:
        """
        Analyse la profondeur du marché pour détecter des opportunités d'arbitrage.
        
        Args:
            depth_levels: Nombre de niveaux à analyser
            
        Returns:
            Dictionnaire avec l'analyse de profondeur
        """
        snapshot = self.order_book_manager.current_snapshot
        if not snapshot or not snapshot.bids or not snapshot.asks:
            return {}
        
        # Extraire les N premiers niveaux de chaque côté
        bids = snapshot.bids[:depth_levels]
        asks = snapshot.asks[:depth_levels]
        
        if not bids or not asks:
            return {}
            
        # Calculer les métriques de profondeur
        bid_volumes = [float(level.volume) for level in bids]
        ask_volumes = [float(level.volume) for level in asks]
        
        bid_prices = [float(level.price) for level in bids]
        ask_prices = [float(level.price) for level in asks]
        
        # Calculer les volumes cumulés
        cum_bid_volume = sum(bid_volumes)
        cum_ask_volume = sum(ask_volumes)
        
        # Calculer les prix moyens pondérés par le volume
        vwap_bid = sum(p * v for p, v in zip(bid_prices, bid_volumes)) / (cum_bid_volume or 1)
        vwap_ask = sum(p * v for p, v in zip(ask_prices, ask_volumes)) / (cum_ask_volume or 1)
        
        # Calculer l'écart moyen entre les niveaux
        bid_spreads = [bid_prices[i] - bid_prices[i+1] for i in range(len(bid_prices)-1)]
        ask_spreads = [ask_prices[i+1] - ask_prices[i] for i in range(len(ask_prices)-1)]
        
        avg_bid_spread = sum(bid_spreads) / len(bid_spreads) if bid_spreads else 0
        avg_ask_spread = sum(ask_spreads) / len(ask_spreads) if ask_spreads else 0
        
        return {
            "bid_volume": cum_bid_volume,
            "ask_volume": cum_ask_volume,
            "volume_imbalance": (cum_bid_volume - cum_ask_volume) / (cum_bid_volume + cum_ask_volume + 1e-10),
            "vwap_bid": vwap_bid,
            "vwap_ask": vwap_ask,
            "avg_bid_spread": avg_bid_spread,
            "avg_ask_spread": avg_ask_spread,
            "spread_ratio": (avg_ask_spread - avg_bid_spread) / ((avg_ask_spread + avg_bid_spread) / 2 + 1e-10)
        }
    
    def _detect_spoofing(self, window: int = 10, threshold: float = 0.7) -> Dict:
        """
        Détecte les tentatives de manipulation de marché (spoofing).
        
        Args:
            window: Taille de la fenêtre d'analyse
            threshold: Seuil de détection (0-1)
            
        Returns:
            Dictionnaire avec les indicateurs de spoofing
        """
        # Cette implémentation est simplifiée et devrait être adaptée
        # en fonction des caractéristiques spécifiques du marché
        
        snapshot = self.order_book_manager.current_snapshot
        if not snapshot or not snapshot.bids or not snapshot.asks:
            return {"spoofing_detected": False, "confidence": 0.0}
        
        # Analyser la forme du carnet d'ordres
        bid_volumes = [float(level.volume) for level in snapshot.bids[:window]]
        ask_volumes = [float(level.volume) for level in snapshot.asks[:window]]
        
        # Détecter les ordres anormalement gros suivis d'une disparition
        spoofing_confidence = 0.0
        
        # Vérifier les pics de volume isolés (simplifié)
        if len(bid_volumes) >= 3:
            avg_volume = sum(bid_volumes[1:]) / (len(bid_volumes) - 1)
            if bid_volumes[0] > avg_volume * 5:  # Ordre 5x plus gros que la moyenne
                spoofing_confidence = min(1.0, bid_volumes[0] / (avg_volume * 10))
        
        if len(ask_volumes) >= 3:
            avg_volume = sum(ask_volumes[1:]) / (len(ask_volumes) - 1)
            if ask_volumes[0] > avg_volume * 5:  # Ordre 5x plus gros que la moyenne
                spoofing_confidence = max(spoofing_confidence, 
                                       min(1.0, ask_volumes[0] / (avg_volume * 10)))
        
        return {
            "spoofing_detected": spoofing_confidence > threshold,
            "confidence": spoofing_confidence,
            "threshold": threshold
        }
    
    async def make_trading_decisions(self, signals: Dict):
        """Prend des décisions de trading basées sur les signaux."""
        if not signals or not signals.get('available', False):
            return
        
        try:
            # Exemple de stratégie: suivre le déséquilibre d'ordres
            order_imbalance = signals.get('order_flow_imbalance', 0)
            
            # Seuil pour éviter le surtrading
            threshold = 0.2
            
            if order_imbalance > threshold:
                # Signal d'achat fort
                logger.info(f"Signal d'achat détecté (imbalance: {order_imbalance:.2f})")
                await self.execute_trade("buy", strength=abs(order_imbalance))
                
            elif order_imbalance < -threshold:
                # Signal de vente fort
                logger.info(f"Signal de vente détecté (imbalance: {order_imbalance:.2f})")
                await self.execute_trade("sell", strength=abs(order_imbalance))
            
            # Autres stratégies basées sur les signaux personnalisés
            custom_signals = signals.get('custom_signals', {})
            if custom_signals.get('order_walls', {}).get('bids'):
                logger.info("Paroi d'achat détectée, potentielle pression haussière")
                
            if custom_signals.get('spoofing_detected', {}).get('spoofing_detected', False):
                logger.warning("Manipulation de marché détectée, prudence requise")
                
        except Exception as e:
            logger.error(f"Erreur lors de la prise de décision: {e}")
    
    async def execute_trade(self, side: str, strength: float):
        """
        Exécute un ordre de trading.
        
        Args:
            side: 'buy' ou 'sell'
            strength: Force du signal (0-1)
        """
        # Implémentation factice - à remplacer par un appel à l'API de trading
        logger.info(f"Exécution d'un ordre {side} avec une force de {strength:.2f}")
        
        # Ici, vous pourriez ajouter la logique pour:
        # 1. Calculer la taille de la position en fonction de la force du signal
        # 2. Vérifier les soldes disponibles
        # 3. Placer l'ordre via l'API de trading
        # 4. Gérer les erreurs et les timeouts
        
        # Exemple avec une API de trading hypothétique:
        # try:
        #     order = await trading_api.place_order(
        #         pair=self.pair,
        #         side=side,
        #         order_type="limit",
        #         price=best_price,
        #         size=calculated_size
        #     )
        #     logger.info(f"Ordre exécuté: {order}")
        # except Exception as e:
        #     logger.error(f"Erreur lors de l'exécution de l'ordre: {e}")


async def main():
    """Fonction principale pour exécuter l'exemple."""
    trader = AdvancedOrderBookTrader(pair="XBT/USD")
    
    try:
        # Démarrer le trader dans une tâche séparée
        task = asyncio.create_task(trader.start())
        
        # Exécuter pendant 60 secondes (pour l'exemple)
        await asyncio.sleep(60)
        
        # Arrêter le trader
        trader.running = False
        await task
        
    except KeyboardInterrupt:
        logger.info("Arrêt demandé par l'utilisateur")
        trader.running = False
        await task
    except Exception as e:
        logger.error(f"Erreur inattendue: {e}", exc_info=True)
    finally:
        # Nettoyage
        logger.info("Arrêt du trader")


if __name__ == "__main__":
    asyncio.run(main())
