import logging
from typing import Dict, List, Optional, Any, AsyncIterator
from datetime import datetime, timedelta
from ..types.market_types import MarketData, Candle, OrderBook as OrderBookType, Trade
from .order_book import OrderBookManager, OrderBookSnapshot
from ..analysis.technical import TechnicalAnalyzer
import asyncio


class MarketDataManager:
    """
    Gestionnaire des données de marché.

    Cette classe gère :
    - La collecte des données de marché
    - Le stockage des données historiques
    - L'analyse en temps réel
    - La mise à jour des données
    """

    def __init__(self,
                 api,  # Instance de l'API du broker
                 config: Optional[Dict] = None):
        """
        Initialise le gestionnaire des données de marché.

        Args:
            api: Instance de l'API du broker
            config: Configuration optionnelle
        """
        self.api = api
        self.config = config or {}
        self.logger = logging.getLogger(__name__)

        # Stockage des données
        self.market_data: Dict[str, MarketData] = {}
        self.candle_history: Dict[str, List[Candle]] = {}
        self.order_books: Dict[str, OrderBookManager] = {}
        self.trades: Dict[str, List[Trade]] = {}
        self._order_book_update_tasks: Dict[str, asyncio.Task] = {}

        # Analyse technique
        self.technical_analyzer = TechnicalAnalyzer()

    async def update_market_data(self, symbol: str) -> MarketData:
        """
        Met à jour les données de marché pour une paire.

        Args:
            symbol: Paire de trading

        Returns:
            Données de marché mises à jour
        """
        try:
            # Récupérer les données de marché
            candles = await self._fetch_candles(symbol)
            order_book = await self._fetch_order_book(symbol)
            trades = await self._fetch_trades(symbol)

            # Analyser les données
            analysis = self._analyze_market(
                candles=candles,
                order_book=order_book,
                trades=trades
            )

            # Mettre à jour les données
            self.market_data[symbol] = {
                'symbol': symbol,
                'candles': candles,
                'order_book': order_book,
                'trades': trades,
                'analysis': analysis,
                'timestamp': datetime.now()
            }

            # Mettre à jour l'historique
            self._update_history(symbol, candles)

            return self.market_data[symbol]

        except Exception as e:
            self.logger.error(
                f"Erreur lors de la mise à jour des données pour {symbol}: {e}")
            return self.market_data.get(symbol, {})

    async def _fetch_candles(self, symbol: str) -> List[Candle]:
        """Récupère les bougies de la paire."""
        try:
            # Récupérer les bougies via l'API
            candles = await self.api.get_candles(
                pair=symbol,
                interval=self.config.get('candle_interval', '1m'),
                since=datetime.now() - timedelta(days=30)
            )

            return candles

        except Exception as e:
            self.logger.error(
                f"Erreur lors de la récupération des bougies {symbol}: {e}")
            return []

    async def _fetch_order_book(self, symbol: str) -> Dict[str, Any]:
        """
        Récupère le carnet d'ordres de la paire.
        
        Args:
            symbol: Symbole de la paire
            
        Returns:
            Dictionnaire contenant les offres (asks), les demandes (bids) et le timestamp
        """
        try:
            # Initialiser le gestionnaire de carnet d'ordres si nécessaire
            if symbol not in self.order_books:
                self.order_books[symbol] = OrderBookManager(symbol, self.api)
                
                # Démarrer la mise à jour automatique si ce n'est pas déjà fait
                if symbol not in self._order_book_update_tasks:
                    self._order_book_update_tasks[symbol] = asyncio.create_task(
                        self.order_books[symbol].start(
                            update_interval=self.config.get('orderbook_update_interval', 1.0)
                        )
                    )
            
            # Récupérer le gestionnaire de carnet d'ordres
            order_book_manager = self.order_books[symbol]
            
            # S'assurer que le carnet est à jour
            await order_book_manager.update()
            
            # Récupérer le snapshot actuel
            snapshot = order_book_manager.current_snapshot
            if snapshot is None:
                raise ValueError("Impossible de récupérer le carnet d'ordres")
                
            # Convertir les données en format compatible avec l'ancienne interface
            return {
                'asks': [{'price': float(level['price']), 'amount': float(level['amount'])} 
                        for level in snapshot.asks],
                'bids': [{'price': float(level['price']), 'amount': float(level['amount'])} 
                        for level in snapshot.bids],
                'timestamp': snapshot.timestamp,
                'metrics': {
                    'spread': float(snapshot.metrics.spread) if snapshot.metrics.spread else None,
                    'imbalance': float(snapshot.metrics.imbalance) if hasattr(snapshot.metrics, 'imbalance') else None,
                    'vwap_bid': float(snapshot.metrics.vwap_bid) if hasattr(snapshot.metrics, 'vwap_bid') else None,
                    'vwap_ask': float(snapshot.metrics.vwap_ask) if hasattr(snapshot.metrics, 'vwap_ask') else None
                }
            }

        except Exception as e:
            self.logger.error(
                f"Erreur lors de la récupération du carnet d'ordres {symbol}: {e}")
            return {
                'asks': [],
                'bids': [],
                'timestamp': datetime.now()
            }

    async def _fetch_trades(self, symbol: str) -> List[Trade]:
        """Récupère les trades récents de la paire."""
        try:
            # Récupérer les trades via l'API
            trades = await self.api.get_trades(
                pair=symbol,
                since=datetime.now() - timedelta(hours=24)
            )

            return trades

        except Exception as e:
            self.logger.error(
                f"Erreur lors de la récupération des trades {symbol}: {e}")
            return []

    def _analyze_market(self,
                        candles: List[Candle],
                        order_book: OrderBookType,
                        trades: List[Trade]) -> Dict[str, Any]:
        """
        Analyse les données de marché.

        Args:
            candles: Liste des bougies
            order_book: Carnet d'ordres
            trades: Liste des trades

        Returns:
            Résultats de l'analyse
        """
        analysis = {}

        # Analyse technique
        if candles:
            technical_indicators = self.technical_analyzer.analyze_candles(
                candles)
            analysis.update(technical_indicators)

        # Analyse du carnet d'ordres
        if order_book:
            analysis.update(self._analyze_order_book(order_book))

        # Analyse des trades
        if trades:
            analysis.update(self._analyze_trades(trades))

        return analysis

    def _analyze_order_book(self, order_book: OrderBookType) -> Dict[str, Any]:
        """Analyse le carnet d'ordres."""
        if not order_book['asks'] or not order_book['bids']:
            return {
                'order_book_imbalance': 0.0,
                'spread': 0.0,
                'best_ask': 0.0,
                'best_bid': 0.0,
                'vwap_ask': 0.0,
                'vwap_bid': 0.0,
                'liquidity_imbalance': 0.0,
                'order_imbalance': 0.0
            }
            
        # Créer un snapshot pour l'analyse
        snapshot = OrderBookSnapshot(
            bids=order_book['bids'],
            asks=order_book['asks'],
            timestamp=order_book.get('timestamp', datetime.now())
        )
        
        # Récupérer les métriques
        metrics = snapshot.metrics
        
        # Calculer le déséquilibre de liquidité
        liquidity_imbalance = 0.0
        if metrics.vwap_ask and metrics.vwap_bid and metrics.mid_price:
            spread = metrics.vwap_ask - metrics.vwap_bid
            if spread > 0:
                liquidity_imbalance = (metrics.mid_price - (metrics.vwap_bid + spread/2)) / (spread/2)
        
        return {
            'order_book_imbalance': float(metrics.imbalance) if metrics.imbalance else 0.0,
            'spread': float(metrics.spread) if metrics.spread else 0.0,
            'best_ask': float(metrics.best_ask) if metrics.best_ask else 0.0,
            'best_bid': float(metrics.best_bid) if metrics.best_bid else 0.0,
            'mid_price': float(metrics.mid_price) if metrics.mid_price else 0.0,
            'vwap_ask': float(metrics.vwap_ask) if metrics.vwap_ask else 0.0,
            'vwap_bid': float(metrics.vwap_bid) if metrics.vwap_bid else 0.0,
            'liquidity_imbalance': liquidity_imbalance,
            'order_imbalance': snapshot.metrics.order_imbalance if hasattr(snapshot.metrics, 'order_imbalance') else 0.0
        }

    def _analyze_trades(self, trades: List[Trade]) -> Dict[str, Any]:
        """Analyse les trades récents."""
        if not trades:
            return {}

        # Calcul des statistiques
        total_volume = sum(float(trade['amount']) for trade in trades)
        buy_volume = sum(float(trade['amount'])
                         for trade in trades if trade['side'] == 'buy')
        sell_volume = total_volume - buy_volume

        # Calcul du ratio acheteurs/vendeurs
        buy_sell_ratio = buy_volume / total_volume if total_volume > 0 else 0

        return {
            'trade_volume': total_volume,
            'buy_volume': buy_volume,
            'sell_volume': sell_volume,
            'buy_sell_ratio': buy_sell_ratio,
            'last_price': float(trades[-1]['price']) if trades else 0.0
        }
        
    def _update_history(self, symbol: str, candles: List[Candle]) -> None:
        """Met à jour l'historique des bougies."""
        if symbol not in self.candle_history:
            self.candle_history[symbol] = []

        # Ajouter les nouvelles bougies
        for candle in candles:
            if candle not in self.candle_history[symbol]:
                self.candle_history[symbol].append(candle)

        # Limiter la taille de l'historique
        max_history = self.config.get('max_history', 1000)
        if len(self.candle_history[symbol]) > max_history:
            self.candle_history[symbol] = self.candle_history[symbol][-max_history:]

    def get_market_data(self, symbol: str) -> Optional[MarketData]:
        """Récupère les données de marché pour une paire."""
        return self.market_data.get(symbol)

    def get_candle_history(self, symbol: str) -> List[Candle]:
        """Récupère l'historique des bougies pour une paire."""
        return self.candle_history.get(symbol, [])

    def get_order_book(self, symbol: str) -> Optional[OrderBookType]:
        """
        Récupère le carnet d'ordres actuel pour une paire.

        Args:
            symbol: Symbole de la paire

        Returns:
            Le carnet d'ordres ou None si non disponible
        """
        if symbol not in self.order_books:
            return None
            
        snapshot = self.order_books[symbol].current_snapshot
        if not snapshot:
            return None
            
        return {
            'asks': snapshot.asks,
            'bids': snapshot.bids,
            'timestamp': snapshot.timestamp
        }
        
    def get_order_book_manager(self, symbol: str) -> Optional[OrderBookManager]:
        """
        Récupère le gestionnaire de carnet d'ordres pour une paire.
        
        Args:
            symbol: Symbole de la paire
            
        Returns:
            L'instance OrderBookManager ou None si non disponible
        """
        return self.order_books.get(symbol)
        
    async def stop_order_book_updates(self, symbol: str = None) -> None:
        """
        Arrête les mises à jour automatiques du carnet d'ordres.
        
        Args:
            symbol: Symbole de la paire (si None, arrête toutes les mises à jour)
        """
        if symbol is not None:
            if symbol in self._order_book_update_tasks:
                task = self._order_book_update_tasks.pop(symbol)
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
                
            if symbol in self.order_books:
                await self.order_books[symbol].stop()
        else:
            # Arrêter toutes les mises à jour
            tasks = list(self._order_book_update_tasks.values())
            self._order_book_update_tasks.clear()
            
            for task in tasks:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
                    
            for ob in self.order_books.values():
                await ob.stop()

    def get_trades(self, symbol: str) -> List[Trade]:
        """Récupère les trades pour une paire."""
        return self.trades.get(symbol, [])

    async def stream_market_data(
            self,
            symbols: List[str],
            interval: str = '1m') -> AsyncIterator[MarketData]:
        """
        Stream des données de marché en temps réel.

        Args:
            symbols: Liste des paires à suivre
            interval: Intervalle des bougies

        Yields:
            Données de marché mises à jour
        """
        while True:
            try:
                for symbol in symbols:
                    data = await self.update_market_data(symbol)
                    yield data

                # Attendre l'intervalle
                await asyncio.sleep(float(interval[:-1]) * 60)

            except Exception as e:
                self.logger.error(f"Erreur dans le stream des données: {e}")
                await asyncio.sleep(10)  # Attendre avant de réessayer
                
    async def close(self):
        """
        Ferme le gestionnaire et libère les ressources.
        
        Cette méthode doit être appelée à l'arrêt de l'application pour éviter les fuites de mémoire.
        """
        try:
            # Arrêter toutes les mises à jour du carnet d'ordres
            await self.stop_order_book_updates()
            
            # Nettoyer les autres ressources
            self.market_data.clear()
            self.candle_history.clear()
            self.trades.clear()
            
            # Nettoyer les gestionnaires de carnet d'ordres
            for ob in list(self.order_books.values()):
                await ob.stop()
            self.order_books.clear()
            
            # Annuler toutes les tâches en attente
            for task in list(self._order_book_update_tasks.values()):
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
            self._order_book_update_tasks.clear()
            
            self.logger.info("Gestionnaire de données de marché arrêté avec succès")
            
        except Exception as e:
            self.logger.error(f"Erreur lors de la fermeture du gestionnaire de données: {e}")
            raise
