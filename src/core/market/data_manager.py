import logging
from typing import Dict, List, Optional, Any, AsyncIterator
from datetime import datetime, timedelta
from ..types.market_data import MarketData, Candle, OrderBook, Trade
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
        self.order_book: Dict[str, OrderBook] = {}
        self.trades: Dict[str, List[Trade]] = {}

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

    async def _fetch_order_book(self, symbol: str) -> OrderBook:
        """Récupère le carnet d'ordres de la paire."""
        try:
            # Récupérer le carnet d'ordres via l'API
            order_book = await self.api.get_order_book(
                pair=symbol,
                depth=self.config.get('order_book_depth', 50)
            )

            return order_book

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
                        order_book: OrderBook,
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

    def _analyze_order_book(self, order_book: OrderBook) -> Dict[str, Any]:
        """Analyse le carnet d'ordres."""
        asks = order_book['asks']
        bids = order_book['bids']

        # Calcul des statistiques
        ask_total = sum(float(level['amount']) for level in asks)
        bid_total = sum(float(level['amount']) for level in bids)

        # Calcul du spread
        spread = float(asks[0]['price']) - float(bids[0]['price'])

        return {
            'order_book_imbalance': (ask_total - bid_total) / (ask_total + bid_total),
            'spread': spread,
            'best_ask': float(asks[0]['price']),
            'best_bid': float(bids[0]['price'])
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
            'last_price': float(trades[-1]['price'])
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

    def get_order_book(self, symbol: str) -> Optional[OrderBook]:
        """Récupère le carnet d'ordres pour une paire."""
        return self.order_book.get(symbol)

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
