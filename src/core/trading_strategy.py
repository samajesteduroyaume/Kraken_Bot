from typing import Dict
import logging
import json
import pandas as pd
import asyncio
from datetime import datetime, timedelta
from .kraken_api import KrakenAPI
from .trading_simulation import TradingSimulation, TradingParameters

logger = logging.getLogger('kraken_strategy')


class TradingStrategy:
    """Classe de gestion de stratégie de trading."""

    def __init__(self, api: KrakenAPI, pair: str, timeframe: int = 60):
        """
        Initialise la stratégie de trading.

        Args:
            api: Instance de KrakenAPI
            pair: Paire de trading
            timeframe: Intervalle de temps en minutes (1, 5, 15, 30, 60, 240, 1440, 10080, 21600)
        """
        self.api = api
        self.pair = pair
        self.timeframe = timeframe
        self.simulation = None
        self.last_update = None
        self.data = None
        self.positions = []
        self.signals = []
        self.metrics = {
            'total_trades': 0,
            'total_profit': 0.0,
            'max_drawdown': 0.0,
            'current_positions': 0
        }

        # Paramètres de trading par défaut
        self.parameters = TradingParameters(
            initial_balance=10000.0,
            risk_per_trade=0.01,
            leverage=1.0,
            stop_loss_percent=0.02,
            take_profit_percent=0.04,
            max_positions=5,
            max_drawdown=0.1,
            max_holding_time=timedelta(days=7)
        )

    async def initialize(self) -> None:
        """Initialise la stratégie et la simulation."""
        self.simulation = TradingSimulation(self.parameters)
        await self.update_data()
        logger.info(
            f"Stratégie initialisée pour {self.pair} avec timeframe {self.timeframe}")

    async def update_data(self) -> pd.DataFrame:
        """
        Met à jour les données historiques.

        Returns:
            DataFrame avec les données OHLC
        """
        try:
            # Convertir l'intervalle en minutes
            interval_map = {
                1: 1,    # 1 minute
                5: 5,    # 5 minutes
                15: 15,  # 15 minutes
                30: 30,  # 30 minutes
                60: 60,  # 1 heure
                240: 240,  # 4 heures
                1440: 1440,  # 1 jour
                10080: 10080,  # 1 semaine
                21600: 21600  # 1 mois
            }

            if self.timeframe not in interval_map:
                raise ValueError(
                    f"Intervalle invalide. Valeurs valides: {list(interval_map.keys())}")

            # Convertir la paire au format Kraken
            kraken_pair = self.pair.replace('/', '').upper()
            if kraken_pair == 'BTCUSD':
                kraken_pair = 'XXBTZUSD'

            # Récupérer les données OHLC
            data = await self.api.get_ohlc_data(
                pair=kraken_pair,
                interval=interval_map[self.timeframe],
                since=int((datetime.now() - timedelta(days=30)).timestamp())
            )

            # Logging détaillé pour déboguer
            logger.debug(f"Données brutes de l'API: {data}")
            logger.debug(f"Type des données: {type(data)}")
            logger.debug(f"Type des données result: {type(data['result'])}")
            logger.debug(
                f"Type des données pair: {type(data['result'][kraken_pair])}")
            if data['result'][kraken_pair]:
                logger.debug(
                    f"Première entrée: {data['result'][kraken_pair][0]}")
                logger.debug(
                    f"Type de la première entrée: {type(data['result'][kraken_pair][0])}")

            # Convertir en DataFrame
            # Les données OHLC retournées par Kraken sont dans l'ordre suivant:
            # [time, open, high, low, close, vwap, volume, count]
            try:
                # Vérifier si les données existent
                if kraken_pair not in data['result']:
                    raise ValueError(
                        f"Paire {kraken_pair} non trouvée dans la réponse")

                # Vérifier si les données sont une liste
                if not isinstance(data['result'][kraken_pair], list):
                    raise ValueError(
                        f"Format des données invalide: {type(data['result'][kraken_pair])}")
                # Convertir les données en DataFrame de manière plus robuste
                df = pd.DataFrame(
                    data['result'][kraken_pair],
                    columns=[
                        'time',
                        'open',
                        'high',
                        'low',
                        'close',
                        'vwap',
                        'volume',
                        'count'])

                # Vérifier que le DataFrame n'est pas vide
                if df.empty:
                    raise ValueError("DataFrame vide après conversion")

                # Vérifier qu'il n'y a pas de valeurs NaN
                if df.isnull().values.any():
                    raise ValueError("Valeurs NaN dans les données OHLC")

                self.last_update = datetime.now()
                return df

            except Exception:
                # Si la conversion échoue, essayer avec json.loads
                try:
                    raw_data = json.loads(json.dumps(
                        data['result'][kraken_pair]))
                    df = pd.DataFrame(
                        raw_data,
                        columns=[
                            'time',
                            'open',
                            'high',
                            'low',
                            'close',
                            'vwap',
                            'volume',
                            'count'])

                    # Vérifier que le DataFrame n'est pas vide
                    if df.empty:
                        raise ValueError("DataFrame vide après conversion")

                    # Vérifier qu'il n'y a pas de valeurs NaN
                    if df.isnull().values.any():
                        raise ValueError("Valeurs NaN dans les données OHLC")

                    self.last_update = datetime.now()
                    return df

                except Exception as e2:
                    logger.error(
                        f"Erreur lors de la conversion des données: {str(e2)}")
                    raise

        except Exception as e:
            logger.error(
                f"Erreur lors de la récupération des données: {str(e)}")
            raise

    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calcule les indicateurs techniques.

        Args:
            data: Données OHLC

        Returns:
            DataFrame avec les indicateurs
        """
        # Vérifier que les colonnes existent
        required_columns = [
            'time',
            'open',
            'high',
            'low',
            'close',
            'vwap',
            'volume',
            'count']
        missing_columns = [
            col for col in required_columns if col not in data.columns]
        if missing_columns:
            raise ValueError(f"Colonnes manquantes: {missing_columns}")

        # Calcul des moyennes mobiles
        data['sma_20'] = data['close'].rolling(window=20).mean()
        data['sma_50'] = data['close'].rolling(window=50).mean()

        # Calcul du RSI
        delta = data['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        data['rsi'] = 100 - (100 / (1 + rs))

        # Calcul des bandes de Bollinger
        data['bb_middle'] = data['close'].rolling(window=20).mean()
        data['bb_std'] = data['close'].rolling(window=20).std()
        data['bb_upper'] = data['bb_middle'] + 2 * data['bb_std']
        data['bb_lower'] = data['bb_middle'] - 2 * data['bb_std']

        return data

    def generate_signals(self, data: pd.DataFrame) -> Dict:
        """
        Génère les signaux de trading.

        Args:
            data: Données avec indicateurs

        Returns:
            Dictionnaire des signaux
        """
        signals = {}

        # Conditions d'entrée
        if data['close'].iloc[-1] > data['sma_20'].iloc[-1] and \
           data['rsi'].iloc[-1] < 30 and \
           data['close'].iloc[-1] < data['bb_lower'].iloc[-1]:
            signals[self.pair] = {
                'action': 'buy',
                'size': 1.0,
                'leverage': self.parameters.leverage,
                'stop_loss': data['close'].iloc[-1] * (1 - self.parameters.stop_loss_percent),
                'take_profit': data['close'].iloc[-1] * (1 + self.parameters.take_profit_percent)
            }

        # Conditions de sortie
        elif data['close'].iloc[-1] < data['sma_20'].iloc[-1] or \
                data['rsi'].iloc[-1] > 70 or \
                data['close'].iloc[-1] > data['bb_upper'].iloc[-1]:
            signals[self.pair] = {
                'action': 'sell'
            }

        return signals

    async def get_metrics(self) -> dict:
        """Récupère les métriques de performance."""
        return {
            'trades': self.metrics['total_trades'],
            'profit': self.metrics['total_profit'],
            'drawdown': self.metrics['max_drawdown'],
            'positions': self.metrics['current_positions']
        }

    def get_open_positions(self) -> list:
        """Récupère les positions ouvertes."""
        return [
            {
                'pair': pos.pair,
                'type': pos.type,
                'size': pos.size,
                'entry_price': pos.entry_price,
                'pnl': pos.calculate_pnl(self.data['close'].iloc[-1])
            }
            for pos in self.positions
        ]

    def get_latest_signals(self) -> list:
        """Récupère les signaux de trading les plus récents."""
        return [
            {
                'symbol': sig.symbol,
                'action': sig.action,
                'price': sig.price,
                'confidence': sig.confidence
            }
            for sig in self.signals[-5:]  # Derniers 5 signaux
        ]

    def get_market_data(self) -> dict:
        """Récupère les données de marché et les indicateurs techniques."""
        try:
            if self.data is None:
                return {
                    'price': 0,
                    'rsi': 0,
                    'macd': 0,
                    'atr': 0
                }

            # Calculer les indicateurs techniques
            close_prices = self.data['close']

            # RSI
            delta = close_prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))

            # MACD
            ema_12 = close_prices.ewm(span=12).mean()
            ema_26 = close_prices.ewm(span=26).mean()
            macd = ema_12 - ema_26

            # ATR
            high = self.data['high']
            low = self.data['low']
            tr = pd.DataFrame()
            tr['h-l'] = high - low
            tr['h-pc'] = abs(high - close_prices.shift(1))
            tr['l-pc'] = abs(low - close_prices.shift(1))
            tr['tr'] = tr[['h-l', 'h-pc', 'l-pc']].max(axis=1)
            atr = tr['tr'].rolling(window=14).mean()

            return {
                'price': close_prices.iloc[-1],
                'rsi': rsi.iloc[-1],
                'macd': macd.iloc[-1],
                'atr': atr.iloc[-1]
            }

        except Exception as e:
            logger.error(f"Erreur lors du calcul des indicateurs: {str(e)}")
            return {
                'price': 0,
                'rsi': 0,
                'macd': 0,
                'atr': 0
            }

    async def run(self) -> None:
        """Exécute la stratégie de trading."""
        try:
            # Récupération des données
            data = await self.update_data()

            # Calcul des indicateurs
            data = self.calculate_indicators(data)

            # Génération des signaux
            self.generate_signals(data)

            # Exécution de la simulation
            metrics = self.simulation.run_simulation(
                data, self.generate_signals)

            # Log des résultats
            logger.info(f"Métriques de la stratégie:\n"
                        f"Trades: {metrics['total_trades']}\n"
                        f"Profit: {metrics['total_profit']:.2f}\n"
                        f"Drawdown: {metrics['max_drawdown']:.2%}\n"
                        f"Positions: {metrics['current_positions']}")

        except Exception as e:
            logger.error(
                f"Erreur lors de l'exécution de la stratégie: {str(e)}")
            raise

    async def monitor(self, interval: int = 3600) -> None:
        """
        Surveille et exécute la stratégie périodiquement.

        Args:
            interval: Intervalle en secondes
        """
        while True:
            try:
                await self.run()
                await asyncio.sleep(interval)
            except Exception as e:
                logger.error(f"Erreur lors de la surveillance: {str(e)}")
                await asyncio.sleep(60)  # Attente avant de réessayer
                continue
