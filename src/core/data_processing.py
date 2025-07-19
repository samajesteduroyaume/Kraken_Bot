"""
Module de traitement des données pour le bot de trading Kraken.
Gère la collecte, le nettoyage et le prétraitement des données.
"""

from typing import Dict, List, Tuple
import pandas as pd
from datetime import datetime, timedelta
import logging
from src.core.api.kraken import KrakenAPI

logger = logging.getLogger(__name__)


class DataProcessor:
    """
    Classe de traitement des données pour le bot de trading.
    """

    def __init__(self, api: 'KrakenAPI', config: Dict):
        """
        Initialise le processeur de données.

        Args:
            api: Instance de l'API Kraken
            config: Configuration du processeur
        """
        self.api = api
        self.config = config
        self.data_cache = {}
        self.last_update = {}

    async def fetch_ohlcv_data(
            self,
            pair: str,
            timeframe: int,
            limit: int = 500) -> pd.DataFrame:
        """
        Récupère les données OHLCV pour une paire donnée.

        Args:
            pair: Paire de trading (ex: 'XXBTZUSD')
            timeframe: Durée en minutes (1, 5, 15, 30, 60, 240, 1440, 10080, 21600)
            limit: Nombre de bougies à récupérer

        Returns:
            DataFrame pandas avec les données OHLCV
        """
        try:
            # Vérifier si les données sont en cache et toujours valides
            cache_key = f"{pair}_{timeframe}"
            now = datetime.now()

            if (cache_key in self.data_cache and cache_key in self.last_update and (
                    now - self.last_update[cache_key]).total_seconds() < self.config['cache_ttl']):
                return self.data_cache[cache_key]

            # Récupérer les données via l'API
            data = await self.api.get_ohlc_data(
                pair=pair,
                interval=timeframe,
                since=int((now - timedelta(days=30)).timestamp())
            )

            # Convertir en DataFrame
            df = pd.DataFrame(
                data,
                columns=[
                    'time',
                    'open',
                    'high',
                    'low',
                    'close',
                    'vwap',
                    'volume',
                    'count'])
            df['time'] = pd.to_datetime(df['time'], unit='s')
            df.set_index('time', inplace=True)

            # Mettre à jour le cache
            self.data_cache[cache_key] = df
            self.last_update[cache_key] = now

            return df

        except Exception as e:
            logger.error(
                f"Erreur lors de la récupération des données OHLCV pour {pair}: {str(e)}")
            raise

    async def calculate_indicators(
            self,
            df: pd.DataFrame,
            indicators: List[str]) -> pd.DataFrame:
        """
        Calcule les indicateurs techniques sur un DataFrame.

        Args:
            df: DataFrame contenant les données OHLCV
            indicators: Liste des indicateurs à calculer

        Returns:
            DataFrame avec les indicateurs ajoutés
        """
        try:
            for indicator in indicators:
                if indicator == 'rsi':
                    df['rsi'] = self.calculate_rsi(df['close'])
                elif indicator == 'macd':
                    df['macd'], df['macd_signal'], df['macd_hist'] = self.calculate_macd(
                        df['close'])
                elif indicator == 'bb':
                    df['bb_upper'], df['bb_middle'], df['bb_lower'] = self.calculate_bollinger_bands(
                        df['close'])
                elif indicator == 'ema':
                    df['ema_20'] = self.calculate_ema(df['close'], 20)
                    df['ema_50'] = self.calculate_ema(df['close'], 50)
                    df['ema_200'] = self.calculate_ema(df['close'], 200)

            return df

        except Exception as e:
            logger.error(f"Erreur lors du calcul des indicateurs: {str(e)}")
            raise

    def calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calcule l'RSI."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    def calculate_macd(self,
                       prices: pd.Series,
                       fast_period: int = 12,
                       slow_period: int = 26,
                       signal_period: int = 9) -> Tuple[pd.Series,
                                                        pd.Series,
                                                        pd.Series]:
        """Calcule le MACD."""
        ema_fast = self.calculate_ema(prices, fast_period)
        ema_slow = self.calculate_ema(prices, slow_period)
        macd = ema_fast - ema_slow
        signal = self.calculate_ema(macd, signal_period)
        hist = macd - signal
        return macd, signal, hist

    def calculate_ema(self, prices: pd.Series, period: int) -> pd.Series:
        """Calcule l'EMA."""
        return prices.ewm(span=period, adjust=False).mean()

    def calculate_bollinger_bands(self,
                                  prices: pd.Series,
                                  period: int = 20,
                                  multiplier: float = 2.0) -> Tuple[pd.Series,
                                                                    pd.Series,
                                                                    pd.Series]:
        """Calcule les bandes de Bollinger."""
        sma = prices.rolling(window=period).mean()
        std = prices.rolling(window=period).std()
        upper = sma + (std * multiplier)
        lower = sma - (std * multiplier)
        return upper, sma, lower
