"""
Implémentation de la stratégie de swing trading.
Cette stratégie vise à capturer les mouvements de prix à court et moyen terme.
"""
from typing import Dict, Optional, Tuple
import pandas as pd
from .base_strategy import BaseStrategy

from typing import Dict, Optional, Tuple
import pandas as pd
from ta.momentum import RSIIndicator, ROCIndicator
from ta.trend import MACD, SMAIndicator
from ta.volatility import AverageTrueRange
import logging

from .base_strategy import BaseStrategy

logger = logging.getLogger(__name__)


class SwingStrategy(BaseStrategy):
    """
    Stratégie de swing trading avec une approche multi-timeframe.

    Cette stratégie vise à capturer les mouvements de prix à court et moyen terme,
    avec une gestion du risque avancée et une confirmation des signaux.
    """

    def __init__(self, config: Optional[Dict] = None):
        """
        Initialise la stratégie de swing trading.

        Args:
            config: Configuration de la stratégie avec les paramètres suivants:
                - rsi_period: Période pour le RSI (par défaut: 14)
                - rsi_overbought: Seuil de surachat RSI (par défaut: 70)
                - rsi_oversold: Seuil de survente RSI (par défaut: 30)
                - macd_fast: Période rapide pour le MACD (par défaut: 12)
                - macd_slow: Période lente pour le MACD (par défaut: 26)
                - macd_signal: Période du signal MACD (par défaut: 9)
                - atr_period: Période pour l'ATR (par défaut: 14)
                - atr_multiplier: Multiplicateur pour le stop loss ATR (par défaut: 2.0)
                - take_profit_multiplier: Multiplicateur du take profit (par défaut: 2.0)
                - risk_per_trade: Risque par trade (par défaut: 1%)
                - max_position_size: Taille maximale de position (par défaut: 15%)
        """
        super().__init__(config)

        # Paramètres par défaut
        self.default_config = {
            'rsi_period': 14,
            'rsi_overbought': 70,
            'rsi_oversold': 30,
            'macd_fast': 12,
            'macd_slow': 26,
            'macd_signal': 9,
            'atr_period': 14,
            'atr_multiplier': 2.0,
            'take_profit_multiplier': 2.0,
            'risk_per_trade': 0.01,  # 1% par trade
            'max_position_size': 0.15,  # 15% du capital
            'confirmation_period': 3,
            'trend_strength_threshold': 0.7
        }

        # Fusion avec la configuration fournie
        self.config = {**self.default_config, **(config or {})}

        # Définir les timeframes nécessaires
        self.timeframes = ['15m', '1h', '4h', '1d']  # Analyse multi-timeframe

        # État de la tendance
        self.trend = {
            'direction': 'neutral',
            'strength': 0.0,
            'last_update': None
        }

    def calculate_indicators(self,
                             data: Dict[str,
                                        pd.DataFrame],
                             order_book: Optional[Dict[str,
                                                       Dict]] = None) -> Dict[str,
                                                                              pd.DataFrame]:
        """
        Calcule les indicateurs techniques et analyse le carnet d'ordres pour la stratégie swing.

        Args:
            data (Dict[str, pd.DataFrame]): Données OHLCV par timeframe
            order_book (Optional[Dict[str, Dict]]): Carnet d'ordres par timeframe

        Returns:
            Dict[str, pd.DataFrame]: Données avec les indicateurs et l'analyse du carnet d'ordres
        """
        if not self.validate_data(data):
            logger.error("Données invalides pour le calcul des indicateurs")
            return data

        for tf, df in data.items():
            # Copie pour éviter les modifications en place
            df = df.copy()

            # RSI
            rsi = RSIIndicator(
                close=df['close'],
                window=self.config['rsi_period'])
            df['rsi'] = rsi.rsi()
            df['rsi_overbought'] = self.config['rsi_overbought']
            df['rsi_oversold'] = self.config['rsi_oversold']

            # MACD
            macd = MACD(
                close=df['close'],
                window_slow=self.config['macd_slow'],
                window_fast=self.config['macd_fast'],
                window_sign=self.config['macd_signal']
            )
            df['macd'] = macd.macd()
            df['macd_signal'] = macd.macd_signal()
            df['macd_diff'] = macd.macd_diff()

            # Moyennes mobiles
            df['sma_50'] = SMAIndicator(
                close=df['close'], window=50).sma_indicator()
            df['sma_200'] = SMAIndicator(
                close=df['close'], window=200).sma_indicator()

            # ATR pour le calcul du risque
            atr = AverageTrueRange(
                high=df['high'],
                low=df['low'],
                close=df['close'],
                window=self.config['atr_period']
            )
            df['atr'] = atr.average_true_range()
            df['atr_multiplier'] = self.config['atr_multiplier']

            # ROC (Rate of Change)
            roc = ROCIndicator(
                close=df['close'],
                window=self.config['rsi_period'])
            df['roc'] = roc.roc()

            # Force de la tendance
            df['trend_strength'] = (
                (df['rsi'] / 100) * 0.3 +
                (df['macd_diff'].abs() / df['macd_signal'].abs()) * 0.3 +
                ((df['sma_50'] - df['sma_200']).abs() / df['sma_200']) * 0.4
            )

            # Confirmation des signaux
            df['signal_confirmation'] = df['trend_strength'].rolling(
                window=self.config['confirmation_period']).mean()

            # Analyse du carnet d'ordres
            if order_book and tf in order_book:
                ob = order_book[tf]

                # Profondeur du carnet d'ordres
                bid_depth = sum(float(order['amount'])
                                for order in ob['bids'][:5])
                ask_depth = sum(float(order['amount'])
                                for order in ob['asks'][:5])

                # Ratio d'offres/demandes
                df['orderbook_ratio'] = bid_depth / (ask_depth + 1e-6)

                # Prix au milieu du carnet
                df['orderbook_mid'] = (
                    float(ob['bids'][0]['price']) + float(ob['asks'][0]['price'])) / 2

                # Distance au prix actuel
                df['orderbook_distance'] = abs(
                    df['close'] - df['orderbook_mid']) / df['close']

            else:
                df['orderbook_ratio'] = 1.0  # Ratio neutre par défaut
                df['orderbook_mid'] = df['close']
                df['orderbook_distance'] = 0.0

            # Score de confiance
            df['confidence_score'] = (
                df['trend_strength'] * 0.4 +
                df['signal_confirmation'] * 0.3 +
                (df['volume_ratio'] - 1) * 0.2 +
                (df['orderbook_ratio'] - 1) * 0.1  # Impact du carnet d'ordres
            )

            # Calcul du stop loss dynamique
            df['stop_loss_level'] = df['close'] - \
                (df['atr'] * self.config['atr_multiplier'])

            # Taille de position optimale
            df['optimal_position_size'] = self.calculate_position_size(df)

            data[tf] = df

        return data

    def calculate_position_size(self, df: pd.DataFrame) -> float:
        """
        Calcule la taille optimale de la position en fonction du risque.

        Args:
            df: Données avec les indicateurs

        Returns:
            Taille optimale de la position
        """
        current_price = df['close'].iloc[-1]
        stop_loss = df['stop_loss_level'].iloc[-1]
        risk_amount = self.config['risk_per_trade'] * self.portfolio_value

        # Calcul du risque potentiel
        potential_risk = abs(current_price - stop_loss)

        # Calcul de la taille de position
        position_size = (risk_amount / potential_risk) * \
            self.config['max_position_size']

        # Application du multiplicateur ATR
        atr_multiplier = df['atr_multiplier'].iloc[-1]
        position_size *= atr_multiplier

        # Limite à la taille maximale
        position_size = min(position_size, self.config['max_position_size'])

        return position_size

    def generate_signals(self,
                         data: Dict[str,
                                    pd.DataFrame],
                         order_book: Optional[Dict[str,
                                                   Dict]] = None) -> Tuple[Dict[str,
                                                                                str],
                                                                           Dict[str,
                                                                                float],
                                                                           Dict[str,
                                                                                float]]:
        """
        Génère les signaux d'achat/vente basés sur les indicateurs calculés.

        Args:
            data: Données avec les indicateurs

        Returns:
            Tuple des signaux, confiances et tailles de position
        """
        if not self.validate_data(data):
            logger.error("Données invalides pour la génération des signaux")
            return {}, {}, {}

        signals = {}
        confidences = {}
        position_sizes = {}

        # Analyse multi-timeframe
        for tf, df in data.items():
            if len(df) < max(
                    self.config['rsi_period'],
                    self.config['macd_slow']):
                continue

            last_row = df.iloc[-1]

            # Conditions de tendance
            trend_up = (
                last_row['sma_50'] > last_row['sma_200'] and
                last_row['macd'] > last_row['macd_signal']
            )

            trend_down = (
                last_row['sma_50'] < last_row['sma_200'] and
                last_row['macd'] < last_row['macd_signal']
            )

            # Conditions de momentum
            rsi_overbought = last_row['rsi'] > last_row['rsi_overbought']
            rsi_oversold = last_row['rsi'] < last_row['rsi_oversold']

            # Confirmation de la tendance
            trend_strength = last_row['trend_strength']
            signal_confirmation = last_row['signal_confirmation']

            # Génération des signaux avec confirmation multi-timeframe
            if tf == '15m':  # Timeframe principale
                if trend_up and rsi_oversold:
                    # Vérification de la confirmation sur les autres timeframes
                    confirmations = []
                    for other_tf in data.keys():
                        if other_tf != '15m':
                            other_df = data[other_tf]
                            other_last = other_df.iloc[-1]
                            if other_last['trend_strength'] > self.config['trend_strength_threshold']:
                                confirmations.append(True)
                            else:
                                confirmations.append(False)

                    if sum(
                            confirmations) / len(confirmations) > self.config['trend_strength_threshold']:
                        signals[tf] = 'buy'
                        confidences[tf] = min(
                            last_row['confidence_score'],
                            trend_strength,
                            signal_confirmation
                        )
                        position_sizes[tf] = last_row['optimal_position_size']
                    else:
                        signals[tf] = 'hold'
                        confidences[tf] = 0.0
                        position_sizes[tf] = 0.0

                elif trend_down and rsi_overbought:
                    confirmations = []
                    for other_tf in data.keys():
                        if other_tf != '15m':
                            other_df = data[other_tf]
                            other_last = other_df.iloc[-1]
                            if other_last['trend_strength'] > self.config['trend_strength_threshold']:
                                confirmations.append(True)
                            else:
                                confirmations.append(False)

                    if sum(
                            confirmations) / len(confirmations) > self.config['trend_strength_threshold']:
                        signals[tf] = 'sell'
                        confidences[tf] = min(
                            last_row['confidence_score'],
                            trend_strength,
                            signal_confirmation
                        )
                        position_sizes[tf] = last_row['optimal_position_size']
                    else:
                        signals[tf] = 'hold'
                        confidences[tf] = 0.0
                        position_sizes[tf] = 0.0

                else:
                    signals[tf] = 'hold'
                    confidences[tf] = 0.0
                    position_sizes[tf] = 0.0

            # Pour les autres timeframes, on utilise uniquement pour la
            # confirmation
            else:
                signals[tf] = 'hold'
                confidences[tf] = 0.0
                position_sizes[tf] = 0.0

        return signals, confidences, position_sizes

    def calculate_signals(self, data: pd.DataFrame) -> Dict[str, any]:
        """
        Calcule les signaux de trading basés sur la stratégie de swing.

        Args:
            data: DataFrame contenant les données OHLCV et indicateurs techniques

        Returns:
            Dictionnaire contenant les signaux et les valeurs des indicateurs
        """
        if data.empty or len(data) < max(
                self.config['rsi_period'],
                self.config['macd_slow']):
            return {'signal': 'HOLD', 'rsi': None, 'macd': None, 'atr': None}

        # Récupérer les dernières valeurs
        current_price = data['close'].iloc[-1]

        # Calculer le RSI
        rsi = self._calculate_rsi(data['close'])

        # Calculer le MACD
        macd_line, signal_line, _ = self._calculate_macd(data['close'])

        # Calculer l'ATR
        atr = self._calculate_atr(data)

        # Détecter les croisements MACD
        macd_cross_up = macd_line.iloc[-2] < signal_line.iloc[-2] and macd_line.iloc[-1] > signal_line.iloc[-1]
        macd_cross_down = macd_line.iloc[-2] > signal_line.iloc[-2] and macd_line.iloc[-1] < signal_line.iloc[-1]

        # Règles d'entrée
        buy_signal = (rsi.iloc[-1] < self.config['rsi_oversold'] and
                      macd_cross_up and
                      current_price > data['sma_50'].iloc[-1])

        sell_signal = (rsi.iloc[-1] > self.config['rsi_overbought'] and
                       macd_cross_down and
                       current_price < data['sma_50'].iloc[-1])

        # Déterminer le signal
        signal = 'HOLD'
        if buy_signal:
            signal = 'BUY'
        elif sell_signal:
            signal = 'SELL'

        return {
            'signal': signal,
            'price': current_price,
            'rsi': rsi.iloc[-1],
            'macd': macd_line.iloc[-1],
            'macd_signal': signal_line.iloc[-1],
            'atr': atr.iloc[-1] if atr is not None else None
        }

    def _calculate_rsi(self, prices: pd.Series) -> pd.Series:
        """Calcule le RSI."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(
            window=self.config['rsi_period']).mean()
        loss = (-delta.where(delta < 0, 0)
                ).rolling(window=self.config['rsi_period']).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    def _calculate_macd(
            self, prices: pd.Series) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calcule le MACD et sa ligne de signal."""
        exp1 = prices.ewm(span=self.config['macd_fast'], adjust=False).mean()
        exp2 = prices.ewm(span=self.config['macd_slow'], adjust=False).mean()
        macd = exp1 - exp2
        signal = macd.ewm(span=self.config['macd_signal'], adjust=False).mean()
        histogram = macd - signal
        return macd, signal, histogram

    def _calculate_atr(self, data: pd.DataFrame) -> pd.Series:
        """Calcule l'Average True Range (ATR)."""
        high = data['high']
        low = data['low']
        close = data['close']

        tr1 = high - low
        tr2 = (high - close.shift()).abs()
        tr3 = (low - close.shift()).abs()

        tr = pd.DataFrame({'tr1': tr1, 'tr2': tr2, 'tr3': tr3}).max(axis=1)
        atr = tr.rolling(window=self.config['atr_period']).mean()

        return atr

    def calculate_position_size(
            self,
            current_price: float,
            balance: float) -> float:
        """
        Calcule la taille de la position basée sur le capital disponible et la volatilité.

        Args:
            current_price: Prix actuel du marché
            balance: Solde disponible pour le trading

        Returns:
            Taille de la position dans l'unité de base
        """
        max_position_value = balance * self.config['max_position_size']
        position_size = max_position_value / current_price
        return position_size

    def calculate_stop_loss(
            self,
            entry_price: float,
            signal: str,
            atr: float = None) -> float:
        """
        Calcule le prix du stop loss basé sur l'ATR.

        Args:
            entry_price: Prix d'entrée de la position
            signal: Type de signal ('BUY' ou 'SELL')
            atr: Valeur de l'ATR (optionnel)

        Returns:
            Prix du stop loss
        """
        if atr is None:
            # Fallback à un pourcentage fixe si l'ATR n'est pas disponible
            stop_pct = 0.02  # 2%
            if signal == 'BUY':
                return entry_price * (1 - stop_pct)
            else:  # SELL
                return entry_price * (1 + stop_pct)

        # Utiliser l'ATR pour le stop loss
        atr_stop = atr * self.config['atr_multiplier']

        if signal == 'BUY':
            return entry_price - atr_stop
        else:  # SELL
            return entry_price + atr_stop

    def calculate_take_profit(
            self,
            entry_price: float,
            stop_loss: float,
            signal: str) -> float:
        """
        Calcule le prix du take profit basé sur le stop loss.

        Args:
            entry_price: Prix d'entrée de la position
            stop_loss: Prix du stop loss
            signal: Type de signal ('BUY' ou 'SELL')

        Returns:
            Prix du take profit
        """
        if signal == 'BUY':
            distance = entry_price - stop_loss
            return entry_price + \
                (distance * self.config['take_profit_multiplier'])
        else:  # SELL
            distance = stop_loss - entry_price
            return entry_price - \
                (distance * self.config['take_profit_multiplier'])
