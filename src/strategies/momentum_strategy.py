"""
Implémentation d'une stratégie de momentum pour le trading.
"""
from typing import Dict, Tuple, Optional
import pandas as pd
from ta.momentum import RSIIndicator, ROCIndicator
from ta.trend import MACD, EMAIndicator, SMAIndicator
from ta.volatility import AverageTrueRange
import logging

from .base_strategy import BaseStrategy

logger = logging.getLogger(__name__)


class MomentumStrategy(BaseStrategy):
    """
    Stratégie de trading basée sur le momentum avec une approche multi-timeframe.

    Cette stratégie utilise une combinaison de RSI, MACD, moyennes mobiles et volatilité
    pour identifier les tendances et générer des signaux d'achat/vente précis.
    Elle inclut une gestion du risque avancée et une confirmation des signaux.
    """

    def __init__(self, config: Optional[Dict] = None):
        """
        Initialise la stratégie de momentum avec la configuration fournie.

        Args:
            config (Optional[Dict]): Configuration de la stratégie
        """
        super().__init__(config)

        # Configuration des indicateurs
        self.indicator_config = {
            'rsi': {
                'period': self.config.get('rsi_period', 14),
                'overbought': self.config.get('rsi_overbought', 70),
                'oversold': self.config.get('rsi_oversold', 30),
                'threshold': self.config.get('rsi_threshold', 50)
            },
            'macd': {
                'fast': self.config.get('macd_fast', 12),
                'slow': self.config.get('macd_slow', 26),
                'signal': self.config.get('macd_signal', 9)
            },
            'ema': {
                'period': self.config.get('ema_period', 20)
            },
            'sma': {
                'period': self.config.get('sma_period', 50)
            },
            'atr': {
                'period': self.config.get('atr_period', 14),
                'multiplier': self.config.get('atr_multiplier', 2.0)
            }
        }

        # Paramètres de confirmation
        self.confirmation_config = {
            'period': self.config.get(
                'confirmation_period', 3), 'trend_strength_threshold': self.config.get(
                'trend_strength_threshold', 0.7)}

        # Validation de la configuration
        self.validate_config()

    def validate_config(self):
        """
        Valide la configuration de la stratégie.
        """
        # Vérification des indicateurs
        for ind, params in self.indicator_config.items():
            for param, value in params.items():
                if not isinstance(value, (int, float)) or value <= 0:
                    raise ValueError(
                        f"Paramètre invalide pour {ind}: {param} = {value}")

        # Vérification des paramètres de confirmation
        if not isinstance(
                self.confirmation_config['period'],
                int) or self.confirmation_config['period'] <= 0:
            raise ValueError(
                f"Période de confirmation invalide: {self.confirmation_config['period']}")

        if not isinstance(
                self.confirmation_config['trend_strength_threshold'],
                float) or not 0 <= self.confirmation_config['trend_strength_threshold'] <= 1:
            raise ValueError(
                f"Seuil de force de tendance invalide: {self.confirmation_config['trend_strength_threshold']}")

    def calculate_indicators(self,
                             data: Dict[str,
                                        pd.DataFrame],
                             order_book: Optional[Dict[str,
                                                       Dict]] = None) -> Dict[str,
                                                                              pd.DataFrame]:
        """
        Calcule les indicateurs techniques et analyse le carnet d'ordres pour la stratégie momentum.

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

            # Calcul des indicateurs
            # RSI
            rsi = RSIIndicator(
                close=df['close'],
                window=self.indicator_config['rsi']['period'])
            df['rsi'] = rsi.rsi()
            df['rsi_overbought'] = self.indicator_config['rsi']['overbought']
            df['rsi_oversold'] = self.indicator_config['rsi']['oversold']
            df['rsi_threshold'] = self.indicator_config['rsi']['threshold']

            # MACD
            macd = MACD(
                close=df['close'],
                window_slow=self.indicator_config['macd']['slow'],
                window_fast=self.indicator_config['macd']['fast'],
                window_sign=self.indicator_config['macd']['signal']
            )
            df['macd'] = macd.macd()
            df['macd_signal'] = macd.macd_signal()
            df['macd_diff'] = macd.macd_diff()

            # Moyennes mobiles
            df['ema'] = EMAIndicator(
                close=df['close'],
                window=self.indicator_config['ema']['period']).ema_indicator()
            df['sma'] = SMAIndicator(
                close=df['close'],
                window=self.indicator_config['sma']['period']).sma_indicator()

            # ATR pour le calcul du risque
            atr = AverageTrueRange(
                high=df['high'],
                low=df['low'],
                close=df['close'],
                window=self.indicator_config['atr']['period']
            )
            df['atr'] = atr.average_true_range()
            df['atr_multiplier'] = self.indicator_config['atr']['multiplier']

            # ROC (Rate of Change)
            roc = ROCIndicator(
                close=df['close'],
                window=self.indicator_config['rsi']['period'])
            df['roc'] = roc.roc()

            # Force de la tendance
            df['trend_strength'] = (
                (df['rsi'] / 100) * 0.3 +
                (df['macd_diff'].abs() / df['macd_signal'].abs()) * 0.3 +
                ((df['ema'] - df['sma']).abs() / df['sma']) * 0.4
            )

            # Confirmation des signaux
            df['signal_confirmation'] = df['trend_strength'].rolling(
                window=self.confirmation_config['period']
            ).mean()

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
                (df['atr'] * self.indicator_config['atr']['multiplier'])

            # Taille de position optimale
            df['optimal_position_size'] = df.apply(
                lambda row: self.calculate_position_size(
                    row['close'], row['stop_loss_level']), axis=1)

            data[tf] = df

        return data

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
            data (Dict[str, pd.DataFrame]): Données avec indicateurs

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
                    self.indicator_config['rsi']['period'],
                    self.indicator_config['macd']['slow']):
                continue

            last_row = df.iloc[-1]

            # Conditions de momentum avec analyse du carnet d'ordres
            momentum_up = (
                last_row['rsi'] < last_row['rsi_oversold'] and
                last_row['macd'] > last_row['macd_signal'] and
                last_row['volume_ratio'] > 1.5 and
                # Plus de demandes que d'offres
                (
                    order_book and tf in order_book and last_row['orderbook_ratio'] > 1.2)
            )

            trend_down = (
                last_row['ema'] < last_row['sma'] and
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
                            if other_last['trend_strength'] > self.confirmation_config['trend_strength_threshold']:
                                confirmations.append(True)
                            else:
                                confirmations.append(False)

                    if sum(
                            confirmations) / len(confirmations) > self.confirmation_config['trend_strength_threshold']:
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
                            if other_last['trend_strength'] > self.confirmation_config['trend_strength_threshold']:
                                confirmations.append(True)
                            else:
                                confirmations.append(False)

                    if sum(
                            confirmations) / len(confirmations) > self.confirmation_config['trend_strength_threshold']:
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

    def calculate_position_size(self, close: float, stop_loss: float) -> float:
        """
        Calcule la taille optimale de la position en fonction du risque.

        Args:
            close (float): Prix de clôture
            stop_loss (float): Niveau de stop loss

        Returns:
            float: Taille optimale de la position
        """
        risk_amount = self.risk_per_trade * self.portfolio_value

        # Calcul du risque potentiel
        potential_risk = abs(close - stop_loss)

        # Calcul de la taille de position
        position_size = (risk_amount / potential_risk) * self.max_position_size

        # Application du multiplicateur ATR
        atr_multiplier = self.indicator_config['atr']['multiplier']
        position_size *= atr_multiplier

        # Limite à la taille maximale
        position_size = min(position_size, self.max_position_size)

        return position_size

    def calculate_confidence(
            self, data: Dict[str, pd.DataFrame], signals: pd.Series) -> pd.Series:
        """
        Calcule un score de confiance pour chaque signal basé sur la convergence des indicateurs.

        Args:
            data (Dict[str, pd.DataFrame]): Données avec indicateurs
            signals (pd.Series): Signaux générés

        Returns:
            pd.Series: Score de confiance pour chaque signal (0.0 à 1.0)
        """
        tf = min(self.timeframes, key=lambda x: int(
            x[:-1]) if x[-1] in ['m', 'h'] else float('inf'))
        df = data[tf].copy()

        # Initialisation de la confiance
        confidence = pd.Series(0.5, index=df.index)

        # Nombre total de conditions (à ajuster selon les indicateurs utilisés)
        total_conditions = 6

        # Calcul du pourcentage de conditions remplies pour les signaux d'achat
        buy_signals = signals == 1
        if buy_signals.any():
            buy_confidence = (
                (df['rsi'] < self.rsi_oversold).astype(int) +
                (df['macd'] > df['macd_signal']).astype(int) +
                (df['ema_short'] > df['ema_medium']).astype(int) +
                (df['ema_medium'] > df['ema_long']).astype(int) +
                (df['stoch_k'] > df['stoch_d']).astype(int) +
                (df['roc'] > 0).astype(int)
            ) / total_conditions
            confidence[buy_signals] = buy_confidence[buy_signals]

        # Calcul du pourcentage de conditions remplies pour les signaux de
        # vente
        sell_signals = signals == -1
        if sell_signals.any():
            sell_confidence = (
                (df['rsi'] > self.rsi_overbought).astype(int) +
                (df['macd'] < df['macd_signal']).astype(int) +
                (df['ema_short'] < df['ema_medium']).astype(int) +
                (df['ema_medium'] < df['ema_long']).astype(int) +
                (df['stoch_k'] < df['stoch_d']).astype(int) +
                (df['roc'] < 0).astype(int)
            ) / total_conditions
            confidence[sell_signals] = sell_confidence[sell_signals]

        # Lissage de la confiance avec une moyenne mobile
        confidence = confidence.rolling(window=3, min_periods=1).mean()

        # Assure que la confiance est entre 0 et 1
        return confidence.clip(0, 1)
