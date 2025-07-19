"""
Implémentation d'une stratégie de breakout pour le trading.
"""
from typing import Dict, List, Tuple, Optional
import pandas as pd
from ta.trend import ADXIndicator
from ta.volatility import BollingerBands, AverageTrueRange
from ta.momentum import RSIIndicator, ROCIndicator
import logging

from .base_strategy import BaseStrategy

logger = logging.getLogger(__name__)


class BreakoutStrategy(BaseStrategy):
    """
    Stratégie de trading basée sur les cassures de prix avec une approche multi-timeframe.

    Cette stratégie identifie les points de cassure importants et génère des signaux
    d'achat/vente basés sur ces cassures, avec une gestion du risque avancée.
    """

    def __init__(self, config: Optional[Dict] = None):
        """
        Initialise la stratégie de breakout avec la configuration fournie.

        Args:
            config (Optional[Dict]): Configuration de la stratégie
        """
        super().__init__(config)

        # Configuration des indicateurs
        self.indicator_config = {
            'adx': {
                'period': self.config.get('adx_period', 14),
                'threshold': self.config.get('adx_threshold', 25)
            },
            'atr': {
                'period': self.config.get('atr_period', 14),
                'multiplier': self.config.get('atr_multiplier', 2.0)
            },
            'bb': {
                'window': self.config.get('bb_window', 20),
                'std': self.config.get('bb_std', 2.0)
            },
            'rsi': {
                'period': self.config.get('rsi_period', 14),
                'overbought': self.config.get('rsi_overbought', 70),
                'oversold': self.config.get('rsi_oversold', 30)
            },
            'roc': {
                'period': self.config.get('roc_period', 14)
            }
        }

        # Configuration de la confirmation
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
        # Validation des indicateurs
        for ind, params in self.indicator_config.items():
            for param, value in params.items():
                if not isinstance(value, (int, float)) or value <= 0:
                    raise ValueError(
                        f"Paramètre invalide pour {ind}: {param} = {value}")

        # Validation des paramètres de confirmation
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
        Calcule les indicateurs techniques et analyse le carnet d'ordres pour la stratégie de breakout.

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

            # ADX pour la force de la tendance
            adx = ADXIndicator(
                high=df['high'],
                low=df['low'],
                close=df['close'],
                window=self.indicator_config['adx']['period']
            )
            df['adx'] = adx.adx()
            df['adx_pos'] = adx.adx_pos()
            df['adx_neg'] = adx.adx_neg()
            df['adx_threshold'] = self.indicator_config['adx']['threshold']

            # ATR pour le calcul du risque
            atr = AverageTrueRange(
                high=df['high'],
                low=df['low'],
                close=df['close'],
                window=self.indicator_config['atr']['period']
            )
            df['atr'] = atr.average_true_range()
            df['atr_multiplier'] = self.indicator_config['atr']['multiplier']

            # Bandes de Bollinger
            bb = BollingerBands(
                close=df['close'],
                window=self.indicator_config['bb']['window'],
                window_dev=self.indicator_config['bb']['std']
            )
            df['bb_upper'] = bb.bollinger_hband()
            df['bb_lower'] = bb.bollinger_lband()
            df['bb_middle'] = bb.bollinger_mavg()
            df['bb_width'] = bb.bollinger_wband()

            # RSI pour la confirmation
            rsi = RSIIndicator(
                close=df['close'],
                window=self.indicator_config['rsi']['period'])
            df['rsi'] = rsi.rsi()
            df['rsi_overbought'] = self.indicator_config['rsi']['overbought']
            df['rsi_oversold'] = self.indicator_config['rsi']['oversold']

            # ROC pour la vitesse de la cassure
            roc = ROCIndicator(
                close=df['close'],
                window=self.indicator_config['roc']['period'])
            df['roc'] = roc.roc()

            # Support et résistance
            df['support'] = df['low'].rolling(
                window=self.indicator_config['atr']['period']).min()
            df['resistance'] = df['high'].rolling(
                window=self.indicator_config['atr']['period']).max()

            # Volume
            df['volume_ma'] = df['volume'].rolling(
                window=self.indicator_config['atr']['period']).mean()
            df['volume_ratio'] = df['volume'] / df['volume_ma']

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

            # Force de la tendance
            df['trend_strength'] = (
                (df['adx'] / 100) * 0.4 +
                (df['rsi'] / 100) * 0.3 +
                (df['volume_ratio'] - 1) * 0.2 +
                (df['orderbook_distance'] * 100) *
                0.1  # Impact du carnet d'ordres
            )

            # Confirmation des signaux
            df['signal_confirmation'] = df['trend_strength'].rolling(
                window=self.confirmation_config['period']
            ).mean()

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
                    self.indicator_config['adx']['period'],
                    self.indicator_config['bb']['window']):
                continue

            last_row = df.iloc[-1]

            # Conditions de tendance
            trend_up = (
                last_row['adx'] > last_row['adx_threshold'] and
                last_row['adx_pos'] > last_row['adx_neg']
            )

            trend_down = (
                last_row['adx'] > last_row['adx_threshold'] and
                last_row['adx_pos'] < last_row['adx_neg']
            )

            # Conditions de breakout avec analyse du carnet d'ordres
            breakout_up = (
                last_row['close'] > last_row['bb_upper'] and
                last_row['close'] > last_row['resistance'] and
                last_row['volume_ratio'] > 1.5 and
                last_row['roc'] > 0 and
                # Plus de demandes que d'offres
                (
                    order_book and tf in order_book and last_row['orderbook_ratio'] > 1.2)
            )

            breakout_down = (
                last_row['close'] < last_row['bb_lower'] and
                last_row['close'] < last_row['support'] and
                last_row['volume_ratio'] > 1.5 and
                last_row['roc'] < 0 and
                # Plus d'offres que de demandes
                (
                    order_book and tf in order_book and last_row['orderbook_ratio'] < 0.8)
            )

            # Confirmation de la tendance
            trend_strength = last_row['trend_strength']
            signal_confirmation = last_row['signal_confirmation']

            # Génération des signaux avec confirmation multi-timeframe
            if tf == '15m':  # Timeframe principale
                if trend_up and breakout_up:
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

                elif trend_down and breakout_down:
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
        Calcule la taille de position optimale en fonction du risque.

        Args:
            close (float): Prix actuel
            stop_loss (float): Niveau de stop loss

        Returns:
            float: Taille de position optimale
        """
        risk_amount = self.risk_per_trade * self.portfolio_value
        potential_risk = abs(close - stop_loss)
        position_size = (risk_amount / potential_risk) * self.max_position_size
        return position_size

    def validate_data(self, data: Dict[str, pd.DataFrame]) -> bool:
        """
        Valide les données OHLCV pour chaque timeframe.

        Args:
            data (Dict[str, pd.DataFrame]): Données par timeframe

        Returns:
            bool: True si les données sont valides, False sinon
        """
        if not isinstance(data, dict):
            logger.error("Les données doivent être un dictionnaire")
            return False

        for tf, df in data.items():
            if not isinstance(df, pd.DataFrame):
                logger.error(f"Données invalides pour le timeframe {tf}")
                return False

            required_columns = ['open', 'high', 'low', 'close', 'volume']
            if not all(col in df.columns for col in required_columns):
                logger.error(f"Colonnes manquantes pour le timeframe {tf}")
                return False

            if len(df) < max(
                    self.indicator_config['adx']['period'],
                    self.indicator_config['bb']['window']):
                logger.error(f"Pas assez de données pour le timeframe {tf}")
                return False

        return True

    def get_required_timeframes(self) -> List[str]:
        """
        Retourne la liste des timeframes nécessaires pour la stratégie.

        Returns:
            List[str]: Liste des timeframes
        """
        return ['15m', '1h', '4h', '1d']

    def get_required_indicators(self) -> List[str]:
        """
        Retourne la liste des indicateurs techniques nécessaires.

        Returns:
            List[str]: Liste des indicateurs
        """
        return ['adx', 'atr', 'bb', 'rsi', 'roc']
