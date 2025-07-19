"""
Implémentation d'une stratégie de retour à la moyenne (Mean Reversion) pour le trading.
"""
from typing import Dict, Tuple, Optional
import pandas as pd
from ta.volatility import BollingerBands
from ta.momentum import RSIIndicator, ROCIndicator
from ta.trend import SMAIndicator
import logging

from .base_strategy import BaseStrategy

logger = logging.getLogger(__name__)


class MeanReversionStrategy(BaseStrategy):
    """
    Stratégie de mean reversion qui identifie les opportunités de trading
    lors des retours à la moyenne.

    Cette stratégie utilise les bandes de Bollinger, le RSI et la volatilité
    pour identifier les points d'entrée et de sortie optimaux.
    """

    def __init__(self, config: Optional[Dict] = None):
        """
        Initialise la stratégie de mean reversion avec la configuration fournie.

        Args:
            config (Optional[Dict]): Configuration de la stratégie
        """
        super().__init__(config)

        # Configuration des indicateurs
        self.indicator_config = {
            'bb': {
                'window': self.config.get('bb_window', 20),
                'std': self.config.get('bb_std', 2.0)
            },
            'rsi': {
                'period': self.config.get('rsi_period', 14),
                'overbought': self.config.get('rsi_overbought', 70),
                'oversold': self.config.get('rsi_oversold', 30)
            },
            'sma': {
                'period': self.config.get('sma_period', 50)
            },
            'roc': {
                'period': self.config.get('roc_period', 14)
            }
        }

        # Configuration de la volatilité
        self.volatility_config = {
            'z_score_threshold': self.config.get(
                'z_score_threshold', 2.0), 'stop_loss_multiplier': self.config.get(
                'stop_loss_multiplier', 1.5)}

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

        # Validation de la volatilité
        if not isinstance(
            self.volatility_config['z_score_threshold'],
            (int,
             float)) or self.volatility_config['z_score_threshold'] <= 0:
            raise ValueError(
                f"Seuil Z-score invalide: {self.volatility_config['z_score_threshold']}")

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
        Calcule les indicateurs techniques et analyse le carnet d'ordres pour la stratégie de mean reversion.

        Args:
            data (Dict[str, pd.DataFrame]): Données OHLCV par timeframe
            order_book (Optional[Dict[str, Dict]]): Carnet d'ordres par timeframe

        Returns:
            Dict[str, pd.DataFrame]: Données avec les indicateurs et l'analyse du carnet d'ordres
        """
        for tf, df in data.items():
            # Copie pour éviter les modifications en place
            df = df.copy()

            # Bandes de Bollinger
            bb = BollingerBands(
                close=df['close'],
                window=self.indicator_config['bb']['window'],
                window_dev=self.indicator_config['bb']['std'])
            df['bb_upper'] = bb.bollinger_hband()
            df['bb_middle'] = bb.bollinger_mavg()
            df['bb_lower'] = bb.bollinger_lband()
            df['bb_width'] = bb.bollinger_wband()

            # RSI
            rsi = RSIIndicator(
                close=df['close'],
                window=self.indicator_config['rsi']['period'])
            df['rsi'] = rsi.rsi()
            df['rsi_overbought'] = self.indicator_config['rsi']['overbought']
            df['rsi_oversold'] = self.indicator_config['rsi']['oversold']

            # SMA
            sma = SMAIndicator(
                close=df['close'],
                window=self.indicator_config['sma']['period'])
            df['sma'] = sma.sma_indicator()

            # ROC (Rate of Change)
            roc = ROCIndicator(
                close=df['close'],
                window=self.indicator_config['roc']['period'])
            df['roc'] = roc.roc()

            # Calcul du Z-score
            df['z_score'] = (df['close'] - df['sma']) / df['bb_width']
            df['z_score_threshold'] = self.volatility_config['z_score_threshold']

            # Volatilité
            df['volatility'] = df['bb_width'] / df['bb_middle']

            # Force de la tendance
            df['trend_strength'] = (
                (df['rsi'] / 100) * 0.4 +
                (1 - df['volatility']) * 0.3 +
                (df['z_score'].abs() / self.volatility_config['z_score_threshold']) * 0.3
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
                (df['bb_width'] * self.volatility_config['stop_loss_multiplier'])

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
                    self.indicator_config['bb']['window'],
                    self.indicator_config['rsi']['period']):
                continue

            last_row = df.iloc[-1]

            # Conditions de tendance
            trend_up = (
                last_row['close'] > last_row['sma'] and
                last_row['rsi'] > last_row['rsi_oversold']
            )

            trend_down = (
                last_row['close'] < last_row['sma'] and
                last_row['rsi'] < last_row['rsi_overbought']
            )

            # Conditions de mean reversion avec analyse du carnet d'ordres
            mean_reversion_up = (
                last_row['close'] < last_row['bb_lower'] and
                last_row['rsi'] < last_row['rsi_oversold'] and
                last_row['volume_ratio'] > 1.5 and
                # Plus de demandes que d'offres
                (
                    order_book and tf in order_book and last_row['orderbook_ratio'] > 1.2)
            )
            last_row['bb_upper']
            last_row['bb_lower']

            # Confirmation de la tendance
            trend_strength = last_row['trend_strength']
            signal_confirmation = last_row['signal_confirmation']

            # Génération des signaux avec confirmation multi-timeframe
            if tf == '15m':  # Timeframe principale
                if trend_up and z_score < - \
                        self.volatility_config['z_score_threshold']:
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

                elif trend_down and z_score > self.volatility_config['z_score_threshold']:
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
            close (float): Prix de clôture
            stop_loss (float): Niveau de stop loss

        Returns:
            float: Taille de position optimale
        """
        risk_amount = self.risk_per_trade * self.portfolio_value
        potential_risk = abs(close - stop_loss)
        position_size = (risk_amount / potential_risk) * self.max_position_size
        return position_size

    def calculate_confidence(
            self, data: Dict[str, pd.DataFrame], signals: pd.Series) -> pd.Series:
        """
        Calcule un score de confiance pour chaque signal basé sur la force du retour à la moyenne.

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

        # Calcul de la confiance pour les signaux d'achat
        buy_signals = signals == 1
        if buy_signals.any():
            # Plus l'écart à la moyenne est grand, plus la confiance est élevée
            buy_confidence = (
                # Distance à la bande inférieure
                0.3 * (1 - (df['close'] / df['bb_low'])) +
                # Distance au RSI
                0.2 * ((self.rsi_oversold - df['rsi']) / self.rsi_oversold) +
                # Distance à la moyenne en écart-type
                0.3 * (-df['zscore'] / 3) +
                0.2 * (df['roc'] / df['roc'].abs().max())  # Force du ROC
            )
            confidence[buy_signals] = buy_confidence[buy_signals].clip(0, 1)

        # Calcul de la confiance pour les signaux de vente
        sell_signals = signals == -1
        if sell_signals.any():
            # Plus l'écart à la moyenne est grand, plus la confiance est élevée
            sell_confidence = (
                # Distance à la bande supérieure
                0.3 * ((df['close'] / df['bb_high']) - 1) +
                # Distance au RSI
                0.2 * ((df['rsi'] - self.rsi_overbought) / (100 - self.rsi_overbought)) +
                # Distance à la moyenne en écart-type
                0.3 * (df['zscore'] / 3) +
                0.2 * (-df['roc'] / df['roc'].abs().max())  # Force du ROC
            )
            confidence[sell_signals] = sell_confidence[sell_signals].clip(0, 1)

        # Lissage de la confiance avec une moyenne mobile
        confidence = confidence.rolling(window=3, min_periods=1).mean()

        # Assure que la confiance est entre 0 et 1
        return confidence.clip(0, 1)
