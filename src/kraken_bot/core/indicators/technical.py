"""
Module d'analyse technique et de prédiction ML pour les données de marché.
"""

import logging
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
from decimal import Decimal
from kraken_bot.core.types.market_types import Candle
from ta import momentum, volatility, trend, volume
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

try:
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.callbacks import EarlyStopping
    _HAS_TF = True
except ImportError:
    _HAS_TF = False


class TechnicalAnalyzer:
    """
    Analyseur technique pour les données de marché.

    Cette classe fournit des méthodes pour calculer différents indicateurs
    techniques sur les données de marché.
    """

    def __init__(self, window_size: int = 20):
        """
        Initialise l'analyseur technique.

        Args:
            window_size: Taille de la fenêtre pour les calculs mobiles
        """
        self.window_size = window_size

    def analyze_candles(self, candles: List[Candle]) -> Dict[str, Any]:
        """
        Analyse une liste de bougies et calcule les indicateurs techniques.

        Args:
            candles: Liste des bougies de marché

        Returns:
            Dictionnaire des indicateurs techniques calculés
        """
        if not candles:
            return {}

        df = pd.DataFrame(candles)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)

        # Calcul des indicateurs
        indicators = {
            # Momentum
            'rsi': momentum.RSIIndicator(df['close'], window=self.window_size).rsi().iloc[-1],
            'stoch_rsi': momentum.StochRSIIndicator(df['close'], window=self.window_size).stochrsi().iloc[-1],
            'awesome_oscillator': momentum.AwesomeOscillatorIndicator(df['high'], df['low'], window1=5, window2=34).awesome_oscillator().iloc[-1],

            # Volatilité
            'bollinger_hband': volatility.BollingerBands(df['close'], window=self.window_size).bollinger_hband().iloc[-1],
            'bollinger_lband': volatility.BollingerBands(df['close'], window=self.window_size).bollinger_lband().iloc[-1],
            'keltner_channel_hband': volatility.KeltnerChannel(df['high'], df['low'], df['close'], window=self.window_size).keltner_channel_hband().iloc[-1],
            'keltner_channel_lband': volatility.KeltnerChannel(df['high'], df['low'], df['close'], window=self.window_size).keltner_channel_lband().iloc[-1],

            # Tendance
            'macd': trend.MACD(df['close']).macd().iloc[-1],
            'macd_signal': trend.MACD(df['close']).macd_signal().iloc[-1],
            'ema_fast': trend.EMAIndicator(df['close'], window=20).ema_indicator().iloc[-1],
            'ema_slow': trend.EMAIndicator(df['close'], window=50).ema_indicator().iloc[-1],

            # Volume
            'volume_adi': volume.AccDistIndexIndicator(df['high'], df['low'], df['close'], df['volume']).acc_dist_index().iloc[-1],
            'volume_obv': volume.OnBalanceVolumeIndicator(df['close'], df['volume']).on_balance_volume().iloc[-1]
        }

        return indicators


class MLPredictor:
    """
    Prédicteur ML pour le trading.

    Cette classe utilise des modèles de machine learning pour prédire les mouvements de prix.
    Elle prend en compte les données techniques, les données de marché et les patterns.
    """

    def __init__(self,
                 config: Optional[Dict] = None):
        """
        Initialise le prédicteur ML.

        Args:
            config: Configuration du prédicteur
        """
        if not _HAS_TF:
            raise ImportError(
                "TensorFlow est requis pour utiliser MLPredictor. Installez-le avec: pip install kraken-trading-bot[ml]")

        self.config = config or {}
        self.logger = logging.getLogger(__name__)

        # Configuration du modèle
        self.model_config = {
            'sequence_length': self.config.get('sequence_length', 60),
            'n_features': self.config.get('n_features', 10),
            'n_units': self.config.get('n_units', 50),
            'dropout': self.config.get('dropout', 0.2),
            'learning_rate': self.config.get('learning_rate', 0.001),
            'batch_size': self.config.get('batch_size', 32),
            'epochs': self.config.get('epochs', 50)
        }

        # Initialiser le modèle
        self.model = self._build_model()

    def _build_model(self) -> 'Sequential':
        """
        Construit le modèle LSTM.

        Returns:
            Modèle LSTM compilé
        """
        model = Sequential()

        # Couche LSTM
        model.add(
            LSTM(
                units=self.model_config['n_units'],
                return_sequences=True,
                input_shape=(
                    self.model_config['sequence_length'],
                    self.model_config['n_features'])))
        model.add(Dropout(self.model_config['dropout']))

        # Couche LSTM supplémentaire
        model.add(LSTM(
            units=self.model_config['n_units'] // 2,
            return_sequences=False
        ))
        model.add(Dropout(self.model_config['dropout']))

        # Couche de sortie
        model.add(Dense(1))

        # Compiler le modèle
        model.compile(
            optimizer='adam',
            loss='mse'
        )

        return model

    def prepare_data(self,
                     data: pd.DataFrame,
                     target_col: str = 'close') -> Tuple[np.ndarray,
                                                         np.ndarray]:
        """
        Prépare les données pour l'entraînement.

        Args:
            data: DataFrame avec les données
            target_col: Colonne cible

        Returns:
            Tuple (X, y)
        """
        # Nettoyer les données
        data = pd.dropna(data)

        # Préparer les séquences
        X, y = [], []
        for i in range(len(data) - self.model_config['sequence_length']):
            X.append(data.iloc[i:i +
                               self.model_config['sequence_length']].values)
            y.append(data[target_col].iloc[i +
                                           self.model_config['sequence_length']])

        return np.array(X), np.array(y)

    def train(self,
              data: pd.DataFrame,
              target_col: str = 'close',
              validation_split: float = 0.2) -> Dict:
        """
        Entraîne le modèle.

        Args:
            data: DataFrame avec les données
            target_col: Colonne cible
            validation_split: Pourcentage de données pour la validation

        Returns:
            Résultats de l'entraînement
        """
        try:
            # Préparer les données
            X, y = self.prepare_data(data, target_col)

            # Diviser les données
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=validation_split, random_state=42
            )

            # Normaliser les données
            scaler = StandardScaler()
            X_train = scaler.fit_transform(
                X_train.reshape(-1, X_train.shape[-1])).reshape(X_train.shape)
            X_val = scaler.transform(
                X_val.reshape(-1, X_val.shape[-1])).reshape(X_val.shape)

            # Entraîner le modèle
            history = self.model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=self.model_config['epochs'],
                batch_size=self.model_config['batch_size'],
                callbacks=[
                    EarlyStopping(
                        monitor='val_loss',
                        patience=10,
                        restore_best_weights=True
                    )
                ]
            )

            # Évaluer le modèle
            train_mse = history.history['loss'][-1]
            val_mse = history.history['val_loss'][-1]

            return {
                'train_mse': train_mse,
                'val_mse': val_mse,
                'history': history.history
            }

        except Exception as e:
            self.logger.error(
                f"Erreur lors de l'entraînement du modèle: {str(e)}")
            raise

    def predict(self,
                data: pd.DataFrame,
                target_col: str = 'close') -> np.ndarray:
        """
        Fait une prédiction.

        Args:
            data: DataFrame avec les données
            target_col: Colonne cible

        Returns:
            Prédictions
        """
        try:
            # Préparer les données
            X, _ = self.prepare_data(data, target_col)

            # Normaliser les données
            scaler = StandardScaler()
            X = scaler.fit_transform(
                X.reshape(-1, X.shape[-1])).reshape(X.shape)

            # Faire la prédiction
            predictions = self.model.predict(X)

            return predictions

        except Exception as e:
            self.logger.error(f"Erreur lors de la prédiction: {str(e)}")
            raise

    def _calculate_trend_strength(self, df: pd.DataFrame) -> float:
        """Calcule la force de la tendance."""
        ema_short = df['close'].ewm(span=10).mean()
        ema_long = df['close'].ewm(span=50).mean()
        return float(
            (ema_short.iloc[-1] - ema_long.iloc[-1]) / df['close'].iloc[-1])

    def _calculate_momentum(self, df: pd.DataFrame) -> float:
        """Calcule la force du momentum."""
        returns = df['close'].pct_change()
        return float(returns.rolling(window=self.window_size).mean().iloc[-1])

    def _calculate_volume_trend(self, df: pd.DataFrame) -> float:
        """Calcule la tendance du volume."""
        volume_ma = df['volume'].rolling(window=self.window_size).mean()
        current_volume = df['volume'].iloc[-1]
        return float(
            (current_volume - volume_ma.iloc[-1]) / volume_ma.iloc[-1])

    def detect_patterns(self, candles: List[Candle]) -> Dict[str, bool]:
        """
        Détecte les patterns de bougies.

        Args:
            candles: Liste des bougies de marché

        Returns:
            Dictionnaire des patterns détectés
        """
        if len(candles) < 3:
            return {}

        df = pd.DataFrame(candles[-3:])  # Prendre les 3 dernières bougies
        patterns = {
            'bullish_engulfing': self._is_bullish_engulfing(df),
            'bearish_engulfing': self._is_bearish_engulfing(df),
            'morning_star': self._is_morning_star(df),
            'evening_star': self._is_evening_star(df),
            'hammer': self._is_hammer(df.iloc[-1]),
            'shooting_star': self._is_shooting_star(df.iloc[-1])
        }

        return patterns

    def _is_bullish_engulfing(self, df: pd.DataFrame) -> bool:
        """Détecte un pattern de Bullish Engulfing."""
        return (
            df.iloc[-2]['close'] < df.iloc[-2]['open'] and
            df.iloc[-1]['close'] > df.iloc[-1]['open'] and
            df.iloc[-1]['close'] > df.iloc[-2]['open'] and
            df.iloc[-1]['open'] < df.iloc[-2]['close']
        )

    def _is_bearish_engulfing(self, df: pd.DataFrame) -> bool:
        """Détecte un pattern de Bearish Engulfing."""
        return (
            df.iloc[-2]['close'] > df.iloc[-2]['open'] and
            df.iloc[-1]['close'] < df.iloc[-1]['open'] and
            df.iloc[-1]['close'] < df.iloc[-2]['open'] and
            df.iloc[-1]['open'] > df.iloc[-2]['close']
        )

    def _is_morning_star(self, df: pd.DataFrame) -> bool:
        """Détecte un pattern de Morning Star."""
        return (
            df.iloc[-3]['close'] < df.iloc[-3]['open'] and
            df.iloc[-2]['high'] < df.iloc[-3]['close'] and
            df.iloc[-1]['close'] > df.iloc[-1]['open'] and
            df.iloc[-1]['close'] > df.iloc[-3]['open']
        )

    def _is_evening_star(self, df: pd.DataFrame) -> bool:
        """Détecte un pattern de Evening Star."""
        return (
            df.iloc[-3]['close'] > df.iloc[-3]['open'] and
            df.iloc[-2]['low'] > df.iloc[-3]['close'] and
            df.iloc[-1]['close'] < df.iloc[-1]['open'] and
            df.iloc[-1]['close'] < df.iloc[-3]['open']
        )

    def _is_hammer(self, candle: Dict[str, Any]) -> bool:
        """Détecte un pattern de Hammer."""
        body = abs(candle['close'] - candle['open'])
        upper_wick = candle['high'] - max(candle['close'], candle['open'])
        lower_wick = min(candle['close'], candle['open']) - candle['low']
        return (
            lower_wick >= 2 * body and
            upper_wick <= body and
            candle['close'] > candle['open']
        )

    def _is_shooting_star(self, candle: Dict[str, Any]) -> bool:
        """Détecte un pattern de Shooting Star."""
        body = abs(candle['close'] - candle['open'])
        upper_wick = candle['high'] - max(candle['close'], candle['open'])
        lower_wick = min(candle['close'], candle['open']) - candle['low']
        return (
            upper_wick >= 2 * body and
            lower_wick <= body and
            candle['close'] < candle['open']
        )

    def detect_divergences(self,
                           candles: List[Candle],
                           indicator_values: List[Decimal]) -> Dict[str,
                                                                    bool]:
        """
        Détecte les divergences entre prix et indicateur.

        Args:
            candles: Liste des bougies de marché
            indicator_values: Liste des valeurs de l'indicateur

        Returns:
            Dictionnaire des divergences détectées
        """
        if len(candles) < self.window_size or len(
                indicator_values) < self.window_size:
            return {}

        price_trend = self._calculate_trend(candles[-self.window_size:])
        indicator_trend = self._calculate_trend(
            indicator_values[-self.window_size:])

        return {
            'bullish_divergence': price_trend < 0 and indicator_trend > 0,
            'bearish_divergence': price_trend > 0 and indicator_trend < 0
        }

    def _calculate_trend(self, values: List[Decimal]) -> float:
        """Calcule la tendance d'une série de valeurs."""
        if len(values) < 2:
            return 0

        return float((values[-1] - values[0]) / values[0])
