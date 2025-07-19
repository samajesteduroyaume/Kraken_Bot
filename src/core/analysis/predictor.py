"""
Module de prédiction ML pour le trading.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, Tuple, Any
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

logger = logging.getLogger(__name__)


class MLPredictor:
    """
    Classe pour la prédiction ML utilisant un modèle LSTM.
    """

    def __init__(self, model_config: Dict[str, Any]):
        """
        Initialise le prédicteur ML.

        Args:
            model_config: Configuration du modèle
        """
        self.model_config = model_config
        self.model = None

    def build_model(self, input_shape: Tuple[int, int]) -> None:
        """
        Construit le modèle LSTM.

        Args:
            input_shape: Shape des données d'entrée
        """
        self.model = Sequential([
            LSTM(self.model_config['units'],
                 return_sequences=True,
                 input_shape=input_shape),
            Dropout(self.model_config['dropout']),
            LSTM(self.model_config['units']),
            Dropout(self.model_config['dropout']),
            Dense(1)
        ])

        self.model.compile(optimizer='adam', loss='mse')

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

    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Entraîne le modèle.

        Args:
            X: Données d'entrée
            y: Données cibles
        """
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=self.model_config['patience'],
            restore_best_weights=True
        )

        self.model.fit(
            X, y,
            epochs=self.model_config['epochs'],
            batch_size=self.model_config['batch_size'],
            validation_split=self.model_config['validation_split'],
            callbacks=[early_stopping]
        )

    def predict(self, data: np.ndarray) -> np.ndarray:
        """
        Fait une prédiction.

        Args:
            data: Données d'entrée pour la prédiction

        Returns:
            Prédiction
        """
        return self.model.predict(data)
