import numpy as np
from typing import Tuple
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from .ml_predictor import MLPredictor
from .technical_analyzer import TechnicalAnalyzer


class AdaptivePrediction:
    """Système de prédiction adaptative qui ajuste automatiquement ses paramètres."""

    def __init__(
        self,
        base_predictor: MLPredictor,
        technical_analyzer: TechnicalAnalyzer,
        lookback_window: int = 30,
        update_frequency: int = 10,
        confidence_threshold: float = 0.7
    ):
        self.base_predictor = base_predictor
        self.technical_analyzer = technical_analyzer
        self.lookback_window = lookback_window
        self.update_frequency = update_frequency
        self.confidence_threshold = confidence_threshold
        self.last_update = 0
        self.current_model = None
        self.metrics = {
            'mse': [],
            'accuracy': [],
            'confidence': []
        }

    def _calculate_confidence(
            self,
            predictions: np.ndarray,
            actual: np.ndarray) -> float:
        """Calcule la confiance dans les prédictions."""
        mse = mean_squared_error(actual, predictions)
        self.metrics['mse'].append(mse)

        # Calculer l'erreur relative
        relative_error = np.abs((actual - predictions) / actual).mean()

        # Calculer la précision
        accuracy = 1 - relative_error
        self.metrics['accuracy'].append(accuracy)

        # Calculer la confiance
        confidence = np.exp(-mse) * accuracy
        self.metrics['confidence'].append(confidence)

        return confidence

    def _update_model(self, data: pd.DataFrame) -> None:
        """Mise à jour du modèle avec les dernières données."""
        # Préparer les données
        features = self.technical_analyzer.calculate_features(data)
        target = data['close'].shift(-1).dropna()

        # Créer un ensemble de validation
        train_size = int(len(features) * 0.8)
        X_train, X_val = features[:train_size], features[train_size:]
        y_train, y_val = target[:train_size], target[train_size:]

        # Entraîner plusieurs modèles
        models = []
        for n_estimators in [50, 100, 200]:
            model = RandomForestRegressor(
                n_estimators=n_estimators,
                random_state=42
            )
            model.fit(X_train, y_train)
            predictions = model.predict(X_val)
            confidence = self._calculate_confidence(predictions, y_val)
            models.append((model, confidence))

        # Sélectionner le meilleur modèle
        self.current_model = max(models, key=lambda x: x[1])[0]

    def _adjust_prediction(
            self,
            prediction: float,
            confidence: float) -> float:
        """Ajuste la prédiction en fonction de la confiance."""
        if confidence < self.confidence_threshold:
            # Si la confiance est faible, réduire l'ampleur de la prédiction
            return prediction * confidence
        else:
            # Si la confiance est élevée, augmenter légèrement la prédiction
            return prediction * (1 + (confidence - 1) * 0.5)

    def predict(self, data: pd.DataFrame) -> Tuple[float, float]:
        """
        Fait une prédiction et retourne la valeur ajustée avec la confiance.

        Args:
            data: DataFrame avec les données historiques

        Returns:
            Tuple (prediction_adjustee, confidence)
        """
        data.index[-1]

        # Vérifier si une mise à jour est nécessaire
        if (len(data) - self.last_update) >= self.update_frequency:
            self._update_model(data)
            self.last_update = len(data)

        # Préparer les données pour la prédiction
        features = self.technical_analyzer.calculate_features(data.tail(1))

        if self.current_model is None:
            # Si pas de modèle, utiliser le prédicteur de base
            prediction = self.base_predictor.predict(data.tail(1))
            confidence = self.confidence_threshold
        else:
            prediction = self.current_model.predict(features)[0]

            # Calculer la confiance sur les données récentes
            recent_data = data.tail(self.lookback_window)
            recent_features = self.technical_analyzer.calculate_features(
                recent_data)
            recent_predictions = self.current_model.predict(recent_features)
            confidence = self._calculate_confidence(
                recent_predictions,
                recent_data['close'].shift(-1).dropna()
            )

        # Ajuster la prédiction
        adjusted_prediction = self._adjust_prediction(prediction, confidence)

        return adjusted_prediction, confidence

    def get_metrics(self) -> dict:
        """Retourne les métriques de performance."""
        return {
            'mse': np.mean(self.metrics['mse']),
            'accuracy': np.mean(self.metrics['accuracy']),
            'confidence': np.mean(self.metrics['confidence'])
        }
