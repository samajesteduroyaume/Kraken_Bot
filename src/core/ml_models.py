"""
Module de modèles d'apprentissage automatique pour le bot de trading Kraken.
Gère les prédictions et les analyses basées sur les modèles ML.
"""

from typing import Dict, Tuple
import logging
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

logger = logging.getLogger(__name__)


class MLModelManager:
    """Gestionnaire de modèles d'apprentissage automatique."""

    def __init__(self, config: Dict):
        """Initialise le gestionnaire de modèles ML."""
        self.config = config
        self.models = {}
        self.scalers = {}
        self.features = config.get('features', [
            'rsi', 'macd', 'bb_upper', 'bb_lower', 'volume', 'price_change'
        ])

    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Prépare les features pour le modèle ML.

        Args:
            df: DataFrame avec les données OHLCV et indicateurs

        Returns:
            DataFrame avec les features prêtes
        """
        try:
            features = df[self.features].copy()
            features['price_change'] = df['close'].pct_change()
            features = features.dropna()

            return features

        except Exception as e:
            logger.error(
                f"Erreur lors de la préparation des features: {str(e)}")
            raise

    def train_model(self,
                    df: pd.DataFrame,
                    pair: str,
                    model_type: str = 'random_forest') -> None:
        """
        Entraîne un modèle ML pour une paire donnée.

        Args:
            df: DataFrame avec les données OHLCV et indicateurs
            pair: Paire de trading
            model_type: Type de modèle ('random_forest' ou 'xgboost')
        """
        try:
            features = self.prepare_features(df)

            # Créer les labels (1 pour hausse, 0 pour baisse)
            labels = (features['price_change'].shift(-1) > 0).astype(int)
            features = features.drop('price_change', axis=1)

            # Diviser les données
            X_train, X_test, y_train, y_test = train_test_split(
                features, labels, test_size=0.2, random_state=42
            )

            # Standardiser les features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            # Entraîner le modèle
            if model_type == 'random_forest':
                model = RandomForestClassifier(
                    n_estimators=100,
                    max_depth=10,
                    random_state=42
                )
            else:  # xgboost
                model = XGBClassifier(
                    n_estimators=100,
                    max_depth=10,
                    random_state=42
                )

            model.fit(X_train_scaled, y_train)

            # Stocker le modèle et le scaler
            self.models[pair] = model
            self.scalers[pair] = scaler

            # Évaluer le modèle
            accuracy = model.score(X_test_scaled, y_test)
            logger.info(
                f"Modèle {model_type} pour {pair} entraîné avec précision: {accuracy:.2f}")

        except Exception as e:
            logger.error(
                f"Erreur lors de l'entraînement du modèle pour {pair}: {str(e)}")
            raise

    def predict(self, df: pd.DataFrame, pair: str) -> Tuple[float, float]:
        """
        Fait une prédiction avec le modèle ML.

        Args:
            df: DataFrame avec les données OHLCV et indicateurs
            pair: Paire de trading

        Returns:
            Tuple (probabilité de hausse, probabilité de baisse)
        """
        try:
            if pair not in self.models:
                logger.error(f"Aucun modèle entraîné pour {pair}")
                return 0.5, 0.5

            features = self.prepare_features(df).iloc[-1:]  # Dernière bougie
            features_scaled = self.scalers[pair].transform(features)

            # Faire la prédiction
            probabilities = self.models[pair].predict_proba(features_scaled)[0]

            return tuple(probabilities)

        except Exception as e:
            logger.error(f"Erreur lors de la prédiction pour {pair}: {str(e)}")
            return 0.5, 0.5
