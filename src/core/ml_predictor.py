import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import logging


class MLPredictor:
    """Prédiction des mouvements de prix avec des modèles ML."""

    def __init__(self):
        """Initialise le prédicteur ML."""
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )
        self.scaler = StandardScaler()
        self.features = []
        self.target = []
        self.logger = logging.getLogger('ml_predictor')

    def prepare_features(self, price_history: pd.Series) -> pd.DataFrame:
        """
        Prépare les features pour le modèle ML.

        Args:
            price_history: Historique des prix

        Returns:
            DataFrame avec les features
        """
        df = pd.DataFrame()

        # Features techniques
        df['returns'] = price_history.pct_change()
        df['volatility'] = df['returns'].rolling(window=20).std()
        df['momentum'] = price_history.rolling(
            window=20).apply(lambda x: x[-1] / x[0] - 1)
        df['rsi'] = self.calculate_rsi(price_history)

        # Retirer les NaN
        df = df.dropna()

        return df

    def calculate_rsi(self, price_history: pd.Series,
                      window: int = 14) -> pd.Series:
        """
        Calcule l'RSI.

        Args:
            price_history: Historique des prix
            window: Fenêtre de calcul

        Returns:
            Série RSI
        """
        delta = price_history.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    def prepare_target(self, price_history: pd.Series,
                       horizon: int = 1) -> pd.Series:
        """
        Prépare la variable cible.

        Args:
            price_history: Historique des prix
            horizon: Horizon de prédiction

        Returns:
            Série avec les mouvements de prix
        """
        returns = price_history.pct_change(horizon).shift(-horizon)
        return (returns > 0).astype(int)

    def train(self, price_history: pd.Series):
        """
        Entraîne le modèle ML.

        Args:
            price_history: Historique des prix
        """
        try:
            # Préparer les données
            features = self.prepare_features(price_history)
            target = self.prepare_target(price_history)

            # Aligner les données
            features = features.iloc[:-1]
            target = target.iloc[:-1]

            # Diviser les données
            X_train, X_test, y_train, y_test = train_test_split(
                features, target, test_size=0.2, random_state=42
            )

            # Normaliser les features
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)

            # Entraîner le modèle
            self.model.fit(X_train_scaled, y_train)

            # Évaluer le modèle
            accuracy = self.model.score(X_test_scaled, y_test)
            self.logger.info(f"Précision du modèle: {accuracy:.2%}")

        except Exception as e:
            self.logger.error(
                f"Erreur lors de l'entraînement du modèle: {str(e)}")
            raise

    def predict(self, price_history: pd.Series) -> float:
        """
        Prédit la direction du prix.

        Args:
            price_history: Historique des prix

        Returns:
            Probabilité de hausse (0-1)
        """
        try:
            # Préparer les features
            features = self.prepare_features(price_history)

            # Normaliser les features
            features_scaled = self.scaler.transform(features.tail(1))

            # Faire la prédiction
            proba = self.model.predict_proba(features_scaled)[0][1]

            return proba

        except Exception as e:
            self.logger.error(f"Erreur lors de la prédiction: {str(e)}")
            return 0.5  # Retourne 0.5 en cas d'erreur (neutre)
