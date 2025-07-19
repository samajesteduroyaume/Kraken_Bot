"""
Module de prédiction ML pour le bot de trading Kraken.
"""
import os
import logging
import numpy as np
import pandas as pd
import joblib
from typing import Optional, Dict, Any, List, Tuple
from datetime import datetime

from .model_manager import ModelManager
from src.core.config import Config
from typing import Tuple, Dict, Any, List
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

logger = logging.getLogger(__name__)


class MLPredictor:
    """
    Prédicteur de tendance de marché basé sur l'apprentissage automatique.
    Utilise plusieurs modèles et sélectionne le meilleur en fonction des performances.
    """

    def __init__(self, db_manager: Optional[Any] = None) -> None:
        """Initialise le prédicteur ML avec gestionnaire de modèles avancé.

        Args:
            db_manager: Gestionnaire de base de données (optionnel)
        """
        self.db_manager = db_manager
        config = Config()
        self.model_dir = config.ml_config.get('model_dir', 'models')
        self.retrain_interval = config.ml_config.get(
            'retrain_interval_hours', 24)

        # Initialiser le gestionnaire de modèles
        self.model_manager = ModelManager(
            models_dir=self.model_dir,
            retrain_interval=self.retrain_interval
        )

        # Modèle de prédiction de risque
        self.risk_model = self._init_risk_model()

        # Configuration du levier
        self.leverage_settings = {
            'max_leverage': 5.0,
            'volatility_window': 20,
            'confidence_threshold': 0.7
        }

        # Configurer les modèles par défaut
        self._setup_default_models()

        # Charger les modèles existants
        self._load_saved_models()

    def _init_risk_model(self) -> Pipeline:
        """Initialise le modèle de prédiction de risque."""
        return Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', RandomForestClassifier(
                n_estimators=100,
                max_depth=5,
                random_state=42,
                class_weight='balanced'
            ))
        ])

    def _load_saved_models(self) -> None:
        """Charge les modèles sauvegardés depuis le répertoire des modèles."""
        try:
            # Charger le modèle de risque s'il existe
            risk_model_path = os.path.join(self.model_dir, 'risk_model.joblib')
            if os.path.exists(risk_model_path):
                self.risk_model = joblib.load(risk_model_path)
                logger.info("Modèle de risque chargé avec succès")

            # Utiliser le ModelManager pour charger les autres modèles
            self.model_manager.load_models()

            # Vérifier si des modèles ont été chargés
            if not self.model_manager.models:
                logger.warning(
                    "Aucun modèle sauvegardé trouvé. Les modèles par défaut seront utilisés.")
            else:
                logger.info(
                    f"{len(self.model_manager.models)} modèles chargés depuis {self.model_dir}")

                # Afficher les informations sur les modèles chargés
                for model_name, model in self.model_manager.models.items():
                    if model.last_trained:
                        logger.info(
                            f"Modèle {model_name} chargé (dernier entraînement: {model.last_trained})")
                    else:
                        logger.info(
                            f"Modèle {model_name} chargé (pas encore entraîné)")

                # Mettre à jour le meilleur modèle
                if self.model_manager.best_model:
                    logger.info(
                        f"Meilleur modèle sélectionné: {self.model_manager.best_model_name} "
                        f"(score: {self.model_manager.best_score:.4f})")

        except Exception as e:
            logger.error(
                f"Erreur lors du chargement des modèles sauvegardés: {e}")
            logger.warning("Les modèles par défaut seront utilisés.")

    def _setup_default_models(self) -> None:
        """Configure les modèles par défaut avec des paramètres optimisés."""
        # Configuration des modèles avec des paramètres optimisés
        model_configs = {
            'random_forest': {
                'n_estimators': 200,
                'max_depth': 10,
                'min_samples_split': 5,
                'min_samples_leaf': 2,
                'class_weight': 'balanced',
                'n_jobs': -1,
                'random_state': 42
            },
            'xgboost': {
                'n_estimators': 150,
                'max_depth': 6,
                'learning_rate': 0.05,
                'subsample': 0.9,
                'colsample_bytree': 0.8,
                'gamma': 1,
                'n_jobs': -1,
                'random_state': 42
            },
            'neural_network': {
                'hidden_layer_sizes': (128, 64, 32),
                'activation': 'relu',
                'solver': 'adam',
                'alpha': 0.0001,
                'batch_size': 32,
                'learning_rate': 'adaptive',
                'max_iter': 300,
                'early_stopping': True,
                'validation_fraction': 0.1,
                'n_iter_no_change': 10,
                'random_state': 42
            }
        }

        # Ajouter les modèles au gestionnaire
        for model_name, params in model_configs.items():
            self.model_manager.add_model(model_name, params)

    async def load_historical_data(
            self, pair: str, days: int = 90) -> Optional[pd.DataFrame]:
        """
        Charge les données historiques depuis la table market_data.
        Args:
            pair (str): Le symbole de la paire (ex: 'XBT/USD').
            days (int): Nombre de jours d'historique à charger.
        Returns:
            Optional[pd.DataFrame]: Données OHLCV sous forme de DataFrame ou None si absence de données.
        """
        try:
            # Récupérer l'id de la paire
            pair_row = await self.db_manager.fetchrow(
                "SELECT id FROM pairs WHERE symbol = $1",
                pair
            )
            if not pair_row:
                logger.warning(
                    f"Paire {pair} non trouvée dans la table pairs"
                )
                return None
            pair_id = pair_row['id']
            query = (
                f"""
                SELECT timestamp, open, high, low, close, volume
                FROM market_data
                WHERE pair_id = $1
                  AND timestamp >= NOW() - INTERVAL '{days} days'
                ORDER BY timestamp ASC
                """
            )
            records = await self.db_manager.fetch(query, pair_id)
            if not records:
                logger.warning(
                    f"Aucune donnée trouvée pour {pair} sur les {days} derniers jours"
                )
                return None
            df = pd.DataFrame(
                records,
                columns=[
                    'timestamp', 'open', 'high', 'low', 'close', 'volume'
                ]
            )
            # Conversion explicite des colonnes numériques
            for col in [
                'open', 'high', 'low', 'close', 'volume'
            ]:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            # Suppression des lignes avec valeurs manquantes
            df = df.dropna(
                subset=['open', 'high', 'low', 'close', 'volume']
            )
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df.set_index('timestamp', inplace=True)
            return df
        except Exception as e:
            logger.error(
                f"Erreur lors du chargement des données pour {pair}: {e}"
            )
            return None

    def _prepare_features(
            self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """Prépare les caractéristiques pour l'entraînement.

        Args:
            df: DataFrame contenant les données OHLCV

        Returns:
            Tuple de (features, target) pour l'entraînement
        """
        try:
            # Calcul des indicateurs techniques
            df = df.copy()

            # Retours logarithmiques
            df['returns'] = np.log(df['close'] / df['close'].shift(1))

            # Volatilité sur 20 périodes
            df['volatility'] = df['returns'].rolling(
                window=20).std() * np.sqrt(252)

            # RSI (14 périodes)
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            df['rsi'] = 100 - (100 / (1 + rs))

            # MACD
            exp1 = df['close'].ewm(span=12, adjust=False).mean()
            exp2 = df['close'].ewm(span=26, adjust=False).mean()
            df['macd'] = exp1 - exp2
            df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()

            # Volume moyen sur 20 périodes
            df['volume_ma20'] = df['volume'].rolling(window=20).mean()

            # Cible: 1 si le prix monte le jour suivant, 0 sinon
            df['target'] = (df['close'].shift(-1) > df['close']).astype(int)

            # Supprimer les valeurs manquantes
            df = df.dropna()

            if df.empty:
                return pd.DataFrame(), pd.Series(dtype=float)

            # Sélection des caractéristiques
            features = [
                'returns',
                'volatility',
                'rsi',
                'macd',
                'macd_signal',
                'volume_ma20']
            X = df[features]
            y = df['target']

            return X, y

        except Exception as e:
            logger.error(
                f"Erreur lors de la préparation des caractéristiques: {e}"
            )
            return pd.DataFrame(), pd.Series(dtype=float)

    async def train(
            self,
            pair: str,
            days: int = 90,
            force_retrain: bool = False) -> Optional[float]:
        """Entraîne le modèle pour une paire donnée.

        Args:
            pair: Paire de trading (ex: 'BTC/USD')
            days: Nombre de jours de données à utiliser
            force_retrain: Forcer l'entraînement même si le modèle est à jour

        Returns:
            Score de précision du modèle ou None en cas d'échec
        """
        try:
            # Vérifier si le réentraînement est nécessaire
            if not force_retrain and not await self._needs_retraining(pair):
                logger.info(f"Le modèle pour {pair} est à jour")
                return await self._get_latest_score(pair)

            # Charger les données
            df = await self.load_historical_data(pair, days)
            if df is None or df.empty:
                logger.error(f"Impossible de charger les données pour {pair}")
                return None

            if len(df) < ML_CONFIG['min_data_points']:
                logger.warning(
                    f"Pas assez de données pour {pair} ({len(df)} < {ML_CONFIG['min_data_points']})")
                return None

            # Préparer les caractéristiques
            X, y = self._prepare_features(df)
            if X.empty or y.empty:
                logger.error("Impossible de préparer les caractéristiques")
                return None

            # Séparer en train/test
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, shuffle=False
            )

            # Entraîner le modèle
            X_train_scaled = self.scaler.fit_transform(X_train)
            self.model.fit(X_train_scaled, y_train)

            # Évaluer le modèle
            X_test_scaled = self.scaler.transform(X_test)
            train_score = accuracy_score(
                y_train, self.model.predict(X_train_scaled))
            test_score = accuracy_score(
                y_test, self.model.predict(X_test_scaled))

            # Enregistrer le modèle
            model_data = {
                'model': self.model,
                'scaler': self.scaler,
                'trained_at': datetime.now(timezone.utc).timestamp(),
                'train_score': train_score,
                'test_score': test_score,
                'data_points': len(X),
                'features': X.columns.tolist(),
                'pair': pair
            }

            await self._save_model(pair, model_data)

            logger.info(
                f"Modèle pour {pair} entraîné avec succès - "
                f"Train: {train_score:.4f}, Test: {test_score:.4f}, "
                f"Données: {len(X)} points"
            )

            return test_score

        except Exception as e:
            logger.error(
                f"Erreur lors de l'entraînement pour {pair}: {e}"
            )
            return None

    async def predict(self, pair: str,
                      df: Optional[pd.DataFrame] = None) -> Optional[float]:
        """Prédit la probabilité de hausse pour une paire donnée.

        Args:
            pair: Paire de trading (ex: 'BTC/USD')
            df: Données récentes (optionnel)

        Returns:
            Probabilité de hausse (0-1) ou None en cas d'erreur
        """
        try:
            # Charger le modèle le plus récent
            model_data = await self._load_latest_model(pair)
            if model_data is None:
                logger.warning(
                    f"Aucun modèle trouvé pour {pair}, entraînement en cours...")
                score = await self.train(pair, force_retrain=True)
                if score is None:
                    return None
                model_data = await self._load_latest_model(pair)
                if model_data is None:
                    return None

            # Charger les données si non fournies
            if df is None:
                # Dernière semaine
                df = await self.load_historical_data(pair, days=7)
                if df is None or df.empty:
                    logger.error(
                        f"Impossible de charger les données pour {pair}")
                    return None

            # Correction : si df est une liste, le convertir en DataFrame
            if isinstance(df, list):
                columns = [
                    'timestamp',
                    'open',
                    'high',
                    'low',
                    'close',
                    'vwap',
                    'volume',
                    'count']
                df = pd.DataFrame(df, columns=columns[:len(df[0])])
                for col in ['open', 'high', 'low', 'close', 'volume']:
                    if col in df.columns:
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                df = df.dropna(
                    subset=[
                        'open',
                        'high',
                        'low',
                        'close',
                        'volume'])
                df['timestamp'] = pd.to_datetime(
                    df['timestamp'], unit='s', errors='coerce')
                df.set_index('timestamp', inplace=True)

            # Préparer les caractéristiques
            X, _ = self._prepare_features(df)
            if X.empty:
                logger.error(
                    "Impossible de préparer les caractéristiques pour la prédiction")
                return None

            # Sélectionner les dernières données
            X_latest = X.iloc[-1:]

            # Vérifier que nous avons toutes les caractéristiques nécessaires
            if X_latest.isnull().any().any():
                logger.warning("Données manquantes pour la prédiction")
                return None

            # Récupérer le scaler et le modèle
            scaler = model_data['scaler']
            model = model_data['model']

            # Normaliser et prédire
            X_scaled = scaler.transform(X_latest)
            proba = model.predict_proba(
                X_scaled)[0][1]  # Probabilité de hausse

            logger.debug(f"Prédiction pour {pair}: {proba:.2f}")
            return float(proba)

        except Exception as e:
            logger.error(
                f"Erreur lors de la prédiction pour {pair}: {e}"
            )
            return None

    async def _needs_retraining(self, pair: str) -> bool:
        """Vérifie si le modèle a besoin d'être réentraîné."""
        try:
            model_data = await self._load_latest_model(pair)
            if model_data is None:
                return True

            # Vérifier l'âge du modèle
            trained_at = datetime.fromtimestamp(
                model_data.get('trained_at', 0), timezone.utc)
            model_age = datetime.now(timezone.utc) - trained_at

            if model_age.days >= ML_CONFIG['max_model_age_days']:
                logger.info(
                    f"Modèle pour {pair} trop ancien ({model_age.days} jours)")
                return True

            # Vérifier le nombre de points de données
            data_points = model_data.get('data_points', 0)
            if data_points < ML_CONFIG['min_data_points']:
                logger.info(
                    f"Modèle pour {pair} a trop peu de données ({data_points} points)")
                return True

            return False

        except Exception as e:
            logger.error(
                f"Erreur lors de la vérification du modèle pour {pair}: {e}"
            )
            return True

    async def _get_latest_score(self, pair: str) -> Optional[float]:
        """Récupère le score du dernier modèle entraîné."""
        try:
            model_data = await self._load_latest_model(pair)
            return model_data.get('test_score') if model_data else None
        except Exception as e:
            logger.error(
                f"Erreur lors de la récupération du score pour {pair}: {e}"
            )
            return None

    async def _get_model_path(
            self,
            pair: str,
            timestamp: Optional[float] = None) -> str:
        """Génère le chemin du fichier de modèle."""
        pair_dir = os.path.join(self.model_dir, pair.replace("/", ""))
        os.makedirs(pair_dir, exist_ok=True)

        if timestamp is None:
            timestamp = datetime.now(timezone.utc).timestamp()

        filename = f"model_{int(timestamp)}.pkl"
        return os.path.join(pair_dir, filename)

    async def _save_model(self, pair: str, model_data: Dict[str, Any]) -> bool:
        """Sauvegarde le modèle sur le disque."""
        try:
            model_path = await self._get_model_path(pair, model_data['trained_at'])
            joblib.dump(model_data, model_path)

            # Nettoyer les anciens modèles (garder les 3 plus récents)
            await self._cleanup_old_models(pair)

            return True

        except Exception as e:
            logger.error(
                f"Erreur lors de la sauvegarde du modèle pour {pair}: {e}"
            )
            return False

    async def _load_latest_model(self, pair: str) -> Optional[Dict[str, Any]]:
        """Charge le modèle le plus récent pour une paire donnée."""
        try:
            pair_dir = os.path.join(self.model_dir, pair.replace("/", ""))
            if not os.path.exists(pair_dir):
                return None

            model_files = [f for f in os.listdir(
                pair_dir) if f.endswith('.pkl')]
            if not model_files:
                return None

            latest_model = max(model_files)
            model_path = os.path.join(pair_dir, latest_model)

            return joblib.load(model_path)

        except Exception as e:
            logger.error(
                f"Erreur lors du chargement du modèle pour {pair}: {e}"
            )
            return None

    async def _cleanup_old_models(self, pair: str, keep: int = 3) -> None:
        """Supprime les anciens modèles, ne garde que les 'keep' plus récents."""
        try:
            pair_dir = os.path.join(self.model_dir, pair.replace("/", ""))
            if not os.path.exists(pair_dir):
                return

            model_files = [f for f in os.listdir(
                pair_dir) if f.endswith('.pkl')]
            if len(model_files) <= keep:
                return

            # Trier par date (du plus ancien au plus récent)
            model_files.sort()

            # Supprimer les plus anciens
            for old_model in model_files[:-keep]:
                os.remove(os.path.join(pair_dir, old_model))

        except Exception as e:
            logger.error(
                f"Erreur lors du nettoyage des anciens modèles pour {pair}: {e}"
            )

    def _evaluate_risk(self, features: Dict[str, Any]) -> float:
        """
        Évalue le risque du marché en fonction des caractéristiques.

        Args:
            features: Caractéristiques du marché

        Returns:
            float: Score de risque entre 0 (faible risque) et 1 (haut risque)
        """
        try:
            # Préparation des caractéristiques pour le modèle de risque
            risk_features = self._prepare_risk_features(features)

            # Prédiction du risque (0=faible risque, 1=haut risque)
            risk_prob = self.risk_model.predict_proba([risk_features])[0][1]

            return float(risk_prob)

        except Exception as e:
            logger.warning(f"Erreur lors de l'évaluation du risque: {e}")
            return 0.5  # Risque moyen par défaut

    def _prepare_risk_features(self, features: Dict[str, Any]) -> List[float]:
        """
        Prépare les caractéristiques pour le modèle de risque.

        Args:
            features: Caractéristiques brutes du marché

        Returns:
            Liste des caractéristiques numériques pour le modèle de risque
        """
        # Exemple de caractéristiques pour l'évaluation du risque
        return [
            features.get('volatility', 0.0),
            features.get('volume_ratio', 1.0),
            features.get('spread', 0.0),
            features.get('rsi', 50.0) / 100.0,  # Normalisation 0-1
            features.get('adx', 25.0) / 100.0,   # Normalisation 0-1
            features.get('atr', 0.0) / features.get('close',
                                                    1.0)  # ATR en pourcentage du prix
        ]

    def _calculate_recommended_leverage(
        self,
        confidence: float,
        risk_score: float,
        volatility: float
    ) -> float:
        """
        Calcule le levier recommandé en fonction de la confiance et du risque.

        Args:
            confidence: Niveau de confiance du signal (0-1)
            risk_score: Score de risque (0-1)
            volatility: Volatilité du marché (0-1)

        Returns:
            float: Levier recommandé
        """
        # Facteurs d'ajustement
        confidence_factor = min(1.0, confidence /
                                self.leverage_settings['confidence_threshold'])
        risk_factor = 1.0 - risk_score  # Moins de levier quand le risque est élevé
        # Moins de levier en cas de forte volatilité
        volatility_factor = 1.0 / (1.0 + volatility * 2)

        # Calcul du levier de base
        base_leverage = self.leverage_settings['max_leverage'] * \
            confidence_factor

        # Ajustement pour le risque et la volatilité
        adjusted_leverage = base_leverage * risk_factor * volatility_factor

        # Application des limites
        return max(
            1.0, min(
                adjusted_leverage, self.leverage_settings['max_leverage']))

    async def train_risk_model(
            self, X: List[Dict[str, float]], y: List[int]) -> Dict[str, Any]:
        """
        Entraîne le modèle de prédiction de risque.

        Args:
            X: Liste de caractéristiques d'entraînement
            y: Étiquettes (0=faible risque, 1=haut risque)

        Returns:
            Dictionnaire contenant les métriques d'entraînement
        """
        try:
            # Préparation des données
            X_processed = [self._prepare_risk_features(x) for x in X]

            # Entraînement du modèle
            self.risk_model.fit(X_processed, y)

            # Sauvegarde du modèle
            os.makedirs(self.model_dir, exist_ok=True)
            joblib.dump(
                self.risk_model,
                os.path.join(
                    self.model_dir,
                    'risk_model.joblib'))

            # Calcul des métriques
            train_score = self.risk_model.score(X_processed, y)

            return {
                'status': 'success',
                'train_score': train_score,
                'model_params': self.risk_model.get_params()
            }

        except Exception as e:
            logger.error(
                f"Erreur lors de l'entraînement du modèle de risque: {e}")
            return {
                'status': 'error',
                'error': str(e)
            }
