import os
import json
import joblib
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple, Optional
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
import logging

# Configuration des dossiers
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)

class ModelVersion:
    """Gestion des versions de modèles."""
    
    def __init__(self, model_id: str):
        self.model_id = model_id
        self.model_dir = MODEL_DIR / model_id
        self.model_dir.mkdir(exist_ok=True)
        self.metadata_file = self.model_dir / "metadata.json"
        self.current_version = self._load_metadata().get('current_version', 0)
    
    def _load_metadata(self) -> dict:
        if self.metadata_file.exists():
            with open(self.metadata_file, 'r') as f:
                return json.load(f)
        return {}
    
    def save_metadata(self, metadata: dict):
        with open(self.metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
    
    def get_next_version(self) -> int:
        self.current_version += 1
        return self.current_version
    
    def get_model_path(self, version: Optional[int] = None) -> Path:
        version = version or self.current_version
        return self.model_dir / f"model_v{version}.joblib"
    
    def save_model(self, model, metrics: dict):
        version = self.get_next_version()
        model_path = self.get_model_path(version)
        
        # Sauvegarder le modèle
        joblib.dump(model, model_path)
        
        # Mettre à jour les métadonnées
        metadata = self._load_metadata()
        metadata['current_version'] = version
        metadata['versions'] = metadata.get('versions', [])
        metadata['versions'].append({
            'version': version,
            'timestamp': datetime.now().isoformat(),
            'metrics': metrics,
            'model_path': str(model_path)
        })
        self.save_metadata(metadata)
        
        return version
    
    def load_model(self, version: Optional[int] = None):
        version = version or self.current_version
        model_path = self.get_model_path(version)
        if not model_path.exists():
            raise ValueError(f"Modèle version {version} non trouvé")
        return joblib.load(model_path)


class MLPredictor:
    """
    Prédiction des mouvements de prix avec des modèles ML.
    
    Gère l'entraînement, la prédiction, le versionnage et le suivi des performances.
    """

    def __init__(
        self,
        model_id: str = "default",
        retrain_interval: int = 7,  # jours
        min_accuracy: float = 0.6,  # précision minimale acceptable
        max_versions: int = 5,      # nombre maximum de versions à conserver
    ):
        """
        Initialise le prédicteur ML avec gestion de version.
        
        Args:
            model_id: Identifiant unique du modèle
            retrain_interval: Intervalle en jours entre deux réentraînements
            min_accuracy: Précision minimale acceptable pour le modèle
            max_versions: Nombre maximum de versions à conserver
        """
        self.model_id = model_id
        self.retrain_interval = retrain_interval
        self.min_accuracy = min_accuracy
        self.max_versions = max_versions
        
        # Gestion des versions
        self.version_manager = ModelVersion(model_id)
        
        # Initialisation du modèle et du scaler
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1  # Utilisation de tous les cœurs
        )
        self.scaler = StandardScaler()
        
        # Suivi des performances
        self.last_trained = None
        self.performance_history = []
        
        # Configuration du logger
        self.logger = logging.getLogger(f'ml_predictor.{model_id}')
        self.logger.info(f"Initialisation du prédicteur ML (ID: {model_id})")
        
        # Charger le dernier modèle entraîné si disponible
        self._load_latest_model()

    def _load_latest_model(self) -> bool:
        """Charge la dernière version du modèle."""
        try:
            self.model = self.version_manager.load_model()
            self.last_trained = datetime.now()
            self.logger.info(f"Modèle chargé (version: {self.version_manager.current_version})")
            return True
        except Exception as e:
            self.logger.warning(f"Impossible de charger le modèle: {e}. Un nouveau modèle sera créé.")
            return False
            
    def _cleanup_old_versions(self):
        """Supprime les anciennes versions du modèle."""
        metadata = self.version_manager._load_metadata()
        versions = metadata.get('versions', [])
        
        if len(versions) > self.max_versions:
            # Trier par version (croissante)
            versions.sort(key=lambda x: x['version'])
            
            # Supprimer les versions les plus anciennes
            for version in versions[:-self.max_versions]:
                try:
                    model_path = Path(version['model_path'])
                    if model_path.exists():
                        model_path.unlink()
                        self.logger.debug(f"Version {version['version']} supprimée")
                except Exception as e:
                    self.logger.error(f"Erreur lors de la suppression de la version {version['version']}: {e}")
            
            # Mettre à jour les métadonnées
            metadata['versions'] = versions[-self.max_versions:]
            self.version_manager.save_metadata(metadata)
    
    def needs_retrain(self) -> bool:
        """Vérifie si le modèle a besoin d'être réentraîné."""
        if not self.last_trained:
            return True
            
        days_since_training = (datetime.now() - self.last_trained).days
        return days_since_training >= self.retrain_interval

    def prepare_features(self, price_history: pd.Series) -> pd.DataFrame:
        """
        Prépare les features pour le modèle ML.

        Args:
            price_history: Historique des prix (pandas Series avec index datetime)

        Returns:
            DataFrame avec les features et index datetime
        """
        if not isinstance(price_history, pd.Series):
            raise ValueError("price_history doit être une pandas Series")
            
        if len(price_history) < 50:  # Minimum pour calculer les indicateurs
            raise ValueError("Pas assez de données historiques (minimum 50 périodes)")
            
        df = pd.DataFrame(index=price_history.index)
        df['price'] = price_history

        try:
            # Features de base
            df['returns'] = price_history.pct_change()
            df['log_returns'] = np.log(price_history / price_history.shift(1))
            
            # Volatilité
            df['volatility_20'] = df['returns'].rolling(window=20).std()
            df['volatility_50'] = df['returns'].rolling(window=50).std()
            
            # Momentum
            for window in [5, 10, 20, 50]:
                df[f'momentum_{window}'] = price_history.pct_change(window)
            
            # RSI
            df['rsi_14'] = self.calculate_rsi(price_history, window=14)
            df['rsi_30'] = self.calculate_rsi(price_history, window=30)
            
            # Moyennes mobiles
            for window in [5, 10, 20, 50, 200]:
                df[f'sma_{window}'] = price_history.rolling(window=window).mean()
                df[f'ema_{window}'] = price_history.ewm(span=window, adjust=False).mean()
            
            # Différences de moyennes mobiles
            df['sma_diff_5_20'] = df['sma_5'] - df['sma_20']
            df['sma_diff_20_50'] = df['sma_20'] - df['sma_50']
            
            # Retirer les NaN créés par les fenêtres mobiles
            df = df.dropna()
            
            # Vérifier qu'il reste assez de données
            if len(df) < 20:
                raise ValueError("Pas assez de données après nettoyage")
                
            return df
            
        except Exception as e:
            self.logger.error(f"Erreur lors de la préparation des features: {str(e)}")
            raise

    def calculate_rsi(self, price_history: pd.Series, window: int = 14) -> pd.Series:
        """
        Calcule le Relative Strength Index (RSI).
        
        Args:
            price_history: Série de prix
            window: Période de calcul (par défaut: 14)
            
        Returns:
            Série pandas contenant les valeurs du RSI
            
        Raises:
            ValueError: Si la fenêtre est trop grande ou les données insuffisantes
        """
        if window < 1:
            raise ValueError("La fenêtre doit être d'au moins 1")
            
        if len(price_history) < window + 1:
            raise ValueError(f"Pas assez de données pour calculer le RSI({window})")
            
        try:
            delta = price_history.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
            
            # Éviter la division par zéro
            rs = gain / loss.replace(0, float('inf'))
            rsi = 100 - (100 / (1 + rs))
            
            # Limiter les valeurs entre 0 et 100
            return rsi.clip(0, 100)
            
        except Exception as e:
            self.logger.error(f"Erreur dans le calcul du RSI: {str(e)}")
            raise

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

    def train(self, training_data: dict):
        """
        Entraîne le modèle ML.

        Args:
            training_data: Dictionnaire contenant :
                - 'X': DataFrame ou array-like, les caractéristiques d'entraînement
                - 'y': array-like, la cible d'entraînement
                - 'test_size': float, proportion du jeu de test (optionnel, 0.2 par défaut)
                - 'random_state': int, seed pour la reproductibilité (optionnel)
        """
        try:
            # Extraire les données d'entraînement
            X = training_data['X']
            y = training_data['y']
            test_size = training_data.get('test_size', 0.2)
            random_state = training_data.get('random_state', 42)
            
            # Convertir en DataFrame/Series si nécessaire
            if not isinstance(X, (pd.DataFrame, np.ndarray)):
                raise ValueError("X doit être un DataFrame ou un array-like")
                
            if not isinstance(y, (pd.Series, np.ndarray, list)):
                raise ValueError("y doit être une Series, un array ou une liste")
            
            # Convertir en numpy array pour le traitement
            if isinstance(X, pd.DataFrame):
                X_values = X.values
            else:
                X_values = np.array(X)
                
            if isinstance(y, (pd.Series, list)):
                y_values = np.array(y)
            else:
                y_values = y
            
            # Vérifier les dimensions
            if len(X_values) != len(y_values):
                raise ValueError("X et y doivent avoir la même longueur")
            
            # Diviser les données
            X_train, X_test, y_train, y_test = train_test_split(
                X_values, y_values, 
                test_size=test_size, 
                random_state=random_state,
                stratify=y_values  # Préserve la distribution des classes
            )

            # Normaliser les features
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)

            # Entraîner le modèle
            self.model.fit(X_train_scaled, y_train)

            # Évaluer le modèle
            train_accuracy = self.model.score(X_train_scaled, y_train)
            test_accuracy = self.model.score(X_test_scaled, y_test)
            
            # Calculer d'autres métriques
            y_pred = self.model.predict(X_test_scaled)
            precision = precision_score(y_test, y_pred, zero_division=0)
            recall = recall_score(y_test, y_pred, zero_division=0)
            
            metrics = {
                'train_accuracy': train_accuracy,
                'test_accuracy': test_accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0,
                'n_samples': len(X_values),
                'n_features': X_values.shape[1] if len(X_values.shape) > 1 else 1,
                'class_distribution': {
                    'class_0': int(np.sum(y_values == 0)),
                    'class_1': int(np.sum(y_values == 1))
                }
            }
            
            self.logger.info(f"Métriques d'entraînement - Précision: {test_accuracy:.2%}, "
                           f"Précision: {precision:.2%}, Rappel: {recall:.2%}")
            
            # Sauvegarder le modèle et les métriques
            self.version_manager.save_model(self.model, metrics)
            
            # Nettoyer les anciennes versions si nécessaire
            self._cleanup_old_versions()
            
            # Mettre à jour le suivi des performances
            self.performance_history.append({
                'timestamp': datetime.now().isoformat(),
                'metrics': metrics,
                'version': self.version_manager.current_version
            })
            
            self.last_trained = datetime.now()
            
            return metrics

        except Exception as e:
            self.logger.error(f"Erreur lors de l'entraînement du modèle: {str(e)}")
            raise

    def predict(self, X) -> np.ndarray:
        """
        Prédit la direction du prix pour plusieurs échantillons.

        Args:
            X: Tableau numpy ou DataFrame contenant les caractéristiques d'entrée

        Returns:
            Tableau numpy des probabilités de hausse (0-1) pour chaque échantillon
        """
        try:
            # Convertir en numpy array si nécessaire
            if isinstance(X, pd.DataFrame):
                X_values = X.values
            else:
                X_values = np.array(X)
            
            # Vérifier les dimensions
            if len(X_values.shape) == 1:
                X_values = X_values.reshape(1, -1)
            
            # Vérifier que nous avons le bon nombre de caractéristiques
            expected_n_features = len(self.scaler.scale_) if hasattr(self.scaler, 'scale_') else X_values.shape[1]
            if X_values.shape[1] != expected_n_features:
                raise ValueError(f"Nombre incorrect de caractéristiques. Reçu {X_values.shape[1]}, attendu {expected_n_features}")
            
            # Normaliser les features
            X_scaled = self.scaler.transform(X_values)
            
            # Faire les prédictions
            if hasattr(self.model, 'predict_proba'):
                probas = self.model.predict_proba(X_scaled)
                # Retourne les probabilités de la classe positive (classe 1)
                if probas.shape[1] > 1:  # Si classification binaire ou multi-classe
                    return probas[:, 1] if probas.shape[1] == 2 else probas
                else:  # Si le modèle ne renvoie qu'une seule probabilité
                    return probas.flatten()
            else:
                # Si le modèle n'a pas de predict_proba, utiliser predict
                return self.model.predict(X_scaled)
                
        except Exception as e:
            self.logger.error(f"Erreur lors de la prédiction: {str(e)}")
            # Retourne un tableau de 0.5 (neutre) de la même longueur que le nombre d'échantillons
            n_samples = len(X) if hasattr(X, '__len__') else 1
            return np.full(n_samples, 0.5, dtype=float)
