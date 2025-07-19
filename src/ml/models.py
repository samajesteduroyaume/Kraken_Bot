"""
Module pour les modèles d'apprentissage automatique du bot de trading.
"""
import numpy as np
from abc import ABC, abstractmethod
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
import joblib
import os
from datetime import datetime


class BaseModel(ABC):
    """Classe de base pour tous les modèles de trading."""

    def __init__(self, name, model_params=None):
        """Initialise le modèle de base.

        Args:
            name (str): Nom du modèle
            model_params (dict, optional): Paramètres du modèle. Defaults to None.
        """
        self.name = name
        self.model = None
        self.scaler = StandardScaler()
        self.model_params = model_params or {}
        self.last_trained = None
        self.performance_metrics = {}

    @abstractmethod
    def create_model(self):
        """Crée une instance du modèle."""

    def preprocess_data(self, X, y=None, fit_scaler=False):
        """Prétraite les données d'entrée.

        Args:
            X (pd.DataFrame): Features
            y (pd.Series, optional): Target. Defaults to None.
            fit_scaler (bool, optional): Si True, ajuste le scaler. Defaults to False.

        Returns:
            tuple: (X_processed, y_processed)
        """
        if fit_scaler:
            X_scaled = self.scaler.fit_transform(X)
        else:
            X_scaled = self.scaler.transform(X)

        if y is not None:
            return X_scaled, y.values
        return X_scaled

    def train(self, X, y, cv_splits=5):
        """Entraîne le modèle avec validation croisée.

        Args:
            X (pd.DataFrame): Features
            y (pd.Series): Target
            cv_splits (int, optional): Nombre de plis pour la validation croisée. Defaults to 5.

        Returns:
            dict: Métriques de performance
        """
        self.model = self.create_model()
        X_processed, y_processed = self.preprocess_data(X, y, fit_scaler=True)

        # Validation croisée
        cv = TimeSeriesSplit(n_splits=cv_splits)
        scoring = [
            'accuracy',
            'precision_weighted',
            'recall_weighted',
            'f1_weighted']

        metrics = {}
        for score in scoring:
            scores = cross_val_score(
                self.model, X_processed, y_processed,
                cv=cv, scoring=score, n_jobs=-1
            )
            metrics[f'cv_{score}_mean'] = np.mean(scores)
            metrics[f'cv_{score}_std'] = np.std(scores)

        # Entraînement final sur toutes les données
        self.model.fit(X_processed, y_processed)
        self.last_trained = datetime.now()
        self.performance_metrics = metrics

        return metrics

    def predict(self, X):
        """Fait des prédictions sur de nouvelles données.

        Args:
            X (pd.DataFrame): Nouvelles données

        Returns:
            np.array: Prédictions
        """
        if self.model is None:
            raise ValueError("Le modèle n'a pas été entraîné.")

        X_processed = self.preprocess_data(X)
        return self.model.predict(X_processed)

    def save(self, directory='models'):
        """Sauvegarde le modèle et le scaler.

        Args:
            directory (str, optional): Dossier de sauvegarde. Defaults to 'models'.

        Returns:
            str: Chemin complet du fichier du modèle sauvegardé
        """
        os.makedirs(directory, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        # Sauvegarder le modèle
        model_path = os.path.join(directory, f'{self.name}_{timestamp}.joblib')
        joblib.dump({
            'model': self.model,
            'scaler': self.scaler,
            'last_trained': self.last_trained,
            'performance_metrics': self.performance_metrics
        }, model_path)

        return model_path

    @classmethod
    def load(cls, filepath):
        """Charge un modèle sauvegardé.

        Args:
            filepath (str): Chemin vers le fichier du modèle

        Returns:
            BaseModel: Instance du modèle chargé
        """
        data = joblib.load(filepath)
        model = cls(name=os.path.basename(filepath).split('_')[0])
        model.model = data['model']
        model.scaler = data['scaler']
        model.last_trained = data['last_trained']
        model.performance_metrics = data.get('performance_metrics', {})

        return model


class RandomForestModel(BaseModel):
    """Modèle Random Forest pour le trading."""

    def __init__(self, model_params=None):
        """Initialise le modèle Random Forest.

        Args:
            model_params (dict, optional): Paramètres du modèle. Defaults to None.
        """
        default_params = {
            'n_estimators': 100,
            'max_depth': 5,
            'random_state': 42,
            'n_jobs': -1,
            'class_weight': 'balanced'
        }

        if model_params:
            default_params.update(model_params)

        super().__init__('random_forest', default_params)

    def create_model(self):
        """Crée une instance du modèle Random Forest."""
        return RandomForestClassifier(**self.model_params)


class XGBoostModel(BaseModel):
    """Modèle XGBoost pour le trading."""

    def __init__(self, model_params=None):
        """Initialise le modèle XGBoost.

        Args:
            model_params (dict, optional): Paramètres du modèle. Defaults to None.
        """
        default_params = {
            'n_estimators': 100,
            'max_depth': 3,
            'learning_rate': 0.1,
            'random_state': 42,
            'n_jobs': -1,
            'scale_pos_weight': 1,
            'use_label_encoder': False,
            'eval_metric': 'logloss'
        }

        if model_params:
            default_params.update(model_params)

        super().__init__('xgboost', default_params)

    def create_model(self):
        """Crée une instance du modèle XGBoost."""
        return XGBClassifier(**self.model_params)


class NeuralNetworkModel(BaseModel):
    """Modèle de réseau de neurones pour le trading."""

    def __init__(self, model_params=None):
        """Initialise le modèle de réseau de neurones.

        Args:
            model_params (dict, optional): Paramètres du modèle. Defaults to None.
        """
        default_params = {
            'hidden_layer_sizes': (100, 50),
            'activation': 'relu',
            'solver': 'adam',
            'alpha': 0.0001,
            'batch_size': 'auto',
            'learning_rate': 'adaptive',
            'max_iter': 200,
            'random_state': 42,
            'early_stopping': True,
            'validation_fraction': 0.1,
            'n_iter_no_change': 10
        }

        if model_params:
            default_params.update(model_params)

        super().__init__('neural_network', default_params)

    def create_model(self):
        """Crée une instance du modèle de réseau de neurones."""
        return MLPClassifier(**self.model_params)


def get_model(model_name, params=None):
    """Factory pour créer des modèles par nom.

    Args:
        model_name (str): Nom du modèle ('random_forest', 'xgboost', 'neural_network')
        params (dict, optional): Paramètres du modèle. Defaults to None.

    Returns:
        BaseModel: Instance du modèle demandé

    Raises:
        ValueError: Si le nom du modèle n'est pas reconnu
    """
    models = {
        'random_forest': RandomForestModel,
        'xgboost': XGBoostModel,
        'neural_network': NeuralNetworkModel
    }

    if model_name not in models:
        raise ValueError(
            f"Modèle non reconnu: {model_name}. Choisissez parmi: {', '.join(models.keys())}")

    return models[model_name](params)
