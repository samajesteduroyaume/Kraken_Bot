"""
Gestionnaire de modèles pour l'entraînement continu et la sélection des modèles.
"""
import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, Optional, Tuple
import logging

from .models import get_model, BaseModel

logger = logging.getLogger(__name__)


class ModelManager:
    """Gère l'entraînement, l'évaluation et la sélection des modèles de trading."""

    def __init__(self, models_dir: str = 'models', retrain_interval: int = 24):
        """Initialise le gestionnaire de modèles.

        Args:
            models_dir (str, optional): Dossier de stockage des modèles. Defaults to 'models'.
            retrain_interval (int, optional): Intervalle de réentraînement en heures. Defaults to 24.
        """
        self.models_dir = models_dir
        self.retrain_interval = retrain_interval
        self.models: Dict[str, BaseModel] = {}
        self.best_model: Optional[BaseModel] = None
        self.best_model_name: Optional[str] = None
        self.model_performance: Dict[str, dict] = {}

        # Créer le dossier des modèles s'il n'existe pas
        os.makedirs(models_dir, exist_ok=True)

    def add_model(
            self,
            model_name: str,
            model_params: Optional[dict] = None) -> None:
        """Ajoute un nouveau modèle au gestionnaire.

        Args:
            model_name (str): Nom du modèle à ajouter
            model_params (dict, optional): Paramètres du modèle. Defaults to None.
        """
        try:
            model = get_model(model_name, model_params)
            self.models[model_name] = model
            self.model_performance[model_name] = {
                'last_trained': None,
                'metrics': {},
                'training_history': []
            }
            logger.info(f"Modèle ajouté: {model_name}")
        except Exception as e:
            logger.error(f"Erreur lors de l'ajout du modèle {model_name}: {e}")

    def train_models(self, X: pd.DataFrame, y: pd.Series,
                     cv_splits: int = 5) -> Dict[str, dict]:
        """Entraîne tous les modèles avec validation croisée.

        Args:
            X (pd.DataFrame): Features
            y (pd.Series): Target
            cv_splits (int, optional): Nombre de plis pour la validation croisée. Defaults to 5.

        Returns:
            Dict[str, dict]: Métriques de performance par modèle
        """
        if not self.models:
            logger.warning(
                "Aucun modèle à entraîner. Ajoutez d'abord des modèles avec add_model().")
            return {}

        performance = {}

        for name, model in self.models.items():
            try:
                logger.info(f"Entraînement du modèle: {name}")
                metrics = model.train(X, y, cv_splits)

                # Mettre à jour les performances
                self.model_performance[name].update({
                    'last_trained': datetime.now(),
                    'metrics': metrics,
                    'training_history': self.model_performance[name].get('training_history', []) + [{
                        'timestamp': datetime.now(),
                        'metrics': metrics
                    }]
                })

                performance[name] = metrics
                logger.info(
                    f"Modèle {name} entraîné avec succès. Précision: {metrics.get('cv_accuracy_mean', 0):.4f}")

                # Sauvegarder le modèle
                model_path = model.save(self.models_dir)
                logger.info(f"Modèle {name} sauvegardé dans {model_path}")

            except Exception as e:
                logger.error(
                    f"Erreur lors de l'entraînement du modèle {name}: {e}",
                    exc_info=True)

        # Mettre à jour le meilleur modèle
        self._update_best_model()

        return performance

    def _update_best_model(self) -> None:
        """Met à jour la référence vers le meilleur modèle basé sur les performances."""
        if not self.model_performance:
            return

        best_score = -1
        best_model_name = None

        for name, perf in self.model_performance.items():
            if not perf['metrics']:
                continue

            # Utiliser la précision comme métrique principale
            score = perf['metrics'].get('cv_accuracy_mean', 0)

            if score > best_score:
                best_score = score
                best_model_name = name

        if best_model_name and best_model_name in self.models:
            self.best_model = self.models[best_model_name]
            self.best_model_name = best_model_name
            logger.info(
                f"Nouveau meilleur modèle: {best_model_name} (score: {best_score:.4f})")

    def should_retrain(self, model_name: Optional[str] = None) -> bool:
        """Vérifie si un modèle (ou le meilleur modèle) doit être réentraîné.

        Args:
            model_name (str, optional): Nom du modèle à vérifier. Si None, vérifie le meilleur modèle.

        Returns:
            bool: True si le modèle doit être réentraîné, False sinon
        """
        if model_name is None:
            if self.best_model_name is None:
                return True
            model_name = self.best_model_name

        if model_name not in self.model_performance:
            return True

        last_trained = self.model_performance[model_name].get('last_trained')
        if last_trained is None:
            return True

        time_since_last_train = datetime.now() - last_trained
        return time_since_last_train > timedelta(hours=self.retrain_interval)

    def get_predictions(self,
                        X: pd.DataFrame,
                        model_name: Optional[str] = None) -> Tuple[np.ndarray,
                                                                   str]:
        """Obtient des prédictions à partir d'un modèle spécifique ou du meilleur modèle.

        Args:
            X (pd.DataFrame): Données d'entrée
            model_name (str, optional): Nom du modèle à utiliser. Si None, utilise le meilleur modèle.

        Returns:
            Tuple[np.ndarray, str]: Prédictions et nom du modèle utilisé
        """
        if model_name is None:
            if self.best_model is None:
                raise ValueError(
                    "Aucun modèle disponible pour les prédictions.")
            model = self.best_model
            model_name = self.best_model_name
        else:
            if model_name not in self.models:
                raise ValueError(f"Modèle {model_name} non trouvé.")
            model = self.models[model_name]

        return model.predict(X), model_name

    def get_model_summary(self) -> dict:
        """Retourne un résumé des performances des modèles.

        Returns:
            dict: Résumé des performances
        """
        summary = {
            'best_model': self.best_model_name,
            'models': {}
        }

        for name, perf in self.model_performance.items():
            summary['models'][name] = {
                'last_trained': perf.get('last_trained', 'Jamais'),
                'metrics': perf.get('metrics', {}),
                'is_best': name == self.best_model_name
            }

        return summary

    def save_models(self) -> Dict[str, str]:
        """Sauvegarde tous les modèles gérés dans le dossier des modèles.

        Returns:
            Dict[str, str]: Dictionnaire des chemins de sauvegarde par nom de modèle
        """
        saved_paths = {}

        for name, model in self.models.items():
            try:
                if model is not None:
                    # Créer un sous-dossier pour chaque modèle
                    model_dir = os.path.join(self.models_dir, name)
                    os.makedirs(model_dir, exist_ok=True)

                    # Générer un nom de fichier avec horodatage
                    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                    filename = f"{name}_{timestamp}.joblib"
                    os.path.join(model_dir, filename)

                    # Sauvegarder le modèle
                    model_path = model.save(directory=model_dir)
                    saved_paths[name] = model_path

                    # Créer un lien symbolique vers la dernière version
                    latest_path = os.path.join(
                        model_dir, f"{name}_latest.joblib")
                    if os.path.exists(latest_path):
                        os.remove(latest_path)
                    os.symlink(os.path.basename(model_path), latest_path)

                    logger.info(f"Modèle {name} sauvegardé dans {model_path}")

            except Exception as e:
                logger.error(
                    f"Erreur lors de la sauvegarde du modèle {name}: {e}")
                saved_paths[name] = f"error: {str(e)}"

        return saved_paths

    def load_models(self) -> None:
        """Charge les modèles sauvegardés depuis le dossier des modèles."""
        if not os.path.exists(self.models_dir):
            logger.warning(
                f"Le dossier des modèles {self.models_dir} n'existe pas.")
            return

        # Parcourir les sous-dossiers de modèles
        for model_name in os.listdir(self.models_dir):
            model_dir = os.path.join(self.models_dir, model_name)
            if not os.path.isdir(model_dir):
                continue

            # Chercher le dernier modèle sauvegardé
            latest_model = os.path.join(
                model_dir, f"{model_name}_latest.joblib")
            if not os.path.exists(latest_model):
                logger.warning(f"Aucun modèle trouvé pour {model_name}")
                continue

            try:
                # Charger le modèle
                model = BaseModel.load(latest_model)
                self.models[model_name] = model

                # Mettre à jour les métriques
                if model_name not in self.model_performance:
                    self.model_performance[model_name] = {
                        'last_trained': model.last_trained,
                        'metrics': model.performance_metrics,
                        'training_history': [{
                            'timestamp': model.last_trained,
                            'metrics': model.performance_metrics
                        }] if model.last_trained else []
                    }

                logger.info(
                    f"Modèle chargé: {model_name} (entraîné le {model.last_trained})")

            except Exception as e:
                logger.error(
                    f"Erreur lors du chargement du modèle {model_name}: {e}")

        # Mettre à jour le meilleur modèle après le chargement
        self._update_best_model()

    def get_model(self, model_name: str) -> Optional[BaseModel]:
        """Récupère un modèle par son nom.

        Args:
            model_name (str): Nom du modèle

        Returns:
            Optional[BaseModel]: Instance du modèle ou None si non trouvé
        """
        return self.models.get(model_name)

    def get_best_model(self) -> Tuple[Optional[BaseModel], Optional[str]]:
        """Récupère le meilleur modèle disponible.

        Returns:
            Tuple[Optional[BaseModel], Optional[str]]: (instance du modèle, nom du modèle)
        """
        return self.best_model, self.best_model_name
