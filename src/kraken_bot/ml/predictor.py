"""
Module de prédiction ML pour le bot de trading Kraken.
"""
import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from kraken_bot.core.config import Config
from .core.model_factory import ModelFactory
from .core.base_model import BaseModel
from .utils.metrics import ModelMetrics
from .utils.preprocessor import DataPreprocessor

logger = logging.getLogger(__name__)


class MLPredictor:
    """
    Prédicteur de tendance de marché basé sur l'apprentissage automatique.
    Utilise plusieurs modèles et sélectionne le meilleur en fonction des performances.
    """

    def __init__(self, config: dict = None) -> None:
        """Initialise le prédicteur ML avec la configuration fournie.

        Args:
            config: Dictionnaire de configuration optionnel avec les clés :
                - model_dir: Répertoire pour sauvegarder les modèles (défaut: 'models')
                - default_model_type: Type de modèle par défaut ('random_forest', 'xgboost', etc.)
                - test_size: Taille du jeu de test (entre 0 et 1, défaut: 0.2)
                - model_params: Paramètres pour chaque type de modèle
        """
        self.config = config if config else {}
        self.model_dir = Path(self.config.get('model_dir', 'models'))
        self.default_model_type = self.config.get('default_model_type', 'random_forest')
        self.test_size = self.config.get('test_size', 0.2)
        self.model_params = self.config.get('model_params', {})
        
        # Initialiser la fabrique de modèles
        self.model_factory = ModelFactory()
        
        # Dictionnaire pour stocker les modèles chargés
        self.models: Dict[str, BaseModel] = {}
        
        # Modèle actuellement sélectionné
        self.current_model: Optional[BaseModel] = None
        self.current_model_name: Optional[str] = None
        
        # Initialiser le prétraitement des données
        self.preprocessor = DataPreprocessor()
        
        # Métriques des modèles
        self.metrics = ModelMetrics()
        
        # Créer le répertoire des modèles s'il n'existe pas
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        # Charger les modèles existants
        self._load_saved_models()
        
        # Configurer le modèle par défaut s'il n'y a pas de modèles chargés
        if not self.models and self.default_model_type:
            self._setup_default_model()
    
    def _setup_default_model(self) -> None:
        """Configure le modèle par défaut si aucun modèle n'est chargé."""
        try:
            model = self.model_factory.create_model(
                model_type=self.default_model_type,
                params=self.model_params.get(self.default_model_type, {})
            )
            self.models[self.default_model_type] = model
            self.current_model = model
            self.current_model_name = self.default_model_type
            logger.info(f"Modèle par défaut {self.default_model_type} initialisé")
        except Exception as e:
            logger.error(f"Erreur lors de l'initialisation du modèle par défaut: {e}")
            raise
        
    def fit(self, X, y, model_name: str = None):
        """Entraîne un nouveau modèle sur les données fournies.
        
        Args:
            X: Données d'entraînement (features)
            y: Cibles d'entraînement (labels)
            model_name: Nom du modèle (optionnel, généré automatiquement si non fourni)
            
        Returns:
            dict: Métriques d'évaluation du modèle
        """
        try:
            if model_name is None:
                model_name = f"model_{int(datetime.now().timestamp())}"
                
            # Créer le modèle avec les paramètres par défaut ou spécifiés
            model_type = self.default_model_type
            model_params = self.model_params.get(model_type, {})
            
            # Créer le pipeline de prétraitement + modèle
            model = Pipeline([
                ('scaler', StandardScaler()),
                ('classifier', RandomForestClassifier(**model_params))
            ])
            
            # Diviser les données en ensembles d'entraînement et de test
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=self.test_size, random_state=42
            )
            
            # Entraîner le modèle
            model.fit(X_train, y_train)
            
            # Évaluer le modèle
            train_accuracy = model.score(X_train, y_train)
            test_accuracy = model.score(X_test, y_test)
            
            # Sauvegarder le modèle
            self.models[model_name] = model
            self.current_model = model
            self.current_model_name = model_name
            
            # Stocker les métriques dans le modèle pour une utilisation ultérieure
            metrics = {
                'train_accuracy': float(train_accuracy),
                'test_accuracy': float(test_accuracy),
                'accuracy': float(test_accuracy),  # Pour la compatibilité avec les tests
                'model_type': model_type
            }
            
            # Ajouter les métriques au modèle et à l'instance
            model.metrics_ = metrics
            self.metrics_ = metrics
            
            # Sauvegarder le modèle sur le disque
            try:
                self.save_model(model_name=model_name)
            except Exception as save_error:
                logger.warning(f"Échec de la sauvegarde du modèle {model_name}: {save_error}")
            
            # Retourner les métriques
            return {
                'model_name': model_name,
                'metrics': metrics,
                'model': model,
                'created_at': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Erreur lors de l'entraînement du modèle: {e}")
            raise
    
    def predict(self, X, model_name: str = None):
        """Effectue des prédictions avec le modèle spécifié ou le modèle courant.
        
        Args:
            X: Données à prédire
            model_name: Nom du modèle à utiliser (optionnel, utilise le modèle courant si non spécifié)
            
        Returns:
            np.ndarray: Prédictions du modèle
        """
        try:
            model = self.current_model
            if model_name is not None and model_name in self.models:
                model = self.models[model_name]
                
            if model is None:
                raise ValueError("Aucun modèle disponible pour la prédiction")
                
            return model.predict(X)
            
        except Exception as e:
            logger.error(f"Erreur lors de la prédiction: {e}")
            raise
            
    def predict_proba(self, X, model_name: str = None):
        """Calcule les probabilités des classes pour les échantillons fournis.
        
        Args:
            X: Données à prédire (forme: [n_samples, n_features])
            model_name: Nom du modèle à utiliser (optionnel, utilise le modèle courant si non spécifié)
            
        Returns:
            np.ndarray: Probabilités des classes (forme: [n_samples, n_classes])
        """
        try:
            model = self.current_model
            if model_name is not None and model_name in self.models:
                model = self.models[model_name]
                
            if model is None:
                raise ValueError("Aucun modèle disponible pour la prédiction")
                
            # Vérifier si le modèle implémente predict_proba
            if hasattr(model, 'predict_proba'):
                return model.predict_proba(X)
            else:
                # Si le modèle n'implémente pas predict_proba, simuler avec une confiance de 100%
                # pour la classe prédite (utile pour les modèles comme SVM sans probabilités)
                predictions = model.predict(X)
                n_samples = len(predictions)
                n_classes = len(np.unique(predictions))
                probas = np.zeros((n_samples, n_classes))
                for i, pred in enumerate(predictions):
                    probas[i, int(pred)] = 1.0
                return probas
                
        except Exception as e:
            logger.error(f"Erreur lors du calcul des probabilités: {e}")
            # En cas d'erreur, retourner des probabilités uniformes
            n_samples = X.shape[0] if hasattr(X, 'shape') else 1
            return np.ones((n_samples, 3)) / 3  # Supposons 3 classes par défaut
    
    def evaluate(self, X, y, model_name: str = None):
        """Évalue le modèle sur les données fournies.
        
        Args:
            X: Données de test
            y: Vraies étiquettes
            model_name: Nom du modèle à évaluer (optionnel, utilise le modèle courant si non spécifié)
            
        Returns:
            dict: Métriques d'évaluation
        """
        try:
            model = self.current_model
            if model_name is not None and model_name in self.models:
                model = self.models[model_name]
                
            if model is None:
                raise ValueError("Aucun modèle disponible pour l'évaluation")
                
            accuracy = model.score(X, y)
            
            return {
                'accuracy': accuracy,
                'model_name': model_name or self.current_model_name,
                'num_samples': len(X)
            }
            
        except Exception as e:
            logger.error(f"Erreur lors de l'évaluation du modèle: {e}")
            raise
    
    def save_model(self, model_name: str = None, path: str = None):
        """Sauvegarde le modèle spécifié ou le modèle courant sur le disque.
        
        Args:
            model_name: Nom du modèle à sauvegarder (optionnel, utilise le modèle courant si non spécifié)
            path: Chemin de destination (optionnel, utilise le répertoire des modèles si non spécifié)
            
        Returns:
            str: Chemin du modèle sauvegardé
        """
        try:
            model_name = model_name or self.current_model_name
            if model_name not in self.models:
                raise ValueError(f"Modèle {model_name} non trouvé")
                
            model = self.models[model_name]
            
            # Utiliser le chemin fourni ou le répertoire par défaut
            if path is not None and os.path.isdir(path):
                # Si un dossier est fourni, créer un sous-dossier avec le nom du modèle
                model_dir = os.path.join(path, model_name)
            else:
                # Sinon utiliser le répertoire des modèles
                model_dir = os.path.join(self.model_dir, model_name)
                
            # Créer le dossier s'il n'existe pas
            os.makedirs(model_dir, exist_ok=True)
            
            # Chemin complet pour le modèle
            model_path = os.path.join(model_dir, 'model.joblib')
            
            # Sauvegarder le modèle
            joblib.dump(model, model_path)
            
            # Sauvegarder également le scaler séparément si c'est un pipeline
            if hasattr(model, 'named_steps') and 'scaler' in model.named_steps:
                scaler_path = os.path.join(model_dir, 'scaler.joblib')
                joblib.dump(model.named_steps['scaler'], scaler_path)
            
            # Sauvegarder les métriques si disponibles
            metrics = {}
            if hasattr(model, 'metrics_'):
                metrics = model.metrics_
            elif hasattr(self, 'metrics_'):
                metrics = self.metrics_
            
            if metrics:
                # Sauvegarder les métriques dans metrics.json
                metrics_path = os.path.join(model_dir, 'metrics.json')
                with open(metrics_path, 'w') as f:
                    json.dump(metrics, f)
                
                # Créer également model_info.json avec les informations du modèle
                model_info = {
                    'name': model_name,
                    'model_type': metrics.get('model_type', 'unknown'),
                    'accuracy': metrics.get('accuracy', 0.0),
                    'train_accuracy': metrics.get('train_accuracy', 0.0),
                    'test_accuracy': metrics.get('test_accuracy', 0.0),
                    'created_at': datetime.now(timezone.utc).isoformat(),
                    'features': [f'feature_{i}' for i in range(10)]  # Exemple de caractéristiques
                }
                
                model_info_path = os.path.join(model_dir, 'model_info.json')
                with open(model_info_path, 'w') as f:
                    json.dump(model_info, f, indent=2)
            
            logger.info(f"Modèle {model_name} sauvegardé avec succès dans {model_dir}")
            return model_dir
            
        except Exception as e:
            logger.error(f"Erreur lors de la sauvegarde du modèle: {e}")
            raise
    
    def load_model(self, model_name: str, path: str = None):
        """Charge un modèle depuis le disque.
        
        Args:
            model_name: Nom à donner au modèle chargé
            path: Chemin du modèle à charger (optionnel, utilise le répertoire des modèles si non spécifié)
            
        Returns:
            Le modèle chargé
        """
        try:
            # Déterminer le chemin du dossier du modèle
            if path is None:
                model_dir = os.path.join(self.model_dir, model_name)
            else:
                model_dir = path if os.path.isdir(path) else os.path.dirname(path)
            
            # Chemin complet vers le fichier du modèle
            model_path = os.path.join(model_dir, 'model.joblib')
            
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Aucun modèle trouvé à l'emplacement: {model_path}")
            
            # Charger le modèle
            model = joblib.load(model_path)
            
            # Charger les métriques si elles existent
            metrics_path = os.path.join(model_dir, 'metrics.json')
            if os.path.exists(metrics_path):
                with open(metrics_path, 'r') as f:
                    metrics = json.load(f)
                if hasattr(model, 'metrics_'):
                    model.metrics_ = metrics
                else:
                    # Si le modèle n'a pas d'attribut metrics_, on le stocke dans l'instance
                    if not hasattr(self, 'metrics_'):
                        self.metrics_ = {}
                    self.metrics_.update(metrics)
            
            # Mettre à jour les attributs
            self.models[model_name] = model
            self.current_model = model
            self.current_model_name = model_name
            
            logger.info(f"Modèle {model_name} chargé avec succès depuis {model_path}")
            return model
            
        except Exception as e:
            logger.error(f"Erreur lors du chargement du modèle: {e}")
            raise
    
    def get_feature_importance(self, model_name: str = None):
        """Récupère l'importance des caractéristiques pour le modèle spécifié.
        
        Args:
            model_name: Nom du modèle (optionnel, utilise le modèle courant si non spécifié)
            
        Returns:
            dict: Dictionnaire des caractéristiques et de leur importance
        """
        try:
            model = self.current_model
            if model_name is not None and model_name in self.models:
                model = self.models[model_name]
                
            if model is None:
                raise ValueError("Aucun modèle disponible")
                
            # Extraire le classificateur du pipeline
            classifier = model.named_steps.get('classifier', model)
            
            if hasattr(classifier, 'feature_importances_'):
                importances = classifier.feature_importances_
                return dict(zip(range(len(importances)), importances))
            else:
                logger.warning("Le modèle ne fournit pas d'importance des caractéristiques")
                return {}
                
        except Exception as e:
            logger.error(f"Erreur lors de la récupération de l'importance des caractéristiques: {e}")
            raise
    
    def get_model_info(self, model_name: str = None):
        """Récupère des informations sur le modèle spécifié.
        
        Args:
            model_name: Nom du modèle (optionnel, utilise le modèle courant si non spécifié)
            
        Returns:
            dict: Informations sur le modèle
        """
        try:
            model = self.current_model
            model_name = model_name or self.current_model_name
            
            if model_name is not None and model_name in self.models:
                model = self.models[model_name]
            elif model is None:
                raise ValueError("Aucun modèle disponible")
                
            # Extraire le classificateur du pipeline
            classifier = model.named_steps.get('classifier', model)
            
            # Vérifier si le modèle est sauvegardé sur disque
            model_path = os.path.join(self.model_dir, model_name)
            created_at = None
            metrics = {}
            
            if os.path.exists(model_path):
                model_file = os.path.join(model_path, 'model.joblib')
                if os.path.exists(model_file):
                    created_at = datetime.fromtimestamp(
                        os.path.getmtime(model_file)
                    ).isoformat()
                
                # Essayer de charger les métriques si elles existent
                metrics_file = os.path.join(model_path, 'metrics.json')
                if os.path.exists(metrics_file):
                    try:
                        with open(metrics_file, 'r') as f:
                            metrics = json.load(f)
                    except Exception as e:
                        logger.warning(f"Impossible de charger les métriques: {e}")
            
            # Obtenir les paramètres du modèle
            model_params = {}
            if hasattr(classifier, 'get_params'):
                model_params = classifier.get_params()
            
            # Obtenir l'importance des caractéristiques si disponible
            feature_importance = {}
            if hasattr(classifier, 'feature_importances_'):
                feature_importance = {
                    f'feature_{i}': float(imp) 
                    for i, imp in enumerate(classifier.feature_importances_)
                }
            
            # Créer le dictionnaire d'informations
            info = {
                'model_type': type(classifier).__name__,
                'model_params': model_params,
                'num_features': getattr(classifier, 'n_features_in_', None),
                'is_fitted': hasattr(classifier, 'classes_') or hasattr(classifier, 'coef_'),
                'created_at': created_at or datetime.now().isoformat(),
                'name': model_name,
                'metrics': metrics,  # Ajout des métriques
                'feature_importance': feature_importance  # Ajout de l'importance des caractéristiques
            }
            
            return info
            
        except Exception as e:
            logger.error(f"Erreur lors de la récupération des informations du modèle: {e}")
            raise
    
    def get_available_models(self):
        """Récupère la liste des modèles disponibles.
        
        Returns:
            list: Liste des dictionnaires d'informations sur les modèles disponibles
        """
        try:
            models_info = []
            
            # Ajouter les modèles en mémoire
            for model_name in self.models:
                models_info.append(self.get_model_info(model_name))
            
            # Ajouter les modèles sur disque
            if os.path.exists(self.model_dir):
                for model_name in os.listdir(self.model_dir):
                    model_path = os.path.join(self.model_dir, model_name)
                    if os.path.isdir(model_path) and model_name not in self.models:
                        try:
                            # Charger le modèle pour obtenir ses informations
                            model = joblib.load(os.path.join(model_path, 'model.joblib'))
                            self.models[model_name] = model
                            models_info.append(self.get_model_info(model_name))
                        except Exception as e:
                            logger.warning(f"Impossible de charger le modèle {model_name}: {e}")
            
            # Supprimer les doublons (en gardant la première occurrence)
            seen = set()
            unique_models = []
            for model in models_info:
                if model['name'] not in seen:
                    seen.add(model['name'])
                    unique_models.append(model)
            
            return unique_models
            
        except Exception as e:
            logger.error(f"Erreur lors de la récupération de la liste des modèles: {e}")
            return []

    def _load_saved_models(self) -> None:
        """Charge les modèles sauvegardés depuis le répertoire des modèles.
        
        Cette méthode parcourt tous les sous-dossiers de model_dir et tente de charger
        les modèles sauvegardés au format model.joblib.
        """
        try:
            # Vérifier si le répertoire des modèles existe
            if not os.path.exists(self.model_dir):
                logger.warning(f"Le répertoire des modèles {self.model_dir} n'existe pas.")
                return
                
            # Parcourir tous les dossiers dans le répertoire des modèles
            model_dirs = [
                d for d in os.listdir(self.model_dir)
                if os.path.isdir(os.path.join(self.model_dir, d))
            ]
            
            if not model_dirs:
                logger.info("Aucun modèle sauvegardé trouvé dans le répertoire des modèles.")
                return
                
            # Charger chaque modèle trouvé
            loaded_models = 0
            for model_name in model_dirs:
                try:
                    model_dir = os.path.join(self.model_dir, model_name)
                    model_path = os.path.join(model_dir, 'model.joblib')
                    
                    # Vérifier si le fichier du modèle existe
                    if not os.path.exists(model_path):
                        logger.debug(f"Aucun modèle trouvé dans {model_dir}")
                        continue
                        
                    # Charger le modèle
                    model = joblib.load(model_path)
                    self.models[model_name] = model
                    loaded_models += 1
                    
                    # Charger les métriques si elles existent
                    metrics_path = os.path.join(model_dir, 'metrics.json')
                    if os.path.exists(metrics_path):
                        with open(metrics_path, 'r') as f:
                            metrics = json.load(f)
                        if hasattr(model, 'metrics_'):
                            model.metrics_ = metrics
                        else:
                            if not hasattr(self, 'metrics_'):
                                self.metrics_ = {}
                            self.metrics_.update(metrics)
                    
                    logger.info(f"Modèle chargé: {model_name}")
                    
                    # Définir le premier modèle chargé comme modèle courant
                    if self.current_model is None:
                        self.current_model = model
                        self.current_model_name = model_name
                        
                except Exception as e:
                    logger.error(f"Erreur lors du chargement du modèle {model_name}: {e}")
            
            if loaded_models > 0:
                logger.info(f"{loaded_models} modèles chargés avec succès depuis {self.model_dir}")
                if self.current_model_name:
                    logger.info(f"Modèle actif: {self.current_model_name}")
            else:
                logger.warning("Aucun modèle n'a pu être chargé depuis le répertoire des modèles.")
                
        except Exception as e:
            logger.error(f"Erreur lors du chargement des modèles sauvegardés: {e}")
            logger.warning("Les modèles par défaut seront utilisés.")

    def _setup_default_models(self) -> None:
        """Configure les modèles par défaut avec des paramètres optimisés.
        
        Cette méthode est conservée pour la compatibilité mais ne fait plus rien
        car les modèles sont maintenant créés à la volée dans la méthode fit().
        """
        pass

    def load_historical_data(
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
            pair_row = self.db_manager.fetchrow(
                "SELECT id FROM trading_pairs WHERE symbol = $1", pair
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
            records = self.db_manager.fetch(query, pair_id)
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

    def _prepare_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """Prépare les caractéristiques pour l'entraînement.

        Args:
            df: DataFrame contenant les données OHLCV

        Returns:
            Tuple de (features, target) pour l'entraînement
        """
        try:
            # Conversion explicite des colonnes numériques
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Suppression des lignes avec valeurs manquantes
            df = df.dropna(subset=['open', 'high', 'low', 'close', 'volume'])
            
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df.set_index('timestamp', inplace=True)

            # Calcul des indicateurs techniques
            df = df.copy()

            # Retours logarithmiques
            df['returns'] = np.log(df['close'] / df['close'].shift(1))

            # Volatilité sur 20 périodes
            df['volatility'] = df['returns'].rolling(window=20).std() * np.sqrt(252)

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
                'volume_ma20'
            ]
            X = df[features]
            y = df['target']

            return X, y

        except Exception as e:
            logger.error(
                f"Erreur lors de la préparation des caractéristiques: {e}"
            )
            return pd.DataFrame(), pd.Series(dtype=float)
            
    def _get_latest_score(self, pair: str) -> Optional[float]:
        """Récupère le score du dernier modèle entraîné."""
        try:
            model_data = self._load_latest_model(pair)
            return model_data.get('test_score') if model_data else None
        except Exception as e:
            logger.error(
                f"Erreur lors de la récupération du score pour {pair}: {e}"
            )
            return None

    def train(self, pair: str, force_retrain: bool = False, days: int = 90) -> Optional[float]:
        """Entraîne un nouveau modèle pour la paire spécifiée.
        
        Args:
            pair: Paire de trading (ex: 'BTC/USD')
            force_retrain: Si True, force le réentraînement même si non nécessaire
            days: Nombre de jours de données à utiliser pour l'entraînement
            
        Returns:
            Score du modèle entraîné ou None en cas d'erreur
        """
        try:
            # Vérifier si un réentraînement est nécessaire
            if not force_retrain and not self._needs_retraining(pair):
                return self._get_latest_score(pair)

            # Charger les données
            df = self.load_historical_data(pair, days)
            if df is None or df.empty:
                logger.error(f"Impossible de charger les données pour {pair}")
                return None

            min_data_points = 100  # Valeur par défaut
            if len(df) < min_data_points:
                logger.warning(
                    f"Pas assez de données pour {pair} ({len(df)} < {min_data_points})")
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

            self._save_model(pair, model_data)

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

    def predict_pair(self, pair: str,
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
            model_data = self._load_latest_model(pair)
            if model_data is None:
                logger.warning(
                    f"Aucun modèle trouvé pour {pair}, entraînement en cours...")
                score = self.train(pair, force_retrain=True)
                if score is None:
                    return None
                model_data = self._load_latest_model(pair)
                if model_data is None:
                    return None

            # Charger les données si non fournies
            if df is None:
                # Dernière semaine
                df = self.load_historical_data(pair, days=7)
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

    def _needs_retraining(self, pair: str) -> bool:
        """Vérifie si le modèle a besoin d'être réentraîné."""
        try:
            model_data = self._load_latest_model(pair)
            if model_data is None:
                return True

            # Vérifier l'âge du modèle
            max_model_age_days = 7  # Valeur par défaut
            trained_at = datetime.fromtimestamp(
                model_data.get('trained_at', 0), timezone.utc)
            model_age = datetime.now(timezone.utc) - trained_at

            if model_age.days >= max_model_age_days:
                logger.info(
                    f"Modèle pour {pair} trop ancien ({model_age.days} jours)")
                return True

            # Vérifier le nombre de points de données
            min_data_points = 100  # Valeur par défaut
            data_points = model_data.get('data_points', 0)
            if data_points < min_data_points:
                logger.info(
                    f"Modèle pour {pair} a trop peu de données ({data_points} points)")
                return True

            return False

        except Exception as e:
            logger.error(
                f"Erreur lors de la vérification du modèle pour {pair}: {e}"
            )
            return True

    def _get_latest_score(self, pair: str) -> Optional[float]:
        """Récupère le score du dernier modèle entraîné."""
        try:
            model_data = self._load_latest_model(pair)
            return model_data.get('test_score') if model_data else None
        except Exception as e:
            logger.error(
                f"Erreur lors de la récupération du score pour {pair}: {e}"
            )
            return None

    def _get_model_path(
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

    def _save_model(self, pair: str, model_data: Dict[str, Any]) -> bool:
        """Sauvegarde le modèle sur le disque."""
        try:
            model_path = self._get_model_path(pair, model_data['trained_at'])
            joblib.dump(model_data, model_path)

            # Nettoyer les anciens modèles (garder les 3 plus récents)
            self._cleanup_old_models(pair)

            return True

        except Exception as e:
            logger.error(
                f"Erreur lors de la sauvegarde du modèle pour {pair}: {e}"
            )
            return False

    def _load_latest_model(self, pair: str) -> Optional[Dict[str, Any]]:
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

    def _cleanup_old_models(self, pair: str, keep: int = 3) -> None:
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

    def train_risk_model(
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
