"""
Module de base pour les modèles de machine learning du bot de trading.

Ce module fournit une classe de base abstraite pour tous les modèles de trading,
ainsi qu'un scaler personnalisé pour une meilleure gestion de la sérialisation.

Classes:
    PersistentStandardScaler: Scaler personnalisé avec une meilleure gestion de la sérialisation.
    BaseModel: Classe de base abstraite pour tous les modèles de trading.
"""
import os
import joblib
import numpy as np
from datetime import datetime
from typing import Dict, Optional, Any
from sklearn.base import BaseEstimator
from sklearn.exceptions import NotFittedError
from sklearn.preprocessing import StandardScaler as SklearnStandardScaler
from sklearn.utils.validation import check_is_fitted
from abc import ABC, abstractmethod

class PersistentStandardScaler(SklearnStandardScaler):
    """StandardScaler personnalisé avec une meilleure gestion de la sérialisation.
    
    Cette classe étend StandardScaler de scikit-learn pour assurer une sérialisation
    et une désérialisation correctes de l'état du scaler. Elle résout les problèmes
    de compatibilité lors de la sauvegarde et du chargement des modèles.
    
    Attributes:
        mean_ (np.ndarray, optional): La moyenne de chaque caractéristique dans l'ensemble d'entraînement.
        scale_ (np.ndarray, optional): L'écart-type de chaque caractéristique dans l'ensemble d'entraînement.
        n_features_in_ (int, optional): Nombre de caractéristiques vues pendant l'entraînement.
        feature_names_in_ (np.ndarray, optional): Noms des caractéristiques vus pendant l'entraînement.
    """
    
    def __init__(self, **kwargs):
        # Appel au constructeur parent avec les paramètres par défaut
        super().__init__(
            with_mean=kwargs.get('with_mean', True),
            with_std=kwargs.get('with_std', True),
            copy=kwargs.get('copy', True)
        )
        # Initialisation des attributs pour éviter les erreurs
        self.mean_ = None
        self.scale_ = None
        self.var_ = None
        self.n_features_in_ = None
        self.n_samples_seen_ = 0
        
        # Forcer l'initialisation des attributs internes de scikit-learn
        self._sklearn_is_fitted = False
        self._sklearn_validation_attributes = ['mean_', 'scale_', 'n_features_in_']
        self._sklearn_outputs = {}
        self._sklearn_validation_parameters = {}
        
        # Stocker les paramètres
        self.with_mean = kwargs.get('with_mean', True)
        self.with_std = kwargs.get('with_std', True)
        self.copy = kwargs.get('copy', True)
    
    def __sklearn_is_fitted__(self):
        """Vérifie si le scaler a été ajusté.
        
        Returns:
            bool: True si le scaler est ajusté, False sinon.
        """
        # Vérification des attributs essentiels
        has_required_attrs = (hasattr(self, 'mean_') and self.mean_ is not None and 
                             hasattr(self, 'scale_') and self.scale_ is not None and
                             hasattr(self, 'n_features_in_') and self.n_features_in_ is not None)
        
        # Si les attributs requis sont présents, on considère que le scaler est ajusté
        if has_required_attrs:
            self._sklearn_is_fitted = True
            return True
            
        # Pour le débogage
        print("⚠️ Attributs manquants dans __sklearn_is_fitted__:")
        print(f"  - mean_ present: {hasattr(self, 'mean_')}, is None: {not hasattr(self, 'mean_') or getattr(self, 'mean_') is None}")
        print(f"  - scale_ present: {hasattr(self, 'scale_')}, is None: {not hasattr(self, 'scale_') or getattr(self, 'scale_') is None}")
        print(f"  - n_features_in_ present: {hasattr(self, 'n_features_in_')}, is None: {not hasattr(self, 'n_features_in_') or getattr(self, 'n_features_in_') is None}")
        
        return False
    
    def __getstate__(self):
        """Sérialisation personnalisée.
        
        Retourne un dictionnaire contenant l'état du scaler.
        """
        state = {}
        # Sauvegarder les paramètres
        state['with_mean'] = self.with_mean
        state['with_std'] = self.with_std
        state['copy'] = self.copy
        
        # Sauvegarder les attributs ajustés s'ils existent
        if hasattr(self, 'mean_'):
            state['mean_'] = self.mean_
        if hasattr(self, 'scale_'):
            state['scale_'] = self.scale_
        if hasattr(self, 'var_'):
            state['var_'] = self.var_
        if hasattr(self, 'n_features_in_'):
            state['n_features_in_'] = self.n_features_in_
        if hasattr(self, 'n_samples_seen_'):
            state['n_samples_seen_'] = self.n_samples_seen_
            
        # Ajouter les attributs internes de scikit-learn
        if hasattr(self, '_sklearn_is_fitted'):
            state['_sklearn_is_fitted'] = self._sklearn_is_fitted
        if hasattr(self, '_sklearn_validation_attributes'):
            state['_sklearn_validation_attributes'] = self._sklearn_validation_attributes
        if hasattr(self, '_sklearn_outputs'):
            state['_sklearn_outputs'] = self._sklearn_outputs
        if hasattr(self, '_sklearn_validation_parameters'):
            state['_sklearn_validation_parameters'] = self._sklearn_validation_parameters
            
        return state
    
    def __setstate__(self, state):
        """Désérialisation personnalisée.
        
        Args:
            state (dict): État du scaler à restaurer.
        """
        # Initialiser les paramètres de base
        self.with_mean = state.get('with_mean', True)
        self.with_std = state.get('with_std', True)
        self.copy = state.get('copy', True)
        
        # Restaurer les attributs ajustés
        self.mean_ = state.get('mean_')
        self.scale_ = state.get('scale_')
        self.var_ = state.get('var_')
        self.n_features_in_ = state.get('n_features_in_')
        self.n_samples_seen_ = state.get('n_samples_seen_', 0)
        
        # Restaurer les attributs internes de scikit-learn
        self._sklearn_is_fitted = state.get('_sklearn_is_fitted', False)
        self._sklearn_validation_attributes = state.get(
            '_sklearn_validation_attributes', 
            ['mean_', 'scale_', 'n_features_in_']
        )
        self._sklearn_outputs = state.get('_sklearn_outputs', {})
        self._sklearn_validation_parameters = state.get('_sklearn_validation_parameters', {})
        
        # Si les attributs essentiels sont présents, forcer l'état fitted
        if (self.mean_ is not None and self.scale_ is not None and 
            self.n_features_in_ is not None):
            self._sklearn_is_fitted = True
            
        # S'assurer que les attributs essentiels sont initialisés
        if not hasattr(self, 'mean_'):
            self.mean_ = None
        if not hasattr(self, 'scale_'):
            self.scale_ = None
        if not hasattr(self, 'var_'):
            self.var_ = None
        if not hasattr(self, 'n_features_in_'):
            self.n_features_in_ = None
        if not hasattr(self, 'n_samples_seen_'):
            self.n_samples_seen_ = 0
            
        # Initialiser les attributs manquants de scikit-learn
        if not hasattr(self, '_sklearn_is_fitted'):
            self._sklearn_is_fitted = False
        if not hasattr(self, '_sklearn_validation_attributes'):
            self._sklearn_validation_attributes = ['mean_', 'scale_', 'n_features_in_']
        if not hasattr(self, '_sklearn_outputs'):
            self._sklearn_outputs = {}
        if not hasattr(self, '_sklearn_validation_parameters'):
            self._sklearn_validation_parameters = {}
            
    def fit(self, X, y=None):
        """Ajuste le StandardScaler sur les données X.
        
        Args:
            X: Données d'entraînement de forme (n_samples, n_features)
            y: Ignoré, présent pour la compatibilité
            
        Returns:
            self: L'instance du scaler ajusté
        """
        # Appeler la méthode fit du parent
        super().fit(X, y)
        
        # S'assurer que les attributs essentiels sont définis
        if not hasattr(self, 'mean_') or self.mean_ is None:
            self.mean_ = np.mean(X, axis=0) if self.with_mean else 0.0
        if not hasattr(self, 'scale_') or self.scale_ is None:
            self.scale_ = np.std(X, axis=0, ddof=0) if self.with_std else 1.0
        if not hasattr(self, 'var_'):
            self.var_ = np.var(X, axis=0, ddof=0) if self.with_std else 1.0
        if not hasattr(self, 'n_features_in_'):
            self.n_features_in_ = X.shape[1] if len(X.shape) > 1 else 1
        if not hasattr(self, 'n_samples_seen_'):
            self.n_samples_seen_ = X.shape[0] if hasattr(X, 'shape') else len(X)
            
        # Définir les attributs internes de scikit-learn
        self._sklearn_is_fitted = True
        self._sklearn_outputs = {}
        self._sklearn_validation_parameters = {}
        self._sklearn_validation_attributes = ['mean_', 'scale_', 'n_features_in_']
            
        return self
        
    def transform(self, X):
        """Transforme les données en utilisant le scaler ajusté.
        
        Args:
            X: Données à transformer de forme (n_samples, n_features)
            
        Returns:
            Données transformées
            
        Raises:
            NotFittedError: Si le scaler n'a pas été ajusté
        """
        # Vérifier si le scaler est ajusté
        if not self._sklearn_is_fitted or self.mean_ is None or self.scale_ is None:
            # Vérifier si les attributs essentiels sont présents malgré _sklearn_is_fitted
            if hasattr(self, 'mean_') and hasattr(self, 'scale_'):
                print("⚠️ Avertissement: scaler marqué comme non ajusté mais les attributs sont présents. Forçage de la transformation...")
                self._sklearn_is_fitted = True
            else:
                raise NotFittedError(
                    "This %(name)s instance is not fitted yet. Call 'fit' with appropriate arguments before using this estimator." % {"name": type(self).__name__}
                )
        
        # Effectuer la transformation manuellement pour éviter les problèmes de validation
        X = np.asarray(X, dtype=np.float64)  # S'assurer que X est un tableau numpy de type float
        
        # Vérifier les dimensions
        if len(X.shape) == 1:
            X = X.reshape(-1, 1)
            
        # Vérifier la cohérence des dimensions
        if hasattr(self, 'n_features_in_') and X.shape[1] != self.n_features_in_:
            raise ValueError(f"X a {X.shape[1]} caractéristiques, mais ce {type(self).__name__} attend {self.n_features_in_} caractéristiques.")
            
        # Effectuer la transformation
        if self.with_mean:
            X = X - self.mean_
        if self.with_std:
            # Éviter la division par zéro
            scale = self.scale_.copy()
            scale[scale == 0.0] = 1.0
            X = X / scale
            
        return X
            
        return X

class BaseModel(ABC):
    """Classe de base abstraite pour tous les modèles de trading.
    
    Cette classe définit l'interface commune que doivent implémenter tous les modèles
    de trading. Elle gère également le prétraitement des données via un scaler
    et fournit des méthodes pour la sérialisation/désérialisation.
    
    Attributes:
        name (str): Nom du modèle.
        model: Instance du modèle sous-jacent (dépend de l'implémentation).
        scaler (PersistentStandardScaler): Scaler pour le prétraitement des données.
        last_trained (datetime, optional): Date et heure du dernier entraînement.
        model_params (dict): Paramètres de configuration du modèle.
    """

    def __init__(self, name: str, model_params: Optional[Dict] = None):
        """Initialise le modèle de base.
        
        Args:
            name: Nom du modèle
            model_params: Paramètres du modèle. Defaults to None.
        """
        self.name = name
        self.model = None
        self.scaler = PersistentStandardScaler()
        self.last_trained = None
        self.model_params = model_params or {}
        self._initialize_model()

    @abstractmethod
    def _initialize_model(self) -> None:
        """Initialise le modèle spécifique.
        Doit être implémenté par les classes filles.
        """
        pass

    def train(self, X: np.ndarray, y: np.ndarray, **kwargs) -> Dict[str, float]:
        """Entraîne le modèle sur les données fournies."""
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y, **kwargs)
        self.last_trained = datetime.now()
        return {"status": "success", "last_trained": str(self.last_trained)}

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Fait des prédictions sur de nouvelles données.

        Args:
            X: Données d'entrée à prédire

        Returns:
            Prédictions du modèle
        """
        if not hasattr(self, 'model') or self.model is None:
            raise NotFittedError("Le modèle n'est pas initialisé. Veuillez d'abord entraîner ou charger un modèle.")
            
        if not hasattr(self, 'scaler') or self.scaler is None:
            raise NotFittedError("Le scaler n'est pas initialisé. Veuillez d'abord entraîner ou charger un modèle.")
            
        try:
            # Utiliser la méthode transform du scaler qui gère automatiquement l'état fitted
            X_scaled = self.scaler.transform(X)
            
            # Prédiction
            return self.model.predict(X_scaled)
            
        except Exception as e:
            print(f"Erreur lors de la prédiction: {str(e)}")
            print(f"Shape de X: {X.shape}")
            if hasattr(self.scaler, 'mean_') and hasattr(self.scaler, 'scale_'):
                print(f"Shape de mean_: {getattr(self.scaler, 'mean_', 'N/A')}")
                print(f"Shape de scale_: {getattr(self.scaler, 'scale_', 'N/A')}")
            raise
            
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Calcule les probabilités des classes pour les échantillons fournis.
        
        Args:
            X: Données d'entrée à prédire
            
        Returns:
            Probabilités des classes pour chaque échantillon
            
        Raises:
            NotFittedError: Si le modèle ou le scaler n'est pas initialisé
        """
        if not hasattr(self, 'model') or self.model is None:
            raise NotFittedError("Le modèle n'est pas initialisé. Veuillez d'abord entraîner ou charger un modèle.")
            
        if not hasattr(self, 'scaler') or self.scaler is None:
            raise NotFittedError("Le scaler n'est pas initialisé. Veuillez d'abord entraîner ou charger un modèle.")
            
        try:
            # Utiliser la méthode transform du scaler qui gère automatiquement l'état fitted
            X_scaled = self.scaler.transform(X)
            
            # Vérifier si le modèle sous-jacent supporte predict_proba
            if hasattr(self.model, 'predict_proba') and callable(getattr(self.model, 'predict_proba')):
                return self.model.predict_proba(X_scaled)
            else:
                # Si le modèle ne supporte pas predict_proba, simuler avec des probabilités basées sur la prédiction
                predictions = self.model.predict(X_scaled)
                # Déterminer le nombre de classes (au moins 2 pour binaire, sinon basé sur les prédictions uniques)
                unique_preds = np.unique(predictions)
                n_classes = max(2, len(unique_preds))  # Au moins 2 classes (binaire)
                
                # Créer un tableau de probabilités simulées (1.0 pour la classe prédite, 0.0 pour les autres)
                probas = np.zeros((len(predictions), n_classes))
                
                # S'assurer que les indices de classe sont valides
                for i, pred in enumerate(predictions):
                    # Convertir la prédiction en entier et s'assurer qu'elle est dans la plage valide
                    pred_idx = int(pred) % n_classes
                    probas[i, pred_idx] = 1.0
                    
                return probas
                
        except Exception as e:
            print(f"Erreur lors du calcul des probabilités: {str(e)}")
            print(f"Shape de X: {X.shape}")
            if hasattr(self.scaler, 'mean_') and hasattr(self.scaler, 'scale_'):
                print(f"Shape de mean_: {getattr(self.scaler, 'mean_', 'N/A')}")
                print(f"Shape de scale_: {getattr(self.scaler, 'scale_', 'N/A')}")
            raise
            
    def save(self, path: str) -> None:
        """Sauvegarde le modèle et le scaler dans le dossier spécifié.
        
        Args:
            path: Chemin du dossier où sauvegarder le modèle
        """
        # Créer le dossier s'il n'existe pas
        os.makedirs(path, exist_ok=True)
        
        # Sauvegarder le modèle
        model_path = os.path.join(path, 'model.joblib')
        joblib.dump(self.model, model_path)
        
        # Sauvegarder le scaler
        scaler_path = os.path.join(path, 'scaler.joblib')
        joblib.dump(self.scaler, scaler_path)
        
        # Sauvegarder les métadonnées
        metadata = {
            'name': self.name,
            'last_trained': self.last_trained.isoformat() if self.last_trained else None,
            'model_params': self.model_params,
            'feature_names': getattr(self, 'feature_names', None)
        }
        metadata_path = os.path.join(path, 'metadata.joblib')
        joblib.dump(metadata, metadata_path)
    
    def load(self, path: str) -> None:
        """Charge le modèle et le scaler depuis le dossier spécifié.
        
        Args:
            path: Chemin du dossier contenant les fichiers du modèle
        """
        # Vérifier que le dossier existe
        if not os.path.exists(path):
            raise FileNotFoundError(f"Le dossier {path} n'existe pas.")
            
        # Charger le modèle
        model_path = os.path.join(path, 'model.joblib')
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Fichier de modèle introuvable: {model_path}")
        self.model = joblib.load(model_path)
        
        # Charger le scaler
        scaler_path = os.path.join(path, 'scaler.joblib')
        if not os.path.exists(scaler_path):
            raise FileNotFoundError(f"Fichier de scaler introuvable: {scaler_path}")
        self.scaler = joblib.load(scaler_path)
        
        # Charger les métadonnées
        metadata_path = os.path.join(path, 'metadata.joblib')
        if os.path.exists(metadata_path):
            metadata = joblib.load(metadata_path)
            self.name = metadata.get('name', self.name)
            last_trained = metadata.get('last_trained')
            if last_trained:
                self.last_trained = datetime.fromisoformat(last_trained)
            self.model_params = metadata.get('model_params', {})
            if 'feature_names' in metadata and metadata['feature_names'] is not None:
                self.feature_names = metadata['feature_names']
