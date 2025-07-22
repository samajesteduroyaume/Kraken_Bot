"""
Module predictor - Implémentation du prédicteur ML principal.
"""
from typing import Dict, Any, Optional, List, Tuple
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from .base_model import BaseModel
from .model_factory import ModelFactory

class MLPredictor:
    """
    Classe principale pour effectuer des prédictions avec des modèles ML.
    Gère le chargement, l'évaluation et l'inférence des modèles.
    """
    
    def __init__(self, model: Optional[BaseModel] = None):
        """
        Initialise le prédicteur avec un modèle optionnel.
        
        Args:
            model: Modèle ML à utiliser pour les prédictions
        """
        self.model = model
        self.scaler = StandardScaler()
        self.is_fitted = False
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Entraîne le modèle sur les données fournies.
        
        Args:
            X: Données d'entraînement
            y: Labels d'entraînement
        """
        if self.model is None:
            raise ValueError("Aucun modèle n'a été chargé")
            
        # Mise à l'échelle des données
        X_scaled = self.scaler.fit_transform(X)
        
        # Entraînement du modèle
        self.model.train(X_scaled, y)
        self.is_fitted = True
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Effectue une prédiction sur les données fournies.
        
        Args:
            X: Données à prédire
            
        Returns:
            Prédictions du modèle
        """
        if not self.is_fitted:
            raise ValueError("Le modèle doit être entraîné avant de faire des prédictions")
            
        # Mise à l'échelle des données
        X_scaled = self.scaler.transform(X)
        
        # Prédiction
        return self.model.predict(X_scaled)
        
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Calcule les probabilités des classes pour les échantillons fournis.
        
        Args:
            X: Données pour lesquelles calculer les probabilités des classes
            
        Returns:
            Probabilités des classes pour chaque échantillon
        """
        if not self.is_fitted:
            raise ValueError("Le modèle doit être entraîné avant de calculer les probabilités")
            
        # Mise à l'échelle des données
        X_scaled = self.scaler.transform(X)
        
        # Calcul des probabilités
        return self.model.predict_proba(X_scaled)
    
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """
        Évalue les performances du modèle sur un ensemble de test.
        
        Args:
            X: Données de test
            y: Vraies étiquettes
            
        Returns:
            Dictionnaire contenant les métriques d'évaluation
        """
        if not self.is_fitted:
            raise ValueError("Le modèle doit être entraîné avant d'être évalué")
            
        # Prédiction
        y_pred = self.predict(X)
        
        # Calcul des métriques
        # (à implémenter selon les besoins spécifiques)
        metrics = {
            'accuracy': np.mean(y_pred == y)
        }
        
        return metrics
    
    def save(self, path: str) -> None:
        """
        Sauvegarde le modèle et le scaler à l'emplacement spécifié.
        
        Args:
            path: Chemin où sauvegarder le modèle
        """
        if self.model is None:
            raise ValueError("Aucun modèle à sauvegarder")
            
        # Créer le répertoire s'il n'existe pas
        import os
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Sauvegarder le modèle et le scaler
        self.model.save(path)
        
        # Sauvegarder le scaler
        import joblib
        scaler_path = os.path.join(os.path.dirname(path), 'scaler.joblib')
        joblib.dump(self.scaler, scaler_path)
    
    @classmethod
    def load(cls, path: str) -> 'MLPredictor':
        """
        Charge un modèle et son scaler depuis l'emplacement spécifié.
        
        Args:
            path: Chemin vers le modèle sauvegardé
            
        Returns:
            Une instance de MLPredictor avec le modèle et le scaler chargés
        """
        # Charger le modèle
        model = ModelFactory.load_model(path)
        
        # Créer une instance de MLPredictor
        predictor = cls(model)
        
        # Charger le scaler
        import joblib
        import os
        scaler_path = os.path.join(os.path.dirname(path), 'scaler.joblib')
        if os.path.exists(scaler_path):
            predictor.scaler = joblib.load(scaler_path)
            predictor.is_fitted = True
        
        return predictor
