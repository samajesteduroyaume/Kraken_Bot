"""
Module de prétraitement des données pour l'apprentissage automatique.
"""
import numpy as np
from sklearn.preprocessing import StandardScaler
from typing import Tuple, Union, Optional

class DataPreprocessor:
    """
    Classe pour le prétraitement des données avant entraînement ou inférence.
    """
    
    def __init__(self):
        """Initialise le prétraitement avec un scaler par défaut."""
        self.scaler = StandardScaler()
        self.is_fitted = False
    
    def fit(self, X: np.ndarray) -> 'DataPreprocessor':
        """
        Ajuste le scaler aux données d'entraînement.
        
        Args:
            X: Données d'entraînement (n_samples, n_features)
            
        Returns:
            self: Instance de DataPreprocessor
        """
        self.scaler.fit(X)
        self.is_fitted = True
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Applique la transformation aux données.
        
        Args:
            X: Données à transformer (n_samples, n_features)
            
        Returns:
            np.ndarray: Données transformées
            
        Raises:
            RuntimeError: Si le prétraitement n'a pas été ajusté
        """
        if not self.is_fitted:
            raise RuntimeError("Le prétraitement doit être ajusté avant la transformation")
        return self.scaler.transform(X)
    
    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """
        Ajuste et applique la transformation aux données.
        
        Args:
            X: Données à transformer (n_samples, n_features)
            
        Returns:
            np.ndarray: Données transformées
        """
        return self.fit(X).transform(X)
    
    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        """
        Applique la transformation inverse aux données.
        
        Args:
            X: Données à inverser (n_samples, n_features)
            
        Returns:
            np.ndarray: Données transformées inversement
            
        Raises:
            RuntimeError: Si le prétraitement n'a pas été ajusté
        """
        if not self.is_fitted:
            raise RuntimeError("Le prétraitement doit être ajusté avant l'inversion")
        return self.scaler.inverse_transform(X)
    
    def get_params(self) -> dict:
        """
        Retourne les paramètres du prétraitement.
        
        Returns:
            dict: Paramètres du prétraitement
        """
        return {
            'scaler_params': {
                'mean_': self.scaler.mean_.tolist() if hasattr(self.scaler, 'mean_') else None,
                'scale_': self.scaler.scale_.tolist() if hasattr(self.scaler, 'scale_') else None,
                'n_features_in_': getattr(self.scaler, 'n_features_in_', None),
                'feature_names_in_': getattr(self.scaler, 'feature_names_in_', None)
            },
            'is_fitted': self.is_fitted
        }
