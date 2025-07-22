"""
Utilitaire pour le chargement et la préparation des données pour l'entraînement des modèles.
"""
import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any, Optional, List
from sklearn.preprocessing import StandardScaler


class DataLoader:
    """Classe utilitaire pour le chargement et la préparation des données."""
    
    def __init__(self, feature_columns: List[str], target_column: str, 
                 sequence_length: int = 30, test_size: float = 0.2):
        """Initialise le chargeur de données.
        
        Args:
            feature_columns: Liste des noms des colonnes de caractéristiques
            target_column: Nom de la colonne cible
            sequence_length: Longueur des séquences pour les modèles séquentiels
            test_size: Proportion des données à utiliser pour le test
        """
        self.feature_columns = feature_columns
        self.target_column = target_column
        self.sequence_length = sequence_length
        self.test_size = test_size
        self.scaler = StandardScaler()
        self.feature_indices = None
        
    def prepare_data(self, df: pd.DataFrame, is_sequential: bool = False) -> Tuple:
        """Prépare les données pour l'entraînement.
        
        Args:
            df: DataFrame contenant les données brutes
            is_sequential: Si True, prépare les données pour un modèle séquentiel
            
        Returns:
            Tuple contenant (X_train, X_test, y_train, y_test) ou séquences pour LSTM
        """
        # Nettoyage des données
        df = df.dropna()
        
        # Séparation des caractéristiques et de la cible
        X = df[self.feature_columns].values
        y = df[self.target_column].values
        
        # Normalisation des caractéristiques
        X = self.scaler.fit_transform(X)
        
        if is_sequential:
            return self._create_sequences(X, y)
        
        # Séparation train/test pour les modèles non séquentiels
        split_idx = int(len(X) * (1 - self.test_size))
        return (
            X[:split_idx], X[split_idx:],
            y[:split_idx], y[split_idx:]
        )
    
    def _create_sequences(self, X: np.ndarray, y: np.ndarray) -> Tuple:
        """Crée des séquences pour les modèles séquentiels comme LSTM."""
        X_seq, y_seq = [], []
        
        for i in range(len(X) - self.sequence_length):
            X_seq.append(X[i:(i + self.sequence_length)])
            y_seq.append(y[i + self.sequence_length - 1])
            
        X_seq = np.array(X_seq)
        y_seq = np.array(y_seq)
        
        # Séparation train/test
        split_idx = int(len(X_seq) * (1 - self.test_size))
        return (
            X_seq[:split_idx], X_seq[split_idx:],
            y_seq[:split_idx], y_seq[split_idx:]
        )
    
    def prepare_inference_data(self, df: pd.DataFrame, is_sequential: bool = False) -> np.ndarray:
        """Prépare les données pour l'inférence.
        
        Args:
            df: DataFrame contenant les données brutes
            is_sequential: Si True, prépare les données pour un modèle séquentiel
            
        Returns:
            Données préparées pour l'inférence
        """
        X = df[self.feature_columns].values
        X = self.scaler.transform(X)
        
        if is_sequential:
            # Pour LSTM, on prend les N dernières séquences
            if len(X) < self.sequence_length:
                # Padding si pas assez de données
                padding = np.zeros((self.sequence_length - len(X), X.shape[1]))
                X = np.vstack([padding, X])
            else:
                X = X[-self.sequence_length:]
            
            return X.reshape(1, self.sequence_length, -1)
        
        return X[-1:]
    
    def get_feature_importance(self, model, feature_names: Optional[List[str]] = None) -> Dict[str, float]:
        """Récupère l'importance des caractéristiques du modèle.
        
        Args:
            model: Modèle entraîné avec une méthode feature_importances_
            feature_names: Noms des caractéristiques. Si None, utilise self.feature_columns
            
        Returns:
            Dictionnaire des caractéristiques et leur importance
        """
        if not hasattr(model, 'feature_importances_'):
            return {}
            
        features = feature_names or self.feature_columns
        return dict(zip(features, model.feature_importances_))
