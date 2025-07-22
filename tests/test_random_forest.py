"""
Tests pour le modèle RandomForest.
"""
import os
import pytest
import numpy as np
import pandas as pd
from pathlib import Path
from unittest.mock import MagicMock, patch

# Ajout du répertoire racine au PYTHONPATH pour les imports
import sys
sys.path.append(str(Path(__file__).parent.parent))

from src.ml.models.random_forest import RandomForestModel

# Données de test
def create_test_data(n_samples=100, n_features=10):
    """Crée des données de test pour les modèles."""
    np.random.seed(42)
    X = np.random.randn(n_samples, n_features)
    y = np.random.randint(0, 2, n_samples)
    return X, y

class TestRandomForestModel:
    """Tests pour la classe RandomForestModel."""
    
    def setup_method(self):
        """Initialisation avant chaque test."""
        self.model = RandomForestModel({
            'n_estimators': 10,  # Moins d'estimateurs pour accélérer les tests
            'max_depth': 3,
            'random_state': 42
        })
    
    def test_initialization(self):
        """Teste l'initialisation du modèle."""
        assert self.model.name == 'random_forest'
        assert self.model.model is not None
        assert self.model.model.n_estimators == 10
        assert self.model.model.max_depth == 3
    
    def test_train_predict(self):
        """Teste l'entraînement et la prédiction."""
        X, y = create_test_data()
        
        # Tester l'entraînement
        result = self.model.train(X, y)
        
        # Vérifier que l'entraînement a réussi
        assert 'status' in result
        assert result['status'] == 'success'
        assert self.model.last_trained is not None
        
        # Tester la prédiction
        X_test, _ = create_test_data(10)
        predictions = self.model.predict(X_test)
        
        # Vérifier que les prédictions ont la bonne forme
        assert len(predictions) == 10
        assert set(predictions).issubset({0, 1})
    
    def test_feature_importances(self):
        """Teste le calcul de l'importance des caractéristiques."""
        X, y = create_test_data()
        self.model.train(X, y)
        
        # Ajouter des noms de caractéristiques
        self.model.feature_names = [f'feature_{i}' for i in range(X.shape[1])]
        
        # Obtenir les importances
        importances = self.model.get_feature_importances()
        
        # Vérifier que les importances sont présentes
        assert len(importances) == X.shape[1]
        assert all(isinstance(v, float) for v in importances.values())
        assert all(k.startswith('feature_') for k in importances.keys())
    
    def test_save_load(self, tmp_path):
        """Teste la sauvegarde et le chargement du modèle."""
        X, y = create_test_data()
        self.model.train(X, y)
        
        # Sauvegarder le modèle
        save_path = tmp_path / "test_model"
        self.model.save(str(save_path))
        
        # Vérifier que les fichiers ont été créés
        assert (save_path / "model.joblib").exists()
        assert (save_path / "scaler.joblib").exists()
        
        # Créer un nouveau modèle et charger l'état sauvegardé
        new_model = RandomForestModel()
        new_model.load(str(save_path))
        
        # Vérifier que le modèle chargé fait les mêmes prédictions
        X_test, _ = create_test_data(5)
        assert np.array_equal(
            self.model.predict(X_test),
            new_model.predict(X_test)
        )
    
    def test_get_params(self):
        """Teste la récupération des paramètres du modèle."""
        params = self.model.get_params()
        
        # Vérifier que les paramètres sont présents
        assert 'n_estimators' in params
        assert 'max_depth' in params
        assert params['n_estimators'] == 10
        assert params['max_depth'] == 3

if __name__ == "__main__":
    pytest.main(["-v", "tests/test_random_forest.py"])
