"""
Tests pour le module predictor.py
"""
import os
import pytest
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

# Ajout du répertoire racine au PYTHONPATH pour les imports
import sys
sys.path.append(str(Path(__file__).parent.parent))

from src.ml.predictor import MLPredictor

# Configuration pour les tests
TEST_MODEL_DIR = "tests/test_models"

# Données de test
def create_test_data(n_samples=100, n_features=10):
    """Crée des données de test pour les modèles."""
    np.random.seed(42)
    X = np.random.randn(n_samples, n_features)
    y = np.random.randint(0, 2, n_samples)
    return X, y

class TestMLPredictor:
    """Tests pour la classe MLPredictor."""
    
    def setup_method(self):
        """Initialisation avant chaque test."""
        # Nettoyer le répertoire de test
        if os.path.exists(TEST_MODEL_DIR):
            for f in Path(TEST_MODEL_DIR).glob("*"):
                if f.is_file():
                    f.unlink()
                else:
                    for sf in f.glob("*"):
                        sf.unlink()
                    f.rmdir()
            Path(TEST_MODEL_DIR).rmdir()
        
        # Créer une instance de MLPredictor pour les tests
        self.predictor = MLPredictor({
            'model_dir': TEST_MODEL_DIR,
            'default_model_type': 'random_forest',
            'test_size': 0.2,
            'model_params': {
                'random_forest': {
                    'n_estimators': 10,  # Utiliser moins d'estimateurs pour les tests
                    'max_depth': 3
                }
            }
        })
    
    def test_initialization(self):
        """Teste l'initialisation du prédicteur."""
        assert isinstance(self.predictor, MLPredictor)
        assert os.path.exists(TEST_MODEL_DIR)
    
    def test_fit_predict(self):
        """Teste l'entraînement et la prédiction d'un modèle."""
        # Créer des données de test
        X, y = create_test_data()
        
        # Tester l'entraînement
        result = self.predictor.fit(X, y, model_name="test_model")
        
        # Vérifier que l'entraînement a réussi
        assert 'metrics' in result
        assert 'accuracy' in result['metrics']
        assert len(self.predictor.models) == 1
        assert 'test_model' in self.predictor.models
        
        # Tester la prédiction
        X_test, _ = create_test_data(10)
        predictions = self.predictor.predict(X_test)
        
        # Vérifier que les prédictions ont la bonne forme
        assert len(predictions) == 10
        assert set(predictions).issubset({0, 1})
    
    def test_evaluate(self):
        """Teste l'évaluation d'un modèle."""
        # Créer des données de test
        X, y = create_test_data()
        
        # Entraîner un modèle
        self.predictor.fit(X, y, model_name="test_model")
        
        # Tester l'évaluation
        X_test, y_test = create_test_data(20)
        metrics = self.predictor.evaluate(X_test, y_test)
        
        # Vérifier que les métriques sont présentes
        assert 'accuracy' in metrics
        assert 0 <= metrics['accuracy'] <= 1
    
    def test_save_load_model(self):
        """Teste la sauvegarde et le chargement d'un modèle."""
        # Créer et entraîner un modèle
        X, y = create_test_data()
        self.predictor.fit(X, y, model_name="test_save_model")
        
        # Vérifier que le modèle a été sauvegardé
        model_dir = Path(TEST_MODEL_DIR) / "test_save_model"
        assert model_dir.exists()
        assert (model_dir / "model_info.json").exists()
        
        # Créer un nouveau prédicteur et charger le modèle
        new_predictor = MLPredictor({'model_dir': TEST_MODEL_DIR})
        
        # Vérifier que le modèle a été chargé
        assert 'test_save_model' in new_predictor.models
    
    def test_feature_importance(self):
        """Teste le calcul de l'importance des caractéristiques."""
        # Créer et entraîner un modèle
        X, y = create_test_data()
        self.predictor.fit(X, y, model_name="test_feature_importance")
        
        # Obtenir l'importance des caractéristiques
        importances = self.predictor.get_feature_importance()
        
        # Vérifier que les importances sont présentes
        assert len(importances) == 10  # 10 caractéristiques dans les données de test
        assert all(isinstance(v, float) for v in importances.values())
    
    def test_model_info(self):
        """Teste la récupération des informations du modèle."""
        # Créer et entraîner un modèle
        X, y = create_test_data()
        self.predictor.fit(X, y, model_name="test_model_info")
        
        # Obtenir les informations du modèle
        model_info = self.predictor.get_model_info()
        
        # Vérifier que les informations sont présentes
        assert 'created_at' in model_info
        assert 'metrics' in model_info
        assert 'feature_importance' in model_info
    
    def test_available_models(self):
        """Teste la récupération de la liste des modèles disponibles."""
        # Créer plusieurs modèles
        X, y = create_test_data()
        for i in range(3):
            self.predictor.fit(X, y, model_name=f"test_model_{i}")
        
        # Obtenir la liste des modèles
        models = self.predictor.get_available_models()
        
        # Vérifier que tous les modèles sont présents
        assert len(models) == 3
        assert all(f"test_model_{i}" in [m['name'] for m in models] for i in range(3))

if __name__ == "__main__":
    pytest.main(["-v", "tests/test_predictor.py"])
