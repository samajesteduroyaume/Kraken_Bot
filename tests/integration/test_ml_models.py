"""
Tests d'intégration pour les modèles ML du bot de trading.
"""
import os
import sys
import numpy as np
import pandas as pd
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock

# Ajouter le répertoire racine au PYTHONPATH
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.ml.core.model_factory import ModelFactory
from src.ml.utils.data_loader import DataLoader

# Configuration des tests
TEST_DATA_DIR = Path(__file__).parent.parent / "test_data"
TEST_MODEL_DIR = Path(__file__).parent.parent / "test_models"
TEST_MODEL_DIR.mkdir(exist_ok=True)

# Données de test simulées
def generate_test_data(n_samples=1000, n_features=10, n_classes=3):
    """Génère des données de test pour les modèles."""
    np.random.seed(42)
    X = np.random.randn(n_samples, n_features)
    y = np.random.randint(0, n_classes, n_samples)
    return X, y

class TestMLModels(unittest.TestCase):
    """Classe de test pour les modèles ML."""
    
    @classmethod
    def setUpClass(cls):
        """Préparation des données de test."""
        cls.X, cls.y = generate_test_data()
        cls.model_types = ['random_forest', 'xgboost', 'neural_network', 'lstm']
        
        # Configuration spécifique pour LSTM
        cls.seq_len = 5  # Nombre de pas de temps
        cls.n_features = 2  # Nombre de caractéristiques par pas de temps
        cls.lstm_config = {
            'input_dim': cls.n_features,  # Doit correspondre au nombre de caractéristiques par pas de temps
            'hidden_dim': 32,
            'num_layers': 2,
            'num_classes': 3,
            'learning_rate': 0.001,
            'epochs': 2,  # Réduit pour les tests
            'batch_size': 32,
            'device': 'cpu'  # Force CPU pour les tests
        }
    
    def test_model_creation(self):
        """Teste la création de chaque type de modèle."""
        for model_type in self.model_types:
            with self.subTest(model_type=model_type):
                model_params = {}
                if model_type == 'lstm':
                    model_params = self.lstm_config
                
                model = ModelFactory.create_model(model_type, model_params)
                self.assertIsNotNone(model)
                self.assertTrue(hasattr(model, 'train'))
                self.assertTrue(hasattr(model, 'predict'))
    
    def test_model_training(self):
        """Teste l'entraînement de chaque type de modèle."""
        for model_type in self.model_types:
            with self.subTest(model_type=model_type):
                model_params = {}
                if model_type == 'lstm':
                    model_params = self.lstm_config
                
                model = ModelFactory.create_model(model_type, model_params)
                
                # Pour LSTM, on a besoin de données séquentielles
                if model_type == 'lstm':
                    # Reshape pour LSTM: (samples, time_steps, features)
                    n_samples = len(self.X) // (self.seq_len * self.n_features)
                    X_reshaped = self.X[:n_samples * self.seq_len * self.n_features].reshape(
                        -1, self.seq_len, self.n_features
                    )
                    # On s'assure d'avoir le même nombre d'échantillons que de labels
                    y_reshaped = self.y[:len(X_reshaped)]
                    model.train(X_reshaped, y_reshaped)
                else:
                    model.train(self.X, self.y)
                
                # Vérifie que le modèle a bien été entraîné
                self.assertTrue(hasattr(model, 'model'))
    
    def test_model_prediction(self):
        """Teste la prédiction de chaque type de modèle."""
        for model_type in self.model_types:
            with self.subTest(model_type=model_type):
                model_params = {}
                if model_type == 'lstm':
                    model_params = self.lstm_config
                
                model = ModelFactory.create_model(model_type, model_params)
                
                # Entraînement
                if model_type == 'lstm':
                    # Préparation des données d'entraînement
                    n_samples = len(self.X) // (self.seq_len * self.n_features)
                    X_reshaped = self.X[:n_samples * self.seq_len * self.n_features].reshape(
                        -1, self.seq_len, self.n_features
                    )
                    y_reshaped = self.y[:len(X_reshaped)]
                    model.train(X_reshaped, y_reshaped)
                    
                    # Prédiction sur les mêmes données (juste pour le test)
                    predictions = model.predict(X_reshaped[:10])  # Seulement 10 échantillons
                else:
                    model.train(self.X, self.y)
                    predictions = model.predict(self.X[:10])
                
                # Vérifie que les prédictions ont la bonne forme
                self.assertEqual(len(predictions), 10)
    
    def test_model_serialization(self):
        """Teste la sérialisation et désérialisation des modèles."""
        for model_type in self.model_types:
            with self.subTest(model_type=model_type):
                model_params = {}
                if model_type == 'lstm':
                    model_params = self.lstm_config
                
                # Création et entraînement
                model = ModelFactory.create_model(model_type, model_params)
                
                if model_type == 'lstm':
                    # Préparation des données d'entraînement
                    n_samples = len(self.X) // (self.seq_len * self.n_features)
                    X_reshaped = self.X[:n_samples * self.seq_len * self.n_features].reshape(
                        -1, self.seq_len, self.n_features
                    )
                    y_reshaped = self.y[:len(X_reshaped)]
                    model.train(X_reshaped, y_reshaped)
                else:
                    model.train(self.X, self.y)
                
                # Créer un répertoire temporaire unique pour ce test
                import tempfile
                temp_dir = Path(tempfile.mkdtemp(prefix=f"test_{model_type}_", dir=str(TEST_MODEL_DIR)))
                print(f"\n[DEBUG] Dossier temporaire créé: {temp_dir}")
                
                try:
                    # Sauvegarder le modèle dans le dossier temporaire
                    model.save(str(temp_dir))
                    print(f"[DEBUG] Modèle sauvegardé dans: {temp_dir}")
                    
                    # Vérifier que le dossier existe
                    self.assertTrue(temp_dir.exists(), f"Le dossier temporaire {temp_dir} n'a pas été créé")
                except Exception as e:
                    print(f"[ERREUR] Erreur lors de la sauvegarde du modèle: {e}")
                    # Nettoyer le dossier temporaire en cas d'erreur
                    import shutil
                    if temp_dir.exists():
                        shutil.rmtree(temp_dir, ignore_errors=True)
                    raise
                
                # Vérifie que le dossier a été créé
                print(f"[DEBUG] Dossier temporaire créé: {temp_dir.exists()}")
                self.assertTrue(temp_dir.exists(), f"Le dossier temporaire {temp_dir} n'a pas été créé")
                
                # Vérifie que les fichiers ont été créés
                if model_type == 'lstm':
                    model_file = temp_dir / 'model.pt'
                else:
                    model_file = temp_dir / 'model.joblib'
                    
                scaler_file = temp_dir / 'scaler.joblib'
                metadata_file = temp_dir / 'metadata.joblib'
                
                # Vérifier l'existence des fichiers
                self.assertTrue(model_file.exists(), f"Fichier modèle manquant: {model_file}")
                self.assertTrue(scaler_file.exists(), f"Fichier scaler manquant: {scaler_file}")
                self.assertTrue(metadata_file.exists(), f"Fichier metadata manquant: {metadata_file}")
                
                # Vérifier que les fichiers ne sont pas vides
                self.assertGreater(os.path.getsize(model_file), 0, f"Le fichier modèle est vide: {model_file}")
                self.assertGreater(os.path.getsize(scaler_file), 0, f"Le fichier scaler est vide: {scaler_file}")
                self.assertGreater(os.path.getsize(metadata_file), 0, f"Le fichier metadata est vide: {metadata_file}")
                
                # Nettoyage en forçant la fermeture des descripteurs de fichiers
                import gc
                import shutil
                
                # Libérer les références au modèle et forcer le garbage collection
                del model
                gc.collect()
                
                # Supprimer le dossier temporaire
                shutil.rmtree(temp_dir, ignore_errors=True)

if __name__ == "__main__":
    unittest.main()
