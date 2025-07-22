"""
Mock pour le module predictor pour les tests unitaires.
Permet d'éviter l'import de TensorFlow pendant les tests.
"""
import sys
from unittest.mock import MagicMock

# Créer un mock pour MLPredictor
class MockMLPredictor:
    def __init__(self, config=None):
        self.config = config or {}
        self.is_trained = True
        
    def predict(self, features):
        return [0.7] * len(features) if features else []
        
    def train(self, X, y):
        return {'train_loss': 0.1, 'val_loss': 0.15, 'accuracy': 0.85}
    
    def save_model(self, filepath):
        pass
        
    def load_model(self, filepath):
        self.is_trained = True
        
    def evaluate(self, X, y):
        return {'test_loss': 0.12, 'test_accuracy': 0.83}

# Créer un mock pour le module
sys.modules['src.core.analysis.predictor'] = MagicMock()
sys.modules['src.core.analysis.predictor'].MLPredictor = MockMLPredictor
