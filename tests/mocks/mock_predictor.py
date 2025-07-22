"""
Mock pour le prédicteur ML utilisé dans les tests.
"""
from typing import Dict, Any, List
import numpy as np
from datetime import datetime

class MockMLPredictor:
    """Mock du prédicteur ML pour les tests."""
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialise le mock du prédicteur ML."""
        self.config = config or {}
        self.is_trained = True
        
    def predict(self, features: np.ndarray) -> np.ndarray:
        """Retourne des prédictions aléatoires pour les tests."""
        if features is None or len(features) == 0:
            return np.array([])
            
        # Retourne des prédictions aléatoires entre -1 et 1
        return np.random.uniform(-1, 1, size=len(features))
    
    def train(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """Simule l'entraînement du modèle."""
        return {
            'train_loss': 0.1,
            'val_loss': 0.15,
            'accuracy': 0.85,
            'timestamp': datetime.now().isoformat()
        }
    
    def save_model(self, filepath: str) -> None:
        """Simule la sauvegarde du modèle."""
        pass
    
    def load_model(self, filepath: str) -> None:
        """Simule le chargement du modèle."""
        self.is_trained = True
    
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """Simule l'évaluation du modèle."""
        return {
            'test_loss': 0.12,
            'test_accuracy': 0.83,
            'precision': 0.84,
            'recall': 0.82,
            'f1_score': 0.83
        }


class MockSignalGenerator:
    """Mock du générateur de signaux pour les tests."""
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialise le mock du générateur de signaux."""
        self.config = config or {}
    
    def generate_signals(self, market_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Génère des signaux de test."""
        if not market_data or 'close' not in market_data or len(market_data['close']) == 0:
            return []
            
        # Génère un signal aléatoire
        action = 'buy' if np.random.random() > 0.5 else 'sell'
        return [{
            'symbol': 'BTC/USD',
            'action': action,
            'price': float(market_data['close'][-1]),
            'confidence': np.random.uniform(0.5, 0.95),
            'reason': 'test_signal',
            'timestamp': datetime.now().isoformat(),
            'indicators': {}
        }]
