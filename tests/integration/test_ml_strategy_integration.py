"""
Tests d'intégration entre les modèles ML et les stratégies de trading.
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
from src.ml.predictor import MLPredictor
from src.strategies.base_strategy import SignalStrength
from src.strategies.core.trend_following import TrendFollowingStrategy
from src.utils.config_loader import ConfigLoader

# Configuration des tests
TEST_MODEL_DIR = Path(__file__).parent.parent / "test_models"
TEST_MODEL_DIR.mkdir(exist_ok=True)

# Données de test simulées
def generate_ohlcv_data(n_samples=100):
    """Génère des données OHLCV simulées pour les tests."""
    np.random.seed(42)
    dates = pd.date_range(start='2023-01-01', periods=n_samples, freq='1h')
    close = np.cumprod(1 + np.random.normal(0.001, 0.01, n_samples)) * 100
    open_prices = close * (1 + np.random.normal(0, 0.005, n_samples))
    high = np.maximum(open_prices, close) * (1 + np.abs(np.random.normal(0, 0.005, n_samples)))
    low = np.minimum(open_prices, close) * (1 - np.abs(np.random.normal(0, 0.005, n_samples)))
    volume = np.random.lognormal(5, 1, n_samples)
    
    df = pd.DataFrame({
        'timestamp': dates,
        'open': open_prices,
        'high': high,
        'low': low,
        'close': close,
        'volume': volume
    })
    df.set_index('timestamp', inplace=True)
    return df

class TestMLStrategyIntegration(unittest.TestCase):
    """Classe de test pour l'intégration ML/stratégies."""
    
    @classmethod
    def setUpClass(cls):
        """Préparation des données et modèles de test."""
        # Générer des données de test
        cls.ohlcv_data = generate_ohlcv_data(1000)
        
        # Initialiser le prédicteur ML
        cls.predictor = MLPredictor({
            'model_dir': str(TEST_MODEL_DIR),
            'default_model_type': 'random_forest',
            'test_size': 0.2,
            'model_params': {
                'random_forest': {
                    'n_estimators': 50,  # Réduit pour les tests
                    'max_depth': 5
                }
            }
        })
        
        # Préparer les données d'entraînement
        X = np.random.randn(100, 10)
        y = np.random.randint(0, 3, 100)  # 3 classes: vente, neutre, achat
        
        # Entraîner le modèle
        cls.predictor.fit(X, y, model_name='test_model')
    
    def test_strategy_with_ml_predictions(self):
        """Teste l'intégration des prédictions ML dans une stratégie."""
        # Configuration de la stratégie
        strategy_config = {
            'symbol': 'BTC/EUR',
            'timeframes': ['1h'],
            'rsi_period': 14,
            'rsi_overbought': 70,
            'rsi_oversold': 30,
            'use_ml': True,
            'position_size': 0.1,
            'ml_integration': {
                'enabled': True,
                'model_name': 'test_model',
                'min_confidence': 0.6,
                'weight': 0.7  # Poids des signaux ML dans la décision finale
            },
            'risk_management': {
                'stop_loss': 0.02,
                'take_profit': 0.04,
                'position_size': 0.1
            }
        }
        
        # Créer une instance de stratégie avec un mock pour le prédicteur ML
        with patch('src.ml.predictor.MLPredictor') as mock_predictor:
            # Configurer le mock pour retourner des prédictions de test
            mock_instance = mock_predictor.return_value
            mock_instance.predict.return_value = np.ones(len(self.ohlcv_data))  # Tous les signaux sont d'achat
            mock_instance.predict_proba.return_value = np.ones((len(self.ohlcv_data), 3)) * 0.8  # Haute confiance
            
            strategy = TrendFollowingStrategy(strategy_config)
            
            # Simuler l'analyse des données OHLCV
            for i in range(10, len(self.ohlcv_data)):
                data = self.ohlcv_data.iloc[i-10:i+1]  # Fenêtre glissante de 10 périodes
                signal = strategy.analyze(data)
                
                # Vérifier que le signal contient une prédiction ML
                self.assertIn('ml_prediction', signal.metadata)
                self.assertIn('ml_confidence', signal.metadata)
                
                # Vérifier que le signal final prend en compte la prédiction ML
                if signal.strength.value > SignalStrength.NEUTRAL.value:
                    self.assertGreater(signal.metadata['ml_confidence'], 0.5)
    
    def test_ml_prediction_affects_signal_strength(self):
        """Vérifie que les prédictions ML influencent la force du signal."""
        # Configuration avec intégration ML
        strategy_config = {
            'symbol': 'BTC/EUR',
            'timeframes': ['1h'],
            'ml_integration': {
                'enabled': True,
                'model_name': 'test_model',
                'min_confidence': 0.6,
                'weight': 0.7  # Poids élevé pour les signaux ML
            }
        }
        
        # Créer une instance de stratégie avec un mock pour le prédicteur ML
        with patch('src.ml.predictor.MLPredictor') as mock_predictor:
            # Configurer le mock pour retourner des prédictions de test
            mock_instance = mock_predictor.return_value
            
            # Créer l'instance de la stratégie
            strategy = TrendFollowingStrategy(strategy_config)
            
            # Premier test: prédiction d'achat avec haute confiance
            # Créer une classe MockScaler personnalisée pour une meilleure simulation
            class MockScaler:
                def __init__(self):
                    self.mean_ = np.zeros(7)  # Moyennes nulles
                    self.scale_ = np.ones(7)  # Écarts-types unitaires
                    self.var_ = np.ones(7)  # Variances unitaires
                    self.n_features_in_ = 7  # Nombre de caractéristiques attendues
                    self._sklearn_is_fitted = True  # Marquer comme ajusté
                
                def transform(self, X):
                    # Vérifier que les données d'entrée ont la bonne forme
                    if X.shape[1] != self.n_features_in_:
                        raise ValueError(f"X a {X.shape[1]} caractéristiques, mais ce PersistentStandardScaler attend {self.n_features_in_} caractéristiques.")
                    # Simuler une transformation (ici, on retourne simplement les données d'entrée)
                    return X
                
                def fit(self, X, y=None):
                    return self
            
            # Configurer le mock pour retourner une prédiction d'achat (1) avec haute confiance
            mock_instance.predict.return_value = np.array([1])  # Signal d'achat (classe 1)
            
            # predict_proba doit retourner une matrice (n_samples, n_classes)
            # Format: [prob_vendre, prob_neutre, prob_acheter]
            mock_instance.predict_proba.return_value = np.array([[0.1, 0.1, 0.8]])  # 80% de confiance pour l'achat
            
            # Créer un mock plus robuste pour le scaler
            class MockScaler:
                def __init__(self):
                    self.mean_ = np.zeros(6)  # Moyennes nulles pour 6 caractéristiques
                    self.scale_ = np.ones(6)   # Écarts-types unitaires
                    self.var_ = np.ones(6)     # Variances unitaires
                    self.n_features_in_ = 6     # Nombre de caractéristiques attendues
                    self.feature_names_in_ = [
                        'returns', 'volatility', 'rsi', 'macd', 
                        'macd_signal', 'volume_ratio'
                    ]
                    self._sklearn_is_fitted = True  # Marquer comme ajusté
                    
                def transform(self, X):
                    # Vérifier que les données d'entrée ont la bonne forme
                    if X.shape[1] != self.n_features_in_:
                        raise ValueError(f"X a {X.shape[1]} caractéristiques, mais ce PersistentStandardScaler attend {self.n_features_in_} caractéristiques.")
                    # Simuler une transformation (ici, on retourne simplement les données d'entrée normalisées)
                    return (X - self.mean_) / self.scale_
                    
                def fit(self, X, y=None):
                    # Simuler l'ajustement avec les données
                    self.mean_ = np.mean(X, axis=0)
                    self.scale_ = np.std(X, axis=0, ddof=0)
                    self.var_ = np.var(X, axis=0, ddof=0)
                    self.n_features_in_ = X.shape[1]
                    self._sklearn_is_fitted = True
                    return self
                    
                def get_params(self, deep=True):
                    # Méthode nécessaire pour la compatibilité avec scikit-learn
                    return {}
            
            # Configurer le mock pour utiliser notre MockScaler personnalisé
            mock_instance.scaler = MockScaler()
            
            # S'assurer que le modèle est défini et a une méthode predict_proba
            if not hasattr(mock_instance, 'predict_proba'):
                # Si le mock n'a pas de méthode predict_proba, en ajouter une
                mock_instance.predict_proba = lambda x: np.array([[0.1, 0.1, 0.8]])
            
            # Créer des données de test avec suffisamment de barres pour calculer tous les indicateurs
            # Utiliser 200 barres pour s'assurer que tous les indicateurs peuvent être calculés
            test_data = self.ohlcv_data.iloc[-200:].copy()
            
            # Calculer les indicateurs techniques nécessaires
            # RSI
            delta = test_data['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14, min_periods=1).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14, min_periods=1).mean()
            rs = gain / loss
            test_data['rsi'] = 100 - (100 / (1 + rs))
            
            # MACD
            exp1 = test_data['close'].ewm(span=12, adjust=False).mean()
            exp2 = test_data['close'].ewm(span=26, adjust=False).mean()
            test_data['macd'] = exp1 - exp2
            test_data['macd_signal'] = test_data['macd'].ewm(span=9, adjust=False).mean()
            
            # ATR
            high_low = test_data['high'] - test_data['low']
            high_close = np.abs(test_data['high'] - test_data['close'].shift())
            low_close = np.abs(test_data['low'] - test_data['close'].shift())
            ranges = pd.concat([high_low, high_close, low_close], axis=1)
            true_range = np.max(ranges, axis=1)
            test_data['atr'] = true_range.rolling(14, min_periods=1).mean()
            
            # Bandes de Bollinger
            test_data['bb_middle'] = test_data['close'].rolling(20, min_periods=1).mean()
            rolling_std = test_data['close'].rolling(20, min_periods=1).std()
            test_data['bb_upper'] = test_data['bb_middle'] + (rolling_std * 2)
            test_data['bb_lower'] = test_data['bb_middle'] - (rolling_std * 2)
            
            # Support et résistance (simplifiés pour le test)
            test_data['support'] = test_data['low'].rolling(10, min_periods=1).min()
            test_data['resistance'] = test_data['high'].rolling(10, min_periods=1).max()
            
            # Remplir les valeurs NaN avec la méthode ffill puis bfill
            test_data = test_data.ffill().bfill()
            
            # S'assurer qu'il n'y a pas de valeurs infinies ou NaN
            test_data = test_data.replace([np.inf, -np.inf], np.nan).dropna()
            
            # Calculer les indicateurs techniques
            test_data['rsi'] = 60  # RSI en territoire haussier (>50)
            
            # MACD croisé à la hausse (MACD > signal)
            test_data['macd'] = np.linspace(8, 12, len(test_data))  # MACD en hausse
            test_data['macd_signal'] = np.linspace(5, 10, len(test_data))  # Signal en dessous de MACD
            
            # Bandes de Bollinger
            test_data['bb_middle'] = test_data['close'].rolling(20).mean()
            test_data['bb_upper'] = test_data['bb_middle'] + 2 * test_data['close'].rolling(20).std()
            test_data['bb_lower'] = test_data['bb_middle'] - 2 * test_data['close'].rolling(20).std()
            
            # ATR pour la volatilité
            test_data['atr'] = test_data['high'] - test_data['low']
            
            # Support et résistance
            test_data['support'] = test_data['low'].rolling(10).min()
            test_data['resistance'] = test_data['high'].rolling(10).max()
            
            # Ajuster le prix de clôture pour qu'il soit au-dessus de la moyenne mobile et de la résistance
            test_data['close'] = test_data[['bb_middle', 'resistance']].max(axis=1) * 1.01
            
            # Volume
            test_data['volume'] = test_data['volume'].rolling(20).mean() * 1.5  # Volume supérieur à la moyenne
            
            # Remplir les valeurs manquantes
            test_data = test_data.fillna(method='bfill').fillna(method='ffill')
            
            # Afficher les données de test pour le débogage
            print("\n=== Données de test ===")
            print(f"Taille des données: {len(test_data)} barres")
            print(f"Colonnes disponibles: {test_data.columns.tolist()}")
            print(f"Close: {test_data['close'].iloc[-1]}")
            if 'bb_middle' in test_data.columns:
                print(f"BB Middle: {test_data['bb_middle'].iloc[-1]}")
            if 'resistance' in test_data.columns:
                print(f"Resistance: {test_data['resistance'].iloc[-1]}")
            if 'rsi' in test_data.columns:
                print(f"RSI: {test_data['rsi'].iloc[-1]}")
            if 'macd' in test_data.columns and 'macd_signal' in test_data.columns:
                print(f"MACD: {test_data['macd'].iloc[-1]}, Signal: {test_data['macd_signal'].iloc[-1]}")
                
            # Afficher les 5 premières et dernières lignes pour le débogage
            print("\n=== Aperçu des données ===")
            print(test_data[['open', 'high', 'low', 'close', 'volume']].head())
            print("...")
            print(test_data[['open', 'high', 'low', 'close', 'volume']].tail())
            
            # Exécuter l'analyse
            signal = strategy.analyze(test_data)
            
            # Afficher les détails du signal pour le débogage
            print("\n=== Signal généré ===")
            print(f"Direction: {signal.direction}")
            print(f"Force: {signal.strength.value}")
            print(f"Métadonnées: {signal.metadata}")
            
            # Vérifier que le signal est haussier et que la force est supérieure à NEUTRAL
            self.assertGreater(signal.strength.value, SignalStrength.NEUTRAL.value, 
                             f"La force du signal devrait être > {SignalStrength.NEUTRAL.value}, mais est {signal.strength.value} - Métadonnées: {signal.metadata}")
            self.assertEqual(signal.direction, 1, f"Le signal devrait être haussier - Direction actuelle: {signal.direction} - Métadonnées: {signal.metadata}")
            
            # Deuxième test: prédiction neutre
            mock_instance.predict.return_value = np.ones(10)  # Signal neutre
            signal = strategy.analyze(data)
            
            # Le signal devrait être neutre
            self.assertEqual(signal.strength, SignalStrength.NEUTRAL)
            self.assertEqual(signal.strength.value, 0)
    
    @classmethod
    def tearDownClass(cls):
        """Nettoyage après les tests."""
        # Supprimer les modèles de test
        test_model_path = TEST_MODEL_DIR / 'test_model'
        if test_model_path.exists():
            for file in test_model_path.glob('*'):
                file.unlink()
            test_model_path.rmdir()

if __name__ == "__main__":
    unittest.main()
