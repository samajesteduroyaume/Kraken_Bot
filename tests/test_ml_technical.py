import unittest
import pandas as pd
import numpy as np
from pathlib import Path
import shutil

# Ajouter le répertoire parent au path pour les imports
import sys
sys.path.append(str(Path(__file__).parent.parent))

from src.core.ml_predictor import MLPredictor, ModelVersion
from src.core.technical_analyzer import TechnicalAnalyzer, IndicatorType, IndicatorConfig, IndicatorManager

class TestMLPredictor(unittest.TestCase):
    """Tests pour la classe MLPredictor."""
    
    @classmethod
    def setUpClass(cls):
        # Configurer le répertoire de test
        cls.test_dir = Path("test_models")
        cls.test_dir.mkdir(exist_ok=True)
        
        # Créer des données de test
        np.random.seed(42)
        n_samples = 1000
        cls.X = pd.DataFrame({
            'feature1': np.random.randn(n_samples),
            'feature2': np.random.randn(n_samples) * 2,
            'feature3': np.random.randn(n_samples) * 0.5
        })
        cls.y = (cls.X['feature1'] + cls.X['feature2'] * 0.5 + np.random.randn(n_samples) * 0.1 > 0).astype(int)
    
    def setUp(self):
        # Réinitialiser le répertoire de test
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)
        self.test_dir.mkdir()
        
        # Initialiser le prédicteur
        self.predictor = MLPredictor(
            model_id="test_model",
            retrain_interval=1,  # 1 jour pour les tests
            min_accuracy=0.6,
            max_versions=3
        )
    
    def test_initialization(self):
        """Teste l'initialisation du prédicteur."""
        self.assertIsNotNone(self.predictor.model)
        self.assertIsNotNone(self.predictor.scaler)
        self.assertEqual(self.predictor.model_id, "test_model")
    
    def test_train_predict(self):
        """Teste l'entraînement et la prédiction."""
        # Entraîner le modèle
        training_data = {
            'X': self.X,
            'y': self.y,
            'test_size': 0.2,
            'random_state': 42
        }
        metrics = self.predictor.train(training_data)
        
        # Vérifier les métriques
        self.assertIn('test_accuracy', metrics)
        self.assertGreaterEqual(metrics['test_accuracy'], 0.5)  # Seuil plus bas pour les tests
        
        # Faire des prédictions
        predictions = self.predictor.predict(self.X[:10].values)  # Convertir en numpy array
        self.assertEqual(len(predictions), 10)
        # Vérifier que toutes les prédictions sont des probabilités entre 0 et 1
        self.assertTrue(all(0.0 <= pred <= 1.0 for pred in predictions))
    
    def test_versioning(self):
        """Teste le versionnage des modèles."""
        # Entraîner plusieurs versions
        versions = []
        training_data = {
            'X': self.X,
            'y': self.y,
            'test_size': 0.2,
            'random_state': 42
        }
        
        for i in range(3):
            metrics = self.predictor.train(training_data)
            versions.append(metrics)
        
        # Vérifier le nombre de versions
        self.assertEqual(len(versions), 3)
        
        # Vérifier que le dossier des modèles existe
        model_dir = Path("models/test_model")
        self.assertTrue(model_dir.exists())
        
        # Vérifier le nombre de fichiers de modèle
        model_files = list(model_dir.glob("*.joblib"))
        self.assertLessEqual(len(model_files), 3)  # max_versions = 3


class TestTechnicalAnalyzer(unittest.TestCase):
    """Tests pour la classe TechnicalAnalyzer."""
    
    @classmethod
    def setUpClass(cls):
        # Créer des données de test
        np.random.seed(42)
        n_samples = 1000
        index = pd.date_range(end=pd.Timestamp.now(), periods=n_samples, freq='D')
        cls.price_data = pd.Series(
            np.cumsum(np.random.randn(n_samples) * 0.01) + 100,
            index=index,
            name='close'
        )
        cls.volume_data = pd.Series(
            np.random.randint(100, 1000, size=n_samples),
            index=index,
            name='volume'
        )
    
    def setUp(self):
        # Initialiser l'analyseur technique
        self.analyzer = TechnicalAnalyzer()
    
    def test_calculate_indicators(self):
        """Teste le calcul des indicateurs."""
        # Créer un DataFrame avec les données de prix et de volume
        df = pd.DataFrame({
            'close': self.price_data,
            'volume': self.volume_data
        })
        
        # Calculer les indicateurs
        indicators = self.analyzer.calculate_indicators(
            price_data=df['close'],
            volume_data=df['volume'],
            indicators=['sma_20', 'rsi_14', 'bb_upper', 'bb_lower']
        )
        
        # Vérifier que nous avons des résultats
        self.assertGreater(len(indicators), 0)
        
        # Vérifier que les indicateurs ont la bonne longueur
        for key, values in indicators.items():
            self.assertIsInstance(values, (pd.Series, np.ndarray))
            self.assertGreater(len(values), 0)
    
    def test_analyze_trend(self):
        """Teste l'analyse de tendance."""
        trend = self.analyzer.analyze_trend(self.price_data)
        self.assertIn(trend, ['bullish', 'bearish', 'neutral'])
    
    def test_analyze_volatility(self):
        """Teste l'analyse de volatilité."""
        volatility = self.analyzer.analyze_volatility(self.price_data)
        self.assertIsInstance(volatility, float)
        self.assertGreaterEqual(volatility, 0.0)


class TestIndicatorManager(unittest.TestCase):
    """Tests pour la classe IndicatorManager."""
    
    def setUp(self):
        # Créer un répertoire de test
        self.test_dir = Path("test_indicators")
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)
        self.test_dir.mkdir()
        
        # Initialiser le gestionnaire d'indicateurs
        self.config_file = self.test_dir / "test_indicators.json"
        self.manager = IndicatorManager(str(self.config_file))
    
    def test_add_remove_indicator(self):
        """Teste l'ajout et la suppression d'indicateurs."""
        # Ajouter un indicateur personnalisé
        indicator = IndicatorConfig(
            name="custom_indicator",
            indicator_type=IndicatorType.CUSTOM,
            params={"param1": 10, "param2": 20},
            display_name="Indicateur personnalisé",
            description="Un indicateur de test"
        )
        self.manager.add_indicator(indicator)
        
        # Vérifier l'ajout
        self.assertIn("custom_indicator", self.manager.indicators)
        
        # Supprimer l'indicateur
        self.assertTrue(self.manager.remove_indicator("custom_indicator"))
        self.assertNotIn("custom_indicator", self.manager.indicators)
    
    def test_save_load_config(self):
        """Teste la sauvegarde et le chargement de la configuration."""
        # Sauvegarder la configuration
        self.assertTrue(self.manager.save_config())
        self.assertTrue(self.config_file.exists())
        
        # Charger la configuration
        new_manager = IndicatorManager(str(self.config_file))
        self.assertTrue(new_manager.load_config())
        self.assertGreater(len(new_manager.indicators), 0)


if __name__ == "__main__":
    unittest.main()
