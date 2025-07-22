"""
Tests unitaires pour le module strategy_switcher.
"""
import unittest
from unittest.mock import MagicMock, patch, Mock
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
import sys
import os

# Ajouter le répertoire parent au chemin pour les imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Importer les mocks avant les modules qui dépendent de TensorFlow
from tests.mocks.mock_predictor import MockMLPredictor, MockSignalGenerator

# Patcher sys.modules pour éviter l'import de TensorFlow
import sys
sys.modules['tensorflow'] = MagicMock()
sys.modules['tensorflow.keras'] = MagicMock()
sys.modules['tensorflow.keras.models'] = MagicMock()

# Importer le mock du module predictor
from tests.mocks.mock_predictor_module import *

# Maintenant importer les modules à tester
from src.core.strategies.strategy_switcher import (
    StrategySwitcher,
    StrategyPerformance,
    MarketCondition
)


class TestStrategyPerformance(unittest.TestCase):
    """Tests pour la classe StrategyPerformance."""
    
    def setUp(self):
        """Initialisation avant chaque test."""
        self.performance = StrategyPerformance(lookback_period=7)  # 7 jours de lookback
        
    @patch('src.core.strategies.strategy_switcher.datetime')
    def test_update_performance(self, mock_datetime):
        """Test de la mise à jour des performances."""
        # Configuration du mock pour datetime
        fixed_time = datetime(2023, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        mock_datetime.now.return_value = fixed_time
        
        # Données de test
        strategy_name = "test_strategy"
        pnl = 0.05  # 5% de profit
        timestamp = datetime.now(timezone.utc)
        
        # Mise à jour des performances avec le timestamp
        self.performance.update_performance(strategy_name, pnl, timestamp)
        
        # Vérification que les performances ont été mises à jour
        self.assertIn(strategy_name, self.performance.performance_data)
        self.assertEqual(len(self.performance.performance_data[strategy_name]), 1)
        self.assertEqual(self.performance.performance_data[strategy_name][0]['pnl'], pnl)
        self.assertEqual(self.performance.performance_data[strategy_name][0]['timestamp'], timestamp)
    
    def test_cleanup_old_data(self):
        """Test du nettoyage des anciennes données."""
        strategy_name = "test_strategy"
        now = datetime.now()
        
        # Ajouter des données anciennes et récentes
        self.performance.performance_data[strategy_name] = [
            {'timestamp': now - timedelta(days=10), 'pnl': 0.01},  # Trop ancien
            {'timestamp': now - timedelta(days=5), 'pnl': 0.02},   # Dans la période
            {'timestamp': now, 'pnl': 0.03}                        # Actuel
        ]
        
        # Nettoyer les anciennes données
        self.performance._cleanup_old_data()
        
        # Vérifier que seules les données récentes sont conservées
        self.assertEqual(len(self.performance.performance_data[strategy_name]), 2)
        self.assertEqual(self.performance.performance_data[strategy_name][0]['pnl'], 0.02)
        self.assertEqual(self.performance.performance_data[strategy_name][1]['pnl'], 0.03)
    
    def test_get_performance_metrics(self):
        """Test du calcul des métriques de performance."""
        strategy_name = "test_strategy"
        
        # Ajouter des données de performance
        self.performance.update_performance(strategy_name, 0.05, datetime.now() - timedelta(days=1))
        self.performance.update_performance(strategy_name, -0.02, datetime.now() - timedelta(hours=12))
        self.performance.update_performance(strategy_name, 0.03, datetime.now())
        
        # Obtenir les métriques
        metrics = self.performance.get_performance_metrics(strategy_name)
        
        # Vérifier les métriques calculées
        self.assertAlmostEqual(metrics['total_return'], 0.06)  # 5% - 2% + 3% = 6%
        self.assertAlmostEqual(metrics['win_rate'], 2/3)       # 2 gains sur 3 trades
        self.assertAlmostEqual(metrics['avg_win'], 0.04)       # (5% + 3%) / 2
        self.assertAlmostEqual(metrics['avg_loss'], -0.02)     # -2%
        self.assertEqual(metrics['num_trades'], 3)


if __name__ == '__main__':
    unittest.main()
