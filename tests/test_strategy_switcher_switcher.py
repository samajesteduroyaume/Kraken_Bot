"""
Tests unitaires pour la classe StrategySwitcher.
"""
import unittest
import logging
from unittest.mock import MagicMock, patch, Mock
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
import sys
import os

# Configurer le niveau de log pour afficher les messages de débogage
logging.basicConfig(level=logging.DEBUG)

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
    MarketCondition
)


class TestStrategySwitcher(unittest.TestCase):
    """Tests pour la classe StrategySwitcher."""
    
    def setUp(self):
        """Initialisation avant chaque test."""
        self.config = {
            'performance_lookback_days': 7,
            'default_strategy_weights': {
                'trend_following': 1.0,
                'mean_reversion': 1.0,
                'momentum': 1.0,
                'volatility_breakout': 1.0,
                'grid_trading': 1.0,
                'ml': 1.0
            },
            'max_active_strategies': 3,
            'min_performance_threshold': 0.0
        }
        
        # Créer une instance de StrategySwitcher avec le mock
        self.switcher = StrategySwitcher(self.config)
        
        # Données de marché de test
        self.market_data = {
            'close': pd.Series([100, 101, 102, 103, 104]),
            'high': pd.Series([101, 102, 103, 104, 105]),
            'low': pd.Series([99, 100, 101, 102, 103]),
            'volume': pd.Series([1000, 1200, 1100, 1300, 1400])
        }
    
    def test_initialization(self):
        """Test de l'initialisation du StrategySwitcher."""
        # Vérifier que le switcher est correctement initialisé avec les stratégies par défaut
        self.assertIsNotNone(self.switcher.performance_tracker)
        self.assertIsNone(self.switcher.last_market_condition)
        
        # Vérifier que les stratégies sont correctement associées aux conditions de marché
        self.assertIn(MarketCondition.TRENDING_UP, self.switcher.strategy_conditions)
        self.assertIn(MarketCondition.TRENDING_DOWN, self.switcher.strategy_conditions)
        self.assertIn(MarketCondition.RANGING, self.switcher.strategy_conditions)
        self.assertIn(MarketCondition.HIGH_VOLATILITY, self.switcher.strategy_conditions)
        self.assertIn(MarketCondition.LOW_VOLATILITY, self.switcher.strategy_conditions)
        self.assertIn(MarketCondition.BREAKOUT, self.switcher.strategy_conditions)
        self.assertIn(MarketCondition.BREAKDOWN, self.switcher.strategy_conditions)
        
        # Vérifier que les stratégies par défaut sont présentes
        default_strategies = ['trend_following', 'mean_reversion', 'momentum', 
                             'volatility_breakout', 'grid_trading', 'ml']
        for strat in default_strategies:
            self.assertIn(strat, self.switcher.strategy_weights)
    
    @patch('src.core.strategies.strategy_switcher.calculate_atr')
    @patch('src.core.strategies.strategy_switcher.calculate_adx')
    def test_analyze_market_conditions_trending_up(self, mock_calculate_adx, mock_calculate_atr):
        """Test de la détection d'une tendance haussière."""
        # Configurer les mocks pour les fonctions de calcul d'indicateurs
        # ATR constant pour simplifier
        mock_calculate_atr.return_value = pd.Series([1.0, 1.0, 1.0, 1.0, 1.0])
        
        # ADX > 25 et +DI > -DI pour indiquer une tendance haussière
        mock_calculate_adx.return_value = (
            pd.Series([30, 31, 32, 33, 34]),  # ADX > 25
            pd.Series([35, 36, 37, 38, 39]),  # +DI
            pd.Series([15, 16, 17, 18, 19])   # -DI
        )
        
        # Créer un vrai DataFrame pandas pour les données de marché
        data = {
            'open': [100, 101, 102, 103, 104],
            'high': [101, 102, 103, 104, 105],
            'low': [99, 100, 101, 102, 103],
            'close': [100.5, 101.5, 102.5, 103.5, 104.5],
            'volume': [1000, 1200, 1100, 1300, 1400]
        }
        df = pd.DataFrame(data)
        
        # Créer un objet qui simule le comportement de rolling pour les tests
        class MockRolling:
            def __init__(self, window):
                self.window = window
                
            def max(self):
                if self.window == 20:
                    return pd.Series([104] * 5)  # Valeur haute pour simuler un breakout
                return pd.Series([0] * 5)
                
            def min(self):
                if self.window == 20:
                    return pd.Series([96] * 5)  # Valeur basse pour simuler un breakdown
                return pd.Series([0] * 5)
                
            def mean(self):
                if self.window == 50:
                    return pd.Series([95, 96, 97, 98, 99])  # SMA50
                elif self.window == 200:
                    return pd.Series([85, 86, 87, 88, 89])  # SMA200
                return pd.Series([0] * 5)
                
            def shift(self, periods=1):
                return self
        
        # Remplacer la méthode rolling du DataFrame pour nos tests
        original_rolling = pd.DataFrame.rolling
        pd.DataFrame.rolling = lambda self, window, min_periods=None: MockRolling(window)
        
        try:
            # Créer un dictionnaire avec la clé 'candles' comme attendu par analyze_market_conditions
            market_data = {'candles': df}
            
            # Analyser les conditions du marché
            condition = self.switcher.analyze_market_conditions(market_data)
            
            # Vérifier que la condition détectée est TRENDING_UP
            self.assertEqual(condition, MarketCondition.TRENDING_UP,
                           "Devrait détecter une tendance haussière avec ADX > 25 et +DI > -DI")
            
            # Vérifier que les mocks ont été appelés correctement
            mock_calculate_atr.assert_called_once()
            mock_calculate_adx.assert_called_once()
            
        finally:
            # Restaurer la méthode rolling originale
            pd.DataFrame.rolling = original_rolling
    
    def test_get_best_strategies(self):
        """Test de la sélection des meilleures stratégies."""
        # Créer un mock de DataFrame avec des données de marché factices
        class MockMarketData:
            def __init__(self):
                self.data = {
                    'close': pd.Series([100, 101, 102, 103, 104]),
                    'high': pd.Series([101, 102, 103, 104, 105]),
                    'low': pd.Series([99, 100, 101, 102, 103]),
                    'volume': pd.Series([1000, 1200, 1100, 1300, 1400]),
                    'adx': pd.Series([30, 32, 34, 36, 38]),
                    'plus_di': pd.Series([35, 36, 37, 38, 39]),
                    'minus_di': pd.Series([15, 16, 17, 18, 19]),
                    'atr': pd.Series([1.0, 1.0, 1.0, 1.0, 1.0])
                }
            
            def __getitem__(self, key):
                return self.data[key]
            
            @property
            def iloc(self):
                return [{
                    'close': 104, 'high': 105, 'low': 103, 'volume': 1400,
                    'adx': 38, 'plus_di': 39, 'minus_di': 19, 'atr': 1.0
                }]
        
        # Créer une instance du mock avec les données de test
        market_data = MockMarketData()
        
        # Simuler une analyse de marché qui retourne une tendance haussière
        with patch.object(self.switcher, 'analyze_market_conditions', 
                         return_value=MarketCondition.TRENDING_UP):
            # Obtenir les meilleures stratégies
            strategies = self.switcher.get_best_strategies(market_data)
            
            # Vérifier que les stratégies sont correctement sélectionnées
            self.assertIsInstance(strategies, list)
            self.assertGreaterEqual(len(strategies), 1)
            
            # Vérifier que chaque stratégie est un tuple (nom, score)
            for strategy in strategies:
                self.assertIsInstance(strategy, tuple)
                self.assertEqual(len(strategy), 2)  # (nom, score)
                self.assertIsInstance(strategy[0], str)    # nom de la stratégie
                self.assertIsInstance(strategy[1], (int, float))  # score
            
            # Vérifier que les stratégies recommandées sont bien celles attendues pour une tendance haussière
            expected_strategies = ['trend_following', 'momentum']
            returned_strategies = [s[0] for s in strategies]
            for expected in expected_strategies:
                self.assertIn(expected, returned_strategies)
    
    def test_adjust_strategy_weights(self):
        """Test de l'ajustement des poids des stratégies."""
        # Créer un mock de DataFrame avec des données de marché factices
        class MockMarketData:
            def __init__(self):
                self.data = {
                    'close': pd.Series([100, 101, 102, 103, 104]),
                    'high': pd.Series([101, 102, 103, 104, 105]),
                    'low': pd.Series([99, 100, 101, 102, 103]),
                    'volume': pd.Series([1000, 1200, 1100, 1300, 1400]),
                    'adx': pd.Series([30, 32, 34, 36, 38]),
                    'plus_di': pd.Series([35, 36, 37, 38, 39]),
                    'minus_di': pd.Series([15, 16, 17, 18, 19]),
                    'atr': pd.Series([1.0, 1.0, 1.0, 1.0, 1.0])
                }
            
            def __getitem__(self, key):
                return self.data[key]
            
            @property
            def iloc(self):
                return [{
                    'close': 104, 'high': 105, 'low': 103, 'volume': 1400,
                    'adx': 38, 'plus_di': 39, 'minus_di': 19, 'atr': 1.0
                }]
        
        # Créer une instance du mock avec les données de test
        market_data = MockMarketData()
        
        # Ajouter des données de performance
        now = datetime.now(timezone.utc)
        
        # Ajouter des performances pour chaque stratégie
        # Pour chaque stratégie, on définit une liste de rendements (certains positifs, d'autres négatifs)
        performances = {
            'trend_following': [0.02, -0.01, 0.015, -0.005, 0.01],  # 60% de win rate
            'mean_reversion': [0.015, -0.01, 0.01, -0.005, 0.005],  # 60% de win rate
            'momentum': [0.025, -0.01, 0.02, -0.005, 0.015],        # 80% de win rate
            'volatility_breakout': [0.01, -0.02, 0.005, -0.01, 0.0], # 40% de win rate
            'grid_trading': [0.018, -0.01, 0.015, -0.005, 0.01],     # 60% de win rate
            'ml': [0.022, -0.01, 0.018, -0.005, 0.012]              # 60% de win rate
        }
        
        # Ajouter plusieurs jours de données pour chaque stratégie
        for strategy_name, pnl_list in performances.items():
            for days_ago, pnl in enumerate(pnl_list):
                self.switcher.performance_tracker.update_performance(
                    strategy_name,
                    pnl=pnl,
                    timestamp=now - timedelta(days=days_ago)
                )
        
        # Simuler une analyse de marché qui retourne une tendance haussière
        with patch.object(self.switcher, 'analyze_market_conditions', 
                         return_value=MarketCondition.TRENDING_UP):
            # Ajuster les poids en fonction des performances
            self.switcher.adjust_strategy_weights(market_data)
            
            # Récupérer les poids mis à jour
            weights = self.switcher.get_strategy_weights()
            
            # Vérifier que tous les poids sont positifs
            for strategy, weight in weights.items():
                self.assertGreaterEqual(weight, 0.0, f"Le poids de {strategy} devrait être >= 0.0")
                self.assertLessEqual(weight, 1.0, f"Le poids de {strategy} devrait être <= 1.0")
            
            # Vérifier que les stratégies recommandées pour une tendance haussière ont un poids plus élevé
            trending_up_strategies = ['trend_following', 'momentum']
            other_strategies = [s for s in weights.keys() if s not in trending_up_strategies]
            
            # Calculer le poids moyen des stratégies de tendance haussière
            avg_trending_up_weight = sum(weights[s] for s in trending_up_strategies) / len(trending_up_strategies)
            # Calculer le poids moyen des autres stratégies
            avg_other_weight = sum(weights[s] for s in other_strategies) / len(other_strategies) if other_strategies else 0
            
            # Vérifier que les stratégies de tendance haussière ont un poids moyen plus élevé
            self.assertGreater(avg_trending_up_weight, avg_other_weight,
                             "Les stratégies de tendance haussière devraient avoir un poids moyen plus élevé")
            
            # Vérifier que les poids sont cohérents avec les performances
            # La stratégie 'momentum' a la meilleure performance (2.5%)
            self.assertEqual(max(weights, key=weights.get), 'momentum',
                           "La stratégie avec la meilleure performance devrait avoir le poids le plus élevé")
            
            # Vérifier que les poids sont normalisés (somme = 1.0)
            total_weight = sum(weights.values())
            self.assertAlmostEqual(total_weight, 1.0, delta=0.01,
                                 msg=f"La somme des poids devrait être égale à 1.0 (actuellement {total_weight})")

if __name__ == '__main__':
    unittest.main()
