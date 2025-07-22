"""
Tests unitaires pour la stratégie de Swing Trading.
"""
import unittest
import pandas as pd
import numpy as np
import logging
from unittest.mock import patch, MagicMock
from datetime import datetime, timedelta

# Configurer les logs pour le débogage
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Importer la stratégie à tester
from src.strategies.base_strategy import SignalStrength
from src.strategies.swing_strategy import SwingStrategy
from src.strategies.swing_config import get_swing_config

class TestSwingStrategy(unittest.TestCase):
    """Classe de test pour la stratégie de Swing Trading."""
    
    def setUp(self):
        """Initialisation avant chaque test."""
        # Créer une configuration de test
        self.test_config = {
            'symbol': 'BTC/EUR',
            'timeframes': ['15m', '1h'],
            'indicators': {
                'rsi': {'period': 14},
                'macd': {'fast_period': 12, 'slow_period': 26, 'signal_period': 9},
                'atr': {'period': 14, 'multiplier': 2.0},
                'sma': {'short_period': 50, 'long_period': 200},
                'bollinger_bands': {'window': 20, 'window_dev': 2.0},
                'mfi': {'period': 14}
            },
            'signal_generation': {
                'required_timeframe_confirmation': 1,
                'min_signal_strength': 0.6,
                'volume_filter': {'enabled': False},
                'trend_filter': {'enabled': True, 'min_trend_strength': 0.6},
                'volatility_filter': {'enabled': False}
            },
            'risk_management': {
                'stop_loss_atr_multiplier': 2.0,
                'take_profit_ratio': 2.0,
                'max_position_size': 0.15,
                'risk_per_trade': 0.01
            }
        }
        
        # Créer des données de test
        self.test_data = self._create_test_data()
        
        # Initialiser la stratégie
        self.strategy = SwingStrategy(config=self.test_config)
    
    def _create_test_data(self):
        """Crée des données de test pour les tests unitaires."""
        # Créer un DataFrame avec des données aléatoires mais réalistes
        np.random.seed(42)
        dates = pd.date_range(end=datetime.now(), periods=500, freq='15min')
        base_price = 50000
        returns = np.random.normal(0.0001, 0.01, len(dates))
        prices = base_price * (1 + np.cumsum(returns))
        
        # Créer des données OHLCV
        df = pd.DataFrame({
            'open': prices * 0.999 + np.random.normal(0, 10, len(dates)),
            'high': prices * 1.001 + np.abs(np.random.normal(0, 15, len(dates))),
            'low': prices * 0.999 - np.abs(np.random.normal(0, 15, len(dates))),
            'close': prices,
            'volume': np.random.lognormal(mean=10, sigma=1, size=len(dates))
        }, index=dates)
        
        # Créer un dictionnaire de DataFrames par timeframe
        data = {
            '15m': df,
            '1h': df.resample('1h').agg({
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last',
                'volume': 'sum'
            }).dropna()
        }
        
        return data
    
    def test_initialization(self):
        """Teste l'initialisation de la stratégie."""
        self.assertEqual(self.strategy.symbol, 'BTC/EUR')
        self.assertIn('15m', self.strategy.config['timeframes'])
        self.assertIn('1h', self.strategy.config['timeframes'])
        self.assertEqual(self.strategy.config['indicators']['rsi']['period'], 14)
    
    def test_calculate_indicators(self):
        """Teste le calcul des indicateurs techniques."""
        # Appeler la méthode à tester
        indicators = self.strategy.calculate_indicators(self.test_data)
        
        # Vérifier que les indicateurs ont été calculés pour chaque timeframe
        self.assertIn('15m', indicators)
        self.assertIn('1h', indicators)
        
        # Vérifier que les indicateurs clés sont présents
        for tf in indicators:
            self.assertIn('rsi', indicators[tf])
            self.assertIn('macd', indicators[tf])
            self.assertIn('atr', indicators[tf])
            self.assertIn('sma_short', indicators[tf])
            self.assertIn('sma_long', indicators[tf])
            self.assertIn('bb_high', indicators[tf])
            self.assertIn('bb_mid', indicators[tf])
            self.assertIn('bb_low', indicators[tf])
            self.assertIn('mfi', indicators[tf])
            self.assertIn('adx', indicators[tf])
            self.assertIn('roc', indicators[tf])
            self.assertIn('trend_strength', indicators[tf])
    
    def test_generate_signals(self):
        """Teste la génération des signaux de trading."""
        # Calculer d'abord les indicateurs
        self.strategy.calculate_indicators(self.test_data)
        
        # Générer les signaux
        signals = self.strategy.generate_signals(self.test_data)
        
        # Vérifier que la méthode retourne une liste
        self.assertIsInstance(signals, list)
        
        # Si des signaux sont générés, vérifier leur structure
        if signals:
            signal = signals[0]
            self.assertIn(signal.direction, [1, -1])  # 1 pour achat, -1 pour vente
            self.assertIsInstance(signal.strength, SignalStrength)
            self.assertIsInstance(signal.price, (int, float, Decimal))
            self.assertIsInstance(signal.timestamp, datetime)
            self.assertIsInstance(signal.stop_loss, (int, float, Decimal, type(None)))
            self.assertIsInstance(signal.take_profit, (int, float, Decimal, type(None)))
            self.assertIsInstance(signal.metadata, (dict, type(None)))
    
    def test_risk_management(self):
        """Teste la gestion des risques."""
        # Tester le calcul de la taille de position
        # Pour ce test, nous allons utiliser directement la configuration de risque
        max_position = self.test_config['risk_management']['max_position_size']
        position_size = self.strategy.calculate_position_size(
            current_price=50000,
            balance=10000  # Utilisation d'une valeur de solde fixe pour le test
        )
        self.assertGreaterEqual(position_size, 0)
        self.assertLessEqual(position_size, max_position)
        
        # Tester le calcul du stop loss avec ATR
        atr_value = 1000  # Valeur ATR pour le test
        stop_loss = self.strategy.calculate_stop_loss(
            entry_price=50000,
            signal='BUY',
            atr=atr_value
        )
        self.assertLess(stop_loss, 50000)  # Pour un achat, le stop loss doit être en dessous du prix d'entrée
        
        # Vérifier que le stop loss est à la bonne distance (ATR * multiplicateur)
        expected_stop_distance = atr_value * self.test_config['risk_management']['stop_loss_atr_multiplier']
        self.assertAlmostEqual(50000 - stop_loss, expected_stop_distance, delta=0.01)
        
        # Tester le calcul du take profit
        take_profit = self.strategy.calculate_take_profit(
            entry_price=50000,
            stop_loss=stop_loss,
            signal='BUY'
        )
        self.assertGreater(take_profit, 50000)  # Pour un achat, le take profit doit être au-dessus du prix d'entrée
        
        # Vérifier que le take profit est au bon ratio par rapport au stop loss
        expected_take_profit = 50000 + (50000 - stop_loss) * self.test_config['risk_management']['take_profit_ratio']
        self.assertAlmostEqual(take_profit, expected_take_profit, delta=0.01)
    
    @patch.object(SwingStrategy, 'calculate_indicators')
    def test_generate_signals_with_mock(self, mock_calculate_indicators):
        """Teste la génération de signaux avec un mock pour calculate_indicators."""
        logger = logging.getLogger(__name__)
        logger.info("Début du test test_generate_signals_with_mock")
        
        # Créer des données de test avec des valeurs qui satisfont toutes les conditions
        test_data = {
            '15m': pd.DataFrame({
                'open': [50000, 50500, 50300, 50100, 50200],
                'high': [50500, 50800, 50600, 50400, 50300],
                'low': [49500, 49800, 49600, 49700, 49800],
                'close': [50200, 50000, 50200, 50000, 50100],  # Prix actuel: 50100
                'volume': [100, 120, 110, 115, 105]  # Volume > min_volume_btc (1.0)
            })
        }
        
        # Configurer le mock pour retourner des indicateurs favorisant un signal d'achat
        mock_indicators = {
            '15m': {
                # RSI < 30 (survente) - condition 1
                'rsi': pd.Series([28, 29, 28, 29, 25]),  # Dernière valeur: 25 (< 30)
                
                # MACD > signal (condition 2) et croisement haussier (condition 3)
                'macd': pd.Series([-50, -30, -10, 10, 12]),  # Dernière valeur: 12
                'macd_signal': pd.Series([-40, -20, 0, 15, 10]),  # Dernière valeur: 10 (MACD > signal)
                'macd_hist': pd.Series([-10, -10, -10, -5, 2]),  # Croisement haussier (négatif à positif)
                
                # ATR pour la volatilité
                'atr': pd.Series([1000, 1050, 1100, 1150, 1200]),  # ~2.4% du prix
                
                # Moyennes mobiles
                'sma_short': pd.Series([48500, 48600, 48700, 48800, 48900]),  # Dernière valeur: 48900
                'sma_long': pd.Series([49000, 48900, 48800, 48700, 48600]),  # Dernière valeur: 48600 (SMA courte > SMA longue)
                
                # ADX > 25 (condition 6)
                'adx': pd.Series([30, 31, 32, 33, 34]),  # Dernière valeur: 34 (> 25)
                
                # MFI < 30 (condition 7)
                'mfi': pd.Series([25, 26, 27, 28, 25]),  # Dernière valeur: 25 (< 30)
                
                # Bandes de Bollinger
                'bb_low': pd.Series([47000, 47200, 47400, 47600, 47000]),  # Dernière valeur: 47000
                'bb_high': pd.Series([51000, 51200, 51400, 51600, 51800]),
                'bb_mid': pd.Series([49000, 49200, 49400, 49600, 49800]),
                'bb_width': pd.Series([0.08, 0.08, 0.08, 0.08, 0.08]),
                
                # Tendance forte pour le filtre
                'trend_strength': pd.Series([0.85, 0.86, 0.87, 0.88, 0.89]),  # > 0.6
                
                # Niveaux de stop loss et take profit
                'stop_loss_level': pd.Series([48000, 48100, 48200, 48300, 48400]),  # Dernière valeur: 48400
                'take_profit_level': pd.Series([52000, 52100, 52200, 52300, 52400])  # Dernière valeur: 52400
            }
        }
        
        # Configurer le mock pour retourner nos indicateurs
        def mock_calculate_indicators_side_effect(data):
            # Retourner directement les indicateurs mockés
            # La clé '15m' contient déjà tous les indicateurs nécessaires
            result = {'15m': mock_indicators['15m']}
            
            # Ajouter des logs pour le débogage
            logger.info(f"Indicateurs retournés par le mock: {list(result.keys())}")
            if '15m' in result:
                logger.info(f"Indicateurs pour 15m: {list(result['15m'].keys())}")
            
            # Mettre à jour les indicateurs de la stratégie
            self.strategy.indicators = result
            return result
            
        mock_calculate_indicators.side_effect = mock_calculate_indicators_side_effect
        
        # Configurer la stratégie pour le test
        self.strategy.signal_config = {
            'required_timeframe_confirmation': 1,  # Un seul timeframe requis pour la confirmation
            'min_signal_strength': 0.6,
            'volume_filter': {
                'enabled': False,  # Désactiver temporairement
                'min_volume_btc': 1.0,  # Volume minimum très bas pour le test
                'volume_period': 20
            },
            'trend_filter': {
                'enabled': False,  # Désactiver temporairement
                'min_trend_strength': 0.6  # Tendance forte requise
            },
            'volatility_filter': {
                'enabled': False,  # Désactiver temporairement
                'min_atr_percent': 0.1,  # Volatilité minimale très basse pour le test
                'max_atr_percent': 10.0  # Volatilité maximale très haute pour le test
            }
        }
        
        # Configurer la gestion des risques
        self.strategy.risk_config = {
            'stop_loss_atr_multiplier': 2.0,
            'take_profit_ratio': 2.0,
            'max_position_size': 0.15,
            'risk_per_trade': 0.01
        }
        
        # Initialiser le portefeuille pour le test
        self.strategy.portfolio_value = 10000.0
        
        # Générer les signaux
        logger.info("Génération des signaux...")
        signals = self.strategy.generate_signals(test_data)
        
        # Afficher les signaux générés
        logger.info(f"Signaux générés: {signals}")
        
        # Vérifier que des signaux ont été générés
        self.assertGreater(len(signals), 0, "Aucun signal généré alors qu'au moins un était attendu")
        
        # Vérifier que le signal est un achat (basé sur les données mockées)
        signal = signals[0]
        self.assertEqual(signal.direction, 1, "Le signal devrait être un achat (direction=1)")
        self.assertEqual(signal.strength, SignalStrength.STRONG, "La force du signal devrait être 'STRONG'")
        
        # Vérifier que les niveaux de stop loss et take profit sont cohérents
        self.assertLess(signal.stop_loss, signal.price, 
                      "Le stop loss devrait être inférieur au prix d'entrée pour un achat")
        self.assertGreater(signal.take_profit, signal.price,
                         "Le take profit devrait être supérieur au prix d'entrée pour un achat")
        
        # Vérifier que le prix d'entrée est cohérent avec les données
        logger.info(f"Prix d'entrée généré: {signal.price}")
        self.assertAlmostEqual(float(signal.price), 50100, delta=100, 
                             msg=f"Le prix d'entrée devrait être proche de 50100, mais est {signal.price}")
        
        # Vérifier que le stop loss est cohérent avec le niveau calculé
        expected_stop_loss = 48400  # Dernière valeur de stop_loss_level dans les données mockées
        logger.info(f"Stop loss généré: {signal.stop_loss}, attendu: {expected_stop_loss}")
        
        # Convertir en float pour la comparaison si nécessaire
        stop_loss_float = float(signal.stop_loss) if hasattr(signal.stop_loss, '__float__') else signal.stop_loss
        self.assertAlmostEqual(stop_loss_float, expected_stop_loss, delta=100,
                             msg=f"Le stop loss devrait être proche de {expected_stop_loss}, mais est {signal.stop_loss}")
        
        # Vérifier que le take profit correspond à la valeur fournie dans take_profit_level
        expected_take_profit = 52400  # Dernière valeur de take_profit_level dans les données mockées
        logger.info(f"Take profit généré: {signal.take_profit}, attendu: {expected_take_profit}")
        
        # Convertir en float pour la comparaison si nécessaire
        take_profit_float = float(signal.take_profit) if hasattr(signal.take_profit, '__float__') else signal.take_profit
        self.assertAlmostEqual(take_profit_float, expected_take_profit, delta=100,
                             msg=f"Le take profit devrait être proche de {expected_take_profit}, mais est {signal.take_profit}")

if __name__ == '__main__':
    unittest.main()
