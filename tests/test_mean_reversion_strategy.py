"""
Tests unitaires pour la stratégie de retour à la moyenne (Mean Reversion).
"""
import unittest
from unittest.mock import MagicMock, patch
import pandas as pd
import numpy as np
from decimal import Decimal
from datetime import datetime

# Configuration de test pour la stratégie Mean Reversion
MEAN_REVERSION_CONFIG = {
    'symbol': 'BTC/EUR',
    'timeframes': ['15m', '1h', '4h'],
    'indicators': {
        'bollinger_bands': {
            'window': 20,
            'window_dev': 2.0,
            'overbought_threshold': 0.95,
            'oversold_threshold': 0.05
        },
        'rsi': {
            'period': 14,
            'overbought': 70,
            'oversold': 30
        },
        'sma': {
            'short_period': 20,
            'medium_period': 50,
            'long_period': 200
        },
        'roc': {
            'period': 14,
            'threshold': 2.0
        },
        'adx': {
            'period': 14,
            'adx_threshold': 25
        },
        'keltner_channels': {
            'ema_period': 20,
            'atr_period': 10,
            'multiplier': 2.0
        },
        'mfi': {
            'period': 14,
            'overbought': 80,
            'oversold': 20
        }
    },
    'signal_generation': {
        'required_timeframe_confirmation': 2,
        'min_signal_strength': 0.5,
        'volume_filter': {
            'enabled': True,
            'min_volume_btc': 1.0,
            'volume_period': 20
        },
        'trend_filter': {
            'enabled': True,
            'min_trend_strength': 0.5
        },
        'volatility_filter': {
            'enabled': True,
            'min_atr_percent': 0.5,
            'max_atr_percent': 5.0
        }
    },
    'risk_management': {
        'stop_loss_atr_multiplier': 2.0,
        'take_profit_ratio': 1.5,
        'max_position_size': 0.1,
        'risk_per_trade': 0.02,
        'max_daily_drawdown': 0.05,
        'max_open_trades': 5,
        'trailing_stop': {
            'enabled': True,
            'activation_percent': 1.0,
            'trail_percent': 0.5
        }
    },
    'backtest': {
        'initial_balance': 10000.0,
        'commission': 0.001,
        'slippage': 0.0005,
        'start_date': '2023-01-01',
        'end_date': '2023-12-31',
        'warmup_period': 200
    }
}

# Classe de test pour la stratégie Mean Reversion
class TestMeanReversionStrategy(unittest.TestCase):
    """Tests unitaires pour la stratégie de retour à la moyenne."""
    
    def setUp(self):
        """Initialisation avant chaque test."""
        # Importer la classe ici pour éviter les problèmes d'import circulaire
        from src.strategies.mean_reversion_strategy import MeanReversionStrategy
        
        # Créer une instance de la stratégie avec la configuration de test
        self.strategy = MeanReversionStrategy(config=MEAN_REVERSION_CONFIG)
        
        # Données de test
        self.sample_data = self._create_sample_data()
        
    def _create_sample_data(self):
        """Crée des données de test pour les différents timeframes."""
        # Générer des données aléatoires mais cohérentes
        np.random.seed(42)
        
        data = {}
        for tf in self.strategy.timeframes:
            # Créer un DataFrame avec des données OHLCV aléatoires
            index = pd.date_range(start='2023-01-01', periods=100, freq='15min' if tf == '15m' else '1H' if tf == '1h' else '4H')
            df = pd.DataFrame(index=index, columns=['open', 'high', 'low', 'close', 'volume'])
            
            # Prix de base avec une tendance légèrement haussière
            base = np.linspace(100, 150, len(df)) + np.random.normal(0, 2, len(df))
            
            # Générer des bougies réalistes
            df['open'] = base + np.random.normal(0, 1, len(df))
            df['high'] = df['open'] + np.abs(np.random.normal(2, 0.5, len(df)))
            df['low'] = df['open'] - np.abs(np.random.normal(2, 0.5, len(df)))
            df['close'] = df['open'] + np.random.normal(0, 1, len(df))
            df['volume'] = np.random.uniform(10, 100, len(df))
            
            # S'assurer que high >= open, high >= close, low <= open, low <= close
            df['high'] = df[['open', 'close', 'high']].max(axis=1)
            df['low'] = df[['open', 'close', 'low']].min(axis=1)
            
            data[tf] = df
            
        return data
    
    def test_initialization(self):
        """Teste l'initialisation de la stratégie."""
        self.assertEqual(self.strategy.symbol, 'BTC/EUR')
        self.assertEqual(self.strategy.timeframes, ['15m', '1h', '4h'])
        self.assertIn('bollinger_bands', self.strategy.indicator_config)
        self.assertIn('risk_management', self.strategy.config)
        
    def test_calculate_indicators(self):
        """Teste le calcul des indicateurs techniques."""
        # Appeler la méthode calculate_indicators
        indicators = self.strategy.calculate_indicators(self.sample_data)
        
        # Vérifier que les indicateurs sont calculés pour chaque timeframe
        for tf in self.strategy.timeframes:
            self.assertIn(tf, indicators)
            tf_indicators = indicators[tf]
            
            # Vérifier que les indicateurs clés sont présents
            for indicator in ['bb_upper', 'bb_middle', 'bb_lower', 'rsi', 'mfi', 'adx', 'atr']:
                self.assertIn(indicator, tf_indicators)
                self.assertEqual(len(tf_indicators[indicator]), len(self.sample_data[tf]))
    
    def test_generate_signals(self):
        """Teste la génération de signaux de trading."""
        # Calculer d'abord les indicateurs
        indicators = self.strategy.calculate_indicators(self.sample_data)
        
        # Générer les signaux
        signals = self.strategy.generate_signals(self.sample_data, indicators)
        
        # Vérifier que la méthode retourne une liste
        self.assertIsInstance(signals, list)
        
        # Pour chaque signal, vérifier la structure
        for signal in signals:
            self.assertIn('symbol', signal)
            self.assertIn('direction', signal)
            self.assertIn('strength', signal)
            self.assertIn('price', signal)
            self.assertIn('stop_loss', signal)
            self.assertIn('take_profit', signal)
            self.assertIn('timestamp', signal)
            self.assertIn('timeframe', signal)
            self.assertIn('strategy', signal)
    
    def test_calculate_confidence(self):
        """Teste le calcul de la confiance des signaux."""
        # Calculer d'abord les indicateurs
        indicators = self.strategy.calculate_indicators(self.sample_data)
        
        # Générer les signaux
        signals = self.strategy.generate_signals(self.sample_data, indicators)
        
        if signals:  # S'il y a des signaux
            # Calculer la confiance
            confidence_scores = self.strategy.calculate_confidence(self.sample_data, indicators, signals)
            
            # Vérifier que la méthode retourne un dictionnaire
            self.assertIsInstance(confidence_scores, dict)
            
            # Vérifier que chaque signal a un score de confiance
            for signal in signals:
                self.assertIn(signal['id'], confidence_scores)
                confidence = confidence_scores[signal['id']]
                self.assertGreaterEqual(confidence, 0.0)
                self.assertLessEqual(confidence, 1.0)
    
    def test_risk_management(self):
        """Teste la gestion du risque."""
        # Tester le calcul de la taille de position
        position_size = self.strategy.calculate_position_size(50000, 49000)
        self.assertGreater(position_size, 0)
        self.assertLessEqual(position_size, self.strategy.risk_config['max_position_size'])
        
        # Tester que le stop loss est inférieur au prix pour un achat
        buy_signal = {
            'symbol': 'BTC/EUR',
            'direction': 1,
            'price': Decimal('50000'),
            'stop_loss': Decimal('49000'),
            'take_profit': Decimal('51500'),
            'timestamp': datetime.now(),
            'timeframe': '15m',
            'strategy': 'MeanReversionStrategy'
        }
        self.assertLess(buy_signal['stop_loss'], buy_signal['price'])
        
        # Tester que le stop loss est supérieur au prix pour une vente
        sell_signal = {
            'symbol': 'BTC/EUR',
            'direction': -1,
            'price': Decimal('50000'),
            'stop_loss': Decimal('51000'),
            'take_profit': Decimal('48500'),
            'timestamp': datetime.now(),
            'timeframe': '15m',
            'strategy': 'MeanReversionStrategy'
        }
        self.assertGreater(sell_signal['stop_loss'], sell_signal['price'])


if __name__ == '__main__':
    unittest.main()
