import pytest
from unittest.mock import MagicMock, patch
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

class TestMultiTimeframeStrategy:
    """Tests pour la stratégie multi-timeframes."""

    @pytest.fixture
    def setup_multi_timeframe(self):
        """Configuration de base pour les tests multi-timeframes."""
        from src.strategies.multi_timeframe_strategy import MultiTimeframeStrategy
        
        # Configuration de base
        config = {
            'enabled': True,
            'weight': 0.3,
            'risk_multiplier': 0.8,
            'timeframes': [
                {
                    'interval': '15m',
                    'weight': 0.2,
                    'indicators': ['rsi', 'bbands']
                },
                {
                    'interval': '1h',
                    'weight': 0.3,
                    'indicators': ['ema', 'macd']
                },
                {
                    'interval': '4h',
                    'weight': 0.5,
                    'indicators': ['ichimoku', 'atr']
                }
            ],
            'confirmation_rules': {
                'min_timeframes': 2,
                'priority': 'higher'
            }
        }
        
        # Données de marché simulées pour différents timeframes
        market_data = {
            '15m': pd.DataFrame({
                'close': [50000, 50100, 50200, 50300, 50400],
                'volume': [100, 120, 115, 130, 125],
                'timestamp': pd.date_range(end=datetime.now(), periods=5, freq='15min')
            }),
            '1h': pd.DataFrame({
                'close': [49500, 49800, 50200, 50500, 50800],
                'volume': [450, 460, 470, 480, 490],
                'timestamp': pd.date_range(end=datetime.now(), periods=5, freq='1h')
            }),
            '4h': pd.DataFrame({
                'close': [49000, 49500, 50000, 50500, 51000],
                'volume': [1800, 1850, 1900, 1950, 2000],
                'timestamp': pd.date_range(end=datetime.now(), periods=5, freq='4h')
            })
        }
        
        # Valeurs simulées pour les indicateurs
        indicator_values = {
            '15m': {
                'rsi': 65.5,
                'bbands_upper': 50600,
                'bbands_middle': 50300,
                'bbands_lower': 50000
            },
            '1h': {
                'ema_20': 50000,
                'ema_50': 49500,
                'macd': 150,
                'macd_signal': 100
            },
            '4h': {
                'ichimoku_conversion': 49800,
                'ichimoku_base': 49600,
                'ichimoku_span_a': 49700,
                'ichimoku_span_b': 49500,
                'atr': 800
            }
        }
        
        strategy = MultiTimeframeStrategy(config)
        
        # Mock des dépendances
        strategy.data_provider = MagicMock()
        strategy.data_provider.get_historical_data.side_effect = lambda symbol, tf, **kwargs: market_data.get(tf)
        strategy.data_provider.calculate_indicators.side_effect = lambda df, indicators: indicator_values.get(indicators[0]['timeframe'], {})
        
        return strategy, market_data, indicator_values

    def test_initialization(self, setup_multi_timeframe):
        """Teste l'initialisation de la stratégie multi-timeframes."""
        strategy, _, _ = setup_multi_timeframe
        
        assert strategy.enabled is True
        assert strategy.weight == 0.3
        assert len(strategy.timeframes) == 3
        assert strategy.confirmation_rules['min_timeframes'] == 2

    def test_timeframe_weights(self, setup_multi_timeframe):
        """Teste la configuration des poids des timeframes."""
        strategy, _, _ = setup_multi_timeframe
        
        # Vérifie que les poids sont correctement normalisés
        total_weight = sum(tf.weight for tf in strategy.timeframes)
        assert abs(total_weight - 1.0) < 0.001  # Tolérance aux erreurs d'arrondi
        
        # Vérifie que les intervalles sont dans l'ordre croissant
        intervals = [tf.interval for tf in strategy.timeframes]
        assert intervals == ['15m', '1h', '4h']

    def test_analyze_timeframe(self, setup_multi_timeframe):
        """Teste l'analyse d'un timeframe individuel."""
        strategy, market_data, indicator_values = setup_multi_timeframe
        
        # Test pour le timeframe 15m
        tf_config = next(tf for tf in strategy.timeframes if tf.interval == '15m')
        analysis = strategy.analyze_timeframe('15m', tf_config, market_data['15m'])
        
        # Vérifications
        assert isinstance(analysis, dict)
        assert 'interval' in analysis
        assert 'signal' in analysis
        assert 'strength' in analysis
        assert 'indicators' in analysis
        assert analysis['interval'] == '15m'

    def test_combine_timeframe_signals(self, setup_multi_timeframe):
        """Teste la combinaison des signaux de différents timeframes."""
        strategy, _, _ = setup_multi_timeframe
        
        # Signaux simulés
        signals = [
            {'interval': '15m', 'signal': 'BUY', 'strength': 0.7},
            {'interval': '1h', 'signal': 'BUY', 'strength': 0.8},
            {'interval': '4h', 'signal': 'SELL', 'strength': 0.6}
        ]
        
        # Test avec priorité aux timeframes plus longs
        combined = strategy.combine_timeframe_signals(signals)
        assert isinstance(combined, dict)
        assert 'signal' in combined
        assert 'strength' in combined
        assert 'timeframe_contributions' in combined
        
        # Le signal final devrait être 'BUY' car la somme pondérée des forces des signaux 'BUY' est plus élevée
        # (0.7*0.2 + 0.8*0.3 = 0.38) > (0.6*0.5 = 0.3)
        assert combined['signal'] == 'BUY'
        assert 0 < combined['strength'] < 1.0

    def test_generate_signals(self, setup_multi_timeframe):
        """Teste la génération de signaux multi-timeframes."""
        strategy, market_data, _ = setup_multi_timeframe
        
        # Configure le mock pour renvoyer les données de marché appropriées
        strategy.data_provider.get_historical_data.side_effect = lambda symbol, interval, **kwargs: market_data.get(interval)
        
        # Configure le mock pour les indicateurs
        def mock_calculate_indicators(df, indicators):
            # Logique simplifiée pour les indicateurs
            if 'rsi' in indicators:
                return {'rsi': 65.0}
            elif 'macd' in indicators:
                return {'macd': 0.5, 'macd_signal': 0.3}
            elif 'atr' in indicators:
                return {'atr': 100.0}
            return {}
            
        strategy.data_provider.calculate_indicators.side_effect = mock_calculate_indicators
        
        # Appel de la méthode à tester
        signals = strategy.generate_signals('BTC/USD')
        
        # Vérifications
        assert isinstance(signals, dict)
        assert 'signal' in signals
        assert 'strength' in signals
        assert 'timeframe_analysis' in signals
        assert len(signals['timeframe_analysis']) == 3, f"Expected 3 timeframes, got {len(signals['timeframe_analysis'])}"

    def test_risk_assessment(self, setup_multi_timeframe):
        """Teste l'évaluation des risques multi-timeframes."""
        strategy, _, _ = setup_multi_timeframe
        
        # Analyse simulée des timeframes
        timeframe_analysis = [
            {'interval': '15m', 'volatility': 0.02, 'trend_strength': 0.7},
            {'interval': '1h', 'volatility': 0.018, 'trend_strength': 0.75},
            {'interval': '4h', 'volatility': 0.015, 'trend_strength': 0.8}
        ]
        
        # Appel de la méthode à tester
        risk = strategy.assess_risk(timeframe_analysis)
        
        # Vérifications
        assert isinstance(risk, dict)
        assert 'composite_volatility' in risk
        assert 'composite_trend' in risk
        assert 'position_size' in risk
        assert 0 < risk['position_size'] <= 1.0

    def test_confirmation_rules(self, setup_multi_timeframe):
        """Teste les règles de confirmation entre timeframes."""
        strategy, _, _ = setup_multi_timeframe
        
        # Cas 1: Confirmation sur 2/3 timeframes
        signals1 = [
            {'interval': '15m', 'signal': 'BUY', 'strength': 0.7},
            {'interval': '1h', 'signal': 'BUY', 'strength': 0.8},
            {'interval': '4h', 'signal': 'SELL', 'strength': 0.6}
        ]
        confirmed1 = strategy.check_confirmation(signals1)
        assert confirmed1 is True  # 2/3 confirment l'achat
        
        # Cas 2: Pas de confirmation (1/3)
        signals2 = [
            {'interval': '15m', 'signal': 'BUY', 'strength': 0.7},
            {'interval': '1h', 'signal': 'HOLD', 'strength': 0.0},
            {'interval': '4h', 'signal': 'SELL', 'strength': 0.6}
        ]
        confirmed2 = strategy.check_confirmation(signals2)
        assert confirmed2 is False

    def test_performance_metrics(self, setup_multi_timeframe):
        """Teste le calcul des métriques de performance."""
        strategy, _, _ = setup_multi_timeframe
        
        # Données de performance simulées
        returns = pd.Series([0.01, -0.005, 0.02, 0.015])
        
        # Appel de la méthode à tester
        metrics = strategy.calculate_performance_metrics(returns)
        
        # Vérifications
        expected_metrics = ['total_return', 'volatility', 'sharpe', 'max_drawdown', 'win_rate']
        for metric in expected_metrics:
            assert metric in metrics, f"La métrique {metric} est manquante"
            assert isinstance(metrics[metric], (float, np.floating)), f"La métrique {metric} devrait être un nombre flottant"
        
        # Vérifications supplémentaires sur les valeurs
        assert -1 <= metrics['total_return'] <= 1, "Le rendement total devrait être entre -100% et 100%"
        assert metrics['volatility'] >= 0, "La volatilité ne peut pas être négative"
        assert 0 <= metrics['win_rate'] <= 1, "Le taux de réussite doit être entre 0 et 1"

    def test_stress_test(self, setup_multi_timeframe):
        """Test de résistance avec des données extrêmes."""
        strategy, market_data, indicator_values = setup_multi_timeframe
        
        # Données de marché extrêmes
        extreme_market_data = {
            '15m': pd.DataFrame({
                'close': [50000, 25000, 50000],  # Crash de 50% puis récupération
                'volume': [100, 500, 400],
                'timestamp': pd.date_range(end=datetime.now(), periods=3, freq='15min')
            })
        }
        strategy.data_provider.get_historical_data.side_effect = lambda symbol, tf, **kwargs: extreme_market_data.get(tf)
        
        # Configuration des retours simulés pour les indicateurs
        strategy.data_provider.calculate_indicators.return_value = {
            'rsi': 25,  # Conditions de survente
            'bbands_upper': 52000,
            'bbands_middle': 37500,
            'bbands_lower': 23000
        }
        
        # Appel de la méthode à tester
        signals = strategy.generate_signals('BTC/USD')
        
        # Vérifications
        assert 'signal' in signals
        assert signals['signal'] in ['BUY', 'SELL', 'HOLD']
        assert 0 <= signals['strength'] <= 1.0

if __name__ == "__main__":
    pytest.main(["-v", "test_multi_timeframe_strategy.py"])
