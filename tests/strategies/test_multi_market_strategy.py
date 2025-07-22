import pytest
from unittest.mock import MagicMock, patch
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

class TestMultiMarketStrategy:
    """Tests pour la stratégie multi-marchés."""

    @pytest.fixture
    def setup_multi_market(self):
        """Configuration de base pour les tests multi-marchés."""
        from src.strategies.multi_market_strategy import MultiMarketStrategy
        
        # Configuration de base
        config = {
            'enabled': True,
            'weight': 0.4,
            'risk_multiplier': 1.2,
            'market_groups': [
                {
                    'name': 'majors',
                    'symbols': ['BTC/USD', 'ETH/USD'],
                    'max_exposure': 0.5  # 50% du capital maximum pour ce groupe
                },
                {
                    'name': 'altcoins',
                    'symbols': ['XRP/USD', 'LTC/USD', 'ADA/USD'],
                    'max_exposure': 0.3
                }
            ],
            'diversification': {
                'max_per_market': 0.2,  # 20% max par marché
                'min_correlation': -0.3  # Corrélation minimale souhaitée
            }
        }
        
        # Données de marché simulées
        market_data = {
            'BTC/USD': pd.DataFrame({
                'close': [50000, 51000, 49000, 49500],
                'volume': [100, 120, 90, 110]
            }),
            'ETH/USD': pd.DataFrame({
                'close': [3000, 3100, 2900, 2950],
                'volume': [500, 520, 480, 510]
            })
        }
        
        # Corrélations simulées
        correlations = {
            ('BTC/USD', 'ETH/USD'): 0.85,
            ('BTC/USD', 'XRP/USD'): 0.65,
            ('ETH/USD', 'XRP/USD'): 0.6
        }
        
        strategy = MultiMarketStrategy(config)
        
        # Mock des dépendances
        strategy.data_provider = MagicMock()
        strategy.data_provider.get_correlation.side_effect = lambda x, y: correlations.get((x, y), 0.1)
        strategy.data_provider.get_historical_data.return_value = market_data
        
        return strategy, market_data

    def test_initialization(self, setup_multi_market):
        """Teste l'initialisation de la stratégie multi-marchés."""
        strategy, _ = setup_multi_market
        
        assert strategy.enabled is True
        assert strategy.weight == 0.4
        assert len(strategy.market_groups) == 2
        assert strategy.diversification['max_per_market'] == 0.2

    def test_calculate_correlations(self, setup_multi_market):
        """Teste le calcul des corrélations entre marchés."""
        strategy, _ = setup_multi_market
        
        # Appel de la méthode à tester
        corr_matrix = strategy.calculate_correlations()
        
        # Vérifications
        assert isinstance(corr_matrix, pd.DataFrame)
        assert 'BTC/USD' in corr_matrix.columns
        assert 'ETH/USD' in corr_matrix.columns
        assert 0.8 < corr_matrix.loc['BTC/USD', 'ETH/USD'] < 0.9

    def test_allocate_capital(self, setup_multi_market):
        """Teste l'allocation de capital entre les marchés."""
        strategy, _ = setup_multi_market
        portfolio_value = 10000
        
        # Appel de la méthode à tester
        allocations = strategy.allocate_capital(portfolio_value)
        
        # Vérifications
        assert isinstance(allocations, dict)
        assert sum(allocations.values()) <= portfolio_value
        
        # Vérification des limites d'exposition par groupe
        majors_allocation = sum(v for k, v in allocations.items() if k in ['BTC/USD', 'ETH/USD'])
        assert majors_allocation <= 5000  # 50% de 10000

    def test_generate_signals(self, setup_multi_market):
        """Teste la génération de signaux multi-marchés."""
        strategy, market_data = setup_multi_market
        
        # Configuration des retours simulés pour les indicateurs
        strategy.data_provider.calculate_indicators.return_value = {
            'BTC/USD': {'rsi': 45, 'macd': 12.5},
            'ETH/USD': {'rsi': 55, 'macd': 8.2}
        }
        
        # Appel de la méthode à tester
        signals = strategy.generate_signals()
        
        # Vérifications
        assert isinstance(signals, dict)
        assert 'BTC/USD' in signals
        assert 'ETH/USD' in signals
        assert 'signal' in signals['BTC/USD']
        assert 'strength' in signals['BTC/USD']

    def test_risk_management(self, setup_multi_market):
        """Teste la gestion des risques multi-marchés."""
        strategy, _ = setup_multi_market
        portfolio = {
            'BTC/USD': {'position': 0.1, 'unrealized_pnl': 0.05},
            'ETH/USD': {'position': 0.15, 'unrealized_pnl': -0.02}
        }
        
        # Appel de la méthode à tester
        risk_assessment = strategy.assess_risk(portfolio)
        
        # Vérifications
        assert isinstance(risk_assessment, dict)
        assert 'total_risk' in risk_assessment
        assert 'per_market_risk' in risk_assessment
        assert risk_assessment['total_risk'] <= 1.0  # Ne peut pas dépasser 100%

    def test_diversification(self, setup_multi_market):
        """Teste la logique de diversification."""
        strategy, _ = setup_multi_market
        
        # Position actuelle concentrée sur BTC
        current_positions = {'BTC/USD': 0.4, 'ETH/USD': 0.1}
        
        # Appel de la méthode à tester
        rebalance_suggestions = strategy.calculate_rebalancing(current_positions)
        
        # Vérifications
        assert isinstance(rebalance_suggestions, dict)
        assert 'BTC/USD' in rebalance_suggestions
        assert rebalance_suggestions['BTC/USD'] < 0  # Devrait suggérer de réduire la position
        assert sum(abs(v) for v in rebalance_suggestions.values()) > 0

    @patch('src.strategies.multi_market_strategy.MultiMarketStrategy.calculate_correlations')
    def test_correlation_handling(self, mock_correlations, setup_multi_market):
        """Teste la gestion des corrélations entre marchés."""
        strategy, _ = setup_multi_market
        
        # Configuration du mock pour les corrélations
        corr_matrix = pd.DataFrame({
            'BTC/USD': [1.0, 0.9],
            'ETH/USD': [0.9, 1.0]
        }, index=['BTC/USD', 'ETH/USD'])
        mock_correlations.return_value = corr_matrix
        
        # Appel de la méthode à tester
        result = strategy.analyze_correlations()
        
        # Vérifications
        assert isinstance(result, dict)
        assert 'high_correlation_pairs' in result
        
        # Vérification que la paire fortement corrélée est présente
        pair_found = any(
            pair[0] == 'BTC/USD' and pair[1] == 'ETH/USD' 
            for pair in result['high_correlation_pairs']
        )
        assert pair_found, "La paire BTC/USD-ETH/USD devrait être détectée comme fortement corrélée"

    def test_performance_metrics(self, setup_multi_market):
        """Teste le calcul des métriques de performance."""
        strategy, _ = setup_multi_market
        
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

    def test_stress_test(self, setup_multi_market):
        """Test de résistance avec des données extrêmes."""
        strategy, _ = setup_multi_market
        
        # Données de marché extrêmes
        extreme_market_data = {
            'BTC/USD': pd.DataFrame({
                'close': [50000, 25000, 50000],  # Crash de 50% puis récupération
                'volume': [100, 500, 400]
            })
        }
        strategy.data_provider.get_historical_data.return_value = extreme_market_data
        
        # Configuration des retours simulés pour les indicateurs
        strategy.data_provider.calculate_indicators.return_value = {
            'BTC/USD': {'rsi': 25, 'macd': -100}  # Conditions de survente
        }
        
        # Appel de la méthode à tester
        signals = strategy.generate_signals()
        
        # Vérifications
        assert 'BTC/USD' in signals
        # Le signal devrait refléter une réaction au crash
        assert signals['BTC/USD']['signal'] in ['SELL', 'HOLD']

if __name__ == "__main__":
    pytest.main(["-v", "test_multi_market_strategy.py"])
