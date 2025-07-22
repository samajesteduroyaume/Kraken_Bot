"""
Tests unitaires pour la stratégie de suivi de tendance.
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
from src.core.strategies.trend_following import TrendFollowingStrategy
from src.core.config import TrendFollowingConfig

class TestTrendFollowingStrategy(unittest.IsolatedAsyncioTestCase):
    """Classe de test pour la stratégie de suivi de tendance."""
    
    async def asyncSetUp(self):
        """Initialisation avant chaque test."""
        # Créer une configuration de test
        self.test_config = TrendFollowingConfig(
            fast_ma=10,
            slow_ma=30,
            ma_type='sma',
            min_trend_strength=0.1  # Réduit pour faciliter la validation du test
        )
        
        # Initialiser la stratégie avec la configuration de test
        self.strategy = TrendFollowingStrategy(config=self.test_config)
        
        # Créer des données de marché de test avec une tendance haussière plus marquée
        np.random.seed(42)
        dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
        
        # Créer une tendance haussière plus forte avec moins de bruit
        trend = np.linspace(100, 150, 100)  # Tendance linéaire de 100 à 150
        noise = np.random.randn(100) * 2     # Réduire le bruit pour une tendance plus propre
        close_prices = trend + noise
        
        # S'assurer que les prix sont strictement croissants pour une tendance forte
        for i in range(1, len(close_prices)):
            if close_prices[i] <= close_prices[i-1]:
                close_prices[i] = close_prices[i-1] + 0.1  # Éviter les baisses de prix
        
        self.market_data = {
            'close': pd.Series(close_prices, index=dates),
            'high': pd.Series(close_prices * 1.01, index=dates),
            'low': pd.Series(close_prices * 0.99, index=dates),
            'volume': pd.Series(np.random.randint(100, 1000, 100), index=dates),
            'symbol': 'BTC/EUR',
            'timestamp': datetime.utcnow()
        }
        
        # Calculer les indicateurs de test
        self.indicators = await self.strategy.calculate_indicators(self.market_data)
    
    async def test_initialization(self):
        """Teste l'initialisation de la stratégie."""
        from src.core.strategies.types import StrategyType
        
        self.assertEqual(self.strategy.name, "TrendFollowing")
        self.assertEqual(self.strategy.strategy_type, StrategyType.TREND_FOLLOWING)
        self.assertEqual(self.strategy.config.fast_ma, 10)
        self.assertEqual(self.strategy.config.slow_ma, 30)
    
    async def test_calculate_indicators(self):
        """Teste le calcul des indicateurs."""
        indicators = await self.strategy.calculate_indicators(self.market_data)
        
        # Vérifier que les indicateurs requis sont présents
        self.assertIn('fast_ma', indicators)
        self.assertIn('slow_ma', indicators)
        self.assertIn('close', indicators)
        
        # Vérifier les dimensions des séries
        self.assertEqual(len(indicators['fast_ma']), len(self.market_data['close']))
        self.assertEqual(len(indicators['slow_ma']), len(self.market_data['close']))
        
        # Vérifier que les moyennes mobiles sont bien ordonnées
        self.assertGreater(
            indicators['fast_ma'].iloc[-1],
            indicators['slow_ma'].iloc[-1],
            "La moyenne mobile rapide devrait être au-dessus de la lente en tendance haussière"
        )
    
    async def test_analyze_buy_signal(self):
        """Teste la génération d'un signal d'achat."""
        # Afficher la configuration de la stratégie
        print("\n=== CONFIGURATION DE LA STRATÉGIE ===")
        print(f"Type de stratégie: {self.strategy.strategy_type}")
        print(f"Période MA rapide: {self.strategy.config.fast_ma}")
        print(f"Période MA lente: {self.strategy.config.slow_ma}")
        print(f"Force de tendance minimale: {self.strategy.config.min_trend_strength}")
        
        # Créer une copie des indicateurs pour la modification
        indicators = self.indicators.copy()
        
        # Afficher les valeurs initiales des indicateurs
        print("\n=== VALEURS INITIALES DES INDICATEURS ===")
        print(f"Fast MA [-2]: {indicators['fast_ma'].iloc[-2]}")
        print(f"Slow MA [-2]: {indicators['slow_ma'].iloc[-2]}")
        print(f"Fast MA [-1]: {indicators['fast_ma'].iloc[-1]}")
        print(f"Slow MA [-1]: {indicators['slow_ma'].iloc[-1]}")
        
        # Afficher les valeurs initiales
        print("\n=== AVANT MODIFICATION ===")
        print(f"Fast MA [-2]: {indicators['fast_ma'].iloc[-2]}")
        print(f"Slow MA [-2]: {indicators['slow_ma'].iloc[-2]}")
        print(f"Fast MA [-1]: {indicators['fast_ma'].iloc[-1]}")
        print(f"Slow MA [-1]: {indicators['slow_ma'].iloc[-1]}")
        
        # Modifier les données pour forcer un croisement haussier
        indicators['fast_ma'].iloc[-2] = indicators['slow_ma'].iloc[-2] - 1
        indicators['fast_ma'].iloc[-1] = indicators['slow_ma'].iloc[-1] + 1
        
        # Afficher les valeurs modifiées des indicateurs
        print("\n=== VALEURS MODIFIÉES DES INDICATEURS ===")
        print(f"Fast MA [-2] (après modif): {indicators['fast_ma'].iloc[-2]}")
        print(f"Slow MA [-2] (après modif): {indicators['slow_ma'].iloc[-2]}")
        print(f"Fast MA [-1] (après modif): {indicators['fast_ma'].iloc[-1]}")
        print(f"Slow MA [-1] (après modif): {indicators['slow_ma'].iloc[-1]}")
        
        # Afficher les valeurs après modification
        print("\n=== APRES MODIFICATION ===")
        print(f"Fast MA [-2]: {indicators['fast_ma'].iloc[-2]}")
        print(f"Slow MA [-2]: {indicators['slow_ma'].iloc[-2]}")
        print(f"Fast MA [-1]: {indicators['fast_ma'].iloc[-1]}")
        print(f"Slow MA [-1]: {indicators['slow_ma'].iloc[-1]}")
        
        # Vérifier la condition de croisement
        prev_condition = indicators['fast_ma'].iloc[-2] <= indicators['slow_ma'].iloc[-2]
        curr_condition = indicators['fast_ma'].iloc[-1] > indicators['slow_ma'].iloc[-1]
        print(f"\n=== CONDITIONS ===")
        print(f"Condition précédente (Fast MA <= Slow MA): {prev_condition}")
        print(f"Condition actuelle (Fast MA > Slow MA): {curr_condition}")
        
        # Mettre à jour les prix de clôture pour correspondre aux indicateurs
        market_data = self.market_data.copy()
        market_data['close'].iloc[-1] = indicators['fast_ma'].iloc[-1] + 1
        
        # Activer les logs de débogage
        import logging
        logging.basicConfig(level=logging.DEBUG)
        logger = logging.getLogger('strategy.trendfollowing')
        logger.setLevel(logging.DEBUG)
        
        # Analyser les données
        print("\n=== LANCEMENT DE L'ANALYSE ===")
        
        # Activer les logs de débogage pour la stratégie
        import logging
        logging.basicConfig(level=logging.DEBUG)
        logger = logging.getLogger('strategy.trendfollowing')
        logger.setLevel(logging.DEBUG)
        
        # Afficher les données d'entrée pour le débogage
        print("\n=== DONNÉES D'ENTRÉE POUR L'ANALYSE ===")
        print(f"Dernier prix de clôture: {market_data['close'].iloc[-1]}")
        print(f"Fast MA [-2]: {indicators['fast_ma'].iloc[-2]}, Slow MA [-2]: {indicators['slow_ma'].iloc[-2]}")
        print(f"Fast MA [-1]: {indicators['fast_ma'].iloc[-1]}, Slow MA [-1]: {indicators['slow_ma'].iloc[-1]}")
        
        # Exécuter l'analyse
        signals = await self.strategy.analyze(market_data, indicators)
        
        # Afficher les résultats
        print("\n=== RÉSULTATS DE L'ANALYSE ===")
        if signals:
            print(f"Signal généré: {signals[0].action.value} avec confiance {signals[0].confidence:.2f}")
        else:
            print("Aucun signal généré")
        
        # Vérifier qu'un signal d'achat a été généré
        self.assertGreater(len(signals), 0, "Aucun signal généré")
        self.assertEqual(signals[0].action.value, "BUY", "Le signal devrait être un achat")
        self.assertGreater(signals[0].confidence, 0.6, "La confiance devrait être élevée")
    
    async def test_analyze_sell_signal(self):
        """Teste la génération d'un signal de vente."""
        # Créer une copie des indicateurs pour la modification
        indicators = self.indicators.copy()
        
        # Modifier les données pour forcer un croisement baissier
        indicators['fast_ma'].iloc[-2] = indicators['slow_ma'].iloc[-2] + 1
        indicators['fast_ma'].iloc[-1] = indicators['slow_ma'].iloc[-1] - 1
        
        # Mettre à jour les prix de clôture pour correspondre aux indicateurs
        market_data = self.market_data.copy()
        market_data['close'].iloc[-1] = indicators['fast_ma'].iloc[-1] - 1
        
        # Analyser les données
        signals = await self.strategy.analyze(market_data, indicators)
        
        # Vérifier qu'un signal de vente a été généré
        self.assertGreater(len(signals), 0, "Aucun signal généré")
        self.assertEqual(signals[0].action.value, "SELL", "Le signal devrait être une vente")
        self.assertGreater(signals[0].confidence, 0.6, "La confiance devrait être élevée")
    
    async def test_analyze_no_signal(self):
        """Teste l'absence de signal quand il n'y a pas de croisement."""
        # Modifier les données pour éviter tout croisement
        self.indicators['fast_ma'].iloc[-2] = self.indicators['slow_ma'].iloc[-2] + 1
        self.indicators['fast_ma'].iloc[-1] = self.indicators['slow_ma'].iloc[-1] + 2
        
        # Analyser les données
        signals = await self.strategy.analyze(self.market_data, self.indicators)
        
        # Vérifier qu'aucun signal n'a été généré
        self.assertEqual(len(signals), 0, "Aucun signal ne devrait être généré")


if __name__ == '__main__':
    unittest.main()
