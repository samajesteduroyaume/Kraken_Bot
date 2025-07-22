"""
Tests unitaires pour la stratégie de Grid Trading.
"""
import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from decimal import Decimal
from enum import Enum
from dataclasses import dataclass
from typing import Dict, Any, List, Optional

# Configuration de test pour la stratégie Grid Trading
GRID_TRADING_CONFIG = {
    "strategy_name": "GridTradingStrategy",
    "enabled": True,
    "is_real_time": True,
    "timeframes": ["15m", "1h"],
    "symbols": ["BTC/USD", "ETH/USD"],
    "parameters": {
        "grid_levels": 5,
        "grid_spacing_pct": 1.5,
        "take_profit_pct": 1.0,
        "stop_loss_pct": 2.0,
        "position_size_pct": 10.0,
        "max_drawdown_pct": 10.0,
        "volatility_lookback": 20,
        "volatility_threshold": 0.5,
        "risk_multiplier": 1.0
    },
    "risk_management": {
        "max_position_size_pct": 10.0,
        "max_drawdown_pct": 10.0,
        "stop_loss_type": "percentage",
        "take_profit_ratio": 1.0
    }
}

# Classes utilitaires pour les tests
class SignalAction(Enum):
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"

@dataclass
class TradingSignal:
    symbol: str
    action: SignalAction
    price: float
    take_profit: Optional[float] = None
    stop_loss: Optional[float] = None
    size: Optional[float] = None
    strategy: Optional[str] = None
    timestamp: Optional[datetime] = None
    metadata: Optional[Dict[str, Any]] = None

# Mock de la stratégie pour les tests
class MockGridTradingStrategy:
    """Mock de la stratégie Grid Trading pour les tests."""
    
    def __init__(self, config=None):
        self.config = config or {}
        self.parameters = self.config.get('parameters', {})
        self.current_grid = []
        self.last_grid_update = None
        self.current_price = 0.0
        self.volatility = 0.0
        self.signals = []
    
    def _generate_grid_levels(self, current_price: float) -> List[Dict]:
        """Génère les niveaux de la grille pour les tests."""
        if current_price <= 0:
            return []
            
        grid = []
        grid_spacing = self.parameters.get('grid_spacing_pct', 1.5) / 100
        take_profit = self.parameters.get('take_profit_pct', 1.0) / 100
        stop_loss = self.parameters.get('stop_loss_pct', 2.0) / 100
        position_size = self.parameters.get('position_size_pct', 10.0) / 100
        grid_levels = self.parameters.get('grid_levels', 5)
        
        # Niveaux d'achat (en dessous du prix actuel)
        for i in range(1, grid_levels + 1):
            level_price = current_price * (1 - i * grid_spacing)
            if level_price <= 0:
                continue
                
            grid.append({
                'type': 'buy',
                'level': i,
                'price': level_price,
                'take_profit': level_price * (1 + take_profit),
                'stop_loss': level_price * (1 - stop_loss),
                'position_size': position_size,
                'active': True
            })
        
        # Niveaux de vente (au-dessus du prix actuel)
        for i in range(1, grid_levels + 1):
            level_price = current_price * (1 + i * grid_spacing)
            
            grid.append({
                'type': 'sell',
                'level': i,
                'price': level_price,
                'take_profit': level_price * (1 - take_profit),
                'stop_loss': level_price * (1 + stop_loss),
                'position_size': position_size,
                'active': True
            })
        
        # Trier la grille par prix (du plus bas au plus haut)
        grid.sort(key=lambda x: x['price'])
        return grid
    
    def update_grid(self, current_price: float):
        """Met à jour la grille avec un nouveau prix."""
        self.current_price = current_price
        self.current_grid = self._generate_grid_levels(current_price)
        self.last_grid_update = datetime.utcnow()

class TestGridTradingStrategy(unittest.TestCase):
    """Classe de tests pour la stratégie de Grid Trading."""
    
    @classmethod
    def setUpClass(cls):
        """Configuration initiale pour les tests."""
        cls.strategy = MockGridTradingStrategy(GRID_TRADING_CONFIG)
        
        # Créer un jeu de données de test
        np.random.seed(42)
        dates = pd.date_range(start='2023-01-01', periods=100, freq='15min')
        prices = np.cumsum(np.random.randn(100) * 0.1 + 0.01) + 100
        
        cls.sample_data = pd.DataFrame({
            'timestamp': dates,
            'open': prices * 0.99,
            'high': prices * 1.01,
            'low': prices * 0.98,
            'close': prices,
            'volume': np.random.randint(100, 1000, size=100)
        })
        
        # Définir le prix actuel pour les tests
        cls.current_price = prices[-1]
        cls.strategy.update_grid(cls.current_price)
    
    def test_initialization(self):
        """Teste l'initialisation de la stratégie."""
        self.assertIsNotNone(self.strategy.config)
        self.assertEqual(self.strategy.parameters['grid_levels'], 5)
        self.assertEqual(self.strategy.parameters['grid_spacing_pct'], 1.5)
        self.assertEqual(self.strategy.parameters['take_profit_pct'], 1.0)
        self.assertEqual(self.strategy.parameters['stop_loss_pct'], 2.0)
        self.assertEqual(self.strategy.parameters['position_size_pct'], 10.0)
    
    def test_grid_generation(self):
        """Teste la génération des niveaux de grille."""
        # Vérifier que la grille a été générée
        self.assertGreater(len(self.strategy.current_grid), 0)
        
        # Vérifier que la grille est triée par prix
        prices = [level['price'] for level in self.strategy.current_grid]
        self.assertEqual(prices, sorted(prices))
        
        # Vérifier que les niveaux d'achat sont en dessous du prix actuel
        buy_levels = [level for level in self.strategy.current_grid if level['type'] == 'buy']
        for level in buy_levels:
            self.assertLessEqual(level['price'], self.current_price)
        
        # Vérifier que les niveaux de vente sont au-dessus du prix actuel
        sell_levels = [level for level in self.strategy.current_grid if level['type'] == 'sell']
        for level in sell_levels:
            self.assertGreaterEqual(level['price'], self.current_price)
    
    def test_grid_levels_calculation(self):
        """Teste le calcul des niveaux de grille."""
        # Tester avec un prix de 100 et un espacement de 1.5%
        test_price = 100.0
        self.strategy.update_grid(test_price)
        
        # Récupérer les niveaux d'achat et de vente
        buy_levels = [level for level in self.strategy.current_grid if level['type'] == 'buy']
        sell_levels = [level for level in self.strategy.current_grid if level['type'] == 'sell']
        
        # Vérifier que nous avons le bon nombre de niveaux d'achat et de vente
        self.assertEqual(len(buy_levels), 5)  # 5 niveaux d'achat
        self.assertEqual(len(sell_levels), 5)  # 5 niveaux de vente
        
        # Trier les niveaux d'achat par niveau (du plus proche au plus éloigné du prix actuel)
        buy_levels_sorted = sorted(buy_levels, key=lambda x: x['level'])
        
        # Vérifier que les prix des niveaux d'achat sont décroissants (du plus haut au plus bas)
        for i in range(1, len(buy_levels_sorted)):
            prev_price = buy_levels_sorted[i-1]['price']
            curr_price = buy_levels_sorted[i]['price']
            self.assertLess(curr_price, prev_price, 
                          f"Le niveau d'achat {i} ({curr_price}) devrait être inférieur au niveau {i-1} ({prev_price})")
        
        # Trier les niveaux de vente par niveau (du plus proche au plus éloigné du prix actuel)
        sell_levels_sorted = sorted(sell_levels, key=lambda x: x['level'])
        
        # Vérifier que les prix des niveaux de vente sont croissants (du plus bas au plus haut)
        for i in range(1, len(sell_levels_sorted)):
            prev_price = sell_levels_sorted[i-1]['price']
            curr_price = sell_levels_sorted[i]['price']
            self.assertGreater(curr_price, prev_price,
                             f"Le niveau de vente {i} ({curr_price}) devrait être supérieur au niveau {i-1} ({prev_price})")
    
    def test_take_profit_and_stop_loss(self):
        """Teste le calcul des niveaux de take profit et stop loss."""
        for level in self.strategy.current_grid:
            if level['type'] == 'buy':
                # Pour les ordres d'achat, le TP doit être au-dessus du prix d'entrée
                self.assertGreater(level['take_profit'], level['price'])
                # Et le SL doit être en dessous
                self.assertLess(level['stop_loss'], level['price'])
            else:  # sell
                # Pour les ordres de vente, le TP doit être en dessous du prix d'entrée
                self.assertLess(level['take_profit'], level['price'])
                # Et le SL doit être au-dessus
                self.assertGreater(level['stop_loss'], level['price'])
    
    def test_grid_update(self):
        """Teste la mise à jour de la grille avec un nouveau prix."""
        # Sauvegarder l'ancienne grille
        old_grid = self.strategy.current_grid.copy()
        
        # Mettre à jour avec un nouveau prix
        new_price = self.current_price * 1.05  # +5%
        self.strategy.update_grid(new_price)
        
        # Vérifier que la grille a été mise à jour
        self.assertNotEqual(len(self.strategy.current_grid), 0)
        
        # Vérifier que les prix ont été mis à jour
        for level in self.strategy.current_grid:
            if level['type'] == 'buy':
                self.assertLess(level['price'], new_price)
            else:  # sell
                self.assertGreater(level['price'], new_price)


if __name__ == '__main__':
    unittest.main()
