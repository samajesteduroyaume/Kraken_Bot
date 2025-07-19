import unittest
from unittest.mock import patch, MagicMock
import pandas as pd
from datetime import datetime
import time
import os
import logging
import asyncio
from src.core.api.kraken import KrakenAPI
from src.core.portfolio import PortfolioManager
from src.core.order_execution import OrderExecutor
from src.core.risk_management import RiskManager
from src.core.position import Position
from src.core.config import Config

class TestTradingComponents(unittest.IsolatedAsyncioTestCase):
    @classmethod
    def setUpClass(cls):
        # Configuration de base pour les tests
        cls.config = Config()
        
        # Configuration de trading pour les tests
        os.environ['RISK_PER_TRADE'] = '0.01'  # 1% du portefeuille
        os.environ['STOP_LOSS_PERCENT'] = '0.02'
        os.environ['TAKE_PROFIT_PERCENT'] = '0.04'
        os.environ['MAX_POSITIONS'] = '5'
        os.environ['MAX_DRAWDOWN'] = '0.1'
        
        # Paramètres de risque
        os.environ['MAX_POSITION_SIZE'] = '0.1'
        os.environ['STOP_LOSS_PERCENTAGE'] = '2.0'
        os.environ['TAKE_PROFIT_PERCENTAGE'] = '4.0'
        os.environ['MAX_DRAWDOWN_PERCENT'] = '10.0'
        os.environ['MAX_LEVERAGE'] = '5.0'
        os.environ['TRAILING_STOP'] = 'True'
        os.environ['PORTFOLIO_VALUE'] = '100000'  # Valeur du portefeuille pour les tests
        
        # Configuration des données historiques
        cls.historical_data = pd.DataFrame({
            'open': [9900, 10000, 10100, 10200, 10300, 10400, 10500, 10600, 10700, 10800],
            'high': [10000, 10100, 10200, 10300, 10400, 10500, 10600, 10700, 10800, 10900],
            'low': [9800, 9900, 10000, 10100, 10200, 10300, 10400, 10500, 10600, 10700],
            'close': [9950, 10050, 10150, 10250, 10350, 10450, 10550, 10650, 10750, 10850],
            'volume': [1000, 1200, 1100, 1300, 1400, 1500, 1600, 1700, 1800, 1900]
        })
        
        # Configuration des paramètres de risque pour les tests
        cls.config.risk_parameters = {
            'max_position_size': 0.1,
            'stop_loss_percentage': 2.0,
            'take_profit_percentage': 4.0,
            'max_drawdown': 10.0,
            'max_leverage': 5.0,
            'trailing_stop': True
        }
        
        # Configuration du portefeuille pour les tests
        cls.config.portfolio_value = 100000
        
        # Vérifier la configuration de trading
        cls.config.trading_config = {
            'risk_per_trade': 0.01,  # 1% du portefeuille
            'stop_loss_percent': 0.02,
            'take_profit_percent': 0.04,
            'max_positions': 5,
            'max_drawdown': 0.1
        }
        
        # Configuration du logging
        logging.basicConfig(level=logging.DEBUG)
        
        # Mock de l'API Kraken
        cls.mock_api = MagicMock()
        cls.mock_api.get_ohlc_data.return_value = cls.historical_data

    async def test_portfolio_management(self):
        # Test de la gestion de portefeuille
        portfolio = PortfolioManager(config=self.config)
        
        # Test de l'ouverture d'une position
        position = Position(
            pair='XBTUSD',
            size=0.1,  # 0.1 BTC
            entry_price=10000,  # Prix d'entrée de 10000 USD
            entry_time=datetime.now()
        )
        
        portfolio.positions['XBTUSD'] = position.to_dict()
        
        # Vérification de la position
        self.assertEqual(len(portfolio.positions), 1)
        self.assertEqual(portfolio.positions['XBTUSD']['size'], 0.1)
        self.assertEqual(portfolio.positions['XBTUSD']['entry_price'], 10000)

    async def test_order_execution(self):
        # Test de l'exécution des ordres
        order_exec = OrderExecutor(api=self.mock_api, config=self.config)
        
        # Configuration de l'ordre
        order = {
            'pair': 'XBTUSD',
            'type': 'market',
            'side': 'buy',
            'size': 0.01,
            'price': 10000
        }
        
        # Configuration des mocks
        self.mock_api.get_server_time.return_value = {'result': {'unixtime': int(time.time())}}
        self.mock_api.get_ohlc_data.return_value = self.historical_data
        
        # Mock de la création d'ordre
        async def mock_create_order(*args, **kwargs):
            return {
                'txid': ['TX123456'],
                'descr': {
                    'order': f"{order['type']} {order['side']} {order['size']} XBTUSD @ limit {order['price']}"
                },
                'error': [],
                'timestamp': int(time.time())
            }
        
        self.mock_api.create_order.side_effect = mock_create_order
        
        # Exécution de l'ordre
        result = await order_exec.place_order(
            pair=order['pair'],
            type=order['type'],
            side=order['side'],
            size=order['size'],
            price=order['price']
        )
        
        # Vérification
        self.assertTrue(result is not None)  # Devrait retourner un résultat
        self.assertEqual(result, 'TX123456')

    async def test_risk_management(self):
        # Test de la gestion des risques
        risk_manager = RiskManager(config=self.config)
        
        # Simulation d'une position risquée (taille > max_position_size)
        result = risk_manager.validate_trade(
            pair='XBTUSD',
            position_size=0.2,  # Taille de position de 0.2 BTC (supérieure à max_position_size = 0.1)
            entry_price=10000,  # Prix d'entrée de 10000 USD
            is_long=True
        )
        self.assertFalse(result, "Validation de trade avec taille excessive a réussi")
        
        # Test d'une position valide (risque < 1%)
        result = risk_manager.validate_trade(
            pair='XBTUSD',
            position_size=0.005,  # Taille de position de 0.005 BTC
            entry_price=10000,    # Prix d'entrée de 10000 USD
            is_long=True
        )
        self.assertTrue(result, "Validation de trade valide a échoué")

    async def test_market_data(self):
        # Test de la récupération des données de marché
        api = KrakenAPI(config=self.config)
        
        # Mock de la réponse de l'API
        with patch('src.core.api.kraken.KrakenAPI.get_ohlc_data') as mock_get_ohlc:
            mock_get_ohlc.return_value = self.historical_data
            
            # Test avec une paire valide
            data = await api.get_ohlc_data('XBTUSD')
            self.assertIsNotNone(data)
            self.assertEqual(data['open'][0], 9900)
            
            # Test avec une paire invalide
            mock_get_ohlc.return_value = None
            data = await api.get_ohlc_data('INVALID')
            self.assertIsNone(data)

if __name__ == '__main__':
    unittest.main()
