"""
Tests d'intégration pour le module order_book avec le reste du système.
"""
import asyncio
import unittest
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch
from typing import Dict, List, Any, Optional

import pandas as pd
import numpy as np

# Import des modules internes
from src.core.market.data_manager import MarketDataManager
from src.core.market.order_book import OrderBookManager, OrderBookSnapshot, PriceLevel, OrderBookMetrics
from src.core.trading.signals import SignalGenerator

# Définition des types pour les tests
Candle = Dict[str, float]
Trade = Dict[str, Any]

class TestOrderBookIntegration(unittest.IsolatedAsyncioTestCase):
    """Tests d'intégration pour le module order_book."""

    async def asyncSetUp(self):
        """Initialisation avant chaque test."""
        # Configuration du mock de l'API
        self.api = AsyncMock()
        self.api.get_order_book.return_value = {
            'XXBTZUSD': {
                'bids': [
                    [str(Decimal('50000.0')), str(Decimal('1.5')), 1234567890],
                    [str(Decimal('49950.0')), str(Decimal('2.0')), 1234567890],
                    [str(Decimal('49900.0')), str(Decimal('3.0')), 1234567890],
                ],
                'asks': [
                    [str(Decimal('50050.0')), str(Decimal('1.2')), 1234567890],
                    [str(Decimal('50100.0')), str(Decimal('2.5')), 1234567890],
                    [str(Decimal('50150.0')), str(Decimal('1.8')), 1234567890],
                ]
            }
        }
        
        # Initialisation du gestionnaire de données de marché
        self.market_data = MarketDataManager(self.api)
        
        # Initialisation du générateur de signaux
        self.signal_generator = SignalGenerator()
        
        # Données OHLCV de test
        self.ohlc_data = {
            '1h': pd.DataFrame({
                'open': [50000, 50100, 50200],
                'high': [50500, 50300, 50400],
                'low': [49900, 50000, 50050],
                'close': [50200, 50150, 50300],
                'volume': [10.5, 8.2, 12.3]
            })
        }
        
    async def test_order_book_integration(self):
        """Test l'intégration complète du carnet d'ordres avec le système."""
        # 1. Mise à jour des données de marché
        symbol = 'XXBTZUSD'
        market_data = await self.market_data.update_market_data(symbol)
        
        # Vérifier que les données de marché contiennent le carnet d'ordres
        self.assertIn('order_book', market_data)
        self.assertIsNotNone(market_data['order_book'])
        
        # 2. Récupérer le gestionnaire de carnet d'ordres
        order_book_manager = self.market_data.get_order_book_manager(symbol)
        self.assertIsNotNone(order_book_manager)
        
        # 3. Vérifier que le snapshot n'est pas vide
        snapshot = order_book_manager.current_snapshot
        self.assertIsNotNone(snapshot)
        self.assertGreater(len(snapshot.bids), 0)
        self.assertGreater(len(snapshot.asks), 0)
        
        # 4. Vérifier les métriques du carnet d'ordres
        metrics = snapshot.metrics
        self.assertIsNotNone(metrics)
        self.assertGreater(metrics.spread, 0)
        self.assertIsNotNone(metrics.imbalance)
        
        # 5. Générer des signaux avec les données du carnet d'ordres
        signals = await self.signal_generator.generate_signals(
            ohlc_data=self.ohlc_data,
            pair=symbol,
            timeframe='1h',
            order_book={
                'bids': [{'price': float(level['price']), 'amount': float(level['amount'])} 
                        for level in snapshot.bids],
                'asks': [{'price': float(level['price']), 'amount': float(level['amount'])} 
                        for level in snapshot.asks],
                'metrics': {
                    'spread': float(metrics.spread) if metrics.spread else 0.0,
                    'imbalance': float(metrics.imbalance) if hasattr(metrics, 'imbalance') else 0.0,
                    'vwap_bid': float(metrics.vwap_bid) if hasattr(metrics, 'vwap_bid') else 0.0,
                    'vwap_ask': float(metrics.vwap_ask) if hasattr(metrics, 'vwap_ask') else 0.0
                }
            }
        )
        
        # 6. Vérifier que les signaux contiennent les métriques du carnet d'ordres
        self.assertIn('order_book_imbalance', signals)
        self.assertIn('order_book_spread', signals)
        self.assertIn('vwap_bid', signals)
        self.assertIn('vwap_ask', signals)
        self.assertIn('liquidity_imbalance', signals)
        
        # 7. Vérifier les signaux générés
        self.assertIn('imbalance_signal', signals)
        self.assertIn('spread_signal', signals)
        self.assertIn('vwap_signal', signals)
        
    async def test_order_book_empty_data(self):
        """Test la gestion des données vides dans le carnet d'ordres."""
        # Simuler une réponse vide de l'API
        self.api.get_order_book.return_value = {'XXBTZUSD': {'bids': [], 'asks': []}}
        
        # Mise à jour des données de marché
        symbol = 'XXBTZUSD'
        market_data = await self.market_data.update_market_data(symbol)
        
        # Vérifier que le gestionnaire gère correctement les données vides
        order_book_manager = self.market_data.get_order_book_manager(symbol)
        self.assertIsNotNone(order_book_manager)
        
        snapshot = order_book_manager.current_snapshot
        self.assertIsNotNone(snapshot)
        self.assertEqual(len(snapshot.bids), 0)
        self.assertEqual(len(snapshot.asks), 0)
        
        # Générer des signaux avec des données vides
        signals = await self.signal_generator.generate_signals(
            ohlc_data=self.ohlc_data,
            pair=symbol,
            timeframe='1h',
            order_book={'bids': [], 'asks': [], 'metrics': {}}
        )
        
        # Vérifier que les signaux ont été générés avec des valeurs par défaut
        self.assertEqual(signals.get('order_book_imbalance', 0), 0.0)
        self.assertEqual(signals.get('order_book_spread', 0), 0.0)
        
    async def test_order_book_error_handling(self):
        """Test la gestion des erreurs dans le carnet d'ordres."""
        # Simuler une erreur de l'API
        self.api.get_order_book.side_effect = Exception("API Error")
        
        # Le gestionnaire doit gérer l'erreur et retourner None
        order_book = await self.market_data._fetch_order_book('XXBTZUSD')
        self.assertIsNone(order_book)
        
        # Vérifier que le gestionnaire est toujours utilisable après une erreur
        self.api.get_order_book.side_effect = None
        self.api.get_order_book.return_value = {
            'XXBTZUSD': {
                'bids': [[str(Decimal('50000.0')), str(Decimal('1.5')), 1234567890]],
                'asks': [[str(Decimal('50050.0')), str(Decimal('1.2')), 1234567890]]
            }
        }
        
        # Nouvelle tentative réussie
        order_book = await self.market_data._fetch_order_book('XXBTZUSD')
        self.assertIsNotNone(order_book)
        self.assertIn('bids', order_book)
        self.assertIn('asks', order_book)

if __name__ == '__main__':
    unittest.main()
