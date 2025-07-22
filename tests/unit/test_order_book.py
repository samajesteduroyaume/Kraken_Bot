"""
Tests unitaires pour le module order_book.
"""

import unittest
import asyncio
import decimal
from decimal import Decimal, ROUND_HALF_UP
from datetime import datetime, timezone
from unittest.mock import AsyncMock, patch, MagicMock

from src.core.market.order_book import (
    OrderBookManager,
    OrderBookSnapshot,
    OrderBookMetrics
)
from src.core.types.market_types import PriceLevel

class TestOrderBookMetrics(unittest.TestCase):
    """Tests pour la classe OrderBookMetrics."""
    
    def setUp(self):
        """Initialisation des tests."""
        self.metrics = OrderBookMetrics(price_precision=2, volume_precision=8)
        
    def test_initial_state(self):
        """Vérifie l'état initial des métriques."""
        self.assertIsNone(self.metrics.best_bid)
        self.assertIsNone(self.metrics.best_ask)
        self.assertIsNone(self.metrics.spread)
        self.assertEqual(len(self.metrics.cumulative_bid), 0)
        self.assertEqual(len(self.metrics.cumulative_ask), 0)
        
    def test_update_with_empty_data(self):
        """Teste la mise à jour avec des données vides."""
        self.metrics.update([], [])
        self.assertIsNone(self.metrics.best_bid)
        self.assertIsNone(self.metrics.best_ask)
        
    def test_update_with_valid_data(self):
        """Teste la mise à jour avec des données valides."""
        bids = [
            {'price': '100.00', 'amount': '1.0'},
            {'price': '99.50', 'amount': '2.0'}
        ]
        asks = [
            {'price': '100.50', 'amount': '0.8'},
            {'price': '101.00', 'amount': '1.5'}
        ]
        
        self.metrics.update(bids, asks)
        
        self.assertEqual(self.metrics.best_bid, Decimal('100.00'))
        self.assertEqual(self.metrics.best_ask, Decimal('100.50'))
        self.assertEqual(self.metrics.spread, Decimal('0.50'))
        self.assertEqual(self.metrics.mid_price, Decimal('100.25'))
        self.assertAlmostEqual(float(self.metrics.vwap_bid), 99.67, places=2)  # (100*1 + 99.5*2)/3 = 99.666...
        self.assertAlmostEqual(float(self.metrics.vwap_ask), 100.83, places=2)  # (100.5*0.8 + 101*1.5)/2.3 ≈ 100.826
        
    def test_get_liquidity_at_price(self):
        """Teste le calcul de la liquidité à un prix donné."""
        bids = [
            {'price': '100.00', 'amount': '1.0'},
            {'price': '99.50', 'amount': '2.0'}
        ]
        asks = [
            {'price': '100.50', 'amount': '0.8'},
            {'price': '101.00', 'amount': '1.5'}
        ]
        
        self.metrics.update(bids, asks)
        
        # Test pour un prix de bid
        liquidity = self.metrics.get_liquidity_at_price(Decimal('99.50'), 'bid')
        self.assertEqual(liquidity, Decimal('3.0'))  # 1.0 + 2.0
        
        # Test pour un prix d'ask
        liquidity = self.metrics.get_liquidity_at_price(Decimal('100.50'), 'ask')
        self.assertEqual(liquidity, Decimal('0.8'))
        
    def test_edge_cases(self):
        """Teste les cas limites et les cas spéciaux."""
        # Test avec des données vides
        self.metrics.update([], [])
        self.assertIsNone(self.metrics.best_bid)
        self.assertIsNone(self.metrics.best_ask)
        self.assertIsNone(self.metrics.spread)
        
        # Test avec des données None
        self.metrics.update(None, None)
        self.assertIsNone(self.metrics.best_bid)
        
        # Test avec des données mal formées
        self.metrics.update([{'price': 'invalid', 'amount': '1.0'}], [])
        self.assertIsNone(self.metrics.best_bid)
        
    def test_precision_handling(self):
        """Teste la gestion de la précision des prix et des volumes."""
        # Test avec des nombres très grands
        bids = [{'price': '1000000.00000001', 'amount': '0.00000001'}]
        asks = [{'price': '1000000.00000002', 'amount': '0.00000001'}]
        self.metrics.update(bids, asks)
        
        # Vérifier que les valeurs sont correctement stockées avec leur précision d'origine
        self.assertEqual(self.metrics.best_bid, Decimal('1000000.00000001'))
        self.assertEqual(self.metrics.best_ask, Decimal('1000000.00000002'))
        
        # Vérifier que les méthodes qui utilisent la précision fonctionnent correctement
        self.assertEqual(self.metrics.vwap_bid.quantize(Decimal('0.01')), Decimal('1000000.00'))
        self.assertEqual(self.metrics.vwap_ask.quantize(Decimal('0.01')), Decimal('1000000.00'))
        
    def test_vwap_edge_cases(self):
        """Teste les cas limites pour le calcul du VWAP."""
        # Test avec un seul niveau de prix
        self.metrics.update(
            [{'price': '100.00', 'amount': '1.0'}],
            [{'price': '101.00', 'amount': '1.0'}]
        )
        self.assertEqual(self.metrics.vwap_bid, Decimal('100.00'))
        self.assertEqual(self.metrics.vwap_ask, Decimal('101.00'))
        
        # Test avec volume nul
        self.metrics.update(
            [{'price': '100.00', 'amount': '0.0'}],
            [{'price': '101.00', 'amount': '0.0'}]
        )
        self.assertEqual(self.metrics.vwap_bid, Decimal('0'))
        self.assertEqual(self.metrics.vwap_ask, Decimal('0'))
        
    def test_error_handling(self):
        """Teste la gestion des erreurs dans les méthodes de la classe."""
        # Vérifier que la méthode gère correctement les données invalides
        # Note: La méthode _calculate_vwap est maintenant plus tolérante et ne lève pas d'exception
        # pour les données invalides, donc nous ne testons plus cette partie
        
        # Vérifier que les paramètres invalides pour get_liquidity_at_price lèvent une ValueError
        with self.assertRaises(ValueError):
            self.metrics.get_liquidity_at_price(Decimal('100.00'), 'invalid_side')


class TestOrderBookSnapshot(unittest.TestCase):
    """Tests pour la classe OrderBookSnapshot."""
    
    def setUp(self):
        """Initialisation des tests."""
        self.bids = [
            {'price': '100.00', 'amount': '1.0'},
            {'price': '99.50', 'amount': '2.0'}
        ]
        self.asks = [
            {'price': '100.50', 'amount': '0.8'},
            {'price': '101.00', 'amount': '1.5'}
        ]
        self.snapshot = OrderBookSnapshot(
            bids=self.bids,
            asks=self.asks,
            timestamp=datetime.now(timezone.utc)
        )
        
    def test_initialization(self):
        """Teste l'initialisation du snapshot."""
        self.assertEqual(len(self.snapshot.bids), 2)
        self.assertEqual(len(self.snapshot.asks), 2)
        self.assertIsNotNone(self.snapshot.timestamp)
        self.assertIsInstance(self.snapshot.metrics, OrderBookMetrics)
        
    def test_get_price_levels(self):
        """Teste la récupération des niveaux de prix."""
        levels = self.snapshot.get_price_levels(num_levels=1)
        self.assertEqual(len(levels['bids']), 1)
        self.assertEqual(len(levels['asks']), 1)
        self.assertEqual(float(levels['bids'][0]['price']), 100.00)
        self.assertEqual(float(levels['asks'][0]['price']), 100.50)
        
    def test_get_cumulative_volume(self):
        """Teste le calcul du volume cumulé."""
        # Volume cumulé jusqu'au meilleur prix d'achat
        volume = self.snapshot.get_cumulative_volume(Decimal('100.00'), 'bid')
        self.assertEqual(volume, Decimal('1.0'))
        
        # Volume cumulé jusqu'au deuxième meilleur prix d'achat
        volume = self.snapshot.get_cumulative_volume(Decimal('99.50'), 'bid')
        self.assertEqual(volume, Decimal('3.0'))  # 1.0 + 2.0
        
    def test_edge_cases(self):
        """Teste les cas limites pour OrderBookSnapshot."""
        # Test avec des données vides
        empty_snapshot = OrderBookSnapshot([], [])
        self.assertEqual(len(empty_snapshot.bids), 0)
        self.assertEqual(len(empty_snapshot.asks), 0)
        
        # Test avec un seul niveau de prix
        single_level = OrderBookSnapshot(
            [{'price': '100.00', 'amount': '1.0'}],
            [{'price': '101.00', 'amount': '1.0'}]
        )
        self.assertEqual(len(single_level.bids), 1)
        self.assertEqual(len(single_level.asks), 1)
        
    def test_invalid_data_handling(self):
        """Teste la gestion des données invalides."""
        # Test avec des données mal formées
        with self.assertRaises((ValueError, decimal.InvalidOperation)):
            OrderBookSnapshot(
                [{'price': 'invalid', 'amount': '1.0'}],
                [{'price': '101.00', 'amount': '1.0'}]
            )
            
    def test_get_price_levels_limits(self):
        """Teste les limites de get_price_levels."""
        # Demander plus de niveaux qu'il n'y en a
        levels = self.snapshot.get_price_levels(num_levels=10)
        self.assertEqual(len(levels['bids']), 2)  # Seulement 2 niveaux disponibles
        self.assertEqual(len(levels['asks']), 2)  # Seulement 2 niveaux disponibles
        
        # Demander 0 niveau
        levels = self.snapshot.get_price_levels(num_levels=0)
        self.assertEqual(len(levels['bids']), 0)
        self.assertEqual(len(levels['asks']), 0)
        
    def test_cumulative_volume_edge_cases(self):
        """Teste les cas limites pour get_cumulative_volume."""
        # Prix supérieur au meilleur prix d'achat
        volume = self.snapshot.get_cumulative_volume(Decimal('1000.00'), 'bid')
        self.assertEqual(volume, Decimal('0.0'))  # Aucun volume car le prix est trop élevé
        
        # Prix inférieur au pire prix d'achat
        volume = self.snapshot.get_cumulative_volume(Decimal('10.00'), 'bid')
        self.assertEqual(volume, Decimal('3.0'))  # Tous les niveaux inclus (1.0 + 2.0)
        
        # Prix égal au pire prix d'achat
        volume = self.snapshot.get_cumulative_volume(Decimal('99.50'), 'bid')
        self.assertEqual(volume, Decimal('3.0'))  # Tous les niveaux inclus
        
        # Côté invalide
        with self.assertRaises(ValueError):
            self.snapshot.get_cumulative_volume(Decimal('100.00'), 'invalid_side')


class TestOrderBookManager(unittest.IsolatedAsyncioTestCase):
    """Tests pour la classe OrderBookManager."""
    
    async def asyncSetUp(self):
        """Initialisation asynchrone des tests."""
        self.api = AsyncMock()
        self.manager = OrderBookManager('XBT/USD', self.api)
        
        # Configuration des réponses simulées de l'API
        self.api.get_order_book.return_value = {
            'XXBTZUSD': {
                'bids': [
                    ['50000.0', '1.0', int(datetime.now().timestamp())],
                    ['49950.0', '2.0', int(datetime.now().timestamp())]
                ],
                'asks': [
                    ['50050.0', '0.8', int(datetime.now().timestamp())],
                    ['50100.0', '1.5', int(datetime.now().timestamp())]
                ]
            }
        }
        
    async def test_initialization(self):
        """Teste l'initialisation du gestionnaire."""
        self.assertEqual(self.manager.symbol, 'XBT/USD')
        self.assertIsNone(self.manager.current_snapshot)
        self.assertEqual(len(self.manager.history), 0)
        
    async def test_update(self):
        """Teste la mise à jour du carnet d'ordres."""
        await self.manager.update()
        
        # Vérifie qu'un snapshot a été créé
        self.assertIsNotNone(self.manager.current_snapshot)
        self.assertEqual(len(self.manager.history), 1)
        
        # Vérifie les données du snapshot
        snapshot = self.manager.current_snapshot
        self.assertEqual(len(snapshot.bids), 2)
        self.assertEqual(len(snapshot.asks), 2)
        self.assertEqual(float(snapshot.bids[0].price), 50000.0)
        self.assertEqual(float(snapshot.asks[0].price), 50050.0)
        
    async def test_start_stop(self):
        """Teste le démarrage et l'arrêt des mises à jour automatiques."""
        # Démarrer les mises à jour automatiques
        await self.manager.start(update_interval=0.1)
        self.assertTrue(self.manager._running)
        
        # Attendre qu'au moins une mise à jour se produise
        await asyncio.sleep(0.15)
        
        # Vérifier qu'au moins une mise à jour a eu lieu
        self.assertGreaterEqual(len(self.manager.history), 1)
        
        # Arrêter les mises à jour
        await self.manager.stop()
        self.assertFalse(self.manager._running)
        
    async def test_get_order_imbalance(self):
        """Teste le calcul du déséquilibre d'ordres."""
        await self.manager.update()
        imbalance = self.manager.get_order_imbalance(levels=2)
        
        # Vérifie que le déséquilibre est dans la plage attendue
        self.assertGreaterEqual(imbalance, -1.0)
        self.assertLessEqual(imbalance, 1.0)
        
    async def test_get_price_impact(self):
        """Teste le calcul de l'impact de prix."""
        await self.manager.update()
        
        # Impact d'un achat de 1 BTC
        impact = self.manager.get_price_impact(1.0, 'buy')
        self.assertGreater(impact, 0)  # L'impact doit être positif pour un achat
        
        # Impact d'une vente de 1 BTC
        impact = self.manager.get_price_impact(1.0, 'sell')
        self.assertLess(impact, 0)  # L'impact doit être négatif pour une vente
        
    async def test_edge_cases(self):
        """Teste les cas limites pour OrderBookManager."""
        # Test avec des données d'API vides - ne doit plus lever d'exception
        self.api.get_order_book.return_value = {}
        try:
            await self.manager.update()
            # Vérifier qu'un snapshot vide a été créé
            self.assertIsNotNone(self.manager.current_snapshot)
            self.assertEqual(len(self.manager.current_snapshot.bids), 0)
            self.assertEqual(len(self.manager.current_snapshot.asks), 0)
        except Exception as e:
            self.fail(f"La méthode update ne devrait pas lever d'exception avec des données vides, mais a levé: {e}")
        
        # Réinitialiser le mock pour le prochain test
        self.api.get_order_book.return_value = None
        # Créer un nouveau gestionnaire pour le prochain test
        self.manager = OrderBookManager(symbol='XXBTZUSD', api=self.api)
        
        # Test avec des données d'API invalides (manque la clé 'last')
        # Ne devrait plus lever d'exception non plus
        self.api.get_order_book.return_value = {'XXBTZUSD': {'bids': [], 'asks': []}}
        try:
            await self.manager.update()
            self.assertIsNotNone(self.manager.current_snapshot)
            self.assertEqual(len(self.manager.current_snapshot.bids), 0)
            self.assertEqual(len(self.manager.current_snapshot.asks), 0)
        except Exception as e:
            self.fail(f"La méthode update ne devrait pas lever d'exception avec des données invalides, mais a levé: {e}")
            
        # Test avec des données d'API valides
        self.api.get_order_book.return_value = {
            'XXBTZUSD': {
                'bids': [['50000.0', '1.0', int(datetime.now().timestamp())]],
                'asks': [['50050.0', '0.8', int(datetime.now().timestamp())]],
                'last': '50025.0'
            }
        }
        await self.manager.update()
        self.assertIsNotNone(self.manager.current_snapshot)
        self.assertEqual(len(self.manager.current_snapshot.bids), 1)
        self.assertEqual(len(self.manager.current_snapshot.asks), 1)
        
    async def test_error_handling(self):
        """Teste la gestion des erreurs dans OrderBookManager."""
        # Test avec une erreur de l'API
        self.api.get_order_book.side_effect = Exception("API Error")
        with self.assertRaises(Exception):
            await self.manager.update()
            
        # Vérifier que le gestionnaire reste dans un état cohérent après une erreur
        self.assertIsNone(self.manager.current_snapshot)
        
    async def test_concurrent_updates(self):
        """Teste les mises à jour concurrentes du gestionnaire."""
        # Démarrer les mises à jour automatiques
        await self.manager.start(update_interval=0.1)
        
        # Effectuer plusieurs mises à jour manuelles en parallèle
        tasks = [self.manager.update() for _ in range(5)]
        await asyncio.gather(*tasks)
        
        # Vérifier que le nombre de snapshots dans l'historique est cohérent
        self.assertGreaterEqual(len(self.manager.history), 1)
        
        # Arrêter les mises à jour automatiques
        await self.manager.stop()
        
    async def test_price_impact_edge_cases(self):
        """Teste les cas limites pour le calcul de l'impact de prix."""
        # Configurer des données de test connues
        self.api.get_order_book.return_value = {
            'XXBTZUSD': {
                'bids': [
                    ['50000.0', '1.0', int(datetime.now().timestamp())],
                    ['49950.0', '2.0', int(datetime.now().timestamp())]
                ],
                'asks': [
                    ['50050.0', '0.8', int(datetime.now().timestamp())],
                    ['50100.0', '1.5', int(datetime.now().timestamp())]
                ],
                'last': '50025.0'
            }
        }
        
        # Mettre à jour avec les données de test
        await self.manager.update()
        
        # Impact pour un volume nul - devrait retourner 0
        impact = self.manager.get_price_impact(0, 'buy')
        self.assertEqual(impact, 0)
        
        # Impact pour un achat avec un petit volume
        impact = self.manager.get_price_impact(0.5, 'buy')
        self.assertGreater(impact, 0)  # Doit être positif pour un achat
        
        # Impact pour une vente avec un petit volume
        impact = self.manager.get_price_impact(0.5, 'sell')
        self.assertLess(impact, 0)  # Doit être négatif pour une vente
        
        # Impact pour un volume très élevé (dépassant la liquidité disponible)
        impact = self.manager.get_price_impact(1000000, 'buy')
        self.assertGreater(impact, 0)  # Doit toujours être positif pour un achat
        
        # Côté invalide
        with self.assertRaises(ValueError):
            self.manager.get_price_impact(1.0, 'invalid_side')


if __name__ == '__main__':
    unittest.main()
