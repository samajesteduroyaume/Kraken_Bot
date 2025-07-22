"""
Tests de performance de base pour le module order_book.
"""
import asyncio
import time
import random
from decimal import Decimal
import statistics
import logging
from typing import List, Dict, Any
from unittest.mock import AsyncMock, MagicMock
import sys
import os

# Ajout du répertoire racine au chemin Python
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Configuration du logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class OrderBookBasicTest:
    """Tests de performance de base pour OrderBookManager."""
    
    def __init__(self):
        self.api = AsyncMock()
        self.symbol = "XXBTZUSD"
        
    def generate_mock_order_book(self, base_price: float = 100.0) -> Dict[str, Any]:
        """Génère un carnet d'ordres factice pour les tests."""
        spread = 0.001  # 0.1% de spread
        mid_price = base_price * (1 + random.uniform(-0.1, 0.1))
        
        def generate_levels(center: float, count: int, step: float, reverse: bool = False) -> list:
            levels = []
            for i in range(1, count + 1):
                price = center + (i * step) * (-1 if reverse else 1)
                amount = random.uniform(0.1, 10.0)
                levels.append([str(Decimal(str(price))), str(Decimal(str(amount))), int(time.time())])
            return levels
        
        return {
            'bids': generate_levels(mid_price * (1 - spread/2), 10, mid_price * 0.001, reverse=True),
            'asks': generate_levels(mid_price * (1 + spread/2), 10, mid_price * 0.001)
        }
    
    async def test_single_order_book(self, num_updates: int = 1000):
        """Test les performances d'un seul carnet d'ordres."""
        from src.core.market.order_book import OrderBookManager
        
        # Configuration du mock
        self.api.get_order_book.return_value = {
            self.symbol: self.generate_mock_order_book()
        }
        
        # Initialisation
        ob_manager = OrderBookManager(self.symbol, self.api)
        
        # Test de performance
        latencies = []
        for _ in range(num_updates):
            start_time = time.perf_counter()
            await ob_manager.update()
            latencies.append((time.perf_counter() - start_time) * 1000)  # en ms
        
        # Nettoyage
        await ob_manager.stop()
        
        # Résultats
        return {
            'test': 'single_order_book',
            'num_updates': num_updates,
            'avg_latency_ms': statistics.mean(latencies),
            'min_latency_ms': min(latencies),
            'max_latency_ms': max(latencies),
            'p95_latency_ms': statistics.quantiles(latencies, n=20)[-1] if len(latencies) >= 20 else 0,
        }
    
    async def test_multiple_order_books(self, num_books: int = 50, updates_per_book: int = 20):
        """Test les performances avec plusieurs carnets d'ordres."""
        from src.core.market.order_book import OrderBookManager
        
        # Configuration du mock pour retourner des données différentes à chaque appel
        def mock_order_book(pair, count=100):
            return {pair: self.generate_mock_order_book(100.0 + hash(pair) % 100)}
        
        self.api.get_order_book = AsyncMock(side_effect=mock_order_book)
        
        # Création des gestionnaires
        managers = [
            OrderBookManager(f"SYM{i:03d}USD", self.api) 
            for i in range(num_books)
        ]
        
        # Test de performance
        latencies = []
        for _ in range(updates_per_book):
            start_time = time.perf_counter()
            
            # Mise à jour en parallèle
            tasks = [m.update() for m in managers]
            await asyncio.gather(*tasks)
            
            latencies.append((time.perf_counter() - start_time) * 1000 / num_books)  # temps moyen par carnet
        
        # Nettoyage
        for m in managers:
            await m.stop()
        
        # Résultats
        return {
            'test': f'multiple_order_books_{num_books}',
            'num_books': num_books,
            'updates_per_book': updates_per_book,
            'avg_latency_ms': statistics.mean(latencies),
            'min_latency_ms': min(latencies),
            'max_latency_ms': max(latencies),
            'p95_latency_ms': statistics.quantiles(latencies, n=20)[-1] if len(latencies) >= 20 else 0,
        }

def print_results(results: Dict[str, Any]):
    """Affiche les résultats des tests."""
    print("\n" + "="*80)
    print("RÉSULTATS DES TESTS DE PERFORMANCE".center(80))
    print("="*80)
    
    print(f"\nTest: {results['test']}")
    if 'num_books' in results:
        print(f"• Nombre de carnets: {results['num_books']}")
    print(f"• Mises à jour: {results.get('num_updates', results.get('updates_per_book', 0))}")
    print("\nLatences (moyenne par opération):")
    print(f"  • Moyenne: {results['avg_latency_ms']:.4f} ms")
    print(f"  • Min/Max: {results['min_latency_ms']:.4f} ms / {results['max_latency_ms']:.4f} ms")
    print(f"  • P95: {results['p95_latency_ms']:.4f} ms")
    
    # Estimation du débit maximal
    if results['avg_latency_ms'] > 0:
        updates_per_second = 1000 / results['avg_latency_ms']
        if 'num_books' in results:
            updates_per_second *= results['num_books']
        print(f"\nDébit estimé: {updates_per_second:.1f} mises à jour/seconde")
    
    print("="*80 + "\n")

async def main():
    """Fonction principale."""
    tester = OrderBookBasicTest()
    
    # Test avec un seul carnet d'ordres
    print("Test avec un seul carnet d'ordres...")
    single_result = await tester.test_single_order_book(num_updates=1000)
    print_results(single_result)
    
    # Test avec plusieurs carnets d'ordres
    for num_books in [10, 50, 100]:
        print(f"\nTest avec {num_books} carnets d'ordres...")
        multi_result = await tester.test_multiple_order_books(num_books=num_books, updates_per_book=50)
        print_results(multi_result)

if __name__ == "__main__":
    asyncio.run(main())
