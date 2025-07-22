"""
Tests de performance pour le module order_book avec un grand nombre de paires.

Ce script évalue les performances du module order_book après les optimisations,
en se concentrant sur :
- Le débit (updates/seconde)
- La latence des opérations
- L'utilisation mémoire
- L'évolutivité avec le nombre de paires
"""
import asyncio
import time
import random
import tracemalloc
import psutil
import gc
from decimal import Decimal, getcontext
import statistics
import logging
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass
from datetime import datetime

import pandas as pd
import numpy as np
from tqdm import tqdm

# Configuration de la précision décimale
getcontext().prec = 12

# Mock des types manquants
from unittest.mock import AsyncMock, MagicMock, patch
import sys
import os

# Ajout du répertoire racine au chemin Python
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Import des modules après avoir configuré le chemin
from src.core.market.data_manager import MarketDataManager
from src.core.market.order_book import OrderBookManager, OrderBookSnapshot, PriceLevel, OrderBookMetrics

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('orderbook_performance.log')
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class PerformanceMetrics:
    """Classe pour stocker les métriques de performance."""
    test_name: str
    num_pairs: int
    num_updates: int
    total_time: float
    avg_latency_ms: float
    min_latency_ms: float
    max_latency_ms: float
    p95_latency_ms: float
    memory_usage_mb: float
    cpu_usage_percent: float
    updates_per_second: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convertit les métriques en dictionnaire."""
        return {
            'test_name': self.test_name,
            'num_pairs': self.num_pairs,
            'num_updates': self.num_updates,
            'total_time_sec': round(self.total_time, 4),
            'avg_latency_ms': round(self.avg_latency_ms, 4),
            'min_latency_ms': round(self.min_latency_ms, 4),
            'max_latency_ms': round(self.max_latency_ms, 4),
            'p95_latency_ms': round(self.p95_latency_ms, 4),
            'memory_usage_mb': round(self.memory_usage_mb, 2),
            'cpu_usage_percent': round(self.cpu_usage_percent, 1),
            'updates_per_second': round(self.updates_per_second, 2)
        }

# Définition des types pour les tests
Candle = Dict[str, float]
Trade = Dict[str, Any]

# Configuration du logging déjà effectuée plus haut

def get_memory_usage() -> float:
    """Retourne l'utilisation mémoire actuelle en Mo."""
    process = psutil.Process()
    return process.memory_info().rss / (1024 * 1024)  # En Mo

def get_cpu_usage() -> float:
    """Retourne l'utilisation CPU actuelle en pourcentage."""
    return psutil.cpu_percent(interval=0.1)

class OrderBookPerformanceTester:
    """
    Classe pour tester les performances du module order_book.
    
    Cette classe permet d'évaluer les performances du carnet d'ordres
    dans différents scénarios de charge.
    """
    
    def __init__(self, api, num_pairs: int = 50):
        """
        Initialise le testeur de performance.
        
        Args:
            api: Mock de l'API
            num_pairs: Nombre de paires à tester
        """
        self.api = api
        self.num_pairs = num_pairs
        self.pairs = [f"SYM{i:03d}USD" for i in range(1, num_pairs + 1)]
        self.market_data = MarketDataManager(api)
        self.results: List[PerformanceMetrics] = []
        
        # Configuration des tests
        self.warmup_iterations = 10
        self.test_iterations = 100
        self.max_pairs = 1000  # Nombre maximum de paires pour les tests d'évolutivité
        
        # Statistiques
        self.start_time = time.time()
        self.memory_samples: List[float] = []
        self.cpu_samples: List[float] = []
    
    def generate_mock_order_book(self, base_price: float = 100.0, num_levels: int = 10) -> Dict[str, Any]:
        """
        Génère un carnet d'ordres factice pour les tests.
        
        Args:
            base_price: Prix de base pour la génération
            num_levels: Nombre de niveaux de prix à générer
            
        Returns:
            Dictionnaire avec les offres (bids) et demandes (asks)
        """
        # Génération plus efficace avec numpy
        np.random.seed(int(time.time()))  # Pour la reproductibilité
        
        # Générer un prix moyen avec une variation aléatoire
        mid_price = base_price * (1 + np.random.uniform(-0.1, 0.1))
        spread = np.random.uniform(0.01, 0.1)  # 0.01% à 0.1% de spread
        
        # Générer les niveaux de prix
        def generate_levels(center: float, count: int, step: float, reverse: bool = False) -> list:
            # Génération vectorisée des prix et volumes
            steps = np.arange(1, count + 1)
            prices = center + (steps * step) * (-1 if reverse else 1)
            amounts = np.random.uniform(0.1, 10.0, count)
            
            # Création des niveaux
            levels = []
            for price, amount in zip(prices, amounts):
                # Utilisation directe des chaînes pour éviter les conversions inutiles
                price_str = f"{price:.8f}"
                amount_str = f"{amount:.8f}"
                levels.append([price_str, amount_str, int(time.time())])
            return levels
        
        # Générer les offres (bids) et demandes (asks)
        price_step = mid_price * 0.001  # Écart de 0.1% entre les niveaux
        
        return {
            'bids': generate_levels(mid_price * (1 - spread/2), num_levels, price_step, reverse=True),
            'asks': generate_levels(mid_price * (1 + spread/2), num_levels, price_step)
        }
    
    async def setup_mock_api(self):
        """Configure le mock de l'API avec des données factices."""
        async def mock_get_order_book(pair, count=100, **kwargs):
            return {
                pair: self.generate_mock_order_book(100.0)
            }
            
        self.api.get_order_book = AsyncMock(side_effect=mock_get_order_book)
    
    async def measure_latency(self, func, *args, **kwargs) -> Tuple[float, Any]:
        """Mesure le temps d'exécution d'une fonction asynchrone."""
        start_time = time.perf_counter()
        result = await func(*args, **kwargs)
        latency_ms = (time.perf_counter() - start_time) * 1000  # en ms
        return latency_ms, result
    
    async def warmup(self, num_iterations: int = 10):
        """Effectue des itérations de préchauffage pour stabiliser les performances."""
        logger.info(f"Démarrage du préchauffage ({num_iterations} itérations)...")
        pair = self.pairs[0]
        
        for i in range(num_iterations):
            await self.market_data.update_market_data(pair)
            if (i + 1) % 10 == 0:
                logger.debug(f"Préchauffage {i+1}/{num_iterations}")
    
    async def test_single_pair_sequential(self, num_updates: int = 100) -> PerformanceMetrics:
        """
        Test les performances avec une seule paire en mise à jour séquentielle.
        
        Args:
            num_updates: Nombre de mises à jour à effectuer
            
        Returns:
            Objet PerformanceMetrics avec les résultats du test
        """
        pair = self.pairs[0]
        latencies = []
        
        # Phase de préchauffage
        await self.warmup(self.warmup_iterations)
        
        # Réinitialisation des statistiques
        self.memory_samples = []
        self.cpu_samples = []
        
        # Début du test
        start_time = time.perf_counter()
        
        # Boucle de test principale
        for i in tqdm(range(num_updates), desc="Test séquentiel à une paire"):
            # Mise à jour des données de marché
            latency, _ = await self.measure_latency(
                self.market_data.update_market_data, pair
            )
            latencies.append(latency)
            
            # Échantillonnage périodique de l'utilisation des ressources
            if i % 10 == 0:
                self.memory_samples.append(get_memory_usage())
                self.cpu_samples.append(get_cpu_usage())
        
        # Calcul des métriques
        total_time = time.perf_counter() - start_time
        
        return PerformanceMetrics(
            test_name='single_pair_sequential',
            num_pairs=1,
            num_updates=num_updates,
            total_time=total_time,
            avg_latency_ms=statistics.mean(latencies),
            min_latency_ms=min(latencies),
            max_latency_ms=max(latencies),
            p95_latency_ms=statistics.quantiles(latencies, n=20)[-1],
            memory_usage_mb=statistics.mean(self.memory_samples) if self.memory_samples else 0,
            cpu_usage_percent=statistics.mean(self.cpu_samples) if self.cpu_samples else 0,
            updates_per_second=num_updates / total_time if total_time > 0 else 0
        )
    
    async def test_multiple_pairs_sequential(self, num_updates: int = 10) -> PerformanceMetrics:
        """
        Test les performances avec plusieurs paires en mise à jour séquentielle.
        
        Args:
            num_updates: Nombre de mises à jour à effectuer par paire
            
        Returns:
            Objet PerformanceMetrics avec les résultats du test
        """
        latencies = []
        
        # Phase de préchauffage
        await self.warmup(self.warmup_iterations)
        
        # Réinitialisation des statistiques
        self.memory_samples = []
        self.cpu_samples = []
        
        # Début du test
        start_time = time.perf_counter()
        
        # Boucle de test principale
        for i in tqdm(range(num_updates), desc=f"Test séquentiel sur {self.num_pairs} paires"):
            for pair_idx, pair in enumerate(self.pairs):
                # Mise à jour des données de marché avec mesure de latence
                latency, _ = await self.measure_latency(
                    self.market_data.update_market_data, pair
                )
                latencies.append(latency)
                
                # Échantillonnage périodique de l'utilisation des ressources
                if (i * len(self.pairs) + pair_idx) % 10 == 0:
                    self.memory_samples.append(get_memory_usage())
                    self.cpu_samples.append(get_cpu_usage())
        
        # Calcul des métriques
        total_updates = num_updates * len(self.pairs)
        total_time = time.perf_counter() - start_time
        
        return PerformanceMetrics(
            test_name='multiple_pairs_sequential',
            num_pairs=len(self.pairs),
            num_updates=total_updates,
            total_time=total_time,
            avg_latency_ms=statistics.mean(latencies),
            min_latency_ms=min(latencies),
            max_latency_ms=max(latencies),
            p95_latency_ms=statistics.quantiles(latencies, n=20)[-1],
            memory_usage_mb=statistics.mean(self.memory_samples) if self.memory_samples else 0,
            cpu_usage_percent=statistics.mean(self.cpu_samples) if self.cpu_samples else 0,
            updates_per_second=total_updates / total_time if total_time > 0 else 0
        )
    
    async def test_multiple_pairs_concurrent(self, num_updates: int = 10) -> PerformanceMetrics:
        """
        Test les performances avec plusieurs paires en mises à jour concurrentes.
        
        Args:
            num_updates: Nombre de mises à jour à effectuer par paire
            
        Returns:
            Objet PerformanceMetrics avec les résultats du test
        """
        # Phase de préchauffage
        await self.warmup(self.warmup_iterations)
        
        # Réinitialisation des statistiques
        self.memory_samples = []
        self.cpu_samples = []
        latencies = []
        
        # Fonction pour mettre à jour une paire unique
        async def update_single_pair(pair: str, update_count: int) -> List[float]:
            pair_latencies = []
            for _ in range(update_count):
                latency, _ = await self.measure_latency(
                    self.market_data.update_market_data, pair
                )
                pair_latencies.append(latency)
            return pair_latencies
        
        # Début du test
        start_time = time.perf_counter()
        
        # Effectuer les mises à jour en parallèle
        tasks = []
        for pair in self.pairs:
            task = asyncio.create_task(update_single_pair(pair, num_updates))
            tasks.append(task)
        
        # Attendre la fin de toutes les tâches et collecter les résultats
        all_latencies = await asyncio.gather(*tasks)
        
        # Aplatir la liste des latences
        latencies = [lat for sublist in all_latencies for lat in sublist]
        
        # Calcul des métriques
        total_updates = num_updates * len(self.pairs)
        total_time = time.perf_counter() - start_time
        
        # Dernier échantillonnage des ressources
        self.memory_samples.append(get_memory_usage())
        self.cpu_samples.append(get_cpu_usage())
        
        return PerformanceMetrics(
            test_name='multiple_pairs_concurrent',
            num_pairs=len(self.pairs),
            num_updates=total_updates,
            total_time=total_time,
            avg_latency_ms=statistics.mean(latencies),
            min_latency_ms=min(latencies),
            max_latency_ms=max(latencies),
            p95_latency_ms=statistics.quantiles(latencies, n=20)[-1],
            memory_usage_mb=statistics.mean(self.memory_samples) if self.memory_samples else 0,
            cpu_usage_percent=statistics.mean(self.cpu_samples) if self.cpu_samples else 0,
            updates_per_second=total_updates / total_time if total_time > 0 else 0
        )
    
    async def test_multiple_pairs_parallel(self, num_updates: int = 10) -> Dict[str, Any]:
        """Test les performances avec plusieurs paires en parallèle."""
        async def update_pair(pair: str) -> float:
            start_time = time.perf_counter()
            await self.market_data.update_market_data(pair)
            return (time.perf_counter() - start_time) * 1000  # en ms
        
        all_latencies = []
        
        for _ in range(num_updates):
            tasks = [update_pair(pair) for pair in self.pairs]
            latencies = await asyncio.gather(*tasks)
            all_latencies.extend(latencies)
        
        return {
            'test': 'multiple_pairs_parallel',
            'num_pairs': len(self.pairs),
            'num_updates': num_updates,
            'avg_latency_ms': statistics.mean(all_latencies),
            'min_latency_ms': min(all_latencies),
            'max_latency_ms': max(all_latencies),
            'p95_latency_ms': statistics.quantiles(all_latencies, n=20)[-1] if len(all_latencies) >= 20 else 0,
        }
    
    async def run_tests(self):
        """Exécute tous les tests de performance."""
        logger.info("Démarrage des tests de performance...")
        
        # Configuration du mock de l'API
        await self.setup_mock_api()
        
        # Tests de performance
        tests = [
            ("single_pair_sequential", "Test avec 1 paire (séquentiel)", self.test_single_pair_sequential),
            (f"{len(self.pairs)}_pairs_sequential", f"Test avec {len(self.pairs)} paires (séquentiel)", self.test_multiple_pairs_sequential),
            (f"{len(self.pairs)}_pairs_parallel", f"Test avec {len(self.pairs)} paires (parallèle)", self.test_multiple_pairs_parallel),
        ]
        
        for test_id, test_name, test_func in tests:
            logger.info(f"Exécution: {test_name}")
            try:
                result = await test_func()
                
                # Créer un objet PerformanceMetrics si le résultat est un dictionnaire
                if isinstance(result, dict):
                    result = PerformanceMetrics(
                        test_name=test_id,
                        num_pairs=result.get('num_pairs', 1),
                        num_updates=result.get('num_updates', 0),
                        total_time=result.get('total_time', 0),
                        avg_latency_ms=result.get('avg_latency_ms', 0),
                        min_latency_ms=result.get('min_latency_ms', 0),
                        max_latency_ms=result.get('max_latency_ms', 0),
                        p95_latency_ms=result.get('p95_latency_ms', 0),
                        memory_usage_mb=result.get('memory_usage_mb', 0),
                        cpu_usage_percent=result.get('cpu_usage_percent', 0),
                        updates_per_second=result.get('updates_per_second', 0)
                    )
                
                self.results.append(result)
                logger.info(f"Résultat - {test_name}:")
                logger.info(f"  • Latence moyenne: {result.avg_latency_ms:.2f} ms")
                logger.info(f"  • Latence min/max: {result.min_latency_ms:.2f} ms / {result.max_latency_ms:.2f} ms")
                logger.info(f"  • Latence P95: {result.p95_latency_ms:.2f} ms")
            except Exception as e:
                logger.error(f"Erreur lors du test {test_name}: {str(e)}")
        
        # Nettoyage
        await self.market_data.close()
        
        return self.results

def print_results_table(results: List[PerformanceMetrics]):
    """
    Affiche les résultats sous forme de tableau.
    
    Args:
        results: Liste des métriques de performance à afficher
    """
    if not results:
        print("Aucun résultat à afficher.")
        return
    
    # Convertir en dictionnaires
    results_dicts = [r.to_dict() for r in results]
    
    # Créer un DataFrame pour un affichage propre
    df = pd.DataFrame(results_dicts)
    
    # Formater les colonnes
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 120)
    pd.set_option('display.float_format', '{:.4f}'.format)
    
    # Afficher les résultats
    print("\n" + "="*120)
    print("RÉSULTATS DES TESTS DE PERFORMANCE - CARNET D'ORDRE OPTIMISÉ")
    print("="*120)
    
    # Vérifier si nous avons des données à afficher
    if len(df) == 0:
        print("Aucune donnée de performance à afficher.")
        return
    
    # Afficher les résultats par type de test
    for test_name, group in df.groupby('test_name'):
        print(f"\n{test_name.upper()}:")
        print("-" * (len(test_name) + 1))
        
        # Sélectionner et formater les colonnes pertinentes
        available_cols = ['num_pairs', 'num_updates', 'total_time_sec', 'avg_latency_ms',
                        'updates_per_second', 'memory_usage_mb', 'cpu_usage_percent']
        
        # Filtrer les colonnes existantes
        cols = [c for c in available_cols if c in group.columns]
        
        if cols:  # Vérifier qu'il y a des colonnes à afficher
            print(group[cols].to_string(index=False))
    
    # Afficher un résumé des performances si nous avons des données
    if not df.empty:
        print("\n" + "="*120)
        print("Résumé des performances moyennes :")
        
        # Créer un résumé des métriques disponibles
        summary_metrics = {}
        
        if 'avg_latency_ms' in df.columns:
            summary_metrics['avg_latency_ms'] = 'mean'
        if 'updates_per_second' in df.columns:
            summary_metrics['updates_per_second'] = 'mean'
        if 'memory_usage_mb' in df.columns:
            summary_metrics['memory_usage_mb'] = 'mean'
        if 'cpu_usage_percent' in df.columns:
            summary_metrics['cpu_usage_percent'] = 'mean'
        
        # Afficher le résumé uniquement si nous avons des métriques à afficher
        if summary_metrics:
            summary = df.groupby('test_name').agg(summary_metrics).round(2)
            print("\n" + str(summary) + "\n")
    
    print("="*120 + "\n")

async def main():
    """Fonction principale."""
    # Création d'un mock de l'API
    api = AsyncMock()
    
    # Nombre de paires à tester (ajuster selon les besoins)
    num_pairs = 50  # Augmenter pour des tests plus intensifs
    
    # Création et exécution du testeur
    tester = OrderBookPerformanceTester(api, num_pairs=num_pairs)
    results = await tester.run_tests()
    
    # Affichage des résultats
    print_results_table(results)

if __name__ == "__main__":
    asyncio.run(main())
