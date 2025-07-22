"""
Configuration des tests pour le bot de trading Kraken.

Ce fichier contient des fixtures et des configurations communes pour les tests.
"""
import os
import pytest
from unittest.mock import patch, MagicMock

# Configuration de test pour les variables d'environnement
TEST_ENV = {
    'KRAKEN_API_KEY': 'test_api_key',
    'KRAKEN_API_SECRET': 'test_api_secret',
    'POSTGRES_USER': 'test_user',
    'POSTGRES_PASSWORD': 'test_password',
    'POSTGRES_DB': 'test_db',
    'POSTGRES_HOST': 'localhost',
    'POSTGRES_PORT': '5432',
    'REDIS_HOST': 'localhost',
    'REDIS_PORT': '6379',
    'REDIS_DB': '1',  # Utiliser une base de données différente pour les tests
}

@pytest.fixture(autouse=True, scope='session')
def setup_test_environment():
    """Configure l'environnement de test."""
    # Sauvegarder l'environnement actuel
    original_env = os.environ.copy()
    
    # Configurer les variables d'environnement de test
    os.environ.update(TEST_ENV)
    
    # Configurer les chemins de fichiers de test
    os.environ['CONFIG_FILE'] = 'tests/test_config.yaml'
    
    yield  # Exécuter les tests
    
    # Restaurer l'environnement d'origine
    os.environ.clear()
    os.environ.update(original_env)

@pytest.fixture(autouse=True)
def mock_redis():
    """Mock pour le client Redis."""
    with patch('redis.Redis') as mock_redis:
        mock_redis.return_value.ping.return_value = True
        yield mock_redis

@pytest.fixture(autouse=True)
def mock_kraken_api():
    """Mock pour l'API Kraken."""
    with patch('src.core.api.kraken.KrakenAPI') as mock_kraken:
        # Configurer les réponses simulées de l'API Kraken
        mock_kraken.return_value.fetch_balance.return_value = {
            'free': {'BTC': 1.5, 'USD': 10000},
            'used': {},
            'total': {'BTC': 1.5, 'USD': 10000},
        }
        yield mock_kraken

@pytest.fixture
def test_config():
    """Retourne une configuration de test."""
    from src.core.config_adapter import Config
    
    # Forcer le rechargement de la configuration avec les variables d'environnement de test
    Config._instance = None
    Config._initialized = False
    
    return Config
