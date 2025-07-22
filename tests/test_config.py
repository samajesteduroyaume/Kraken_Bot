import os
import unittest
from unittest.mock import patch
from src.core.config_adapter import Config  # Migré vers le nouvel adaptateur


class TestConfig(unittest.TestCase):
    def setUp(self):
        # Sauvegarde des variables d'environnement existantes
        self.original_env = os.environ.copy()

    def tearDown(self):
        # Restauration des variables d'environnement
        os.environ.clear()
        os.environ.update(self.original_env)

    def test_main_config(self):
        # Test de la configuration principale
        os.environ['KRAKEN_API_KEY'] = 'test_key'
        os.environ['KRAKEN_API_SECRET'] = 'test_secret'
        os.environ['POSTGRES_PASSWORD'] = 'test_password'
        
        config = Config
        self.assertEqual(config.api_config['api_key'], 'test_key')
        self.assertEqual(config.api_config['api_secret'], 'test_secret')
        self.assertEqual(config.db_config['password'], 'test_password')

        # Test de configuration invalide
        os.environ['KRAKEN_API_KEY'] = ''
        os.environ['KRAKEN_API_SECRET'] = ''
        os.environ['POSTGRES_PASSWORD'] = ''
        
        with self.assertRaises(ValueError):
            config = Config

if __name__ == '__main__':
    unittest.main()
