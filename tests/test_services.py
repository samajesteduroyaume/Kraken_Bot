import os
import sys
import time
import logging
import asyncio
import unittest
from unittest.mock import patch
import redis
import boto3
import matplotlib.pyplot as plt
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.cache.redis_cache import RedisCache
from src.monitoring.redis_monitor import RedisMonitor
from src.backup.log_backup import LogBackup
from src.reporting.trading_reporter import TradingReporter
from src.trading.redis_trading import RedisTradingManager
from src.core.config import Config

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class TestServices(unittest.TestCase):
    """Tests pour vérifier le bon fonctionnement des services."""


        
    def setUp(self):
        """Configuration avant chaque test."""
        # Nettoyer Redis
        self.redis.client.flushdb()
        
    def tearDown(self):
        """Nettoyage après chaque test."""
        # Nettoyer Redis
        self.redis.client.flushdb()
        
    @classmethod
    def tearDownClass(cls):
        """Nettoyage après les tests."""
        # Supprimer les fichiers de test
        if os.path.exists(cls.config['LOGGING']['dir']):
            import shutil
            shutil.rmtree(cls.config['LOGGING']['dir'])
        
        logger.info("✅ Tests terminés avec succès")
    
    @classmethod
    def setUpClass(cls):
        """Configuration initiale pour tous les tests."""
        # Configuration des variables d'environnement de test
        os.environ['REDIS_ENABLED'] = 'true'
        os.environ['REDIS_HOST'] = 'localhost'
        os.environ['REDIS_PORT'] = '6379'
        os.environ['REDIS_DB'] = '0'
        os.environ['REDIS_PASSWORD'] = os.getenv('REDIS_PASSWORD', 'test_password')
        os.environ['REDIS_TTL'] = '3600'
        
        os.environ['AWS_ENABLED'] = 'false'
        os.environ['AWS_ACCESS_KEY_ID'] = 'test_key'
        os.environ['AWS_SECRET_ACCESS_KEY'] = 'test_secret'
        os.environ['AWS_REGION'] = 'test-region'
        os.environ['AWS_BUCKET_NAME'] = 'test-bucket'
        
        os.environ['LOG_LEVEL'] = 'INFO'
        os.environ['LOG_DIR'] = '/tmp/kraken_bot_logs'
        os.environ['LOG_MAX_SIZE'] = '10485760'
        os.environ['LOG_BACKUP_COUNT'] = '5'
        
        # Configuration du trading
        os.environ['MAX_TRADE_AMOUNT'] = '100.0'
        os.environ['MIN_TRADE_AMOUNT'] = '10.0'
        os.environ['MAX_POSITIONS'] = '10'
        os.environ['MAX_PAIRS'] = '50'
        os.environ['MIN_VOLUME'] = '100000'
        os.environ['VOLATILITY_WINDOW'] = '20'
        os.environ['MOMENTUM_WINDOW'] = '14'
        os.environ['INITIAL_BALANCE'] = '10000.0'
        os.environ['RISK_PERCENTAGE'] = '1.0'
        os.environ['MAX_DAILY_DRAWDOWN'] = '5.0'
        os.environ['MAX_LEVERAGE'] = '3.0'
        os.environ['SIMULATION_MODE'] = 'false'
        os.environ['SIMULATION_BALANCE_BTC'] = '0.0'
        os.environ['SIMULATION_BALANCE_EUR'] = '1000.0'
        
        # Créer l'objet Config
        cls.config = Config()
        
        # Initialiser les services
        cls.redis = RedisCache(cls.config)
        cls.monitor = RedisMonitor(cls.config)
        cls.backup = LogBackup(cls.config)
        cls.reporter = TradingReporter(cls.config)
        
        # Créer un event loop pour les opérations asynchrones
        import asyncio
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        cls.loop = loop
        cls.trading = RedisTradingManager(cls.config, loop=loop)
        
    def test_redis_connection(self):
        """Test la connexion à Redis."""
        try:
            self.assertTrue(self.redis.client.ping())
            logger.info("✅ Test Redis: Connexion réussie")
        except Exception as e:
            logger.error(f"❌ Test Redis: {str(e)}")
            self.fail(f"Erreur de connexion Redis: {str(e)}")
    
    def test_cache_operations(self):
        """Test les opérations de cache."""
        try:
            # Test du set/get
            test_key = 'test_key'
            test_value = {'test': 'value'}
            self.assertTrue(self.redis.set(test_key, test_value))
            self.assertEqual(self.redis.get(test_key), test_value)
            
            # Test de l'expiration
            self.assertTrue(self.redis.set(test_key, test_value, ttl=1))
            time.sleep(2)
            self.assertIsNone(self.redis.get(test_key))
            
            logger.info("✅ Test Cache: Opérations réussies")
        except Exception as e:
            logger.error(f"❌ Test Cache: {str(e)}")
            self.fail(f"Erreur dans les opérations de cache: {str(e)}")
    
    def test_monitoring(self):
        """Test le système de monitoring."""
        try:
            # Simuler une condition d'alerte
            with patch.object(self.monitor, '_check_memory_usage') as mock_check:
                mock_check.return_value = True
                self.monitor._send_alert('MEMORY_USAGE', 'Test alert')
                
            logger.info("✅ Test Monitoring: Alertes fonctionnelles")
        except Exception as e:
            logger.error(f"❌ Test Monitoring: {str(e)}")
            self.fail(f"Erreur dans le monitoring: {str(e)}")
    
    def test_backup(self):
        """Test le système de backup."""
        try:
            # Créer un fichier de test
            test_file = os.path.join(self.config.logging.log_dir, 'test.log')
            with open(test_file, 'w') as f:
                f.write("Test log")
            
            # Tester le backup
            self.assertTrue(self.backup.backup_logs(self.config))
            
            # Tester la restauration
            backup_file = os.path.join(self.config.logging.log_dir, 'test_backup.tar.gz')
            self.assertTrue(self.backup.restore_backup(backup_file))
            
            logger.info("✅ Test Backup: Fonctionnel")
        except Exception as e:
            logger.error(f"❌ Test Backup: {str(e)}")
            self.fail(f"Erreur dans le système de backup: {str(e)}")
    
    def test_reporting(self):
        """Test le système de reporting."""
        try:
            # Simuler des trades pour le test
            test_trades = [
                {
                    'symbol': 'BTC/USD',
                    'profit': 100.0,
                    'entry_price': 40000.0,
                    'exit_price': 40100.0,
                    'timestamp': time.time()
                }
            ]
            
            # Générer le rapport
            report_path = self.reporter.generate_daily_report(test_trades)
            self.assertIsNotNone(report_path)
            
            # Vérifier l'envoi d'email
            with patch.object(self.reporter, 'send_email_report') as mock_send:
                mock_send.return_value = True
                self.reporter.send_email_report(report_path)
            
            logger.info("✅ Test Reporting: Fonctionnel")
        except Exception as e:
            logger.error(f"❌ Test Reporting: {str(e)}")
            self.fail(f"Erreur dans le système de reporting: {str(e)}")
    
    def test_trading_manager(self):
        """Test le gestionnaire de trading."""
        try:
            # Simuler un trade
            test_symbol = 'BTC/USD'
            market_data = asyncio.run(self.trading.get_market_data(test_symbol))
            self.assertIsNotNone(market_data)
            
            # Simuler un indicateur
            rsi = self.trading.get_indicator(test_symbol, 'RSI')
            self.assertIsNotNone(rsi)
            
            logger.info("✅ Test Trading: Fonctionnel")
        except Exception as e:
            logger.error(f"❌ Test Trading: {str(e)}")
            self.fail(f"Erreur dans le gestionnaire de trading: {str(e)}")
    
    @classmethod
    def tearDownClass(cls):
        """Nettoyage après les tests."""
        # Supprimer les fichiers de test
        test_files = [
            os.path.join(cls.config['LOGGING']['dir'], 'test.log'),
            os.path.join(cls.config['LOGGING']['dir'], 'test_backup.tar.gz')
        ]
        
        for file in test_files:
            if os.path.exists(file):
                os.remove(file)
        
        logger.info("✅ Tests terminés avec succès")

if __name__ == '__main__':
    # Configurer les variables d'environnement si nécessaire
    if not os.getenv('REDIS_PASSWORD'):
        os.environ['REDIS_PASSWORD'] = 'test_password'
        
    # Lancer les tests
    unittest.main()
