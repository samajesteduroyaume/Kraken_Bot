"""
Tests pour l'adaptateur de configuration.

Ces tests vérifient que l'adaptateur de configuration fonctionne correctement
avec la nouvelle configuration YAML tout en maintenant la compatibilité avec
l'ancien système.
"""
import os
import sys
import pytest
from pathlib import Path

# Ajouter le répertoire racine au PYTHONPATH
root_dir = str(Path(__file__).parent.parent)
if root_dir not in sys.path:
    sys.path.append(root_dir)

from src.core.config_adapter import Config

def test_config_adapter_initialization():
    """Teste l'initialisation de l'adaptateur de configuration."""
    # Vérifier que l'instance est correctement initialisée
    assert Config is not None
    
    # Vérifier que les configurations principales sont chargées
    assert hasattr(Config, 'api_config')
    assert hasattr(Config, 'trading_config')
    assert hasattr(Config, 'db_config')
    assert hasattr(Config, 'ml_config')
    assert hasattr(Config, 'log_config')
    assert hasattr(Config, 'credentials')
    assert hasattr(Config, 'reporting_config')

def test_api_config():
    """Teste la configuration de l'API."""
    api_config = Config.api_config
    
    # Vérifier les clés requises
    assert 'api_key' in api_config
    assert 'api_secret' in api_config
    assert 'base_url' in api_config
    assert 'version' in api_config
    
    # Vérifier les valeurs par défaut
    assert api_config['base_url'] == 'https://api.kraken.com'
    assert api_config['version'] == 'v0'
    assert isinstance(api_config['timeout'], int)

def test_trading_config():
    """Teste la configuration du trading."""
    trading_config = Config.trading_config
    
    # Vérifier les clés requises
    required_keys = [
        'pairs', 'min_score', 'min_volume', 'risk_per_trade',
        'stop_loss_percent', 'take_profit_percent', 'max_positions',
        'max_drawdown', 'max_holding_time', 'max_concurrent_requests',
        'analysis_timeout', 'max_pairs_to_analyze', 'exclude_illiquid',
        'min_daily_trades', 'min_market_cap', 'exclude_stablecoins',
        'exclude_leveraged'
    ]
    
    for key in required_keys:
        assert key in trading_config, f"Clé manquante: {key}"
    
    # Vérifier les types
    assert isinstance(trading_config['pairs'], list)
    assert isinstance(trading_config['min_score'], float)
    assert isinstance(trading_config['min_volume'], float)
    assert isinstance(trading_config['risk_per_trade'], float)
    assert isinstance(trading_config['max_positions'], int)

def test_db_config():
    """Teste la configuration de la base de données."""
    db_config = Config.db_config
    
    # Vérifier les clés requises
    required_keys = [
        'user', 'password', 'host', 'port', 'database', 'enabled'
    ]
    
    for key in required_keys:
        assert key in db_config, f"Clé manquante: {key}"
    
    # Vérifier les valeurs par défaut
    assert db_config['host'] == 'localhost'
    assert db_config['port'] == 5432
    assert db_config['database'] == 'kraken_bot'
    assert isinstance(db_config['enabled'], bool)

def test_ml_config():
    """Teste la configuration du machine learning."""
    ml_config = Config.ml_config
    
    # Vérifier les clés requises
    required_keys = [
        'enabled', 'model_path', 'window_size', 'train_size',
        'test_size', 'n_estimators', 'max_depth', 'retrain_interval'
    ]
    
    for key in required_keys:
        assert key in ml_config, f"Clé manquante: {key}"
    
    # Vérifier les types
    assert isinstance(ml_config['enabled'], bool)
    assert isinstance(ml_config['window_size'], int)
    assert isinstance(ml_config['train_size'], float)
    assert isinstance(ml_config['n_estimators'], int)

def test_log_config():
    """Teste la configuration des logs."""
    log_config = Config.log_config
    
    # Vérifier les clés requises
    required_keys = [
        'level', 'log_dir', 'max_bytes', 'backup_count',
        'console', 'file', 'format', 'date_format'
    ]
    
    for key in required_keys:
        assert key in log_config, f"Clé manquante: {key}"
    
    # Vérifier les valeurs par défaut
    assert log_config['level'] in ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
    assert log_config['log_dir'] == 'logs'
    assert isinstance(log_config['max_bytes'], int)
    assert isinstance(log_config['backup_count'], int)

def test_credentials():
    """Teste les informations d'identification."""
    credentials = Config.credentials
    
    # Vérifier les clés requises
    assert 'api_key' in credentials
    assert 'api_secret' in credentials

def test_reporting_config():
    """Teste la configuration des rapports."""
    reporting_config = Config.reporting_config
    
    # Vérifier les clés requises
    required_keys = [
        'enabled', 'report_dir', 'email_enabled', 'email_recipients',
        'daily_report', 'weekly_report', 'monthly_report'
    ]
    
    for key in required_keys:
        assert key in reporting_config, f"Clé manquante: {key}"
    
    # Vérifier les types
    assert isinstance(reporting_config['enabled'], bool)
    assert isinstance(reporting_config['email_enabled'], bool)
    assert isinstance(reporting_config['email_recipients'], list)

def test_dict_access():
    """Teste l'accès aux configurations comme à un dictionnaire."""
    # Test avec une clé existante
    assert 'api_key' in Config.api_config
    
    # Test avec une clé inexistante
    with pytest.raises(KeyError):
        _ = Config['cle_inexistante']
    
    # Test avec la méthode get()
    assert Config.get('api_key') is not None
    assert Config.get('cle_inexistante', 'valeur_par_defaut') == 'valeur_par_defaut'

def test_environment_variables():
    """Teste la récupération des variables d'environnement."""
    # Tester avec une variable existante
    os.environ['TEST_VAR'] = 'test_value'
    assert Config.get_environment_variable('TEST_VAR') == 'test_value'
    
    # Tester avec une valeur par défaut
    assert Config.get_environment_variable('NON_EXISTENT_VAR', 'default') == 'default'
    
    # Nettoyer
    del os.environ['TEST_VAR']

if __name__ == "__main__":
    pytest.main([__file__, '-v'])
