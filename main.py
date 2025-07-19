#!/usr/bin/env python3
"""
Script principal pour lancer le trader automatisé Kraken.

Ce script initialise et lance le trader avec les paramètres de configuration par défaut.
"""
import yaml
import json
import os
import signal
import sys
import asyncio
import logging
import logging.config
from typing import Dict, Any, Optional
from datetime import datetime, timedelta
from pathlib import Path

# Configuration du logging avant les autres imports
import coloredlogs
from dotenv import load_dotenv

# Configuration du logging avec couleurs
coloredlogs.install(
    level='INFO',
    fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    level_styles={
        'debug': {'color': 'blue'},
        'info': {'color': 'green'},
        'warning': {'color': 'yellow'},
        'error': {'color': 'red'},
        'critical': {'color': 'red', 'bold': True}
    }
)

# Imports des modules du projet
from src.ui.shell import TradingShell
from src.core.database import init_db
from src.core.config import Config
from src.core.api.kraken import KrakenAPI
from src.core.trading.advanced_trader import AdvancedAITrader
from src.core.services.pair_selector import PairSelector
from src.core.simulation_mode import SimulationConfig, SimulationMode
from src.core.database import db_manager

# Configuration du logger principal
logger = logging.getLogger(__name__)

# Charger les variables d'environnement depuis .env
load_dotenv('.env')
logger.info(f"Variables d'environnement chargées depuis .env: POSTGRES_USER={os.environ.get('POSTGRES_USER', 'non défini')}")



# Ajouter le dossier parent du projet au PYTHONPATH
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Ajouter le dossier src au PYTHONPATH
def setup_colored_logging():
    """Configure le logging avec des couleurs pour une meilleure lisibilité."""
    try:
        import coloredlogs
        coloredlogs.install(
            level='INFO',
            fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
            level_styles={
                'debug': {'color': 'blue'},
                'info': {'color': 'green'},
                'warning': {'color': 'yellow'},
                'error': {'color': 'red'},
                'critical': {'color': 'red', 'bold': True}
            }
        )
    except ImportError:
        pass

def setup_logging():
    """Configure le logging pour le bot."""
    config = Config()
    log_dir = os.path.expanduser(os.getenv('LOG_DIR', '~/kraken_bot_logs'))
    os.makedirs(log_dir, exist_ok=True)
    
    # Configuration du logging
    logging.basicConfig(
        level=logging.DEBUG,  # Forcer le niveau DEBUG
        format=config.logging.format,
        handlers=[
            logging.FileHandler(os.path.join(log_dir, 'trading_bot.log')),
            logging.StreamHandler()
        ]
    )
    
    # Configuration spécifique pour le logger root
    root_logger = logging.getLogger()
    root_logger.setLevel(config.logging.level)
    
    # Configuration pour les loggers spécifiques
    for name in ['urllib3', 'aiohttp', 'asyncio']:
        logging.getLogger(name).setLevel(logging.WARNING)
    
    # Activer les logs de débogage pour le module pair_selector
    logging.getLogger('src.core.services.pair_selector').setLevel(logging.DEBUG)
    
    # Configuration des logs de la bibliothèque requests
    requests_log = logging.getLogger("urllib3")
    requests_log.setLevel(logging.WARNING)
    requests_log.propagate = True

def load_environment():
    """Charge les variables d'environnement depuis le fichier .env"""
    env_path = os.path.join(os.path.dirname(__file__), '.env')
    if os.path.exists(env_path):
        load_dotenv(env_path)
        logger.info("Fichier .env chargé avec succès")
    else:
        logger.warning("Aucun fichier .env trouvé, utilisation des variables d'environnement système")

def check_environment():
    """Vérifie que les variables d'environnement requises sont définies"""
    required_vars = {
        'KRAKEN_API_KEY': 'Clé API Kraken manquante',
        'KRAKEN_API_SECRET': 'Secret API Kraken manquant',
        'POSTGRES_USER': "Nom d'utilisateur PostgreSQL manquant",
        'POSTGRES_PASSWORD': "Mot de passe PostgreSQL manquant",
        'POSTGRES_DB': "Nom de la base de données PostgreSQL manquant"
    }
    
    missing_vars = []
    for var, message in required_vars.items():
        if not os.getenv(var):
            missing_vars.append((var, message))
    
    if missing_vars:
        errors = "\n".join([f"- {msg}" for _, msg in missing_vars])
        logger.error(f"Configuration invalide:\n{errors}")
        sys.exit(1)
    
    # Vérification des types et des valeurs
    try:
        config = Config()
        if not config.api.key or not config.api.secret:
            logger.error("Les clés API Kraken ne peuvent pas être vides")
            sys.exit(1)
        
        if not config.db_config['user'] or not config.db_config['password']:
            logger.error("Les identifiants de base de données ne peuvent pas être vides")
            sys.exit(1)
    except Exception as e:
        logger.error(f"Erreur lors de la validation de la configuration: {str(e)}")
        sys.exit(1)
        logger.info("Veuillez créer un fichier .env avec vos identifiants Kraken")
        logger.info("Exemple de contenu pour .env:")
        logger.info("KRAKEN_API_KEY=votre_cle_api")
        logger.info("KRAKEN_API_SECRET=votre_cle_secrete")
        return False
    return True

def get_trading_config() -> dict:
    """
    Récupère la configuration du trading depuis les variables d'environnement.
    
    Returns:
        dict: Dictionnaire contenant la configuration du trading
    """
    return {
        'trading_pair': os.getenv('TRADING_PAIR', 'BTC/USD').strip(),
        'trading_amount': float(os.getenv('TRADING_AMOUNT', '0.01')),
        'trading_fee': float(os.getenv('TRADING_FEE', '0.0026')),
        'risk_level': os.getenv('RISK_LEVEL', 'medium').lower(),
        'quote_currency': os.getenv('QUOTE_CURRENCY', 'USD').upper(),
        'min_volume_btc': float(os.getenv('MIN_VOLUME_BTC', '5.0')),
        'max_spread_pct': float(os.getenv('MAX_SPREAD_PCT', '0.5')),
        'min_trades_24h': int(os.getenv('MIN_TRADES_24H', '100')),
        'custom_pairs': [p.strip() for p in os.getenv('CUSTOM_PAIRS', '').split(',') if p.strip()],
        'max_daily_drawdown': float(os.getenv('MAX_DAILY_DRAWDOWN', '5.0')),
        'risk_percentage': float(os.getenv('RISK_PERCENTAGE', '1.0')),
        'max_leverage': float(os.getenv('MAX_LEVERAGE', '1.0'))
    }

async def main_async():
    """
    Fonction principale asynchrone du bot de trading.
    
    Gère l'initialisation, le démarrage et l'arrêt propre du bot.
    """
    trader = None
    shell = None
    start_time = datetime.now()
    
    try:
        # Vérifier et initialiser la base de données
        logger.info("🔧 Initialisation de la base de données...")
        try:
            await init_db()
            logger.info("✅ Base de données initialisée avec succès")
        except Exception as e:
            logger.error(f"❌ Erreur lors de l'initialisation de la base de données: {e}")
            raise
        
        # Vérifier les variables d'environnement
        logger.info("🔍 Vérification des variables d'environnement...")
        if not all([os.getenv('KRAKEN_API_KEY'), os.getenv('KRAKEN_API_SECRET')]):
            error_msg = "❌ Erreur: Les clés API KRAKEN_API_KEY et KRAKEN_API_SECRET sont requises"
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        # Vérifier la connexion à la base de données
        try:
            # Assuming db_manager is defined elsewhere or needs to be initialized
            # For now, we'll just check if it's available
            if 'db_manager' in locals() and db_manager:
                await db_manager.execute("SELECT 1")
                logger.info("✅ Connexion à la base de données établie")
            else:
                logger.warning("⚠️  Gestionnaire de base de données non initialisé, impossible de vérifier la connexion.")
        except Exception as e:
            logger.error(f"❌ Erreur de connexion à la base de données: {e}")
            raise
        
        # Initialiser la configuration
        logger.info("⚙️ Initialisation de la configuration...")
        config = Config()
        
        # Initialiser l'API Kraken
        logger.info("🔌 Initialisation de l'API Kraken...")
        kraken_api = KrakenAPI(
            api_key=config.api_config['api_key'],
            api_secret=config.api_config['api_secret']
        )
        
        # Charger la configuration YAML
        config_path = os.path.join(os.path.dirname(__file__), 'config', 'config.yaml')
        with open(config_path, 'r') as f:
            config_data = yaml.safe_load(f)
        
        # Récupérer les paires de trading depuis la configuration
        trading_pairs = config_data.get('trading', {}).get('pairs', [])
        logger.info(f"📊 Paires chargées depuis la configuration: {trading_pairs}")
        
        # Initialiser le PairSelector avec les paires configurées
        logger.info("🔍 Initialisation du sélecteur de paires...")
        simulation_config = SimulationConfig(
            mode=SimulationMode.REAL_TIME,
            available_pairs=trading_pairs
        )
        pair_selector = PairSelector(
            api=kraken_api,
            config=simulation_config,
            max_pairs=50,
            min_volume=100000,
            volatility_window=20,
            momentum_window=14
        )
        
        # Configuration du trader
        TRADING_CONFIG = get_trading_config()
        
        # Récupérer les paires valides
        valid_pairs = await pair_selector.get_valid_pairs()
        logger.info(f"✅ {len(valid_pairs)} paires valides récupérées pour le trading")
        
        # Ajouter les paires valides à la configuration
        TRADING_CONFIG['trading_pairs'] = valid_pairs
        
        # Initialiser le trader avec la configuration mise à jour
        logger.info("🤖 Initialisation du trader...")
        trader = AdvancedAITrader(
            api=kraken_api,
            config=TRADING_CONFIG,
            db_manager=db_manager
        )
        
        # S'assurer que les paires valides sont bien dans la configuration du trader
        trader.config['trading_pairs'] = valid_pairs
        
        # Initialiser l'interface CLI
        logger.info("🖥️ Initialisation de l'interface CLI...")
        shell = TradingShell(trader=trader)
            
        # Charger la configuration et récupérer les paires valides
        logger.info("⚙️  Chargement de la configuration...")
        try:
            # Récupérer les paires valides directement via get_valid_pairs
            trading_config = config_data.get('trading', {})
            min_score = trading_config.get('min_score', 0.0)
            min_volume = trading_config.get('min_volume', 0)
            valid_pairs = await pair_selector.get_valid_pairs(min_score=min_score)
            logger.info(f"✅ Configuration chargée: {len(valid_pairs)} paires Kraken valides (chargement dynamique)")
            
            # Afficher les paires sélectionnées
            if valid_pairs:
                logger.info("📊 Paires sélectionnées pour le trading:")
                for i, pair in enumerate(valid_pairs, 1):
                    logger.info(f"   {i}. {pair}")
            else:
                logger.warning("⚠️  Aucune paire valide trouvée avec les critères actuels")
                
        except Exception as e:
            logger.error(f"❌ Erreur lors du chargement de la configuration: {e}", exc_info=True)
            raise
        
        # Initialiser l'API Kraken
        logger.info("🔌 Initialisation de l'API Kraken...")
        try:
            async with KrakenAPI(
                api_key=os.getenv('KRAKEN_API_KEY'),
                api_secret=os.getenv('KRAKEN_API_SECRET')
            ) as kraken_api:
                # Tester la connexion à l'API
                await kraken_api.get_server_time()
                logger.info("✅ Connexion à l'API Kraken établie")
        except Exception as e:
            logger.error(f"❌ Erreur lors de l'initialisation de l'API Kraken: {e}")
            raise
        
        # Initialiser le prédicteur de marché
        logger.info("🧠 Initialisation du prédicteur de marché...")
        try:
            from src.ml.predictor import MLPredictor
            predictor = MLPredictor(db_manager=db_manager)
            logger.info("✅ Prédicteur de marché initialisé")
        except Exception as e:
            logger.error(f"❌ Erreur lors de l'initialisation du prédicteur: {e}")
            raise
        
        # Le sélecteur de paires a déjà été initialisé plus haut
        # avec la configuration appropriée
        
        # Initialiser l'interface CLI
        logger.info("🖥️  Initialisation de l'interface CLI...")
        shell = TradingShell(trader=trader)
        
        # Démarrer l'interface CLI
        await shell.run()
        
        # Les paires valides ont déjà été récupérées plus haut avec pair_selector.get_valid_pairs()
        # et stockées dans valid_pairs
        
        # Initialiser le trader
        logger.info("🤖 Initialisation du trader...")
        try:
            trader = AdvancedAITrader(
                api=kraken_api,
                predictor=predictor,
                config=config,  # Passer la configuration au trader
                db_manager=db_manager  # Passer le gestionnaire de base de données
            )
            logger.info("✅ Trader initialisé avec succès")
        except Exception as e:
            logger.error(f"❌ Erreur lors de l'initialisation du trader: {e}")
            raise
        
        # Afficher le résumé de démarrage
        logger.info("\n" + "="*60)
        logger.info(f"🚀 DÉMARRAGE DU BOT DE TRADING - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info("="*60)
        logger.info(f"📈 Paires actives: {', '.join(valid_pairs)}")
        logger.info(f"⏱  Timeframe: {TRADING_CONFIG.get('timeframe', 'N/A')} minutes")
        logger.info(f"🎯 Niveau de risque: {config['risk_level'].upper()}")
        logger.info(f"💵 Balance initiale: {config.get('initial_balance', 'N/A')} {config.get('quote_currency', 'N/A')}")
        logger.info("="*60 + "\n")

        # Démarrer le trader de manière asynchrone sur les paires sélectionnées
        # Créer un mapping des symboles pour l'API Kraken
        symbol_mapping = {pair: pair.replace('/', '') for pair in valid_pairs}
        await trader.run()
        
    except KeyboardInterrupt:
        logger.info("\n🛑 Arrêt demandé par l'utilisateur")
    except Exception as e:
        logger.error(f"\n❌ ERREUR CRITIQUE: {e}", exc_info=True)
        # Sauvegarder l'erreur pour l'analyse ultérieure
        try:
            error_log = {
                'timestamp': datetime.now().isoformat(),
                'error': str(e),
                'type': type(e).__name__,
                'traceback': str(sys.exc_info()[2]) # Corrected traceback retrieval
            }
            with open(os.path.join('logs', 'error_log.json'), 'a') as f:
                f.write(json.dumps(error_log) + '\n')
        except Exception as log_error:
            logger.error(f"Échec de l'enregistrement du journal d'erreur: {log_error}")
        raise
    finally:
        end_time = datetime.now()
        duration = end_time - start_time
        
        logger.info("\n" + "="*60)
        logger.info(f"🛑 ARRÊT DU BOT DE TRADING - {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"⏱  Durée de fonctionnement: {duration}")
        
        if trader is not None:
            try:
                logger.info("Arrêt en cours du trader...")
                if asyncio.iscoroutinefunction(trader.stop):
                    await trader.stop()
                else:
                    trader.stop()
                logger.info("✅ Trader arrêté avec succès")
            except Exception as e:
                logger.error(f"❌ Erreur lors de l'arrêt du trader: {e}")
            
            # Sauvegarder les logs et les statistiques
            try:
                save_trading_summary()
                logger.info("✅ Résumé de trading sauvegardé")
            except Exception as e:
                logger.error(f"❌ Erreur lors de la sauvegarde du résumé: {e}")
        
        # Fermer la connexion à la base de données
        try:
            await db_manager.close()
            logger.info("✅ Connexion à la base de données fermée")
        except Exception as e:
            logger.error(f"❌ Erreur lors de la fermeture de la base de données: {e}")
        
        logger.info("="*60)

def pair_selector_summary(selector) -> str:
    """Génère un résumé des analyses du sélecteur de paires."""
    if not selector or not hasattr(selector, '_pair_metrics'):
        return "Aucune analyse disponible"
        
    summary = ["\n=== Analyse des paires ===\n"]
    
    # Trier les paires par score de marché décroissant
    pairs = sorted(
        [(k, v) for k, v in selector._pair_metrics.items() if k in selector._recommended_pairs],
        key=lambda x: x[1].get('market_score', 0),
        reverse=True
    )
    
    for pair_id, metrics in pairs[:10]:  # Afficher les 10 meilleures paires
        pair = selector._recommended_pairs.get(pair_id)
        if not pair:
            continue
            
        analysis = pair.analysis
        trend_arrow = '↑' if analysis.trend == 'up' else '↓' if analysis.trend == 'down' else '→'
        
        summary.append(
            f"{pair_id} | "
            f"Score: {metrics.get('market_score', 0):.2f} | "
            f"Prix: {metrics.get('last_price', 0):.2f} {pair_id[-3:]} | "
            f"Tendance: {trend_arrow} {analysis.trend_strength*100:.1f}% | "
            f"Volatilité: {analysis.volatility*100:.2f}% | "
            f"Volume: {metrics.get('volume_24h', 0):.2f} BTC"
        )
    
    return "\n".join(summary)

def save_trading_summary():
    """Sauvegarde un résumé de la session de trading."""
    try:
        summary = {
            'end_time': datetime.utcnow().isoformat(),
            'pairs_traded': [],
            'performance': {}
        }
        
        # Ici, vous pourriez ajouter plus de données de performance
        # à partir de l'instance du trader
        
        with open('trading_summary.json', 'w') as f:
            json.dump(summary, f, indent=2)
            
        logger.info("Résumé de trading sauvegardé")
    except Exception as e:
        logger.error(f"Erreur lors de la sauvegarde du résumé: {e}")
        
    # Créer un gestionnaire de signaux pour l'arrêt propre
    def signal_handler(sig, frame):
        logger.info("\nSignal d'arrêt reçu, arrêt en cours...")
        raise KeyboardInterrupt()
        
    # Enregistrer les gestionnaires de signaux
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

if __name__ == "__main__":
    try:
        asyncio.run(main_async())
    except KeyboardInterrupt:
        logger.info("\nArrêt demandé par l'utilisateur")
    except Exception as e:
        logger.error(f"Erreur lors de l'exécution du bot : {e}")
