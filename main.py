#!/usr/bin/env python3
"""
Script principal pour le Kraken Trading Bot.

Ce module initialise et lance le bot de trading avec une architecture modulaire,
en utilisant l'injection de dépendances pour une meilleure maintenabilité.
"""
from __future__ import annotations

import asyncio
import logging
import os
import signal
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Dict, Optional

import coloredlogs
import sentry_sdk
from dotenv import load_dotenv
from loguru import logger

# Configuration du logging structuré avec Loguru
logger.remove()  # Supprime le gestionnaire par défaut
logger.add(
    sys.stderr,
    format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
    "<level>{level: <8}</level> | "
    "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
)

# Configuration de coloredlogs pour les logs de bibliothèques tierces
coloredlogs.install(
    level="INFO",
    fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level_styles={
        "debug": {"color": "blue"},
        "info": {"color": "green"},
        "warning": {"color": "yellow"},
        "error": {"color": "red"},
        "critical": {"color": "red", "bold": True},
    },
)

# Chargement des variables d'environnement
dotenv_path = Path(__file__).parent / ".env"
load_dotenv(dotenv_path=dotenv_path)

# Configuration de Sentry pour la surveillance des erreurs
sentry_sdk.init(
    dsn=os.getenv("SENTRY_DSN"),
    traces_sample_rate=1.0,
    profiles_sample_rate=1.0,
    environment=os.getenv("ENV", "development"),
)

# Import après configuration de l'environnement
from src.core.container import Container
from src.core.config import settings
from src.core.exceptions import ConfigError
from src.core.strategy_loader import StrategyLoader
from src.infrastructure.database import init_db, migrate_db
from src.infrastructure.kraken import KrakenClient
from src.ui.shell import TradingShell

# Initialisation du conteneur d'injection de dépendances
container = Container()
container.config.from_dict(settings.dict())

class KrakenTradingBot:
    """Classe principale du bot de trading Kraken."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialise le bot avec une configuration optionnelle."""
        self.config = config or {}
        self.running = False
        self.strategies = []
        self.kraken_client: Optional[KrakenClient] = None
        self.strategy_loader = StrategyLoader()

    async def initialize(self) -> None:
        """Initialise les composants du bot."""
        logger.info("🚀 Initialisation du Kraken Trading Bot")

        # Initialisation de la base de données
        await self._initialize_database()

        # Initialisation du client Kraken
        self.kraken_client = await self._initialize_kraken_client()

        # Chargement des stratégies
        await self._load_strategies()

        logger.info("✅ Initialisation terminée avec succès")

    async def _initialize_database(self) -> None:
        """Initialise la base de données et applique les migrations."""
        logger.info("🔌 Connexion à la base de données...")
        await init_db()
        await migrate_db()
        logger.info("✅ Base de données initialisée")

    async def _initialize_kraken_client(self) -> KrakenClient:
        """Initialise et retourne le client Kraken."""
        logger.info("🔌 Connexion à l'API Kraken...")
        client = KrakenClient()
        await client.initialize()
        logger.info("✅ Connecté à l'API Kraken")
        return client

    async def _load_strategies(self) -> None:
        """Charge et initialise les stratégies de trading."""
        logger.info("🔄 Chargement des stratégies...")
        self.strategies = await self.strategy_loader.load_strategies()
        logger.info(f"✅ {len(self.strategies)} stratégies chargées")

    async def start(self) -> None:
        """Démarre le bot de trading."""
        if self.running:
            logger.warning("Le bot est déjà en cours d'exécution")
            return

        self.running = True
        logger.info("🚀 Démarrage du bot de trading")

        try:
            # Démarrer les stratégies
            for strategy in self.strategies:
                await strategy.initialize()
                await strategy.start()
                logger.info(f"▶️ Stratégie démarrée: {strategy.name}")

            # Boucle principale
            while self.running:
                await asyncio.sleep(1)

        except asyncio.CancelledError:
            logger.info("Arrêt propre du bot demandé")
        except Exception as e:
            logger.error(f"Erreur dans la boucle principale: {e}")
            sentry_sdk.capture_exception(e)
            raise
        finally:
            await self.stop()

    async def stop(self) -> None:
        """Arrête le bot de trading de manière propre."""
        if not self.running:
            return

        logger.info("🛑 Arrêt du bot de trading...")
        self.running = False

        # Arrêt des stratégies
        for strategy in self.strategies:
            try:
                await strategy.stop()
                logger.info(f"⏹️ Stratégie arrêtée: {strategy.name}")
            except Exception as e:
                logger.error(f"Erreur lors de l'arrêt de la stratégie {strategy.name}: {e}")
                sentry_sdk.capture_exception(e)

        # Fermeture du client Kraken
        if self.kraken_client:
            await self.kraken_client.close()

        logger.info("✅ Arrêt du bot terminé")


def check_environment() -> None:
    """
    Vérifie que toutes les variables d'environnement requises sont définies.
    
    Raises:
        ConfigError: Si des variables obligatoires sont manquantes
    """
    required_vars = [
        'KRAKEN_API_KEY',
        'KRAKEN_PRIVATE_KEY',
        'DATABASE_URL',
        'REDIS_URL'
    ]
    
    missing = [var for var in required_vars if not os.getenv(var)]
    if missing:
        error_msg = f"Variables d'environnement manquantes: {', '.join(missing)}"
        logger.error(error_msg)
        raise ConfigError(error_msg)
        
    logger.info("✅ Vérification de l'environnement terminée")


def get_trading_config() -> dict[str, Any]:
    """
    Récupère et valide la configuration du trading depuis les variables d'environnement.
    
    Returns:
        dict: Configuration du trading avec des valeurs typées
        
    Raises:
        ConfigError: Si la configuration est invalide
    """
    try:
        config = {
            'api_key': os.getenv('KRAKEN_API_KEY'),
            'private_key': os.getenv('KRAKEN_PRIVATE_KEY'),
            'trading_pairs': [p.strip() for p in os.getenv('TRADING_PAIRS', 'XBT/USD,ETH/USD').split(',')],
            'timeframe': os.getenv('TIMEFRAME', '1h'),
            'max_open_trades': int(os.getenv('MAX_OPEN_TRADES', '3')),
            'risk_per_trade': float(os.getenv('RISK_PER_TRADE', '1.0')),
            'dry_run': os.getenv('DRY_RUN', 'true').lower() == 'true',
            'log_level': os.getenv('LOG_LEVEL', 'INFO'),
            'redis_url': os.getenv('REDIS_URL', 'redis://localhost:6379/0'),
            'database_url': os.getenv('DATABASE_URL'),
            'sentry_dsn': os.getenv('SENTRY_DSN', ''),
            'environment': os.getenv('ENV', 'development'),
            'max_position_size': float(os.getenv('MAX_POSITION_SIZE', '10000')),  # en USD
            'max_drawdown': float(os.getenv('MAX_DRAWDOWN', '10.0')),  # en pourcentage
            'enable_telegram': os.getenv('ENABLE_TELEGRAM', 'false').lower() == 'true',
            'telegram_token': os.getenv('TELEGRAM_TOKEN'),
            'telegram_chat_id': os.getenv('TELEGRAM_CHAT_ID'),
        }
        
        # Validation des paires de trading
        if not config['trading_pairs'] or not all(config['trading_pairs']):
            raise ConfigError("Aucune paire de trading valide spécifiée")
            
        # Validation des clés API en mode production
        if not config['dry_run'] and (not config['api_key'] or not config['private_key']):
            raise ConfigError("Les clés d'API Kraken sont requises en mode production")
            
        return config
        
    except ValueError as e:
        raise ConfigError(f"Erreur de configuration: {str(e)}")


async def setup_bot() -> KrakenTradingBot:
    """
    Configure et initialise le bot de trading.
    
    Returns:
        KrakenTradingBot: Instance du bot configuré
    """
    # Chargement de la configuration
    load_environment()
    check_environment()
    config = get_trading_config()
    
    # Configuration du logging
    logger.remove()
    logger.add(
        sys.stderr,
        level=config['log_level'],
        format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
    )
    
    # Configuration de Sentry pour la production
    if config['sentry_dsn'] and config['environment'] == 'production':
        sentry_sdk.init(
            dsn=config['sentry_dsn'],
            environment=config['environment'],
            traces_sample_rate=1.0,
        )
    
    # Création et initialisation du bot
    bot = KrakenTradingBot(config)
    await bot.initialize()
    
    return bot


def handle_shutdown(signum, frame) -> None:
    """Gestionnaire pour les signaux d'arrêt."""
    logger.warning(f"Signal {signum} reçu, préparation de l'arrêt...")
    raise KeyboardInterrupt("Arrêt demandé par l'utilisateur")


async def main_async() -> None:
    """
    Fonction principale asynchrone du bot de trading.
    
    Gère l'initialisation, le démarrage et l'arrêt propre du bot.
    """
    bot = None
    
    try:
        # Configuration et initialisation
        bot = await setup_bot()
        
        # Enregistrement des gestionnaires de signaux
        for sig in (signal.SIGINT, signal.SIGTERM):
            signal.signal(sig, handle_shutdown)
        
        # Démarrer le bot
        await bot.start()
        
    except asyncio.CancelledError:
        logger.info("Arrêt propre du bot demandé")
        logger.info("✅ Tous les composants sont prêts")
        logger.info("🚀 Démarrage du bot de trading...")
        
        await shell.run()
        
    except KeyboardInterrupt:
        logger.info("\n🛑 Arrêt demandé par l'utilisateur")
    except Exception as e:
        logger.error(f"\n❌ ERREUR CRITIQUE: {e}", exc_info=True)
        raise

def main():
    """
    Point d'entrée principal du programme.
    
    Configure le logging, charge les variables d'environnement et lance la boucle asynchrone.
    """
    try:
        # Chargement des variables d'environnement
        load_environment()
        
        # Démarrage de la boucle asynchrone
        asyncio.run(main_async())
        
    except Exception as e:
        logger.error(f"\n❌ ERREUR LORS DE L'EXÉCUTION: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()
