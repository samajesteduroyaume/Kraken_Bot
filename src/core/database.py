"""
Gestionnaire de base de données asynchrone pour le bot de trading Kraken.
"""

import logging
import asyncpg
import time
from typing import Any, List, Optional, Dict
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from src.core.config import Config
import traceback

# Alias pour les imports longs
DatabaseConnection = asyncpg.Connection
DatabasePool = asyncpg.Pool

logger = logging.getLogger(__name__)

def get_db_config():
    """Récupère la configuration de la base de données de manière dynamique."""
    return Config().db_config


class DatabaseManager:
    """
    Gestionnaire de base de données PostgreSQL pour le bot Kraken.

    Gère la connexion, l'insertion et la récupération de données de marché et de trading.

    Attributes:
        _instance: Instance unique du gestionnaire (pattern Singleton)
        _pool: Pool de connexions à la base de données
        _initialized: Indique si l'initialisation est terminée
    """
    _instance = None
    _pool: Optional[DatabasePool] = None
    _initialized: bool = False

    def __new__(cls) -> 'DatabaseManager':
        """Implémentation du pattern Singleton."""
        if cls._instance is None:
            cls._instance = super(DatabaseManager, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        """Initialise le gestionnaire de base de données."""
        if self._initialized:
            return
        self._pool: Optional[DatabasePool] = None
        self.connected = False
        self._initialized = True
        # Ne pas charger la configuration ici, elle sera chargée dynamiquement

    async def connect(self) -> asyncpg.Pool:
        """Établit la connexion à la base de données."""
        if self._pool is not None:
            return self._pool

        try:
            logger.info("Connexion à la base de données PostgreSQL...")
            # Charger la configuration à chaque tentative de connexion
            db_config = get_db_config()
            logger.info(f"Tentative de connexion à la base de données avec l'utilisateur: {db_config['user']}")
            
            self._pool = await asyncpg.create_pool(
                user=db_config['user'],
                password=db_config['password'],
                host=db_config['host'],
                port=db_config['port'],
                database=db_config['name'],
                min_size=1,
                max_size=10,
                command_timeout=30,
                timeout=30,
                statement_cache_size=0,
            )

            # Tester la connexion
            async with self._pool.acquire() as conn:
                await conn.execute('SELECT 1')

            self.connected = True
            logger.info("Connecté à PostgreSQL avec succès")

            # Initialiser la base de données
            await self._initialize_database()

            print('DEBUG: FIN CONNECT')
            return self._pool

        except Exception as e:
            logger.error(f"Erreur de connexion à PostgreSQL: {e}")
            self.connected = False
            raise

    async def _initialize_database(self):
        """
        Initialise les tables de la base de données si elles n'existent pas.
        """
        try:
            async with self._pool.acquire() as conn:
                logger.info(
                    "Vérification et création des tables si nécessaire...")

                # Activer l'extension uuid-ossp si elle n'est pas déjà activée
                await conn.execute('CREATE EXTENSION IF NOT EXISTS "uuid-ossp"')

                # Création de la table des paires
                await conn.execute('''
                    CREATE TABLE IF NOT EXISTS pairs (
                        id SERIAL PRIMARY KEY,
                        symbol VARCHAR(20) NOT NULL UNIQUE,
                        base_currency VARCHAR(10) NOT NULL,
                        quote_currency VARCHAR(10) NOT NULL,
                        is_active BOOLEAN DEFAULT TRUE,
                        min_trade_size DECIMAL(20, 10),
                        max_trade_size DECIMAL(20, 10),
                        tick_size DECIMAL(20, 10),
                        created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                        updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                        CONSTRAINT unique_pair_currency UNIQUE (base_currency, quote_currency)
                    )
                ''')

                # Création de la table des données de marché (OHLCV)
                await conn.execute('''
                    CREATE TABLE IF NOT EXISTS market_data (
                        id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
                        pair_id INTEGER NOT NULL,
                        timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
                        open DECIMAL(20, 10) NOT NULL,
                        high DECIMAL(20, 10) NOT NULL,
                        low DECIMAL(20, 10) NOT NULL,
                        close DECIMAL(20, 10) NOT NULL,
                        volume DECIMAL(30, 10) NOT NULL,
                        vwap DECIMAL(20, 10),
                        count INTEGER,
                        created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                        CONSTRAINT fk_market_data_pair FOREIGN KEY (pair_id) REFERENCES pairs(id) ON DELETE CASCADE,
                        CONSTRAINT unique_market_data UNIQUE (pair_id, timestamp)
                    )
                ''')

                # Création de la table des trades
                await conn.execute('''
                    CREATE TABLE IF NOT EXISTS trades (
                        id VARCHAR(50) PRIMARY KEY,
                        order_id VARCHAR(50),
                        pair_id INTEGER NOT NULL,
                        type VARCHAR(4) NOT NULL CHECK (type IN ('buy', 'sell')),
                        order_type VARCHAR(10) NOT NULL,
                        price DECIMAL(20, 8) NOT NULL,
                        volume DECIMAL(30, 8) NOT NULL,
                        cost DECIMAL(30, 8) NOT NULL,
                        fee DECIMAL(30, 8) NOT NULL,
                        timestamp BIGINT NOT NULL,
                        created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                        updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                        CONSTRAINT fk_trade_pair FOREIGN KEY (pair_id) REFERENCES pairs(id) ON DELETE CASCADE
                    )
                ''')

                # Création de la table des ordres
                await conn.execute('''
                    CREATE TABLE IF NOT EXISTS orders (
                        id VARCHAR(50) PRIMARY KEY,
                        pair_id INTEGER NOT NULL,
                        type VARCHAR(4) NOT NULL CHECK (type IN ('buy', 'sell')),
                        order_type VARCHAR(20) NOT NULL,
                        price DECIMAL(20, 8) NOT NULL,
                        stop_price DECIMAL(20, 8),
                        volume DECIMAL(30, 8) NOT NULL,
                        filled_volume DECIMAL(30, 8) DEFAULT 0,
                        remaining_volume DECIMAL(30, 8) GENERATED ALWAYS AS (volume - COALESCE(filled_volume, 0)) STORED,
                        status VARCHAR(20) NOT NULL,
                        leverage INTEGER DEFAULT 1,
                        fee DECIMAL(30, 8) DEFAULT 0,
                        timestamp BIGINT NOT NULL,
                        created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                        updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                        closed_at TIMESTAMP WITH TIME ZONE,
                        CONSTRAINT fk_order_pair FOREIGN KEY (pair_id) REFERENCES pairs(id) ON DELETE CASCADE
                    )
                ''')

                # Création de la table des signaux de trading
                await conn.execute('''
                    CREATE TABLE IF NOT EXISTS trading_signals (
                        id BIGSERIAL PRIMARY KEY,
                        pair_id INTEGER NOT NULL,
                        signal_type VARCHAR(20) NOT NULL,
                        price DECIMAL(20, 8) NOT NULL,
                        volume DECIMAL(30, 8) NOT NULL,
                        timestamp BIGINT NOT NULL,
                        is_executed BOOLEAN DEFAULT FALSE,
                        executed_at TIMESTAMP WITH TIME ZONE,
                        order_id VARCHAR(50),
                        created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                        CONSTRAINT fk_signal_pair FOREIGN KEY (pair_id) REFERENCES pairs(id) ON DELETE CASCADE
                    )
                ''')

                # Création de la table des logs d'erreurs
                await conn.execute('''
                    CREATE TABLE IF NOT EXISTS error_logs (
                        id BIGSERIAL PRIMARY KEY,
                        source VARCHAR(50) NOT NULL,
                        level VARCHAR(20) NOT NULL,
                        message TEXT NOT NULL,
                        traceback TEXT,
                        created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
                    )
                ''')

                # Création de la table des sessions de trading
                await conn.execute("""
                    CREATE TABLE IF NOT EXISTS trading_sessions (
                        id SERIAL PRIMARY KEY,
                        start_time TIMESTAMP NOT NULL,
                        end_time TIMESTAMP,
                        pairs TEXT[],
                        strategy TEXT,
                        initial_balance DECIMAL(20,10),
                        final_balance DECIMAL(20,10),
                        trades JSONB,
                        performance_metrics JSONB,
                        risk_metrics JSONB,
                        created_at TIMESTAMP DEFAULT NOW()
                    )
                """)

                # Création de la table des trades
                await conn.execute("""
                    CREATE TABLE IF NOT EXISTS trades (
                        id SERIAL PRIMARY KEY,
                        session_id INTEGER REFERENCES trading_sessions(id),
                        pair TEXT NOT NULL,
                        side TEXT NOT NULL,
                        amount DECIMAL(20,10) NOT NULL,
                        entry_price DECIMAL(20,10) NOT NULL,
                        exit_price DECIMAL(20,10),
                        pnl DECIMAL(20,10),
                        leverage DECIMAL(10,2),
                        entry_time TIMESTAMP NOT NULL,
                        exit_time TIMESTAMP,
                        status TEXT NOT NULL,
                        strategy TEXT,
                        indicators JSONB,
                        created_at TIMESTAMP DEFAULT NOW()
                    )
                """)

                # Création de la table des indicateurs techniques
                await conn.execute("""
                    CREATE TABLE IF NOT EXISTS technical_indicators (
                        id SERIAL PRIMARY KEY,
                        pair TEXT NOT NULL,
                        timeframe TEXT NOT NULL,
                        indicator_type TEXT NOT NULL,
                        parameters JSONB,
                        values JSONB,
                        created_at TIMESTAMP DEFAULT NOW(),
                        updated_at TIMESTAMP DEFAULT NOW()
                    )
                """)

                # Création de la table des données agrégées
                await conn.execute("""
                    CREATE TABLE IF NOT EXISTS market_data_aggregated (
                        id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
                        pair_id INTEGER NOT NULL,
                        timeframe INTERVAL NOT NULL,
                        timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
                        open DECIMAL(20, 10) NOT NULL,
                        high DECIMAL(20, 10) NOT NULL,
                        low DECIMAL(20, 10) NOT NULL,
                        close DECIMAL(20, 10) NOT NULL,
                        volume DECIMAL(30, 10) NOT NULL,
                        created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                        CONSTRAINT fk_market_data_aggregated_pair FOREIGN KEY (pair_id) REFERENCES pairs(id) ON DELETE CASCADE,
                        CONSTRAINT unique_market_data_aggregated UNIQUE (pair_id, timeframe, timestamp)
                    )
                """)

                # Création de la table de gestion des risques
                await conn.execute("""
                    CREATE TABLE IF NOT EXISTS risk_management (
                        id SERIAL PRIMARY KEY,
                        pair TEXT NOT NULL,
                        max_position_size DECIMAL(20, 10) NOT NULL,
                        stop_loss_level DECIMAL(5, 2) NOT NULL,
                        take_profit_level DECIMAL(5, 2) NOT NULL,
                        risk_percentage DECIMAL(5, 2) NOT NULL,
                        created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                        updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
                    )
                """)

                # Création de la table d'audit
                await conn.execute("""
                    CREATE TABLE IF NOT EXISTS audit_logs (
                        id BIGSERIAL PRIMARY KEY,
                        table_name TEXT NOT NULL,
                        record_id TEXT NOT NULL,
                        action TEXT NOT NULL,
                        old_data JSONB,
                        new_data JSONB,
                        created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                        user_id TEXT
                    )
                """)

                # Création des index pour optimiser les performances
                await conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_market_data_pair_timestamp ON market_data(pair_id, timestamp DESC);
                    CREATE INDEX IF NOT EXISTS idx_trades_pair_timestamp ON trades(pair_id, created_at DESC);
                    CREATE INDEX IF NOT EXISTS idx_orders_status ON orders(status);
                    CREATE INDEX IF NOT EXISTS idx_trading_signals_created_at ON trading_signals(created_at DESC);
                """)

                # Création de la table des signaux de trading
                await conn.execute("""
                    CREATE TABLE IF NOT EXISTS trading_signals (
                        id SERIAL PRIMARY KEY,
                        pair TEXT NOT NULL,
                        timeframe TEXT NOT NULL,
                        signal_type TEXT NOT NULL,
                        signal_value TEXT NOT NULL,
                        confidence_score DECIMAL(5,2),
                        created_at TIMESTAMP DEFAULT NOW(),
                        updated_at TIMESTAMP DEFAULT NOW()
                    )
                """)

                # Création de la table de sentiment
                await conn.execute("""
                    CREATE TABLE IF NOT EXISTS sentiment_analysis (
                        id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
                        source TEXT NOT NULL,
                        sentiment_score DECIMAL(5,2) NOT NULL,
                        confidence_score DECIMAL(5,2) NOT NULL,
                        keywords TEXT[],
                        timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
                        pair TEXT,
                        created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                        updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
                    )
                """)

                # Création de la table de carnet d'ordres
                await conn.execute("""
                    CREATE TABLE IF NOT EXISTS market_depth (
                        id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
                        pair TEXT NOT NULL,
                        timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
                        bids JSONB NOT NULL,
                        asks JSONB NOT NULL,
                        total_volume DECIMAL(30, 10) NOT NULL,
                        created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
                    )
                """)

                # Création de la table de performance historique
                await conn.execute("""
                    CREATE TABLE IF NOT EXISTS historical_performance (
                        id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
                        session_id INTEGER REFERENCES trading_sessions(id),
                        pair TEXT NOT NULL,
                        timeframe INTERVAL NOT NULL,
                        start_time TIMESTAMP WITH TIME ZONE NOT NULL,
                        end_time TIMESTAMP WITH TIME ZONE NOT NULL,
                        initial_balance DECIMAL(20,10) NOT NULL,
                        final_balance DECIMAL(20,10) NOT NULL,
                        pnl DECIMAL(20,10) NOT NULL,
                        max_drawdown DECIMAL(5,2),
                        sharpe_ratio DECIMAL(5,2),
                        win_rate DECIMAL(5,2),
                        created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
                    )
                """)

                # Création de la table de stratégies
                await conn.execute("""
                    CREATE TABLE IF NOT EXISTS strategies (
                        id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
                        name TEXT NOT NULL,
                        description TEXT,
                        parameters JSONB NOT NULL,
                        risk_profile JSONB,
                        performance_metrics JSONB,
                        is_active BOOLEAN DEFAULT TRUE,
                        created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                        updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
                    )
                """)

                # Création de la table de cache
                await conn.execute("""
                    CREATE TABLE IF NOT EXISTS data_cache (
                        id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
                        key TEXT NOT NULL UNIQUE,
                        value JSONB NOT NULL,
                        expires_at TIMESTAMP WITH TIME ZONE NOT NULL,
                        created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                        updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
                    )
                """)

                # Création des index supplémentaires
                await conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_sentiment_pair_timestamp ON sentiment_analysis(pair, timestamp DESC);
                    CREATE INDEX IF NOT EXISTS idx_market_depth_pair_timestamp ON market_depth(pair, timestamp DESC);
                    CREATE INDEX IF NOT EXISTS idx_historical_performance_pair ON historical_performance(pair);
                    CREATE INDEX IF NOT EXISTS idx_strategies_name ON strategies(name);
                    CREATE INDEX IF NOT EXISTS idx_data_cache_key ON data_cache(key);
                """)

                # Création de la table des positions
                await conn.execute("""
                    CREATE TABLE IF NOT EXISTS positions (
                        id SERIAL PRIMARY KEY,
                        session_id INTEGER REFERENCES trading_sessions(id),
                        pair TEXT NOT NULL,
                        side TEXT NOT NULL,
                        entry_price DECIMAL(20,10) NOT NULL,
                        current_price DECIMAL(20,10),
                        amount DECIMAL(20,10) NOT NULL,
                        leverage DECIMAL(10,2),
                        entry_time TIMESTAMP NOT NULL,
                        exit_time TIMESTAMP,
                        status TEXT NOT NULL,
                        created_at TIMESTAMP DEFAULT NOW()
                    )
                """)

                # Création de la table des ordres
                await conn.execute("""
                    CREATE TABLE IF NOT EXISTS orders (
                        id SERIAL PRIMARY KEY,
                        session_id INTEGER REFERENCES trading_sessions(id),
                        pair TEXT NOT NULL,
                        order_type TEXT NOT NULL,
                        side TEXT NOT NULL,
                        amount DECIMAL(20,10) NOT NULL,
                        price DECIMAL(20,10),
                        status TEXT NOT NULL,
                        created_at TIMESTAMP DEFAULT NOW(),
                        updated_at TIMESTAMP DEFAULT NOW()
                    )
                """)

                # Création de la table de l'état du trading
                await conn.execute("""
                    CREATE TABLE IF NOT EXISTS trading_status (
                        id SERIAL PRIMARY KEY,
                        timestamp TIMESTAMP NOT NULL,
                        status TEXT NOT NULL,
                        balances JSONB,
                        active_orders JSONB,
                        open_positions JSONB,
                        last_update TIMESTAMP DEFAULT NOW(),
                        created_at TIMESTAMP DEFAULT NOW()
                    )
                """)

                # Création des index pour améliorer les performances
                await conn.execute('''
                    CREATE INDEX IF NOT EXISTS idx_market_data_pair_timestamp
                    ON market_data (pair_id, timestamp DESC)
                ''')

                await conn.execute('''
                    CREATE INDEX IF NOT EXISTS idx_trades_pair_created
                    ON trades (pair_id, created_at DESC)
                ''')

                await conn.execute('''
                    CREATE INDEX IF NOT EXISTS idx_orders_status
                    ON orders (status) WHERE status IN ('open', 'pending')
                ''')

                logger.info("Vérification des tables terminée avec succès")

        except Exception as e:
            logger.error(
                f"Erreur lors de l'initialisation de la base de données: {e}")
            raise

    async def close(self) -> None:
        """Ferme la connexion à la base de données."""
        if self._pool is not None:
            await self._pool.close()
            self._pool = None
            self.connected = False
            logger.info("Connexion à la base de données fermée")

    async def backup_database(self, backup_path: str = None) -> str:
        """
        Crée une sauvegarde de la base de données.

        Args:
            backup_path (str, optional): Chemin personnalisé pour la sauvegarde

        Returns:
            str: Chemin du fichier de sauvegarde
        """
        try:
            import os
            import subprocess

            # Construire le chemin de sauvegarde
            if backup_path is None:
                backup_dir = "database_backups"
                if not os.path.exists(backup_dir):
                    os.makedirs(backup_dir)

                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                backup_path = os.path.join(
                    backup_dir, f"kraken_bot_backup_{timestamp}.sql")

            # Construire la commande pg_dump
            cmd = [
                "pg_dump",
                "-h", self.config["connection"]["host"],
                "-p", str(self.config["connection"]["port"]),
                "-U", self.config["connection"]["user"],
                "-d", self.config["connection"]["database"],
                "-F", "c",  # Format compressé
                "-f", backup_path
            ]

            # Exécuter la commande
            result = subprocess.run(cmd, capture_output=True)

            if result.returncode != 0:
                logger.error(
                    f"Erreur lors de la sauvegarde: {result.stderr.decode()}")
                raise Exception("La sauvegarde a échoué")

            logger.info(f"Sauvegarde créée avec succès: {backup_path}")
            return backup_path

        except Exception as e:
            logger.error(f"Erreur lors de la création de la sauvegarde: {e}")
            raise

    async def reset_database(
            self,
            confirm: bool = False,
            backup: bool = True) -> None:
        """
        Réinitialise complètement la base de données.
        ATTENTION : Cette opération est destructive et supprime toutes les données !

        Args:
            confirm (bool): Confirmation explicite pour exécuter la réinitialisation
            backup (bool): Si True, crée une sauvegarde avant la réinitialisation

        Raises:
            ValueError: Si confirm est False
            Exception: Si la sauvegarde échoue
        """
        if not confirm:
            raise ValueError(
                "La réinitialisation nécessite une confirmation explicite (confirm=True)")

        if backup:
            try:
                backup_path = await self.backup_database()
                logger.info(
                    f"Sauvegarde créée avant la réinitialisation: {backup_path}")
            except Exception as e:
                logger.error(f"Erreur lors de la création de la sauvegarde: {e}")
                raise

        async with self.get_connection() as conn:
            try:
                # Sauvegarde avant la réinitialisation
                if backup:
                    await self.backup_database()

                # Supprimer les tables dans l'ordre inverse de leur dépendance
                await conn.execute("DROP TABLE IF EXISTS trading_signals CASCADE")
                await conn.execute("DROP TABLE IF EXISTS risk_parameters CASCADE")
                await conn.execute("DROP TABLE IF EXISTS market_data CASCADE")
                await conn.execute("DROP TABLE IF EXISTS error_logs CASCADE")
                await conn.execute("DROP TABLE IF EXISTS trading_status CASCADE")

                # Réinitialiser les séquences
                try:
                    await conn.execute("DROP SEQUENCE IF EXISTS trading_signals_id_seq CASCADE")
                    await conn.execute("DROP SEQUENCE IF EXISTS market_data_id_seq CASCADE")
                    await conn.execute("DROP SEQUENCE IF EXISTS error_logs_id_seq CASCADE")
                    await conn.execute("DROP SEQUENCE IF EXISTS trading_status_id_seq CASCADE")
                except Exception as e:
                    logger.error(f"Erreur lors de la réinitialisation des séquences: {e}")

                # Réinitialiser les vues matérielles
                try:
                    await conn.execute("REFRESH MATERIALIZED VIEW CONCURRENTLY IF EXISTS mv_market_data")
                except Exception as e:
                    logger.error(f"Erreur lors de la réinitialisation des vues: {e}")

                logger.info("Base de données réinitialisée avec succès")

                # Réinitialiser l'état de la connexion
                self._initialized = False
                await self._initialize_database()
                logger.info("Tables réinitialisées et recréées")

                # Créer un log de l'opération
                await conn.execute("""
                    INSERT INTO error_logs (source, level, message)
                    VALUES ($1, $2, $3)
                """, "DATABASE_RESET", "INFO", "Base de données réinitialisée avec succès")
            except Exception as e:
                logger.error(f"Erreur lors de la réinitialisation de la base de données: {e}")
                raise

    @asynccontextmanager
    async def get_connection(self) -> asyncpg.Connection:
        """Fournit une connexion de la pool avec gestion de contexte."""
        if self._pool is None:
            await self.connect()

        conn = await self._pool.acquire()
        try:
            async with conn.transaction():
                yield conn
        except Exception as e:
            logger.error(f"Erreur lors de l'exécution de la requête: {e}")
            raise
        finally:
            await self._pool.release(conn)

    async def execute(self, query: str, *args) -> str:
        """Exécute une requête et retourne le statut."""
        async with self.get_connection() as conn:
            result = await conn.execute(query, *args)
            return result

    async def fetch(self, query: str, *args) -> List[asyncpg.Record]:
        """Exécute une requête et retourne plusieurs lignes."""
        async with self.get_connection() as conn:
            return await conn.fetch(query, *args)

    async def get_market_data(
            self,
            pair: str,
            timeframe: str,
            start_time: datetime,
            end_time: datetime):
        """Récupère les données de marché pour une paire et une période donnée."""
        async with self.get_connection() as conn:
            pair_id = await conn.fetchval("""
                SELECT id FROM pairs WHERE symbol = $1
            """, pair)

            if not pair_id:
                raise ValueError(f"Paire {pair} non trouvée")

            return await conn.fetch("""
                SELECT * FROM market_data
                WHERE pair_id = $1
                AND timestamp >= $2
                AND timestamp <= $3
                ORDER BY timestamp ASC
            """, pair_id, start_time, end_time)

    async def get_technical_indicators(
            self,
            pair: str,
            indicator_type: str,
            start_time: datetime,
            end_time: datetime):
        """Récupère les indicateurs techniques pour une paire et une période donnée."""
        async with self.get_connection() as conn:
            return await conn.fetch("""
                SELECT * FROM technical_indicators
                WHERE pair = $1
                AND indicator_type = $2
                AND created_at >= $3
                AND created_at <= $4
                ORDER BY created_at ASC
            """, pair, indicator_type, start_time, end_time)

    async def insert_technical_indicator(
            self,
            pair: str,
            timeframe: str,
            indicator_type: str,
            parameters: dict,
            values: dict):
        """Insère un indicateur technique dans la base de données."""
        async with self.get_connection() as conn:
            await conn.execute("""
                INSERT INTO technical_indicators (pair, timeframe, indicator_type, parameters, values)
                VALUES ($1, $2, $3, $4, $5)
            """, pair, timeframe, indicator_type, parameters, values)

    async def get_market_depth(self, pair: str, limit: int = 1):
        """Récupère les données du carnet d'ordres pour une paire."""
        async with self.get_connection() as conn:
            return await conn.fetch("""
                SELECT * FROM market_depth
                WHERE pair = $1
                ORDER BY timestamp DESC
                LIMIT $2
            """, pair, limit)

    async def get_sentiment_analysis(
            self,
            pair: str = None,
            source: str = None,
            limit: int = 10):
        """Récupère les analyses de sentiment."""
        async with self.get_connection() as conn:
            query = "SELECT * FROM sentiment_analysis"
            params = []

            if pair or source:
                conditions = []
                if pair:
                    conditions.append("pair = $1")
                    params.append(pair)
                if source:
                    conditions.append("source = $2")
                    params.append(source)
                query += " WHERE " + " AND ".join(conditions)

            query += " ORDER BY timestamp DESC LIMIT $" + str(len(params) + 1)
            params.append(limit)

            return await conn.fetch(query, *params)

    async def fetchrow(self, query: str, *args) -> Optional[asyncpg.Record]:
        """Exécute une requête et retourne une seule ligne."""
        async with self.get_connection() as conn:
            return await conn.fetchrow(query, *args)

    async def fetchval(self, query: str, *args) -> Any:
        """Exécute une requête et retourne une valeur scalaire."""
        async with self.get_connection() as conn:
            return await conn.fetchval(query, *args)

    async def delete_old_ohlcv_data(
            self,
            days: int = 60,
            interval: int = None):
        """
        Supprime les données OHLCV de plus de X jours.

        Args:
            days (int): Nombre de jours à conserver (par défaut: 60)
            interval (int, optional): Intervalle de temps en minutes pour filtrer les données à supprimer
        """
        try:
            async with self.get_connection() as conn:
                # Vérifier si la colonne interval_minutes existe
                table_columns = await conn.fetch(
                    "SELECT column_name FROM information_schema.columns "
                    "WHERE table_name = 'market_data' AND column_name = 'interval_minutes'"
                )
                has_interval_column = bool(table_columns)

                # Calculer la date de coupure
                cutoff_timestamp = int(
                    (datetime.now() - timedelta(days=days)).timestamp())

                # Construire et exécuter la requête de suppression
                if has_interval_column and interval is not None:
                    result = await conn.execute(
                        """
                        WITH deleted AS (
                            DELETE FROM market_data
                            WHERE timestamp < to_timestamp($1) AND interval_minutes = $2
                            RETURNING *
                        ) SELECT COUNT(*) FROM deleted
                        """,
                        cutoff_timestamp, interval
                    )
                else:
                    # Version sans le filtre d'intervalle
                    result = await conn.execute(
                        """
                        WITH deleted AS (
                            DELETE FROM market_data
                            WHERE timestamp < to_timestamp($1)
                            RETURNING *
                        ) SELECT COUNT(*) FROM deleted
                        """,
                        cutoff_timestamp
                    )

                # Récupérer le nombre de lignes supprimées
                deleted_count = int(
                    result.split()[1])  # Format: 'DELETE COUNT'
                logger.info(
                    f"Suppression de {deleted_count} entrées de plus de {days} jours")

        except Exception as e:
            logger.error(
                f"Erreur lors de la suppression des anciennes données: {e}")

    async def insert_ohlcv_data(
            self,
            pair_symbol: str,
            ohlcv: list,
            interval: int = None):
        """
        Insère une liste de bougies OHLCV dans la table market_data pour la paire donnée.

        Args:
            pair_symbol (str): Le symbole de la paire (ex: 'XBT/USD').
            ohlcv (list): Liste de bougies [timestamp, open, high, low, close, vwap, volume, count].
            interval (int, optional): Intervalle de temps en minutes pour les données OHLCV.
        """
        if not ohlcv:
            logger.warning(f"Aucune donnée OHLCV à insérer pour {pair_symbol}")
            return

        logger.debug(
            f"Tentative d'insertion de {len(ohlcv)} bougies OHLCV pour {pair_symbol} (intervalle: {interval} min)")
        logger.debug(f"Première bougie: {ohlcv[0] if ohlcv else 'Aucune'}")
        logger.debug(f"Dernière bougie: {ohlcv[-1] if ohlcv else 'Aucune'}")

        try:
            # Récupérer l'id de la paire
            async with self.get_connection() as conn:
                # Vérifier d'abord si la paire existe
                pair_row = await conn.fetchrow("SELECT id FROM pairs WHERE symbol = $1", pair_symbol)
                if not pair_row:
                    logger.error(
                        f"Paire {pair_symbol} non trouvée dans la table pairs.")
                    # Essayer d'insérer la paire si elle n'existe pas
                    try:
                        await conn.execute(
                            "INSERT INTO pairs (symbol, base_asset, quote_asset) VALUES ($1, $2, $3)",
                            pair_symbol,
                            pair_symbol.split('/')[0] if '/' in pair_symbol else pair_symbol,
                            pair_symbol.split(
                                '/')[1] if '/' in pair_symbol and len(pair_symbol.split('/')) > 1 else 'USD'
                        )
                        pair_row = await conn.fetchrow("SELECT id FROM pairs WHERE symbol = $1", pair_symbol)
                        logger.info(
                            f"Nouvelle paire {pair_symbol} insérée avec succès.")
                    except Exception as e:
                        logger.error(
                            f"Échec de l'insertion de la paire {pair_symbol}: {str(e)}")
                        return

                if not pair_row:
                    logger.error(
                        f"Impossible de récupérer l'ID pour la paire {pair_symbol}")
                    return

                pair_id = pair_row['id']

                # Vérifier si la colonne interval_minutes existe dans la table
                try:
                    table_columns = await conn.fetch(
                        "SELECT column_name FROM information_schema.columns "
                        "WHERE table_name = 'market_data'"
                    )
                    has_interval_column = any(
                        col['column_name'] == 'interval_minutes' for col in table_columns)
                    logger.debug(
                        f"La table market_data a une colonne interval_minutes: {has_interval_column}")
                except Exception as e:
                    logger.error(
                        f"Erreur lors de la vérification des colonnes de la table market_data: {str(e)}")
                    has_interval_column = False

                # Compter le nombre de bougies insérées
                inserted_count = 0
                skipped_count = 0

                # Préparer la requête d'insertion
                if has_interval_column and interval is not None:
                    insert_query = """
                    INSERT INTO market_data
                    (pair_id, timestamp, open, high, low, close, vwap, volume, count, interval_minutes)
                    VALUES ($1, to_timestamp($2), $3, $4, $5, $6, $7, $8, $9, $10)
                    ON CONFLICT (pair_id, timestamp, interval_minutes)
                    DO UPDATE SET
                        open = EXCLUDED.open,
                        high = EXCLUDED.high,
                        low = EXCLUDED.low,
                        close = EXCLUDED.close,
                        vwap = EXCLUDED.vwap,
                        volume = EXCLUDED.volume,
                        count = EXCLUDED.count
                    """
                else:
                    insert_query = """
                    INSERT INTO market_data
                    (pair_id, timestamp, open, high, low, close, vwap, volume, count)
                    VALUES ($1, to_timestamp($2), $3, $4, $5, $6, $7, $8, $9)
                    ON CONFLICT (pair_id, timestamp)
                    DO UPDATE SET
                        open = EXCLUDED.open,
                        high = EXCLUDED.high,
                        low = EXCLUDED.low,
                        close = EXCLUDED.close,
                        vwap = EXCLUDED.vwap,
                        volume = EXCLUDED.volume,
                        count = EXCLUDED.count
                    """

                # Insérer chaque bougie
                for candle in ohlcv:
                    # Vérifier que la bougie a le bon format
                    if len(candle) < 8:
                        logger.warning(f"Format de bougie invalide: {candle}")
                        skipped_count += 1
                        continue

                    try:
                        # Préparer les données de la bougie
                        timestamp = int(candle[0])
                        open_price = float(candle[1])
                        high = float(candle[2])
                        low = float(candle[3])
                        close = float(candle[4])
                        vwap = float(candle[5]) if len(
                            candle) > 5 and candle[5] is not None else None
                        volume = float(candle[6]) if len(
                            candle) > 6 and candle[6] is not None else None
                        count = int(candle[7]) if len(
                            candle) > 7 and candle[7] is not None else None

                        # Exécuter la requête d'insertion
                        if has_interval_column and interval is not None:
                            await conn.execute(
                                insert_query,
                                pair_id, timestamp, open_price, high, low, close,
                                vwap, volume, count, interval
                            )
                        else:
                            await conn.execute(
                                insert_query,
                                pair_id, timestamp, open_price, high, low, close,
                                vwap, volume, count
                            )

                        inserted_count += 1

                    except Exception as e:
                        logger.error(
                            f"Erreur lors de l'insertion de la bougie {candle}: {str(e)}")
                        skipped_count += 1
                        continue

                # Supprimer les anciennes données
                if interval is not None:
                    try:
                        await self.delete_old_ohlcv_data(days=60, interval=interval)
                    except Exception as e:
                        logger.error(
                            f"Erreur lors de la suppression des anciennes données: {str(e)}")

                logger.info(
                    f"Insertion terminée pour {pair_symbol}: {inserted_count} bougies insérées, {skipped_count} ignorées")

        except Exception as e:
            logger.error(
                f"Erreur critique lors de l'insertion des données OHLCV pour {pair_symbol}: {str(e)}")
            logger.debug(f"Traceback: {traceback.format_exc()}")
            raise

    async def save_trading_status(self, status_data: Dict[str, Any]):
        """
        Sauvegarde l'état actuel du trading.

        Args:
            status_data (Dict): Données d'état du trading
        """
        async with self.get_connection() as conn:
            await conn.execute("""
                INSERT INTO trading_status (status_data)
                VALUES ($1)
                ON CONFLICT (id) DO UPDATE SET
                    status_data = EXCLUDED.status_data,
                    updated_at = NOW()
            """, status_data)

    async def get_risk_parameters(self, pair: str):
        """
        Récupère les paramètres de risque pour une paire.

        Args:
            pair (str): Paire de trading

        Returns:
            dict: Paramètres de risque
        """
        async with self.get_connection() as conn:
            return await conn.fetchrow("""
                SELECT * FROM risk_management
                WHERE pair = $1
            """, pair)

    async def update_risk_parameters(
            self,
            pair: str,
            max_position_size: float,
            stop_loss: float,
            take_profit: float,
            risk_percentage: float):
        """
        Met à jour les paramètres de risque pour une paire.

        Args:
            pair (str): Paire de trading
            max_position_size (float): Taille maximale de position
            stop_loss (float): Niveau de stop loss
            take_profit (float): Niveau de take profit
            risk_percentage (float): Pourcentage de risque
        """
        async with self.get_connection() as conn:
            await conn.execute("""
                INSERT INTO risk_management (pair, max_position_size, stop_loss_level, take_profit_level, risk_percentage)
                VALUES ($1, $2, $3, $4, $5)
                ON CONFLICT (pair) DO UPDATE SET
                    max_position_size = EXCLUDED.max_position_size,
                    stop_loss_level = EXCLUDED.stop_loss_level,
                    take_profit_level = EXCLUDED.take_profit_level,
                    risk_percentage = EXCLUDED.risk_percentage,
                    updated_at = NOW()
            """, pair, max_position_size, stop_loss, take_profit, risk_percentage)

    async def insert_trading_signal(
            self,
            pair: str,
            timeframe: str,
            signal_type: str,
            signal_value: str,
            confidence: float):
        """
        Insère un signal de trading dans la base de données.

        Args:
            pair (str): Paire de trading
            timeframe (str): Intervalle de temps
            signal_type (str): Type de signal
            signal_value (str): Valeur du signal
            confidence (float): Score de confiance
        """
        async with self.get_connection() as conn:
            await conn.execute("""
                INSERT INTO trading_signals (pair, timeframe, signal_type, signal_value, confidence_score)
                VALUES ($1, $2, $3, $4, $5)
            """, pair, timeframe, signal_type, signal_value, confidence)

    async def get_recent_signals(self, pair: str, limit: int = 10):
        """
        Récupère les signaux de trading récents pour une paire.

        Args:
            pair (str): Paire de trading
            limit (int): Nombre maximum de signaux

        Returns:
            list: Liste des signaux récents
        """
        async with self.get_connection() as conn:
            return await conn.fetch("""
                SELECT * FROM trading_signals
                WHERE pair = $1
                ORDER BY created_at DESC
                LIMIT $2
            """, pair, limit)

    async def update_signal_status(
            self,
            signal_id: int,
            is_executed: bool,
            order_id: str = None):
        """
        Met à jour le statut d'un signal de trading.

        Args:
            signal_id (int): ID du signal
            is_executed (bool): Statut d'exécution
            order_id (str, optional): ID de l'ordre associé
        """
        async with self.get_connection() as conn:
            await conn.execute("""
                UPDATE trading_signals
                SET is_executed = $2,
                    executed_at = NOW(),
                    order_id = $3
                WHERE id = $1
            """, signal_id, is_executed, order_id)

    async def save_trading_status(self, status_data: Dict[str, Any]) -> None:
        """
        Sauvegarde l'état actuel du trading."""
        try:
            async with self.get_connection() as conn:
                await conn.execute("""
                    INSERT INTO trading_status (
                        timestamp, status, balances, active_orders,
                        open_positions, last_update
                    ) VALUES (
                        to_timestamp($1), $2, $3, $4, $5, to_timestamp($6)
                    )
                """, time.time(), status_data['status'],
                    status_data['balances'], status_data['active_orders'],
                    status_data['open_positions'], time.time())
                logger.info("État du trading sauvegardé avec succès")
        except Exception as e:
            logger.error(
                f"Erreur lors de la sauvegarde de l'état du trading: {e}")
            raise


# Instance globale
db_manager = DatabaseManager()


async def init_db() -> DatabaseManager:
    """
    Initialise la connexion à la base de données et crée les tables si nécessaire.
    """
    try:
        logger.info("🔌 Connexion à la base de données...")
        await db_manager.connect()
        logger.info("✅ Base de données initialisée avec succès")
        print('DEBUG: FIN INIT_DB')
        print('DEBUG: AVANT RETURN INIT_DB')
        return db_manager
    except Exception as e:
        error_msg = f"❌ Échec de l'initialisation de la base de données: {str(e)}"
        logger.error(error_msg, exc_info=True)
        raise RuntimeError(error_msg) from e


async def close_db():
    """Ferme la connexion à la base de données."""
    await db_manager.close()

__all__ = ['db_manager', 'init_db', 'close_db', 'DatabaseManager']
