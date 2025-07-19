from typing import Dict, Any, List
import asyncpg
import logging
from datetime import datetime


class DatabaseManager:
    """Gestionnaire de base de données pour le trading."""

    def __init__(self, config: Dict[str, Any]):
        """Initialise le gestionnaire de base de données."""
        self.config = config
        self.pool = None
        self.logger = logging.getLogger('database_manager')

    async def connect(self):
        """Établit la connexion à la base de données."""
        try:
            self.pool = await asyncpg.create_pool(
                user=self.config.get('db_user', 'postgres'),
                password=self.config.get('db_password', ''),
                database=self.config.get('db_name', 'trading_bot'),
                host=self.config.get('db_host', 'localhost'),
                port=self.config.get('db_port', 5432)
            )
            self.logger.info("Connexion à la base de données établie")

        except Exception as e:
            self.logger.error(
                f"Erreur lors de la connexion à la base de données: {str(e)}")
            raise

    async def create_tables(self):
        """Crée les tables nécessaires dans la base de données."""
        try:
            async with self.pool.acquire() as conn:
                await conn.execute('''
                    CREATE TABLE IF NOT EXISTS trades (
                        id SERIAL PRIMARY KEY,
                        timestamp TIMESTAMP,
                        pair TEXT,
                        side TEXT,
                        entry_price DECIMAL,
                        exit_price DECIMAL,
                        size DECIMAL,
                        pnl DECIMAL,
                        leverage DECIMAL,
                        signal TEXT,
                        metrics JSONB
                    );
                ''')

                await conn.execute('''
                    CREATE TABLE IF NOT EXISTS metrics (
                        id SERIAL PRIMARY KEY,
                        timestamp TIMESTAMP,
                        metric_name TEXT,
                        value DECIMAL,
                        period TEXT
                    );
                ''')

            self.logger.info("Tables créées avec succès")

        except Exception as e:
            self.logger.error(
                f"Erreur lors de la création des tables: {str(e)}")
            raise

    async def record_trade(self, trade_data: Dict[str, Any]):
        """
        Enregistre un trade dans la base de données.

        Args:
            trade_data: Dictionnaire avec les données du trade
        """
        try:
            async with self.pool.acquire() as conn:
                await conn.execute('''
                    INSERT INTO trades (
                        timestamp, pair, side, entry_price, exit_price,
                        size, pnl, leverage, signal, metrics
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
                ''',
                                   trade_data['timestamp'],
                                   trade_data['pair'],
                                   trade_data['side'],
                                   trade_data['entry_price'],
                                   trade_data['exit_price'],
                                   trade_data['size'],
                                   trade_data['pnl'],
                                   trade_data['leverage'],
                                   trade_data['signal'],
                                   trade_data['metrics']
                                   )

            self.logger.info(
                f"Trade enregistré: {trade_data['pair']} {trade_data['side']}")

        except Exception as e:
            self.logger.error(
                f"Erreur lors de l'enregistrement du trade: {str(e)}")
            raise

    async def record_metrics(self, metrics: Dict[str, Any]):
        """
        Enregistre des métriques dans la base de données.

        Args:
            metrics: Dictionnaire des métriques à enregistrer
        """
        try:
            async with self.pool.acquire() as conn:
                for metric_name, value in metrics.items():
                    await conn.execute('''
                        INSERT INTO metrics (
                            timestamp, metric_name, value, period
                        ) VALUES ($1, $2, $3, $4)
                    ''',
                                       datetime.now(),
                                       metric_name,
                                       value,
                                       metrics.get('period', 'daily')
                                       )

            self.logger.info("Métriques enregistrées")

        except Exception as e:
            self.logger.error(
                f"Erreur lors de l'enregistrement des métriques: {str(e)}")
            raise

    async def get_trades(self, pair: str = None,
                         period: str = 'all') -> List[Dict[str, Any]]:
        """
        Récupère les trades de la base de données.

        Args:
            pair: Paire de trading (optionnel)
            period: Période à récupérer (optionnel)

        Returns:
            Liste des trades
        """
        try:
            query = "SELECT * FROM trades"
            params = []

            if pair:
                query += " WHERE pair = $1"
                params.append(pair)

            async with self.pool.acquire() as conn:
                trades = await conn.fetch(query, *params)
                return [dict(trade) for trade in trades]

        except Exception as e:
            self.logger.error(
                f"Erreur lors de la récupération des trades: {str(e)}")
            return []

    async def get_metrics(self, metric_name: str = None,
                          period: str = 'all') -> List[Dict[str, Any]]:
        """
        Récupère les métriques de la base de données.

        Args:
            metric_name: Nom de la métrique (optionnel)
            period: Période à récupérer (optionnel)

        Returns:
            Liste des métriques
        """
        try:
            query = "SELECT * FROM metrics"
            params = []

            if metric_name:
                query += " WHERE metric_name = $1"
                params.append(metric_name)

            async with self.pool.acquire() as conn:
                metrics = await conn.fetch(query, *params)
                return [dict(metric) for metric in metrics]

        except Exception as e:
            self.logger.error(
                f"Erreur lors de la récupération des métriques: {str(e)}")
            return []
