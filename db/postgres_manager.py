import os
import logging
import asyncpg
from typing import List, Dict, Any
import pandas as pd
from contextlib import asynccontextmanager

# Configuration du logger
logger = logging.getLogger(__name__)

class DatabaseManager:
    """Gestionnaire de base de données PostgreSQL asynchrone."""
    
    _instance = None
    _pool = None
    
    def __new__(cls):
        """Implémentation du pattern Singleton."""
        if cls._instance is None:
            cls._instance = super(DatabaseManager, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        """Initialise la configuration de la base de données."""
        if self._initialized:
            return
            
        self._initialized = True
        self.connected = False
        self._cache = {}
        
        # Configuration de la connexion
        self.db_config = {
            'database': os.getenv('POSTGRES_DB', 'kraken_db'),
            'user': os.getenv('POSTGRES_USER', 'postgres'),
            'password': os.getenv('POSTGRES_PASSWORD', 'postgres'),
            'host': os.getenv('POSTGRES_HOST', 'localhost'),
            'port': os.getenv('POSTGRES_PORT', '5432')
        }
    
    async def connect(self):
        """Établit la connexion à la base de données."""
        if self._pool is not None:
            return self._pool
            
        try:
            logger.info("Connexion à la base de données PostgreSQL...")
            self._pool = await asyncpg.create_pool(
                min_size=1,
                max_size=10,
                **self.db_config
            )
            
            # Tester la connexion
            async with self._pool.acquire() as conn:
                await conn.execute('SELECT 1')
                
            self.connected = True
            logger.info("Connecté à PostgreSQL avec succès")
            
            # Initialiser la base de données
            await self._initialize_database()
            
            return self._pool
            
        except Exception as e:
            logger.error(f"Erreur de connexion à PostgreSQL: {e}")
            self.connected = False
            raise
    
    async def _initialize_database(self):
        """Initialise les tables de la base de données si elles n'existent pas."""
        async with self._pool.acquire() as conn:
            # Création de la table des paires
            await conn.execute('''
                CREATE TABLE IF NOT EXISTS pairs (
                    id VARCHAR(20) PRIMARY KEY,
                    base_currency VARCHAR(10) NOT NULL,
                    quote_currency VARCHAR(10) NOT NULL,
                    status VARCHAR(20) DEFAULT 'online',
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
                )
            ''')
            
            # Création de la table des données de marché
            await conn.execute('''
                CREATE TABLE IF NOT EXISTS market_data (
                    id SERIAL PRIMARY KEY,
                    pair_id VARCHAR(20) REFERENCES pairs(id) ON DELETE CASCADE,
                    timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
                    open DECIMAL(24, 8) NOT NULL,
                    high DECIMAL(24, 8) NOT NULL,
                    low DECIMAL(24, 8) NOT NULL,
                    close DECIMAL(24, 8) NOT NULL,
                    volume DECIMAL(32, 8) NOT NULL,
                    vwap DECIMAL(24, 8),
                    count INTEGER,
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                    UNIQUE(pair_id, timestamp)
                )
            ''')
            
            # Création d'un index pour accélérer les requêtes par paire et par date
            await conn.execute('''
                CREATE INDEX IF NOT EXISTS idx_market_data_pair_timestamp 
                ON market_data(pair_id, timestamp DESC)
            ''')
    
    @asynccontextmanager
    async def get_connection(self):
        """Fournit une connexion de la pool avec gestion de contexte."""
        if self._pool is None:
            await self.connect()
            
        conn = await self._pool.acquire()
        try:
            # Désactiver le mode autocommit pour gérer manuellement les transactions
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
            
    async def executemany(self, query: str, args_list: List[tuple]) -> List[str]:
        """
        Exécute une requête plusieurs fois avec différents jeux de paramètres.
        
        Args:
            query: La requête SQL à exécuter
            args_list: Liste de tuples, chaque tuple contient les paramètres pour une exécution
            
        Returns:
            Liste des résultats de chaque exécution
        """
        results = []
        async with self.get_connection() as conn:
            # Préparer la requête une seule fois
            stmt = await conn.prepare(query)
            
            # Exécuter pour chaque jeu de paramètres
            for args in args_list:
                result = await stmt.fetchval(*args)
                results.append(result)
                
        return results
    
    async def fetch(self, query: str, *args) -> List[asyncpg.Record]:
        """Exécute une requête et retourne toutes les lignes."""
        async with self.get_connection() as conn:
            return await conn.fetch(query, *args)
    
    async def fetchrow(self, query: str, *args) -> asyncpg.Record:
        """Exécute une requête et retourne une seule ligne."""
        async with self.get_connection() as conn:
            return await conn.fetchrow(query, *args)
    
    async def fetchval(self, query: str, *args) -> Any:
        """Exécute une requête et retourne une valeur scalaire."""
        async with self.get_connection() as conn:
            return await conn.fetchval(query, *args)
    
    async def get_market_data(self, pair: str, limit: int = 100) -> pd.DataFrame:
        """Récupère les données de marché pour une paire."""
        query = """
            SELECT m.*, p.id as pair, p.base_currency, p.quote_currency
            FROM market_data m
            JOIN pairs p ON m.pair_id = p.id
            WHERE p.id = $1
            ORDER BY m.timestamp DESC
            LIMIT $2
        """
        
        try:
            rows = await self.fetch(query, pair, limit)
            if not rows:
                return pd.DataFrame()
                
            # Convertir en DataFrame
            df = pd.DataFrame([dict(row) for row in rows])
            
            # Convertir les types si nécessaire
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            # Trier par timestamp (au cas où)
            df = df.sort_values('timestamp', ascending=True)
            
            return df
            
        except Exception as e:
            logger.error(f"Erreur lors de la récupération des données de marché pour {pair}: {e}")
            return pd.DataFrame()
    
    async def get_available_pairs(self) -> List[Dict[str, Any]]:
        """Récupère la liste des paires disponibles."""
        query = """
            SELECT id, base_currency as base, quote_currency as quote, status
            FROM pairs
            WHERE status = 'online'
            ORDER BY id
        """
        
        try:
            rows = await self.fetch(query)
            return [dict(row) for row in rows]
        except Exception as e:
            logger.error(f"Erreur lors de la récupération des paires: {e}")
            return []
    
    async def insert_market_data(self, pair_id: str, data: Dict[str, Any]) -> bool:
        """Insère ou met à jour des données de marché."""
        query = """
            INSERT INTO market_data 
            (pair_id, timestamp, open, high, low, close, volume, vwap, count)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
            ON CONFLICT (pair_id, timestamp) 
            DO UPDATE SET 
                open = EXCLUDED.open,
                high = EXCLUDED.high,
                low = EXCLUDED.low,
                close = EXCLUDED.close,
                volume = EXCLUDED.volume,
                vwap = EXCLUDED.vwap,
                count = EXCLUDED.count
            RETURNING id
        """
        
        try:
            await self.execute(
                query,
                pair_id,
                data['timestamp'],
                data['open'],
                data['high'],
                data['low'],
                data['close'],
                data['volume'],
                data.get('vwap'),
                data.get('count')
            )
            return True
        except Exception as e:
            logger.error(f"Erreur lors de l'insertion des données de marché pour {pair_id}: {e}")
            return False
    
    async def close(self):
        """Ferme la connexion à la base de données."""
        if self._pool:
            await self._pool.close()
            self._pool = None
            self.connected = False
            logger.info("Connexion à la base de données fermée")

# Instance globale du gestionnaire de base de données
db_manager = DatabaseManager()

# Fonctions d'aide pour la compatibilité avec le code existant
def get_db_manager():
    """Retourne l'instance du gestionnaire de base de données."""
    return db_manager
