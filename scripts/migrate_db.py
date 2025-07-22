#!/usr/bin/env python3
"""
Script de migration de la base de données pour harmoniser les schémas.
"""
import asyncio
import sys
from pathlib import Path

# Ajouter le répertoire parent au path
sys.path.append(str(Path(__file__).parent.parent))

from src.core.database import db_manager

async def create_migrations_table():
    """Crée la table de suivi des migrations si elle n'existe pas."""
    await db_manager.execute("""
        CREATE TABLE IF NOT EXISTS schema_migrations (
            id SERIAL PRIMARY KEY,
            version INTEGER NOT NULL UNIQUE,
            description TEXT NOT NULL,
            applied_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
        )
    """)

async def get_current_version():
    """Récupère la version actuelle du schéma."""
    try:
        result = await db_manager.fetchval(
            "SELECT version FROM schema_migrations ORDER BY version DESC LIMIT 1"
        )
        return result if result is not None else 0
    except Exception as e:
        # Si la table n'existe pas, on considère la version 0
        return 0

async def apply_migration(version, description, queries):
    """Applique une migration et enregistre son application."""
    current_version = await get_current_version()
    if version <= current_version:
        print(f"La migration {version} est déjà appliquée (version actuelle: {current_version})")
        return

    print(f"Application de la migration {version}: {description}")
    
    async with db_manager.get_connection() as conn:
        # Désactiver temporairement les contraintes de clé étrangère
        await conn.execute("SET CONSTRAINTS ALL DEFERRED")
        
        # Exécuter chaque requête de la migration
        for query in queries:
            print(f"  Exécution: {query[:100]}...")
            await conn.execute(query)
        
        # Enregistrer la migration
        await conn.execute(
            "INSERT INTO schema_migrations (version, description) VALUES ($1, $2)",
            version, description
        )
    
    print(f"Migration {version} appliquée avec succès")

async def migrate():
    """Exécute les migrations nécessaires."""
    try:
        await db_manager.connect()
        await create_migrations_table()
        
        # Migration 1: Harmonisation de la table trades
        await apply_migration(
            version=1,
            description="Harmonisation de la table trades avec le schéma de DatabaseManager",
            queries=[
                """
                -- Supprimer les contraintes existantes
                ALTER TABLE IF EXISTS trades 
                DROP CONSTRAINT IF EXISTS trades_pkey,
                DROP CONSTRAINT IF EXISTS fk_trade_pair,
                DROP CONSTRAINT IF EXISTS trades_type_check;
                """,
                """
                -- Renommer la table existante pour sauvegarde
                ALTER TABLE IF EXISTS trades RENAME TO trades_old;
                """,
                """
                -- Créer la nouvelle table avec le schéma cible
                CREATE TABLE IF NOT EXISTS trades (
                    id VARCHAR(50) PRIMARY KEY,
                    order_id VARCHAR(50),
                    pair_id INTEGER NOT NULL,
                    type VARCHAR(4) NOT NULL,
                    order_type VARCHAR(10) NOT NULL,
                    price NUMERIC(20,8) NOT NULL,
                    volume NUMERIC(30,8) NOT NULL,
                    cost NUMERIC(30,8) NOT NULL,
                    fee NUMERIC(30,8) NOT NULL,
                    timestamp BIGINT NOT NULL,
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                    CONSTRAINT fk_trade_pair FOREIGN KEY (pair_id) REFERENCES pairs(id) ON DELETE CASCADE,
                    CONSTRAINT trades_type_check CHECK (type IN ('buy', 'sell'))
                );
                """,
                """
                -- Créer les index
                CREATE INDEX IF NOT EXISTS idx_trades_pair_created ON trades(pair_id, created_at DESC);
                CREATE INDEX IF NOT EXISTS idx_trades_pair_timestamp ON trades(pair_id, created_at DESC);
                """,
                """
                -- Migrer les données si la table old existe
                DO $$
                BEGIN
                    IF EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = 'trades_old') THEN
                        -- Convertir les données de l'ancienne table vers la nouvelle
                        INSERT INTO trades (
                            id, pair_id, type, order_type, price, volume, cost, fee, timestamp, created_at, updated_at
                        )
                        SELECT 
                            id::TEXT, 
                            pair_id, 
                            LOWER(side) as type,
                            'market' as order_type,
                            price,
                            amount as volume,
                            price * amount as cost,
                            0 as fee,
                            EXTRACT(EPOCH FROM timestamp)::BIGINT * 1000 as timestamp,
                            timestamp as created_at,
                            timestamp as updated_at
                        FROM trades_old;
                        
                        -- Supprimer l'ancienne table
                        DROP TABLE trades_old;
                    END IF;
                END $$;
                """
            ]
        )
        
        # Migration 2: Mise à jour de la table pairs si nécessaire
        await apply_migration(
            version=2,
            description="Mise à jour de la table pairs avec les champs manquants",
            queries=[
                """
                ALTER TABLE pairs 
                ADD COLUMN IF NOT EXISTS is_active BOOLEAN DEFAULT TRUE,
                ADD COLUMN IF NOT EXISTS min_trade_size NUMERIC(20,10),
                ADD COLUMN IF NOT EXISTS max_trade_size NUMERIC(20,10),
                ADD COLUMN IF NOT EXISTS tick_size NUMERIC(20,10);
                """
            ]
        )
        
        print("Migration terminée avec succès")
        return True
        
    except Exception as e:
        print(f"Erreur lors de la migration: {str(e)}", file=sys.stderr)
        return False
    finally:
        await db_manager.close()

if __name__ == "__main__":
    exit(0 if asyncio.run(migrate()) else 1)
