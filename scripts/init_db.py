#!/usr/bin/env python3
"""
Script d'initialisation de la base de données.
"""
import asyncio
import os
import sys
from pathlib import Path

# Ajouter le répertoire parent au path
sys.path.append(str(Path(__file__).parent.parent))

from src.core.database import db_manager

async def initialize():
    """Initialise la base de données."""
    try:
        await db_manager.connect()
        # Créer les tables si elles n'existent pas
        await db_manager.execute("""
            CREATE TABLE IF NOT EXISTS pairs (
                id SERIAL PRIMARY KEY,
                symbol VARCHAR(20) NOT NULL UNIQUE,
                base_currency VARCHAR(10) NOT NULL,
                quote_currency VARCHAR(10) NOT NULL,
                active BOOLEAN DEFAULT true,
                min_order_size DECIMAL(20, 8),
                max_order_size DECIMAL(20, 8),
                price_precision INTEGER,
                quantity_precision INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        await db_manager.execute("""
            CREATE TABLE IF NOT EXISTS trades (
                id SERIAL PRIMARY KEY,
                pair_id INTEGER REFERENCES pairs(id),
                side VARCHAR(4) NOT NULL,
                price DECIMAL(20, 8) NOT NULL,
                amount DECIMAL(20, 8) NOT NULL,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                status VARCHAR(20) DEFAULT 'open',
                strategy VARCHAR(50),
                stop_loss DECIMAL(20, 8),
                take_profit DECIMAL(20, 8),
                close_price DECIMAL(20, 8),
                close_time TIMESTAMP,
                pnl DECIMAL(20, 8)
            )
        """)

        await db_manager.execute("""
            CREATE TABLE IF NOT EXISTS ohlcv (
                id SERIAL PRIMARY KEY,
                pair_id INTEGER REFERENCES pairs(id),
                timestamp BIGINT NOT NULL,
                open DECIMAL(20, 8) NOT NULL,
                high DECIMAL(20, 8) NOT NULL,
                low DECIMAL(20, 8) NOT NULL,
                close DECIMAL(20, 8) NOT NULL,
                volume DECIMAL(30, 8) NOT NULL,
                count INTEGER,
                interval VARCHAR(10) NOT NULL,
                UNIQUE(pair_id, timestamp, interval)
            )
        """)
        print("Base de données initialisée avec succès")
        return True
    except Exception as e:
        print(f"Erreur lors de l'initialisation: {str(e)}", file=sys.stderr)
        return False
    finally:
        await db_manager.close()

if __name__ == "__main__":
    exit(0 if asyncio.run(initialize()) else 1)
