#!/usr/bin/env python3
"""
Script pour tester la connexion à la base de données.
"""
import asyncio
import os
import sys
from pathlib import Path

# Ajouter le répertoire parent au path
sys.path.append(str(Path(__file__).parent.parent))

from src.core.database import db_manager

async def test_connection():
    """Teste la connexion à la base de données."""
    try:
        await db_manager.connect()
        result = await db_manager.execute("SELECT 1")
        print(f"Connexion réussie: {result}")
        return True
    except Exception as e:
        print(f"Erreur de connexion: {str(e)}", file=sys.stderr)
        return False
    finally:
        await db_manager.close()

if __name__ == "__main__":
    exit(0 if asyncio.run(test_connection()) else 1)
