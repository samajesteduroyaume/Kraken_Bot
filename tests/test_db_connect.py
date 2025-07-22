import asyncio
from src.core.database import db_manager

async def main():
    print("DEBUG: AVANT CONNECT")
    await db_manager.connect()
    print("DEBUG: APRES CONNECT")

if __name__ == "__main__":
    asyncio.run(main()) 