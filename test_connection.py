import asyncio
import logging
from aiohttp import ClientError, ClientResponseError, ClientTimeout
from src.core.config import Config
from src.core.simulation_mode import SimulationConfig, SimulationMode
from src.core.api.kraken import KrakenAPI
from src.core.services.multi_pair_trader import MultiPairTrader

# Configurer le logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def test_connection():
    
    # Initialiser la configuration
    config = Config()
    sim_config = SimulationConfig(mode=SimulationMode.REAL_TIME)
    
    # Configurer l'API Kraken
    api_config = {
        'credentials': {
            'api_key': config.api_config['api_key'],
            'api_secret': config.api_config['api_secret']
        },
        'base_url': 'https://api.kraken.com',
        'version': '0',
        'timeout': 30
    }
    
    # Initialiser l'API
    api = KrakenAPI(api_config)
    
    # Obtenir le solde initial depuis Kraken
    async with api:
        balance = await api.get_balance()
        initial_balance = float(balance.get('ZUSD', 10000.0))  # Utiliser 10000 USD si aucun solde n'est trouvé
        logger.info(f"Solde initial depuis Kraken: {initial_balance:.2f} USD")
    
    # Initialiser le trader avec le solde initial
    trader = MultiPairTrader(
        api=api,
        config=sim_config
    )
    
    # Initialiser le trader
    await trader.initialize()

if __name__ == "__main__":
    asyncio.run(test_connection())
