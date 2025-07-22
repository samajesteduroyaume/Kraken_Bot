"""
Script pour vérifier les paires de trading disponibles sur Kraken.
"""
import asyncio
import aiohttp
import json
import logging
from typing import List, Dict, Any
import os
import sys

# Ajouter le répertoire racine au PYTHONPATH
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# URL de l'API Kraken
KRAKEN_API_URL = "https://api.kraken.com/0/public"

async def get_available_pairs() -> List[str]:
    """Récupère la liste des paires disponibles sur Kraken."""
    url = f"{KRAKEN_API_URL}/AssetPairs"
    async with aiohttp.ClientSession() as session:
        try:
            async with session.get(url) as response:
                data = await response.json()
                if data.get('error'):
                    logger.error(f"Erreur de l'API: {data['error']}")
                    return []
                return list(data.get('result', {}).keys())
        except Exception as e:
            logger.error(f"Erreur lors de la récupération des paires: {e}")
            return []

async def check_pair(session: aiohttp.ClientSession, pair: str) -> bool:
    """Vérifie si une paire est valide sur Kraken."""
    url = f"{KRAKEN_API_URL}/Ticker"
    params = {'pair': pair}
    try:
        async with session.get(url, params=params) as response:
            data = await response.json()
            return 'result' in data and pair in data['result']
    except Exception as e:
        logger.error(f"Erreur lors de la vérification de la paire {pair}: {e}")
        return False

async def get_valid_pairs(pairs: List[str]) -> List[str]:
    """Filtre les paires pour ne garder que celles qui sont valides."""
    valid_pairs = []
    async with aiohttp.ClientSession() as session:
        # Vérifier chaque paire
        tasks = [check_pair(session, pair) for pair in pairs]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filtrer les paires valides
        for i, is_valid in enumerate(results):
            if isinstance(is_valid, bool) and is_valid:
                valid_pairs.append(pairs[i])
                logger.info(f"✅ Paire valide: {pairs[i]}")
            else:
                logger.warning(f"❌ Paire invalide ou erreur: {pairs[i]}")
    
    return valid_pairs

def get_kraken_symbol(base: str, quote: str = 'USD') -> str:
    """Convertit une paire standard en format Kraken."""
    # Mappage des symboles courants vers le format Kraken
    symbol_map = {
        'BTC': 'XBT',  # Bitcoin
        'XBT': 'XBT',  # Bitcoin (format alternatif)
        'ETH': 'ETH',  # Ethereum
        'XRP': 'XRP',  # Ripple
        'LTC': 'LTC',  # Litecoin
        'BCH': 'BCH',  # Bitcoin Cash
        'ADA': 'ADA',  # Cardano
        'DOT': 'DOT',  # Polkadot
        'LINK': 'LINK',  # Chainlink
        'UNI': 'UNI',  # Uniswap
        'SOL': 'SOL',  # Solana
        'DOGE': 'XDG',  # Dogecoin
        'SHIB': 'SHIB',  # Shiba Inu
        'TRX': 'TRX',  # Tron
        'MATIC': 'MATIC',  # Polygon
        'AVAX': 'AVAX',  # Avalanche
        'EUR': 'ZEUR',  # Euro
        'USD': 'ZUSD',  # Dollar US
        'GBP': 'ZGBP',  # Livre sterling
        'JPY': 'ZJPY',  # Yen japonais
    }
    
    base = symbol_map.get(base.upper(), base.upper())
    quote = symbol_map.get(quote.upper(), quote.upper())
    
    # Format Kraken: XBTZEUR, ETHZUSD, etc.
    return f"{base}{quote}"

def load_pairs_from_config() -> List[str]:
    """Charge les paires depuis la configuration actuelle et les convertit au format Kraken."""
    try:
        from src.config.trading_pairs_config import load_trading_pairs_config
        config = load_trading_pairs_config()
        pairs = []
        
        # Récupérer les paires de chaque catégorie
        for category in ['high_liquidity', 'medium_liquidity', 'low_liquidity']:
            for pair in config.get(category, []):
                # Convertir le format "BTC/USD" en ["BTC", "USD"]
                if '/' in pair:
                    base, quote = pair.split('/')
                    kraken_pair = get_kraken_symbol(base, quote)
                    pairs.append(kraken_pair)
        
        # Ajouter des paires supplémentaires couramment utilisées
        additional_pairs = [
            'XBTUSDT', 'XBTUSDC', 'ETHUSDT', 'ETHUSDC', 'XBTUSDT',
            'XBTUSDC', 'XBTDAI', 'ETHDAI', 'USDTZUSD', 'USDCZUSD'
        ]
        pairs.extend(additional_pairs)
        
        return list(set(pairs))  # Éliminer les doublons
    except Exception as e:
        logger.error(f"Erreur lors du chargement des paires depuis la configuration: {e}")
        return []

async def main():
    logger.info("Démarrage de la vérification des paires...")
    
    # Charger les paires depuis la configuration actuelle
    current_pairs = load_pairs_from_config()
    logger.info(f"{len(current_pairs)} paires chargées depuis la configuration")
    
    # Récupérer les paires disponibles sur Kraken
    available_pairs = await get_available_pairs()
    logger.info(f"{len(available_pairs)} paires disponibles sur Kraken")
    
    # Filtrer les paires actuelles pour ne garder que celles qui sont disponibles
    valid_pairs = [p for p in current_pairs if p in available_pairs]
    
    # Afficher les résultats
    logger.info(f"\nRésultats de la vérification:")
    logger.info(f"- Paires dans la configuration: {len(current_pairs)}")
    logger.info(f"- Paires valides sur Kraken: {len(valid_pairs)}")
    logger.info(f"- Taux de réussite: {len(valid_pairs)/len(current_pairs)*100:.1f}%")
    
    # Afficher les paires valides
    logger.info("\nPaires valides:")
    for pair in sorted(valid_pairs):
        logger.info(f"- {pair}")
    
    # Sauvegarder les paires valides dans un fichier
    with open('valid_pairs.json', 'w') as f:
        json.dump({"valid_pairs": valid_pairs}, f, indent=2)
    logger.info("\nListe des paires valides sauvegardée dans 'valid_pairs.json'")

if __name__ == "__main__":
    asyncio.run(main())
