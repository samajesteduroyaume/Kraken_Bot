import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import pandas as pd
from datetime import datetime
from src.core.api.kraken import KrakenAPI
from dotenv import load_dotenv

# Charger les variables d'environnement
load_dotenv()

PAIR = os.getenv('TRAINING_PAIR', 'BTC/USD')
OUTPUT = os.getenv('TRAINING_DATA_CSV', f'data/{PAIR.replace("/", "_")}_ohlc.csv')
INTERVAL = int(os.getenv('TRAINING_INTERVAL', '1440'))  # 1 jour par défaut
DAYS = int(os.getenv('TRAINING_DAYS', '365'))  # Nombre de jours d'historique

async def fetch_ohlc(pair, interval, days):
    api = KrakenAPI()
    async with api:
        now = int(datetime.now().timestamp())
        since = now - days * 24 * 3600
        ohlc_data = await api.get_ohlc_data(pair=pair, interval=interval, since=since)
        if not ohlc_data or 'result' not in ohlc_data:
            raise ValueError('Aucune donnée OHLC reçue de l\'API Kraken')
        result = ohlc_data['result']
        # Trouver la bonne clé de paire
        pair_key = next((k for k in result if pair.replace('/', '') in k or pair in k), None)
        if not pair_key:
            raise KeyError(f'Paire {pair} non trouvée dans la réponse Kraken. Clés: {list(result.keys())}')
        df = pd.DataFrame(result[pair_key], columns=['time', 'open', 'high', 'low', 'close', 'vwap', 'volume', 'count'])
        df['time'] = pd.to_datetime(df['time'], unit='s')
        df.set_index('time', inplace=True)
        return df

if __name__ == '__main__':
    import asyncio
    print(f'Téléchargement des données OHLC pour {PAIR}...')
    df = asyncio.run(fetch_ohlc(PAIR, INTERVAL, DAYS))
    print(f'{len(df)} lignes téléchargées.')
    df.to_csv(OUTPUT)
    print(f'Données sauvegardées dans {OUTPUT}') 