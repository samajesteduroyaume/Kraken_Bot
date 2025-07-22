"""
Exemple d'utilisation du module de carnet d'ordres avancé.

Ce script montre comment utiliser les fonctionnalités avancées du carnet d'ordres,
y compris la mise à jour en temps réel, l'analyse des métriques et la visualisation.
"""

import asyncio
import logging
import sys
from decimal import Decimal
from datetime import datetime

# Ajouter le répertoire racine au PYTHONPATH
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core.api.kraken_api import KrakenAPI
from src.core.market.data_manager import MarketDataManager

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

async def main():
    """Fonction principale de l'exemple."""
    # Initialiser l'API Kraken
    api = KrakenAPI()
    
    # Initialiser le gestionnaire de données de marché
    config = {
        'candle_interval': '5m',
        'order_book_depth': 100,
        'order_book_update_interval': 1.0,
        'max_history': 1000
    }
    data_manager = MarketDataManager(api, config)
    
    # Paires à surveiller
    symbols = ['XBT/USD', 'ETH/USD']
    
    try:
        # Démarrer la surveillance des paires
        for symbol in symbols:
            await data_manager.update_market_data(symbol)
            
            # Afficher les informations de base
            order_book = data_manager.get_order_book(symbol)
            if order_book:
                best_bid = order_book['bids'][0]['price'] if order_book['bids'] else 'N/A'
                best_ask = order_book['asks'][0]['price'] if order_book['asks'] else 'N/A'
                spread = float(best_ask) - float(best_bid) if best_ask != 'N/A' and best_bid != 'N/A' else 0
                
                print(f"\n=== {symbol} ===")
                print(f"Meilleur achat: {best_bid}")
                print(f"Meilleure vente: {best_ask}")
                print(f"Spread: {spread:.2f}")
                
                # Afficher les métriques avancées
                ob_manager = data_manager.get_order_book_manager(symbol)
                if ob_manager and ob_manager.current_snapshot:
                    metrics = ob_manager.current_snapshot.metrics
                    print("\nMétriques avancées:")
                    print(f"- Imbalance: {metrics.imbalance:.2%}" if hasattr(metrics, 'imbalance') else "- Imbalance: N/A")
                    print(f"- VWAP Achat: {metrics.vwap_bid:.2f}" if hasattr(metrics, 'vwap_bid') else "- VWAP Achat: N/A")
                    print(f"- VWAP Vente: {metrics.vwap_ask:.2f}" if hasattr(metrics, 'vwap_ask') else "- VWAP Vente: N/A")
                    
                    # Calculer le déséquilibre sur 5 niveaux
                    imbalance = ob_manager.current_snapshot.metrics.get_order_imbalance(5)
                    print(f"- Déséquilibre (5 niveaux): {imbalance:.2%}")
                    
                    # Calculer l'impact de prix pour un ordre de 1 BTC
                    impact_achat = ob_manager.current_snapshot.metrics.get_price_impact(1.0, 'buy')
                    impact_vente = ob_manager.current_snapshot.metrics.get_price_impact(1.0, 'sell')
                    print(f"- Impact prix achat 1 BTC: {impact_achat:.4%}")
                    print(f"- Impact prix vente 1 BTC: {impact_vente:.4%}")
        
        # Attendre les mises à jour pendant 30 secondes
        print("\nSurveillance active pendant 30 secondes...")
        await asyncio.sleep(30)
        
    except KeyboardInterrupt:
        print("\nArrêt demandé par l'utilisateur...")
    except Exception as e:
        logger.error(f"Erreur dans l'exemple: {e}", exc_info=True)
    finally:
        # Nettoyer les ressources
        await data_manager.close()
        print("Nettoyage terminé.")

if __name__ == "__main__":
    asyncio.run(main())
