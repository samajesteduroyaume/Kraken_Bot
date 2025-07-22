"""
Visualisation du carnet d'ordres avec Matplotlib.

Ce script montre comment visualiser le carnet d'ordres avec des graphiques
interactifs qui mettent à jour en temps réel via WebSocket.
"""

import asyncio
import json
import logging
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.gridspec import GridSpec
from typing import Dict, List, Optional, Any, Callable, Awaitable

from src.core.market.order_book_websocket import OrderBookWebSocket

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

class OrderBookVisualizer:
    """Classe pour visualiser le carnet d'ordres en temps réel."""
    
    def __init__(self, symbol: str = 'XBT/USD', depth: int = 20):
        """
        Initialise le visualiseur de carnet d'ordres.
        
        Args:
            symbol: Paire de trading à visualiser (ex: 'XBT/USD')
            depth: Nombre de niveaux de prix à afficher
        """
        self.symbol = symbol
        self.depth = depth
        self.fig = None
        self.ax1 = None  # Graphique du carnet d'ordres
        self.ax2 = None  # Graphique du spread et du déséquilibre
        self.ax3 = None  # Graphique du volume cumulé
        self.bid_bars = None
        self.ask_bars = None
        self.spread_line = None
        self.imbalance_line = None
        self.cumulative_bid = None
        self.cumulative_ask = None
        self.websocket = None
        self.update_interval = 1.0  # secondes
        self.running = False
        
        # Données du carnet d'ordres
        self.bids = []
        self.asks = []
        self.spread_history = []
        self.imbalance_history = []
        self.max_history = 100  # Nombre maximum de points d'historique à afficher
        
    async def start(self):
        """Démarre la visualisation."""
        self.running = True
        
        # Initialiser la figure
        self._init_figure()
        
        # Démarrer le WebSocket
        self.websocket = OrderBookWebSocket(
            symbol=self.symbol,
            callback=self._on_orderbook_update,
            depth=100  # Récupérer plus de données que nécessaire pour une meilleure visualisation
        )
        
        # Démarrer la mise à jour de l'interface
        await self._run_visualization()
    
    def _init_figure(self):
        """Initialise la figure Matplotlib."""
        plt.ion()  # Mode interactif
        self.fig = plt.figure(figsize=(14, 10))
        self.fig.suptitle(f'Carnet d\'ordres en temps réel - {self.symbol}', fontsize=16)
        
        # Configuration de la grille
        gs = GridSpec(3, 1, height_ratios=[2, 1, 1])
        
        # Graphique du carnet d'ordres
        self.ax1 = self.fig.add_subplot(gs[0])
        self.ax1.set_title('Profondeur du marché')
        self.ax1.set_xlabel('Prix')
        self.ax1.set_ylabel('Volume')
        self.ax1.grid(True, alpha=0.3)
        
        # Graphique du spread et du déséquilibre
        self.ax2 = self.fig.add_subplot(gs[1])
        self.ax2.set_title('Spread et Déséquilibre')
        self.ax2.grid(True, alpha=0.3)
        
        # Graphique du volume cumulé
        self.ax3 = self.fig.add_subplot(gs[2])
        self.ax3.set_title('Volume cumulé')
        self.ax3.set_xlabel('Niveau de prix')
        self.ax3.set_ylabel('Volume cumulé')
        self.ax3.grid(True, alpha=0.3)
        
        plt.tight_layout()
    
    async def _on_orderbook_update(self, data: Dict[str, Any]):
        """Traite une mise à jour du carnet d'ordres."""
        try:
            # Mettre à jour les données
            self.bids = sorted(
                [{'price': float(b['price']), 'amount': float(b['amount'])} 
                 for b in data.get('bids', [])],
                key=lambda x: -x['price']
            )[:self.depth]
            
            self.asks = sorted(
                [{'price': float(a['price']), 'amount': float(a['amount'])} 
                 for a in data.get('asks', [])],
                key=lambda x: x['price']
            )[:self.depth]
            
            # Mettre à jour l'historique du spread et du déséquilibre
            if 'metrics' in data and data['metrics'].get('spread') is not None:
                self.spread_history.append(data['metrics']['spread'])
                if len(self.spread_history) > self.max_history:
                    self.spread_history.pop(0)
            
            if 'metrics' in data and 'imbalance' in data['metrics'] and data['metrics']['imbalance'] is not None:
                self.imbalance_history.append(data['metrics']['imbalance'])
                if len(self.imbalance_history) > self.max_history:
                    self.imbalance_history.pop(0)
            
        except Exception as e:
            logger.error(f"Erreur lors de la mise à jour du carnet d'ordres: {e}", exc_info=True)
    
    def _update_plot(self, frame):
        """Met à jour les graphiques."""
        if not self.bids or not self.asks:
            return
            
        try:
            # Nettoyer les axes
            for ax in [self.ax1, self.ax2, self.ax3]:
                ax.clear()
            
            # 1. Graphique de profondeur du marché
            bid_prices = [b['price'] for b in self.bids]
            bid_volumes = [b['amount'] for b in self.bids]
            ask_prices = [a['price'] for a in self.asks]
            ask_volumes = [a['amount'] for a in self.asks]
            
            # Afficher les barres d'offre et de demande
            self.bid_bars = self.ax1.barh(
                [str(p) for p in bid_prices],
                bid_volumes,
                color='green',
                alpha=0.6,
                label='Offres (Bid)'
            )
            
            self.ask_bars = self.ax1.barh(
                [str(p) for p in ask_prices],
                ask_volumes,
                color='red',
                alpha=0.6,
                label='Demandes (Ask)'
            )
            
            self.ax1.set_title(f'Profondeur du marché - {self.symbol}')
            self.ax1.set_xlabel('Volume')
            self.ax1.set_ylabel('Prix')
            self.ax1.legend()
            self.ax1.grid(True, alpha=0.3)
            
            # 2. Graphique du spread et du déséquilibre
            if self.spread_history:
                spread_color = 'tab:blue'
                spread_ax = self.ax2
                
                # Tracer le spread
                spread_ax.plot(
                    self.spread_history,
                    color=spread_color,
                    label='Spread',
                    alpha=0.7
                )
                spread_ax.set_ylabel('Spread', color=spread_color)
                spread_ax.tick_params(axis='y', labelcolor=spread_color)
                
                # Ajouter un deuxième axe y pour le déséquilibre
                if self.imbalance_history:
                    imbalance_ax = spread_ax.twinx()
                    imbalance_color = 'tab:red'
                    
                    imbalance_ax.plot(
                        self.imbalance_history,
                        color=imbalance_color,
                        label='Déséquilibre',
                        alpha=0.7
                    )
                    imbalance_ax.set_ylabel('Déséquilibre', color=imbalance_color)
                    imbalance_ax.tick_params(axis='y', labelcolor=imbalance_color)
                    imbalance_ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
                    
                    # Ajouter une légende combinée
                    lines1, labels1 = spread_ax.get_legend_handles_labels()
                    lines2, labels2 = imbalance_ax.get_legend_handles_labels()
                    spread_ax.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
                
                spread_ax.set_title('Évolution du spread et du déséquilibre')
                spread_ax.grid(True, alpha=0.3)
            
            # 3. Graphique du volume cumulé
            if bid_prices and ask_prices:
                # Calculer les volumes cumulés
                cum_bid_volumes = np.cumsum(bid_volumes[::-1])[::-1]  # Cumul inverse pour les bids
                cum_ask_volumes = np.cumsum(ask_volumes)  # Cumul normal pour les asks
                
                # Tracer les volumes cumulés
                self.cumulative_bid, = self.ax3.plot(
                    bid_prices,
                    cum_bid_volumes,
                    'g-',
                    label='Volume cumulé (Bid)',
                    alpha=0.8
                )
                
                self.cumulative_ask, = self.ax3.plot(
                    ask_prices,
                    cum_ask_volumes,
                    'r-',
                    label='Volume cumulé (Ask)',
                    alpha=0.8
                )
                
                # Ajouter une ligne verticale pour le prix actuel
                mid_price = (bid_prices[0] + ask_prices[0]) / 2
                self.ax3.axvline(
                    x=mid_price,
                    color='k',
                    linestyle='--',
                    alpha=0.5,
                    label=f'Prix actuel: {mid_price:.2f}'
                )
                
                self.ax3.set_title('Volume cumulé par niveau de prix')
                self.ax3.set_xlabel('Prix')
                self.ax3.set_ylabel('Volume cumulé')
                self.ax3.legend()
                self.ax3.grid(True, alpha=0.3)
            
            # Ajuster l'espacement
            plt.tight_layout()
            
        except Exception as e:
            logger.error(f"Erreur lors de la mise à jour du graphique: {e}", exc_info=True)
    
    async def _run_visualization(self):
        """Boucle principale de visualisation."""
        # Créer une animation qui met à jour le graphique toutes les 500ms
        ani = FuncAnimation(
            self.fig,
            self._update_plot,
            interval=500,  # ms
            cache_frame_data=False
        )
        
        # Afficher le graphique
        plt.show(block=True)
        
        # Attendre que la visualisation soit terminée
        while self.running and plt.fignum_exists(self.fig.number):
            await asyncio.sleep(0.1)
        
        # Nettoyer
        if self.websocket:
            await self.websocket.disconnect()
    
    async def stop(self):
        """Arrête la visualisation."""
        self.running = False
        if self.websocket:
            await self.websocket.disconnect()
        plt.close('all')


async def main():
    """Fonction principale."""
    # Créer et démarrer le visualiseur
    visualizer = OrderBookVisualizer(symbol='XBT/USD', depth=10)
    
    try:
        await visualizer.start()
    except KeyboardInterrupt:
        print("\nArrêt du visualiseur...")
    except Exception as e:
        print(f"Erreur: {e}")
    finally:
        await visualizer.stop()


if __name__ == "__main__":
    # Désactiver les logs verbeux de Matplotlib
    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    
    # Exécuter le visualiseur
    asyncio.run(main())
