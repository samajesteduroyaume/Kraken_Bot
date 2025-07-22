"""
Exemple d'implémentation d'un système de trading adaptatif
qui utilise plusieurs stratégies et s'adapte aux conditions de marché.
"""
import os
import sys
import logging
import time
import pandas as pd
from datetime import datetime, timedelta, timezone
from decimal import Decimal
import ccxt

# Ajout du répertoire racine au PYTHONPATH
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.strategies.adaptive_strategy_manager import AdaptiveStrategyManager
from src.strategies.momentum_strategy import MomentumStrategy
from src.strategies.mean_reversion_strategy import MeanReversionStrategy
from src.strategies.breakout_strategy import BreakoutStrategy
from src.strategies.grid_strategy import GridStrategy
from src.strategies.swing_strategy import SwingStrategy

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('adaptive_trading.log')
    ]
)
logger = logging.getLogger(__name__)

class AdaptiveTradingBot:
    """Bot de trading qui utilise le gestionnaire de stratégies adaptatives."""
    
    def __init__(self, exchange_id: str = 'kraken', symbol: str = 'BTC/USD', timeframe: str = '1h'):
        """
        Initialise le bot de trading adaptatif.
        
        Args:
            exchange_id: Identifiant de l'échange (par défaut: 'kraken')
            symbol: Paire de trading (par défaut: 'BTC/USD')
            timeframe: Période de temps pour les bougies (par défaut: '1h')
        """
        self.exchange_id = exchange_id
        self.symbol = symbol
        self.timeframe = timeframe
        self.exchange = self._init_exchange()
        self.strategies = {}
        self.adaptive_manager = None
        self.position = None
        self.balance = 10000.0  # Solde initial en USD
        self.initial_balance = self.balance
        self.trade_history = []
        self.setup_strategies()
    
    def _init_exchange(self):
        """Initialise la connexion à l'échange."""
        try:
            exchange_class = getattr(ccxt, self.exchange_id)
            exchange = exchange_class({
                'enableRateLimit': True,
                'options': {
                    'adjustForTimeDifference': True,
                }
            })
            
            # Charger les marchés (nécessaire pour certains échanges)
            exchange.load_markets()
            
            logger.info(f"Connecté à {self.exchange_id}")
            return exchange
            
        except Exception as e:
            logger.error(f"Erreur lors de l'initialisation de l'échange: {str(e)}")
            raise
    
    def setup_strategies(self):
        """Initialise les stratégies de trading."""
        # Configuration des stratégies
        momentum_config = {
            'rsi_period': 14,
            'rsi_overbought': 70,
            'rsi_oversold': 30,
            'macd_fast': 12,
            'macd_slow': 26,
            'macd_signal': 9,
            'atr_period': 14,
            'adx_period': 14,
            'adx_threshold': 25
        }
        
        mean_reversion_config = {
            'bollinger_period': 20,
            'bollinger_std': 2.0,
            'rsi_period': 14,
            'rsi_overbought': 70,
            'rsi_oversold': 30,
            'take_profit': 1.5,
            'stop_loss': 1.0
        }
        
        breakout_config = {
            'resistance_period': 20,
            'support_period': 20,
            'volume_ma_period': 20,
            'min_volume_multiplier': 1.5,
            'confirmation_bars': 2
        }
        
        grid_config = {
            'grid_upper_price': 50000.0,
            'grid_lower_price': 30000.0,
            'grid_levels': 10,
            'position_size': 0.1,  # 10% du solde par position
            'take_profit_pct': 2.0,
            'stop_loss_pct': 1.0
        }
        
        swing_config = {
            'ema_fast': 9,
            'ema_slow': 21,
            'rsi_period': 14,
            'rsi_overbought': 70,
            'rsi_oversold': 30,
            'atr_period': 14,
            'atr_multiplier': 2.0
        }
        
        # Initialisation des stratégies
        self.strategies = {
            'momentum': MomentumStrategy(momentum_config),
            'mean_reversion': MeanReversionStrategy(mean_reversion_config),
            'breakout': BreakoutStrategy(breakout_config),
            'grid': GridStrategy(grid_config),
            'swing': SwingStrategy(swing_config)
        }
        
        # Initialisation du gestionnaire adaptatif
        self.adaptive_manager = AdaptiveStrategyManager(
            strategies=self.strategies,
            initial_capital=self.balance
        )
        
        logger.info("Stratégies initialisées avec succès")
    
    def fetch_ohlcv(self, limit: int = 200) -> pd.DataFrame:
        """Récupère les données OHLCV depuis l'échange."""
        try:
            ohlcv = self.exchange.fetch_ohlcv(
                self.symbol, 
                self.timeframe, 
                limit=limit
            )
            
            df = pd.DataFrame(
                ohlcv, 
                columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
            )
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            # Conversion des types
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = df[col].astype(float)
                
            return df
            
        except Exception as e:
            logger.error(f"Erreur lors de la récupération des données OHLCV: {str(e)}")
            return pd.DataFrame()
    
    def execute_trade(self, signal, current_price):
        """Exécute un trade basé sur le signal reçu."""
        if not signal:
            return
            
        # Calcul de la taille de la position (1% du solde par défaut)
        position_size = (self.balance * 0.01) / current_price
        
        # Simulation d'exécution de l'ordre
        order = {
            'symbol': self.symbol,
            'side': 'buy' if signal.direction > 0 else 'sell',
            'price': float(current_price),
            'amount': float(position_size),
            'cost': float(position_size * current_price),
            'timestamp': datetime.now(timezone.utc),
            'strategy': signal.metadata.get('strategy', 'unknown') if hasattr(signal, 'metadata') else 'unknown'
        }
        
        # Mise à jour du solde (simplifié)
        if signal.direction > 0:  # Achat
            self.balance -= order['cost']
            self.position = {
                'entry_price': current_price,
                'size': position_size,
                'entry_time': datetime.utcnow(),
                'stop_loss': signal.stop_loss,
                'take_profit': signal.take_profit
            }
        else:  # Vente
            if self.position:
                profit_pct = ((current_price - self.position['entry_price']) / 
                             self.position['entry_price']) * 100
                self.balance += self.position['size'] * current_price
                order['profit_pct'] = profit_pct
                self.position = None
        
        # Enregistrement du trade
        self.trade_history.append(order)
        logger.info(
            f"Ordre exécuté: {order['side']} {order['amount']} {self.symbol} "
            f"à {order['price']} (coût: {order['cost']:.2f} USD)"
        )
    
    def run(self, duration_days: int = 7):
        """Exécute le bot pour une durée donnée."""
        self.end_time = datetime.now(timezone.utc) + timedelta(days=duration_days)
        self.duration_days = duration_days  # Sauvegarder pour le rapport
        logger.info(f"Démarrage du bot de trading jusqu'au {self.end_time}")
        
        try:
            while datetime.now(timezone.utc) < self.end_time:
                # Récupération des données du marché
                ohlcv_data = self.fetch_ohlcv(limit=200)
                if ohlcv_data.empty:
                    logger.warning("Aucune donnée reçue, nouvelle tentative dans 1 minute...")
                    time.sleep(60)
                    continue
                
                # Conversion en format attendu par le gestionnaire
                data = {self.timeframe: ohlcv_data}
                
                # Génération des signaux
                signals = self.adaptive_manager.get_strategy_signals(self.symbol, ohlcv_data)
                
                # Exécution des trades
                current_price = Decimal(str(ohlcv_data['close'].iloc[-1]))
                if signals:
                    for signal in signals:
                        self.execute_trade(signal, current_price)
                
                # Pause avant la prochaine itération
                time.sleep(60)  # Vérification toutes les minutes
                
        except KeyboardInterrupt:
            logger.info("Arrêt demandé par l'utilisateur")
        except Exception as e:
            logger.error(f"Erreur inattendue: {str(e)}", exc_info=True)
        finally:
            self.shutdown()
    
    def shutdown(self):
        """Arrête le bot et affiche un rapport de performance."""
        logger.info("Arrêt du bot de trading...")
        
        # Calcul des statistiques de performance
        total_trades = len(self.trade_history)
        winning_trades = len([t for t in self.trade_history if t.get('profit_pct', 0) > 0])
        losing_trades = total_trades - winning_trades
        win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
        
        total_return = ((self.balance - self.initial_balance) / self.initial_balance) * 100
        
        # Calcul de la durée d'exécution
        duration = self.duration_days  # Utiliser la durée prévue au lieu de la calculer
        
        # Affichage du rapport
        print("\n" + "="*50)
        print(f"Rapport de performance - {datetime.now(timezone.utc)}")
        print("="*50)
        print(f"Période: {duration} jours")
        print(f"Solde initial: {self.initial_balance:.2f} USD")
        print(f"Solde final: {self.balance:.2f} USD")
        print(f"Rendement total: {total_return:.2f}%")
        print(f"Nombre total de trades: {total_trades}")
        print(f"Taux de réussite: {win_rate:.1f}%")
        print(f"Trades gagnants: {winning_trades}")
        print(f"Trades perdants: {losing_trades}")
        
        # Affichage des stratégies actives
        active_strategies = self.adaptive_manager.get_active_strategies()
        if active_strategies:
            print("\nStratégies actives:")
            for name, weight in active_strategies.items():
                print(f"  - {name}: {weight*100:.1f}%")
        
        print("="*50 + "\n")

if __name__ == "__main__":
    # Configuration
    SYMBOL = "BTC/USD"
    TIMEFRAME = "1h"
    DURATION_DAYS = 7  # Durée d'exécution en jours
    
    # Initialisation et exécution du bot
    bot = AdaptiveTradingBot(symbol=SYMBOL, timeframe=TIMEFRAME)
    bot.run(duration_days=DURATION_DAYS)
