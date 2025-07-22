# 🛠️ Guide Complet de Création de Stratégies

Ce guide explique comment concevoir, développer, tester et déployer des stratégies de trading personnalisées pour Kraken_Bot. Il couvre les meilleures pratiques, les modèles de conception et les outils disponibles pour vous aider à créer des stratégies robustes et performantes.

## 🏗️ Architecture des Stratégies

### Hiérarchie des Classes

```mermaid
classDiagram
    BaseStrategy <|-- BaseIndicator
    BaseStrategy <|-- ExampleStrategy
    BaseStrategy : +strategy_name: str
    BaseStrategy : +timeframes: list[str]
    BaseStrategy : +symbols: list[str]
    BaseStrategy : +analyze()
    BaseStrategy : +on_tick()
    BaseStrategy : +on_bar()
    BaseStrategy : +on_order()
    BaseStrategy : +on_position()
    
    class ExampleStrategy {
        +parameters: dict
        +indicators: dict
        +initialize()
        +calculate_signals()
        +manage_risk()
    }
```

### Cycle de Vie d'une Stratégie

1. **Initialisation** : Chargement des paramètres et configuration
2. **Démarrage** : Connexion aux flux de données
3. **Exécution** : Analyse des marchés et génération de signaux
4. **Gestion des positions** : Ouverture/fermeture selon les signaux
5. **Arrêt** : Clôture propre des positions et libération des ressources

## 🧩 Structure d'une Stratégie

Chaque stratégie doit hériter de la classe `BaseStrategy` et implémenter les méthodes requises :

```python
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from ..core.strategy import BaseStrategy
from ..core.types import (
    Symbol, Timeframe, Signal, SignalType, Position, Order, OrderType, 
    OrderSide, OrderStatus, Candle, Ticker, Balance
)
from ..core.events import Event, EventType
from ..indicators.ta import SMA, RSI, MACD, BBANDS, ATR, Supertrend, Ichimoku, ADX, StochasticRSI
from ..risk.manager import RiskManager
from ..utils.logging import get_logger

@dataclass
class AdvancedTradingStrategy(BaseStrategy):
    """
    Stratégie de trading avancée avec support multi-indicateurs et gestion des risques.
    
    Cette stratégie combine plusieurs indicateurs techniques pour générer des signaux
    de trading plus fiables et intègre une gestion avancée des risques.
    """
    # Paramètres de trading
    fast_ma: int = 9
    slow_ma: int = 21
    rsi_period: int = 14
    atr_period: int = 14
    adx_period: int = 14
    supertrend_period: int = 10
    supertrend_multiplier: float = 3.0
    stoch_rsi_k: int = 3
    stoch_rsi_d: int = 3
    stoch_rsi_smooth: int = 3
    
    # Paramètres de gestion des risques
    risk_reward_ratio: float = 2.0
    max_position_size: float = 2.0  # % du capital
    max_daily_drawdown: float = 5.0  # %
    max_drawdown: float = 15.0  # %
    
    # Paramètres internes
    indicators: Dict[str, Any] = field(default_factory=dict, init=False)
    risk_manager: RiskManager = field(default_factory=RiskManager, init=False)
    logger = get_logger(__name__)
    
    def __post_init__(self):
        """Initialisation de la stratégie."""
        super().__init__()
        self._setup_indicators()
        self._validate_parameters()
    
    def _setup_indicators(self):
        """Initialise les indicateurs techniques."""
        self.indicators = {
            # Moyennes mobiles
            'sma_fast': SMA(period=self.fast_ma),
            'sma_slow': SMA(period=self.slow_ma),
            
            # Momentum
            'rsi': RSI(period=self.rsi_period),
            'stoch_rsi': StochasticRSI(
                period=self.rsi_period,
                k=self.stoch_rsi_k,
                d=self.stoch_rsi_d,
                smooth_k=self.stoch_rsi_smooth
            ),
            'macd': MACD(fast=12, slow=26, signal=9),
            
            # Volatilité
            'atr': ATR(period=self.atr_period),
            'bbands': BBANDS(period=20, std=2.0),
            'supertrend': Supertrend(
                period=self.supertrend_period,
                multiplier=self.supertrend_multiplier
            ),
            
            # Tendance
            'adx': ADX(period=self.adx_period),
            'ichimoku': Ichimoku()
        }
    
    def _validate_parameters(self):
        """Valide les paramètres de la stratégie."""
        if self.fast_ma >= self.slow_ma:
            raise ValueError("La période de la MA rapide doit être inférieure à celle de la MA lente")
        if not 0 < self.max_position_size <= 100:
            raise ValueError("La taille de position doit être entre 0 et 100%")
    
    async def on_start(self):
        """Appelé au démarrage de la stratégie."""
        self.logger.info(f"Démarrage de la stratégie {self.name}")
        await self.risk_manager.initialize()
    
    async def on_stop(self):
        """Appelé à l'arrêt de la stratégie."""
        self.logger.info(f"Arrêt de la stratégie {self.name}")
        await self.close_all_positions()
    
    async def on_tick(self, ticker: Ticker):
        """Traite chaque tick de prix (pour le trading haute fréquence)."""
        # Implémentation optionnelle pour le trading haute fréquence
        pass
    
    async def on_bar(self, symbol: Symbol, timeframe: Timeframe, candle: Candle):
        """Traite chaque nouvelle bougie."""
        try:
            # 1. Mettre à jour les indicateurs
            df = await self._update_indicators(symbol, timeframe, candle)
            
            # 2. Vérifier les conditions d'entrée/sortie
            signal = await self._evaluate_conditions(symbol, timeframe, df)
            
            # 3. Gérer le risque et exécuter les trades
            if signal:
                await self._manage_trade(signal, df)
                
        except Exception as e:
            self.logger.error(f"Erreur dans on_bar: {str(e)}", exc_info=True)
            await self.on_error(e)
    
    async def _update_indicators(self, symbol: Symbol, timeframe: Timeframe, candle: Candle) -> pd.DataFrame:
        """Met à jour les indicateurs avec les dernières données."""
        # Récupérer l'historique des bougies (au moins 200 bougies pour les indicateurs)
        df = await self.data_provider.get_ohlcv(symbol, timeframe, limit=200)
        
        if df is None or len(df) < 50:  # Pas assez de données
            return None
            
        # Calculer tous les indicateurs
        for name, indicator in self.indicators.items():
            if hasattr(indicator, 'calculate'):
                df[name] = indicator.calculate(df)
        
        return df
    
    async def _evaluate_conditions(self, symbol: Symbol, timeframe: Timeframe, df: pd.DataFrame) -> Optional[Signal]:
        """Évalue les conditions de trading et génère un signal."""
        if df is None or len(df) < 100:  # Attendre suffisamment de données
            return None
            
        last_candle = df.iloc[-1]
        prev_candle = df.iloc[-2]
        
        # Récupérer les valeurs des indicateurs
        sma_fast = last_candle.get('sma_fast')
        sma_slow = last_candle.get('sma_slow')
        rsi = last_candle.get('rsi')
        stoch_k = last_candle.get('stoch_k')
        stoch_d = last_candle.get('stoch_d')
        adx = last_candle.get('adx')
        supertrend = last_candle.get('supertrend')
        
        # Conditions d'achat
        trend_up = sma_fast > sma_slow
        momentum_up = rsi > 50 and stoch_k > stoch_d
        volatility_ok = last_candle['atr'] > df['atr'].mean() * 0.7  # Volatilité suffisante
        
        buy_condition = (
            trend_up and 
            momentum_up and
            volatility_ok and
            last_candle['close'] > supertrend and
            adx > 25  # Tendance forte
        )
        
        # Conditions de vente
        trend_down = sma_fast < sma_slow
        momentum_down = rsi < 50 or stoch_k < stoch_d
        
        sell_condition = (
            trend_down or 
            last_candle['close'] < supertrend or
            (adx < 20 and rsi > 70)  # Marché sans tendance et surachat
        )
        
        # Générer le signal avec force basée sur la convergence des indicateurs
        signal_strength = 0
        if buy_condition:
            signal_strength = self._calculate_signal_strength(df, 'buy')
            if signal_strength > 0.7:  # Seuil de confirmation
                return Signal(
                    symbol=symbol,
                    signal_type=SignalType.BUY,
                    price=last_candle['close'],
                    timestamp=datetime.utcnow(),
                    timeframe=timeframe,
                    strength=signal_strength,
                    metadata={
                        'indicators': {
                            'sma_fast': sma_fast,
                            'sma_slow': sma_slow,
                            'rsi': rsi,
                            'stoch_k': stoch_k,
                            'stoch_d': stoch_d,
                            'adx': adx,
                            'supertrend': supertrend
                        }
                    }
                )
                
        elif sell_condition:
            signal_strength = self._calculate_signal_strength(df, 'sell')
            if signal_strength > 0.7:  # Seuil de confirmation
                return Signal(
                    symbol=symbol,
                    signal_type=SignalType.SELL,
                    price=last_candle['close'],
                    timestamp=datetime.utcnow(),
                    timeframe=timeframe,
                    strength=signal_strength,
                    metadata={
                        'indicators': {
                            'sma_fast': sma_fast,
                            'sma_slow': sma_slow,
                            'rsi': rsi,
                            'stoch_k': stoch_k,
                            'stoch_d': stoch_d,
                            'adx': adx,
                            'supertrend': supertrend
                        }
                    }
                )
        
        return None
    
    def _calculate_signal_strength(self, df: pd.DataFrame, signal_type: str) -> float:
        """Calcule la force du signal basée sur la convergence des indicateurs."""
        last = df.iloc[-1]
        strength = 0.0
        total_weight = 0
        
        # Poids des différents facteurs
        weights = {
            'trend': 0.3,
            'momentum': 0.25,
            'volatility': 0.2,
            'volume': 0.15,
            'market_condition': 0.1
        }
        
        # 1. Tendance (30%)
        if signal_type == 'buy':
            if last['close'] > last['sma_slow']:
                strength += 0.3 * weights['trend']
            if last['sma_fast'] > last['sma_slow']:
                strength += 0.2 * weights['trend']
            if last['close'] > last['supertrend']:
                strength += 0.3 * weights['trend']
            if last['adx'] > 25:  # Tendance forte
                strength += 0.2 * weights['trend']
        
        # 2. Momentum (25%)
        if signal_type == 'buy' and last['rsi'] > 50:
            strength += (last['rsi'] - 50) / 50 * weights['momentum']
        
        # 3. Volatilité (20%)
        atr_ratio = last['atr'] / df['atr'].mean()
        if 0.7 < atr_ratio < 2.0:  # Volatilité dans une fourchette acceptable
            strength += weights['volatility'] * (1 - abs(1 - atr_ratio))
            
        # 4. Volume (15%)
        volume_ma = df['volume'].rolling(window=20).mean().iloc[-1]
        if last['volume'] > volume_ma:
            strength += weights['volume']
            
        # 5. Conditions de marché (10%)
        if last['adx'] > 25:  # Marché en tendance
            strength += 0.5 * weights['market_condition']
            
        return min(max(strength, 0.0), 1.0)  # Normaliser entre 0 et 1
    
    async def _manage_trade(self, signal: Signal, df: pd.DataFrame):
        """Gère l'exécution des trades en fonction du signal."""
        try:
            # Vérifier les limites de risque
            if not await self.risk_manager.can_open_trade(signal.symbol, signal):
                return
            
            # Calculer la taille de position
            position_size = await self._calculate_position_size(signal, df)
            
            if position_size <= 0:
                return
            
            # Créer l'ordre avec gestion avancée du risque
            order = await self._create_order(signal, position_size, df)
            
            if order:
                # Exécuter l'ordre
                await self.execute_order(order)
                
                # Enregistrer le signal pour analyse ultérieure
                await self._log_signal(signal, order)
                
        except Exception as e:
            self.logger.error(f"Erreur dans _manage_trade: {str(e)}", exc_info=True)
            await self.on_error(e)
    
    async def _create_order(self, signal: Signal, position_size: float, df: pd.DataFrame) -> Optional[Order]:
        """Crée un ordre avec gestion avancée du risque."""
        try:
            # Calculer les niveaux de stop-loss et take-profit
            stop_loss = self._calculate_stop_loss(signal, df)
            take_profit = self._calculate_take_profit(signal, df)
            
            # Déterminer le type d'ordre (market, limit, etc.)
            order_type = OrderType.MARKET  # Par défaut, ordre au marché
            
            # Pour les ordres limités, calculer un prix d'entrée optimal
            limit_price = None
            if order_type == OrderType.LIMIT:
                spread = (df['close'].iloc[-1] - df['open'].iloc[-1]) / df['open'].iloc[-1]
                if signal.signal_type == SignalType.BUY:
                    limit_price = signal.price * (1 - spread/2)  # Acheter en dessous du marché
                else:
                    limit_price = signal.price * (1 + spread/2)  # Vendre au-dessus du marché
            
            # Créer l'ordre
            order = Order(
                symbol=signal.symbol,
                side=OrderSide.BUY if signal.signal_type == SignalType.BUY else OrderSide.SELL,
                type=order_type,
                quantity=position_size,
                price=limit_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                time_in_force='GTC',  # Good Till Cancelled
                leverage=1.0,  # Pas d'effet de levier par défaut
                reduce_only=False,
                close_position=False,
                client_order_id=f"{signal.symbol}-{int(datetime.utcnow().timestamp())}",
                metadata={
                    'strategy': self.name,
                    'signal_strength': signal.strength,
                    'indicators': signal.metadata.get('indicators', {}) if signal.metadata else {}
                }
            )
            
            return order
            
        except Exception as e:
            self.logger.error(f"Erreur dans _create_order: {str(e)}", exc_info=True)
            return None
    
    async def _log_signal(self, signal: Signal, order: Order):
        """Enregistre les détails du signal pour analyse ultérieure."""
        log_entry = {
            'timestamp': datetime.utcnow().isoformat(),
            'symbol': signal.symbol,
            'signal_type': signal.signal_type.name,
            'price': signal.price,
            'strength': signal.strength,
            'order_id': order.id,
            'position_size': order.quantity,
            'stop_loss': order.stop_loss,
            'take_profit': order.take_profit,
            'indicators': signal.metadata.get('indicators', {}) if signal.metadata else {}
        }
        
        # Enregistrer dans la base de données ou un fichier de log
        await self.data_logger.log_signal(log_entry)
    
    async def _calculate_position_size(self, signal: Signal, df: pd.DataFrame) -> float:
        """Calcule la taille de position optimale en fonction du risque."""
        try:
            # Récupérer le solde disponible
            balance = await self.exchange.get_balance()
            if not balance or balance.total_balance <= 0:
                self.logger.error("Solde non disponible")
                return 0.0
                
            equity = balance.total_balance
            
            # Calculer le montant à risquer (1% du capital par défaut)
            risk_per_trade = self.risk_manager.get_risk_per_trade()
            risk_amount = equity * (risk_per_trade / 100.0)
            
            # Calculer le stop-loss en points
            stop_loss = self._calculate_stop_loss(signal, df)
            if stop_loss is None:
                return 0.0
                
            risk_per_unit = abs(signal.price - stop_loss)
            if risk_per_unit <= 0:
                return 0.0
                
            # Calculer la taille de position
            position_size = risk_amount / risk_per_unit
            
            # Convertir en nombre d'unités (arrondi à la décimale appropriée)
            tick_size = self.exchange.get_symbol_info(signal.symbol).get('tick_size', 0.00000001)
            if tick_size > 0:
                position_size = round(position_size / tick_size) * tick_size
            
            # Appliquer la limite de taille de position
            max_size = (equity * (self.max_position_size / 100.0)) / signal.price
            position_size = min(position_size, max_size)
            
            # Vérifier les limites de l'échange
            min_size = self.exchange.get_symbol_info(signal.symbol).get('min_size', 0)
            if position_size < min_size:
                return 0.0
                
            return position_size
            
        except Exception as e:
            self.logger.error(f"Erreur dans _calculate_position_size: {str(e)}", exc_info=True)
            return 0.0
    
    def _calculate_stop_loss(self, signal: Signal, df: pd.DataFrame) -> Optional[float]:
        """Calcule le niveau de stop-loss optimal."""
        try:
            last_candle = df.iloc[-1]
            atr = last_candle.get('atr', 0)
            
            if signal.signal_type == SignalType.BUY:
                # Stop à 1.5 ATR sous le plus bas récent
                recent_low = df['low'].iloc[-10:].min()
                return recent_low - (atr * 1.5)
            else:
                # Stop à 1.5 ATR au-dessus du plus haut récent
                recent_high = df['high'].iloc[-10:].max()
                return recent_high + (atr * 1.5)
                
        except Exception as e:
            self.logger.error(f"Erreur dans _calculate_stop_loss: {str(e)}", exc_info=True)
            return None
    
    def _calculate_take_profit(self, signal: Signal, df: pd.DataFrame) -> Optional[float]:
        """Calcule le niveau de take-profit optimal."""
        try:
            stop_loss = self._calculate_stop_loss(signal, df)
            if stop_loss is None:
                return None
                
            if signal.signal_type == SignalType.BUY:
                return signal.price + (self.risk_reward_ratio * (signal.price - stop_loss))
            else:
                return signal.price - (self.risk_reward_ratio * (stop_loss - signal.price))
                
        except Exception as e:
            self.logger.error(f"Erreur dans _calculate_take_profit: {str(e)}", exc_info=True)
            return None
    
    async def on_order_update(self, order: Order):
        """Appelé lors de la mise à jour d'un ordre."""
        self.logger.info(f"Ordre mis à jour: {order}")
        
        if order.status == OrderStatus.FILLED:
            self.logger.info(f"Ordre exécuté: {order.side} {order.quantity} {order.symbol} à {order.price}")
            
            # Envoyer une notification
            await self.notification_manager.send_notification(
                level="INFO",
                title=f"Ordre exécuté: {order.side} {order.symbol}",
                message=f"Quantité: {order.quantity} au prix: {order.price}",
                data=order.to_dict()
            )
    
    async def on_position_update(self, position: Position):
        """Appelé lors de la mise à jour d'une position."""
        self.logger.info(f"Position mise à jour: {position}")
        
        # Mettre à jour le trailing stop si nécessaire
        if position.is_open and position.symbol in self.trailing_stops:
            await self._update_trailing_stop(position)
    
    async def _update_trailing_stop(self, position: Position):
        """Met à jour le trailing stop d'une position ouverte."""
        try:
            symbol_info = self.exchange.get_symbol_info(position.symbol)
            if not symbol_info:
                return
                
            ticker = await self.exchange.get_ticker(position.symbol)
            if not ticker:
                return
                
            current_price = ticker['last']
            trailing_info = self.trailing_stops[position.symbol]
            
            # Calculer le nouveau stop
            if position.side == 'long':
                new_stop = current_price - (trailing_info['atr'] * 2)  # 2 ATR sous le prix actuel
                if new_stop > position.stop_loss and new_stop > position.entry_price * 0.99:  # Ne pas remonter le stop trop tôt
                    await self.exchange.update_position(
                        position.id,
                        stop_loss=new_stop
                    )
            else:  # short
                new_stop = current_price + (trailing_info['atr'] * 2)  # 2 ATR au-dessus du prix actuel
                if (new_stop < position.stop_loss or position.stop_loss is None) and new_stop < position.entry_price * 1.01:
                    await self.exchange.update_position(
                        position.id,
                        stop_loss=new_stop
                    )
                    
        except Exception as e:
            self.logger.error(f"Erreur dans _update_trailing_stop: {str(e)}", exc_info=True)
    
    async def on_error(self, error: Exception):
        """Gestion des erreurs de la stratégie."""
        self.logger.error(f"Erreur dans la stratégie: {str(error)}", exc_info=True)
        
        # Envoyer une alerte
        await self.notification_manager.send_alert(
            level="ERROR",
            message=f"Erreur dans la stratégie {self.name}: {str(error)}",
            details={"traceback": str(error.__traceback__)}
        )
    
    def get_metrics(self) -> Dict[str, Any]:
        """Retourne les métriques de performance de la stratégie."""
        return {
            "sharpe_ratio": self.performance_analyzer.sharpe_ratio(),
            "sortino_ratio": self.performance_analyzer.sortino_ratio(),
            "max_drawdown": self.performance_analyzer.max_drawdown(),
            "win_rate": self.performance_analyzer.win_rate(),
            "profit_factor": self.performance_analyzer.profit_factor(),
            "total_return": self.performance_analyzer.total_return(),
            "annualized_return": self.performance_analyzer.annualized_return(),
            "volatility": self.performance_analyzer.volatility(),
            "calmar_ratio": self.performance_analyzer.calmar_ratio(),
            "number_of_trades": self.performance_analyzer.number_of_trades(),
            "average_trade": self.performance_analyzer.average_trade(),
            "average_win": self.performance_analyzer.average_win(),
            "average_loss": self.performance_analyzer.average_loss(),
            "long_short_ratio": self.performance_analyzer.long_short_ratio()
        }
```

## Étapes pour Ajouter une Nouvelle Stratégie

### 1. Créer un Nouveau Fichier

Créez un nouveau fichier dans `src/strategies/` avec un nom descriptif (ex: `my_strategy.py`).

### 2. Implémenter la Stratégie

Suivez la structure ci-dessus pour implémenter votre logique de trading.

### 3. Enregistrer la Stratégie

Ajoutez votre stratégie au dictionnaire `STRATEGIES` dans `src/strategies/__init__.py` :

```python
from .my_strategy import MyCustomStrategy

STRATEGIES = {
    'momentum': MomentumStrategy,
    'mean_reversion': MeanReversionStrategy,
    'my_custom_strategy': MyCustomStrategy,  # Nouvelle stratégie
}
```

### 4. Créer un Fichier de Configuration

Créez un fichier de configuration dans `config/strategies/my_custom_strategy.yaml` :

```yaml
# Paramètres par défaut de la stratégie
param1: 1.0
param2: 14

# Paramètres de risque
risk:
  stop_loss: -0.05
  take_profit: 0.10
  max_position_size: 1000

# Paires et timeframes
pairs:
  - BTC/EUR
  - ETH/EUR
  
timeframes:
  - 1h
  - 4h

# Activer/désactiver
enabled: false
```

## Bonnes Pratiques

1. **Documentation** : Documentez clairement votre stratégie
2. **Tests** : Ajoutez des tests unitaires
3. **Backtesting** : Testez sur des données historiques
4. **Gestion des Erreurs** : Gérez toutes les erreurs potentielles
5. **Logging** : Utilisez le système de logging intégré

## Exemple Complet : Stratégie de Croisement de Moyennes Mobiles

```python
from ..core.strategy import BaseStrategy
from ..core.types import Symbol, Timeframe, Signal, SignalType, Position

class MovingAverageCrossover(BaseStrategy):
    """
    Stratégie de croisement de moyennes mobiles.
    
    Achete quand la moyenne mobile rapide croise au-dessus de la lente.
    Vends quand la moyenne mobile rapide croise en-dessous de la lente.
    """
    
    def __init__(self, fast_ma: int = 9, slow_ma: int = 21):
        super().__init__()
        self.fast_ma = fast_ma
        self.slow_ma = slow_ma
        self.name = f"MA_Crossover_{fast_ma}_{slow_ma}"
    
    async def analyze(self, symbol: Symbol, timeframe: Timeframe) -> Optional[Signal]:
        # Récupérer les données OHLCV
        df = await self.data_provider.get_ohlcv(symbol, timeframe)
        
        if len(df) < self.slow_ma + 1:
            return None
        
        # Calculer les moyennes mobiles
        df['fast_ma'] = df['close'].rolling(window=self.fast_ma).mean()
        df['slow_ma'] = df['close'].rolling(window=self.slow_ma).mean()
        
        # Détecter les croisements
        df['position'] = 0
        df.loc[df['fast_ma'] > df['slow_ma'], 'position'] = 1
        df.loc[df['fast_ma'] <= df['slow_ma'], 'position'] = -1
        
        # Détecter les changements de position
        df['signal'] = df['position'].diff()
        
        # Générer un signal si croisement détecté
        if df['signal'].iloc[-1] > 0:
            return Signal(
                symbol=symbol,
                signal_type=SignalType.BUY,
                strength=1.0,
                timestamp=df.index[-1],
                metadata={
                    'fast_ma': df['fast_ma'].iloc[-1],
                    'slow_ma': df['slow_ma'].iloc[-1]
                }
            )
        elif df['signal'].iloc[-1] < 0:
            return Signal(
                symbol=symbol,
                signal_type=SignalType.SELL,
                strength=1.0,
                timestamp=df.index[-1],
                metadata={
                    'fast_ma': df['fast_ma'].iloc[-1],
                    'slow_ma': df['slow_ma'].iloc[-1]
                }
            )
        
        return None
```

## Tests et Validation

1. **Tests Unitaires** :
   ```python
   # tests/strategies/test_my_strategy.py
   def test_ma_crossover():
       strategy = MovingAverageCrossover(fast_ma=9, slow_ma=21)
       # Ajoutez des tests unitaires ici
   ```

2. **Backtesting** :
   ```bash
   python -m src.core.backtesting.backtester --strategy my_custom_strategy --pairs "BTC/EUR,ETH/EUR" --timerange 20230101-20231231
   ```

## Prochaines Étapes

- [Architecture du Projet](architecture.md)
- [Guide d'API](../api_reference/overview.md)
- [Tests et Intégration Continue](testing.md)
