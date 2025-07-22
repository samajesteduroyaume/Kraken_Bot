import logging
from typing import Dict, List, Tuple
from decimal import Decimal
from datetime import datetime
import pandas as pd
from kraken_bot.core.types import MarketData, TradeSignal, Position, TradingConfig, TradingMetrics
from kraken_bot.core.types.types import Trade
from ..analysis import MLPredictor
from kraken_bot.core.signals.generator import SignalGenerator
from kraken_bot.core.strategies.manager import StrategyManager
from src.utils import helpers
import numpy as np


class Backtester:
    """
    Système de backtesting pour le trading.

    Cette classe permet de tester les stratégies sur des données historiques.
    Elle gère :
    - Le backtesting des stratégies
    - L'analyse des performances
    - La comparaison des stratégies
    - La validation des paramètres
    """

    def __init__(self,
                 config: TradingConfig,
                 predictor: MLPredictor,
                 signal_generator: SignalGenerator,
                 strategy_manager: StrategyManager):
        """
        Initialise le backtester.

        Args:
            config: Configuration de trading
            predictor: Prédicteur ML
            signal_generator: Générateur de signaux
            strategy_manager: Gestionnaire de stratégies
        """
        self.config = config
        self.predictor = predictor
        self.signal_generator = signal_generator
        self.strategy_manager = strategy_manager
        self.logger = logging.getLogger(__name__)

        # Historique des trades
        self.trade_history: List[Trade] = []

        # Positions actuelles
        self.positions: Dict[str, Position] = {}

        # Métriques de performance
        self.metrics: TradingMetrics = {
            'total_trades': 0,
            'win_rate': 0.0,
            'avg_profit': Decimal('0'),
            'avg_loss': Decimal('0'),
            'max_drawdown': Decimal('0'),
            'sharpe_ratio': 0.0,
            'sortino_ratio': 0.0,
            'calmar_ratio': 0.0,
            'profit_factor': 0.0,
            'roi': Decimal('0')
        }

    async def run_backtest(self,
                           market_data: MarketData,
                           start_date: datetime,
                           end_date: datetime) -> Dict:
        """
        Exécute un backtest sur les données historiques.

        Args:
            market_data: Données de marché
            start_date: Date de début
            end_date: Date de fin

        Returns:
            Résultats du backtest
        """
        try:
            # Préparer les données
            candles = self._filter_candles_by_date(
                market_data['candles'], start_date, end_date)

            if not candles:
                return self.metrics

            # Initialiser le compte
            balance = Decimal(str(self.config['initial_balance']))

            # Parcourir les données
            for i in range(len(candles)):
                current_candle = candles[i]

                # Générer les signaux
                signals = self.strategy_manager.generate_signals({
                    'symbol': market_data['symbol'],
                    'candles': candles[max(0, i - 100):i + 1],
                    'analysis': self._calculate_indicators(candles[max(0, i - 100):i + 1])
                })

                # Exécuter les trades
                trades = self._execute_trades(signals, current_candle, balance)

                # Mettre à jour les positions
                self._update_positions(trades)

                # Mettre à jour le solde
                balance = self._update_balance(trades, balance)

                # Mettre à jour les métriques
                self._update_metrics(trades)

            return self.metrics

        except Exception as e:
            self.logger.error(f"Erreur lors du backtest: {e}")
            return self.metrics

    def _filter_candles_by_date(self,
                                candles: List[Dict],
                                start_date: datetime,
                                end_date: datetime) -> List[Dict]:
        """Filtre les bougies par date."""
        return [candle for candle in candles if start_date <=
                datetime.fromisoformat(candle['timestamp']) <= end_date]

    def _calculate_indicators(self, candles: List[Dict]) -> Dict:
        """Calcule les indicateurs techniques."""
        df = pd.DataFrame(candles)

        # Calculer les indicateurs
        indicators = {
            'rsi': self._calculate_rsi(df),
            'macd': self._calculate_macd(df),
            'bb': self._calculate_bollinger_bands(df),
            'atr': self._calculate_atr(df),
            'volatility': self._calculate_volatility(df)
        }

        return indicators

    def _calculate_rsi(self, df: pd.DataFrame) -> float:
        """Calcule le RSI."""
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        return float(100 - (100 / (1 + rs)).iloc[-1])

    def _calculate_macd(self, df: pd.DataFrame) -> Tuple[float, float, float]:
        """Calcule le MACD."""
        ema12 = df['close'].ewm(span=12).mean()
        ema26 = df['close'].ewm(span=26).mean()
        macd = ema12 - ema26
        signal = macd.ewm(span=9).mean()
        hist = macd - signal

        return float(macd.iloc[-1]), float(signal.iloc[-1]
                                           ), float(hist.iloc[-1])

    def _calculate_bollinger_bands(
            self, df: pd.DataFrame) -> Tuple[float, float, float]:
        """Calcule les bandes de Bollinger."""
        sma = df['close'].rolling(window=20).mean()
        std = df['close'].rolling(window=20).std()
        upper = sma + (std * 2)
        lower = sma - (std * 2)

        return float(upper.iloc[-1]), float(sma.iloc[-1]
                                            ), float(lower.iloc[-1])

    def _calculate_atr(self, df: pd.DataFrame) -> float:
        """Calcule l'ATR."""
        high = df['high']
        low = df['low']
        close = df['close'].shift()

        tr1 = high - low
        tr2 = abs(high - close)
        tr3 = abs(low - close)

        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=14).mean()

        return float(atr.iloc[-1])

    def _calculate_volatility(self, df: pd.DataFrame) -> float:
        """Calcule la volatilité."""
        returns = df['close'].pct_change()
        return float(returns.std() * 100)

    def _execute_trades(self,
                        signals: List[TradeSignal],
                        current_candle: Dict,
                        balance: Decimal) -> List[Trade]:
        """Exécute les trades basés sur les signaux."""
        trades = []

        for signal in signals:
            try:
                # Calculer la taille de position
                position_size = self._calculate_position_size(
                    signal,
                    balance,
                    current_candle['close']
                )

                # Créer le trade
                trade = {
                    'id': len(self.trade_history),
                    'symbol': signal['symbol'],
                    'price': Decimal(str(current_candle['close'])),
                    'amount': position_size,
                    'side': signal['action'],
                    'timestamp': datetime.fromisoformat(current_candle['timestamp']),
                    'confidence': signal['confidence'],
                    'reason': signal['reason']
                }

                trades.append(trade)
                self.trade_history.append(trade)

            except Exception as e:
                self.logger.error(f"Erreur lors de l'exécution du trade: {e}")

        return trades

    def _calculate_position_size(self,
                                 signal: TradeSignal,
                                 balance: Decimal,
                                 price: Decimal) -> Decimal:
        """Calcule la taille de position."""
        risk_percentage = self.config['risk_per_trade']
        risk_amount = balance * Decimal(str(risk_percentage)) / Decimal('100')

        # Calculer le stop-loss
        stop_loss = self._calculate_stop_loss(signal, price)
        stop_distance = abs(price - stop_loss)

        # Calculer la taille de position
        position_size = risk_amount / stop_distance

        # Appliquer les limites
        position_size = min(
            position_size,
            Decimal(str(self.config['max_position_size']))
        )
        position_size = max(
            position_size,
            Decimal(str(self.config['min_position_size']))
        )

        return position_size

    def _calculate_stop_loss(
            self,
            signal: TradeSignal,
            price: Decimal) -> Decimal:
        """Calcule le niveau de stop-loss."""
        atr = signal.get('indicators', {}).get('atr', price * Decimal('0.02'))

        if signal['action'] == 'buy':
            return price - atr * Decimal('1.5')

        return price + atr * Decimal('1.5')

    def _update_positions(self, trades: List[Trade]) -> None:
        """Met à jour les positions."""
        for trade in trades:
            self.positions[trade['id']] = {
                'symbol': trade['symbol'],
                'entry_price': trade['price'],
                'amount': trade['amount'],
                'side': trade['side'],
                'timestamp': trade['timestamp']
            }

    def _update_balance(
            self,
            trades: List[Trade],
            balance: Decimal) -> Decimal:
        """Met à jour le solde."""
        for trade in trades:
            if trade['side'] == 'buy':
                balance -= trade['price'] * trade['amount']
            else:
                balance += trade['price'] * trade['amount']

        return balance

    def _update_metrics(self, trades: List[Trade]) -> None:
        """Met à jour les métriques de performance."""
        if not trades:
            return
        pnls = [float(t['pnl']) for t in trades]
        self.metrics['total_trades'] += len(trades)
        self.metrics['avg_profit'] = float(
            np.mean([p for p in pnls if p > 0])) if any(p > 0 for p in pnls) else 0.0
        self.metrics['avg_loss'] = float(
            np.mean([p for p in pnls if p < 0])) if any(p < 0 for p in pnls) else 0.0
        self.metrics['win_rate'] = helpers.calculate_win_rate(
            [{'pnl': p} for p in pnls]) * 100 if pnls else 0.0
        self.metrics['profit_factor'] = helpers.calculate_profit_factor(
            [{'pnl': p} for p in pnls])
        self.metrics['sharpe_ratio'] = helpers.calculate_sharpe_ratio(pnls)
        self.metrics['sortino_ratio'] = helpers.calculate_sortino_ratio(pnls)
        self.metrics['max_drawdown'] = helpers.calculate_drawdown(pnls)
        self.metrics['calmar_ratio'] = self.metrics['sharpe_ratio'] / \
            self.metrics['max_drawdown'] if self.metrics['max_drawdown'] else 0.0
        self.metrics['roi'] = sum(pnls) / float(
            self.config['initial_balance']) if hasattr(
            self.config, 'initial_balance') and self.config.initial_balance else 0.0

    def generate_report(self) -> Dict:
        """Génère un rapport de performance."""
        report = {
            'metrics': self.metrics,
            'trades': len(self.trade_history),
            'winning_trades': len([t for t in self.trade_history if t['side'] == 'buy']),
            'losing_trades': len([t for t in self.trade_history if t['side'] == 'sell']),
            'average_profit': float(self.metrics['avg_profit']),
            'average_loss': float(self.metrics['avg_loss']),
            'win_rate': float(self.metrics['win_rate']),
            'profit_factor': float(self.metrics['profit_factor']),
            'sharpe_ratio': float(self.metrics['sharpe_ratio']),
            'sortino_ratio': float(self.metrics['sortino_ratio']),
            'calmar_ratio': float(self.metrics['calmar_ratio']),
            'max_drawdown': float(self.metrics['max_drawdown']),
            'roi': float(self.metrics['roi'])
        }

        return report

    async def optimize_parameters(self,
                                  market_data: MarketData,
                                  strategy_name: str,
                                  start_date: datetime,
                                  end_date: datetime) -> Dict:
        """
        Optimise les paramètres d'une stratégie.

        Args:
            market_data: Données de marché
            strategy_name: Nom de la stratégie
            start_date: Date de début
            end_date: Date de fin

        Returns:
            Paramètres optimisés
        """
        try:
            # Obtenir les paramètres initiaux
            params = self.strategy_manager.strategy_params[strategy_name]

            # Définir les plages de paramètres
            param_ranges = {
                'fast_ma': range(5, 20, 5),
                'slow_ma': range(20, 100, 10),
                'rsi_threshold': range(50, 90, 10),
                'atr_period': range(10, 30, 5),
                'confidence_threshold': np.arange(0.5, 1.0, 0.1)
            }

            # Tester les combinaisons de paramètres
            best_params = params
            best_score = float('-inf')

            for fast_ma in param_ranges['fast_ma']:
                for slow_ma in param_ranges['slow_ma']:
                    # Mettre à jour les paramètres
                    params['fast_ma'] = fast_ma
                    params['slow_ma'] = slow_ma

                    # Exécuter le backtest
                    self.strategy_manager.strategy_params[strategy_name] = params
                    metrics = await self.run_backtest(market_data, start_date, end_date)

                    # Calculer le score
                    score = metrics['sharpe_ratio'] + metrics['profit_factor']

                    # Mettre à jour les meilleurs paramètres
                    if score > best_score:
                        best_score = score
                        best_params = params.copy()

            return best_params

        except Exception as e:
            self.logger.error(
                f"Erreur lors de l'optimisation des paramètres: {e}")
            return self.strategy_manager.strategy_params.get(strategy_name, {})
