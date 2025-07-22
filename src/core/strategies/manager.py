import logging
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import pandas as pd
from src.core.types.market_data import MarketData
from src.core.types import TradeSignal
from src.core.config.strategy_config import BaseStrategyConfig as StrategyConfig
from src.core.analysis.predictor import MLPredictor
from src.core.signals.generator import SignalGenerator
from .strategy_switcher import StrategySwitcher, MarketCondition


class StrategyManager:
    """
    Gestionnaire des stratégies de trading.

    Cette classe gère :
    - La sélection des stratégies
    - L'optimisation des paramètres
    - L'exécution des stratégies
    - La gestion des performances
    """

    def __init__(self,
                 config: StrategyConfig,
                 predictor: MLPredictor,
                 signal_generator: SignalGenerator):
        """
        Initialise le gestionnaire des stratégies.

        Args:
            config: Configuration des stratégies
            predictor: Prédicteur ML
            signal_generator: Générateur de signaux
        """
        self.config = config
        self.predictor = predictor
        self.signal_generator = signal_generator
        self.logger = logging.getLogger(__name__)

        # Stratégies disponibles
        self.strategies = {
            'trend_following': self._trend_following_strategy,
            'mean_reversion': self._mean_reversion_strategy,
            'momentum': self._momentum_strategy,
            'volatility_breakout': self._volatility_breakout_strategy,
            'ml': self._ml_strategy
        }

        # Paramètres des stratégies
        self.strategy_params = {
            'trend_following': {
                'fast_ma': 10,
                'slow_ma': 50,
                'risk_multiplier': 1.5
            },
            'mean_reversion': {
                'lookback': 20,
                'std_multiplier': 2.0,
                'entry_threshold': 2.0
            },
            'momentum': {
                'period': 14,
                'rsi_threshold': 70,
                'volatility_filter': 1.5
            },
            'volatility_breakout': {
                'atr_period': 14,
                'atr_multiplier': 2.0,
                'volatility_threshold': 1.5
            },
            'ml': {
                'confidence_threshold': 0.7,
                'prediction_window': 10,
                'risk_adjustment': 1.2
            }
        }

        # Initialiser le système de switching de stratégies
        self.strategy_switcher = StrategySwitcher(
            config=config.get('strategy_switcher', {})
        )
        
        # Performance des stratégies
        self.strategy_performance: Dict[str, Dict] = {}
        
        # Dernières stratégies actives
        self.last_active_strategies: List[str] = []

    def generate_signals(self, market_data: MarketData) -> List[TradeSignal]:
        """
        Génère les signaux de trading pour une paire.

        Args:
            market_data: Données de marché

        Returns:
            Liste des signaux générés
        """
        try:
            # Initialiser la liste des signaux
            signals = []
            
            # Mettre à jour les poids des stratégies en fonction des conditions du marché
            self.strategy_switcher.adjust_strategy_weights(market_data)
            
            # Obtenir les stratégies actives (soit configurées, soit sélectionnées par le système)
            if self.config.get('enable_strategy_switching', False):
                # Utiliser le système de switching pour sélectionner les meilleures stratégies
                best_strategies = self.strategy_switcher.get_best_strategies(
                    market_data,
                    top_n=self.config.get('max_active_strategies', 3)
                )
                active_strategies = [s[0] for s in best_strategies]
                self.last_active_strategies = active_strategies
            else:
                # Utiliser les stratégies configurées manuellement
                active_strategies = self.config.get('active_strategies', [])
            
            # Obtenir les indicateurs techniques
            indicators = market_data.get('analysis', {})
            
            # Appliquer les stratégies actives
            for strategy_name in active_strategies:
                if strategy_name in self.strategies:
                    strategy_signals = self.strategies[strategy_name](
                        market_data,
                        indicators
                    )

                    if strategy_signals:
                        signals.extend(strategy_signals)

            # Générer les signaux ML (toujours actif)
            ml_signals = self._generate_ml_signals(market_data)
            if ml_signals:
                signals.extend(ml_signals)
                
            # Mettre à jour les poids des stratégies si nécessaire
            if signals and self.config.get('enable_strategy_switching', False):
                self._update_strategy_weights_based_on_signals(signals)

            # Filtrer et combiner les signaux
            return self._filter_and_combine_signals(signals)

        except Exception as e:
            self.logger.error(f"Erreur lors de la génération des signaux: {e}")
            return []

    def _trend_following_strategy(self,
                                  market_data: MarketData,
                                  indicators: Dict) -> List[TradeSignal]:
        """Stratégie de suivi de tendance."""
        try:
            # Obtenir les paramètres
            params = self.strategy_params['trend_following']

            # Calculer les MAs
            df = pd.DataFrame(market_data['candles'])
            df['fast_ma'] = df['close'].rolling(
                window=params['fast_ma']).mean()
            df['slow_ma'] = df['close'].rolling(
                window=params['slow_ma']).mean()

            # Dernières valeurs
            last_ma = df.iloc[-1]

            # Générer le signal
            if last_ma['fast_ma'] > last_ma['slow_ma']:
                return [{
                    'symbol': market_data['symbol'],
                    'action': 'buy',
                    'price': last_ma['close'],
                    'confidence': float((last_ma['fast_ma'] - last_ma['slow_ma']) / last_ma['close']),
                    'reason': 'Trend Following',
                    'timestamp': datetime.now()
                }]

            elif last_ma['fast_ma'] < last_ma['slow_ma']:
                return [{
                    'symbol': market_data['symbol'],
                    'action': 'sell',
                    'price': last_ma['close'],
                    'confidence': float((last_ma['slow_ma'] - last_ma['fast_ma']) / last_ma['close']),
                    'reason': 'Trend Following',
                    'timestamp': datetime.now()
                }]

            return []

        except Exception as e:
            self.logger.error(
                f"Erreur dans la stratégie de suivi de tendance: {e}")
            return []

    def _mean_reversion_strategy(self,
                                 market_data: MarketData,
                                 indicators: Dict) -> List[TradeSignal]:
        """Stratégie de réversion à la moyenne."""
        try:
            # Obtenir les paramètres
            params = self.strategy_params['mean_reversion']

            # Calculer la moyenne et l'écart-type
            df = pd.DataFrame(market_data['candles'])
            df['returns'] = df['close'].pct_change()
            df['mean'] = df['returns'].rolling(
                window=params['lookback']).mean()
            df['std'] = df['returns'].rolling(window=params['lookback']).std()

            # Dernières valeurs
            last = df.iloc[-1]

            # Générer le signal
            if last['returns'] > params['entry_threshold'] * last['std']:
                return [
                    {
                        'symbol': market_data['symbol'],
                        'action': 'sell',
                        'price': last['close'],
                        'confidence': float(
                            (last['returns'] -
                             params['entry_threshold'] *
                                last['std']) /
                            last['close']),
                        'reason': 'Mean Reversion',
                        'timestamp': datetime.now()}]

            elif last['returns'] < -params['entry_threshold'] * last['std']:
                return [
                    {
                        'symbol': market_data['symbol'],
                        'action': 'buy',
                        'price': last['close'],
                        'confidence': float(
                            (-last['returns'] - params['entry_threshold'] * last['std']) / last['close']),
                        'reason': 'Mean Reversion',
                        'timestamp': datetime.now()}]

            return []

        except Exception as e:
            self.logger.error(
                f"Erreur dans la stratégie de réversion à la moyenne: {e}")
            return []

    def _momentum_strategy(self,
                           market_data: MarketData,
                           indicators: Dict) -> List[TradeSignal]:
        """Stratégie de momentum."""
        try:
            # Obtenir les paramètres
            params = self.strategy_params['momentum']

            # Calculer le RSI
            df = pd.DataFrame(market_data['candles'])
            df['returns'] = df['close'].pct_change()
            df['rsi'] = indicators.get(
                'rsi', df['returns'].rolling(
                    window=params['period']).mean())

            # Dernières valeurs
            last = df.iloc[-1]

            # Générer le signal
            if last['rsi'] > params['rsi_threshold']:
                return [{
                    'symbol': market_data['symbol'],
                    'action': 'sell',
                    'price': last['close'],
                    'confidence': float((last['rsi'] - params['rsi_threshold']) / 100),
                    'reason': 'Momentum',
                    'timestamp': datetime.now()
                }]

            elif last['rsi'] < 100 - params['rsi_threshold']:
                return [{
                    'symbol': market_data['symbol'],
                    'action': 'buy',
                    'price': last['close'],
                    'confidence': float((params['rsi_threshold'] - last['rsi']) / 100),
                    'reason': 'Momentum',
                    'timestamp': datetime.now()
                }]

            return []

        except Exception as e:
            self.logger.error(f"Erreur dans la stratégie de momentum: {e}")
            return []

    def _volatility_breakout_strategy(self,
                                      market_data: MarketData,
                                      indicators: Dict) -> List[TradeSignal]:
        """Stratégie de breakout de volatilité."""
        try:
            # Obtenir les paramètres
            params = self.strategy_params['volatility_breakout']

            # Calculer l'ATR
            df = pd.DataFrame(market_data['candles'])
            df['atr'] = indicators.get(
                'atr', df['close'].rolling(
                    window=params['atr_period']).std())

            # Dernières valeurs
            last = df.iloc[-1]

            # Générer le signal
            if last['close'] > last['close'].shift(
                    1) + params['atr_multiplier'] * last['atr']:
                return [{
                    'symbol': market_data['symbol'],
                    'action': 'buy',
                    'price': last['close'],
                    'confidence': float(last['atr'] / last['close']),
                    'reason': 'Volatility Breakout',
                    'timestamp': datetime.now()
                }]

            elif last['close'] < last['close'].shift(1) - params['atr_multiplier'] * last['atr']:
                return [{
                    'symbol': market_data['symbol'],
                    'action': 'sell',
                    'price': last['close'],
                    'confidence': float(last['atr'] / last['close']),
                    'reason': 'Volatility Breakout',
                    'timestamp': datetime.now()
                }]

            return []

        except Exception as e:
            self.logger.error(
                f"Erreur dans la stratégie de breakout de volatilité: {e}")
            return []

    def _ml_strategy(self, market_data: MarketData) -> List[TradeSignal]:
        """Stratégie basée sur le ML."""
        try:
            # Obtenir la prédiction
            prediction = self.predictor.predict(market_data)

            # Obtenir les paramètres
            params = self.strategy_params['ml']

            # Générer le signal
            if prediction['confidence'] > params['confidence_threshold']:
                return [{
                    'symbol': market_data['symbol'],
                    'action': prediction['action'],
                    'price': prediction['current_price'],
                    'confidence': prediction['confidence'],
                    'reason': 'ML Prediction',
                    'timestamp': datetime.now(),
                    'prediction': prediction
                }]

            return []

        except Exception as e:
            self.logger.error(f"Erreur dans la stratégie ML: {e}")
            return []

    def _generate_ml_signals(
            self,
            market_data: MarketData) -> List[TradeSignal]:
        """Génère les signaux basés sur le ML."""
        try:
            # Obtenir les signaux ML
            ml_signals = self.signal_generator._generate_ml_signal(market_data)

            if ml_signals:
                return [ml_signals]

            return []

        except Exception as e:
            self.logger.error(
                f"Erreur lors de la génération des signaux ML: {e}")
            return []

    def _update_strategy_weights_based_on_signals(self, signals: List[TradeSignal]):
        """
        Met à jour les poids des stratégies en fonction des signaux générés.
        
        Args:
            signals: Liste des signaux générés
        """
        if not signals:
            return
            
        # Calculer le PnL total pour chaque stratégie
        strategy_pnl = {}
        
        for signal in signals:
            strategy_name = signal.get('strategy', 'unknown')
            pnl = signal.get('pnl', 0)
            
            if strategy_name not in strategy_pnl:
                strategy_pnl[strategy_name] = {
                    'total_pnl': 0,
                    'count': 0
                }
                
            strategy_pnl[strategy_name]['total_pnl'] += pnl
            strategy_pnl[strategy_name]['count'] += 1
        
        # Mettre à jour les performances des stratégies
        for strategy_name, data in strategy_pnl.items():
            if data['count'] > 0:
                avg_pnl = data['total_pnl'] / data['count']
                self.strategy_switcher.update_strategy_performance(
                    strategy_name,
                    {
                        'pnl': avg_pnl,
                        'timestamp': datetime.now()
                    }
                )
                
        # Ajuster les poids des stratégies
        self.strategy_switcher.adjust_strategy_weights(None)

    def _filter_and_combine_signals(
            self, signals: List[TradeSignal]) -> List[TradeSignal]:
        """
        Filtre et combine les signaux.

        Args:
            signals: Liste des signaux à filtrer

        Returns:
            Liste des signaux filtrés et combinés
        """
        if not signals:
            return []
            
        try:
            # Grouper les signaux par symbole et action
            signals_by_symbol = {}
            for signal in signals:
                key = (signal['symbol'], signal['action'])
                if key not in signals_by_symbol:
                    signals_by_symbol[key] = []
                signals_by_symbol[key].append(signal)
            
            # Combiner les signaux pour chaque paire/action
            combined_signals = []
            for (symbol, action), signal_group in signals_by_symbol.items():
                # Trier par confiance décroissante
                signal_group.sort(key=lambda x: x.get('confidence', 0), reverse=True)
                
                # Prendre le signal le plus fort
                combined_signal = signal_group[0].copy()
                
                # Si plusieurs signaux, combiner les raisons
                if len(signal_group) > 1:
                    reasons = set(s.get('reason', '') for s in signal_group)
                    combined_signal['reason'] = ', '.join(reasons)
                
                combined_signals.append(combined_signal)
            
            return combined_signals
            
        except Exception as e:
            self.logger.error(f"Erreur lors du filtrage des signaux: {e}")
            return []

    def optimize_parameters(self,
                            market_data: MarketData,
                            strategy_name: str) -> Dict:
        """
        Optimise les paramètres d'une stratégie.

        Args:
            market_data: Données de marché
            strategy_name: Nom de la stratégie

        Returns:
            Paramètres optimisés
        """
        try:
            # À implémenter avec une méthode d'optimisation
            return self.strategy_params[strategy_name]

        except Exception as e:
            self.logger.error(
                f"Erreur lors de l'optimisation des paramètres: {e}")
            return self.strategy_params.get(strategy_name, {})

    def evaluate_performance(self,
                             market_data: MarketData,
                             signals: List[TradeSignal]) -> Dict:
        """
        Évalue la performance des signaux.

        Args:
            market_data: Données de marché
            signals: Liste des signaux

        Returns:
            Métriques de performance
        """
        try:
            # À implémenter avec le calcul des métriques
            return {}

        except Exception as e:
            self.logger.error(
                f"Erreur lors de l'évaluation de la performance: {e}")
            return {}
