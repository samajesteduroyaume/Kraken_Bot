from typing import Dict, Callable, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass
import pandas as pd
from .position import Position
from .backtesting import Backtester as BacktestManager
from .adaptive_leverage import AdaptiveLeverage
from src.core.api.kraken import KrakenAPI
from .ml_predictor import MLPredictor
from .technical_analyzer import TechnicalAnalyzer
from .sentiment_analyzer import SentimentAnalyzer
from .signal_generator import SignalGenerator
from .database_manager import DatabaseManager
from .metrics_manager import MetricsManager
from src.core.logging.logging import LoggerManager
from .config import Config
from src.utils import helpers

# Configuration globale
config = Config
logger = LoggerManager('kraken_simulation').get_logger('kraken_simulation')


@dataclass
class TradingParameters:
    """Paramètres de trading"""
    api: Optional[KrakenAPI] = None
    risk_per_trade: float = 0.02
    stop_loss_percent: float = 0.02
    take_profit_percent: float = 0.04
    max_positions: int = 5
    max_drawdown: float = 0.1
    max_holding_time: timedelta = timedelta(days=1)
    adaptive_leverage: Optional[AdaptiveLeverage] = None
    backtest_manager: Optional[BacktestManager] = None
    ml_predictor: Optional[MLPredictor] = None
    technical_analyzer: Optional[TechnicalAnalyzer] = None
    sentiment_analyzer: Optional[SentimentAnalyzer] = None
    signal_generator: Optional[SignalGenerator] = None
    database_manager: Optional[DatabaseManager] = None
    metrics_manager: Optional[MetricsManager] = None
    logger: Optional[LoggerManager] = None
    config: Optional[Config] = None
    utils: Optional[helpers] = None
    current_balance: float = 0.0
    initial_balance: float = 0.0

    @classmethod
    async def initialize(cls, **kwargs):
        """Initialise les paramètres de trading de manière asynchrone."""
        params = cls(**kwargs)
        await params._initialize_async()
        return params

    async def _initialize_async(self):
        """Initialisation asynchrone des paramètres."""
        if self.api:
            # Initialiser le logger
            if not self.logger:
                self.logger = LoggerManager()

            # Initialiser la session async avec l'API
            async with self.api:
                # Initialiser le solde à partir de l'API Kraken
                balance = await self.api.get_balance()
                self.current_balance = float(
                    balance.get('ZUSD', 0)) if balance else 0.0
                self.initial_balance = self.current_balance
                self.logger.get_logger('trading_params').info(
                    f"Solde initial depuis Kraken: {self.current_balance:.2f} USD")

                # Initialisation du levier adaptatif
                if not self.adaptive_leverage:
                    self.adaptive_leverage = AdaptiveLeverage(
                        min_leverage=1.0,
                        max_leverage=5.0,
                        volatility_threshold=0.02,
                        risk_factor=self.risk_per_trade
                    )

                # Initialisation du prédicteur ML
                if not self.ml_predictor:
                    self.ml_predictor = MLPredictor()

                # Initialisation de l'analyse technique
                if not self.technical_analyzer:
                    self.technical_analyzer = TechnicalAnalyzer()
        else:
            raise ValueError(
                "L'API Kraken est requise pour utiliser le solde réel")

    def validate(self) -> None:
        """Valide les paramètres de trading."""
        if self.initial_balance <= 0:
            raise ValueError("Le solde initial doit être positif")

        if not 0 < self.risk_per_trade <= 1:
            raise ValueError("Le risque par trade doit être entre 0 et 1")

        if self.leverage < 1:
            raise ValueError("Le levier doit être supérieur ou égal à 1")

        if self.stop_loss_percent <= 0:
            raise ValueError("Le stop-loss doit être positif")

        if self.take_profit_percent <= 0:
            raise ValueError("Le take-profit doit être positif")

        if self.max_positions <= 0:
            raise ValueError(
                "Le nombre maximum de positions doit être positif")

        if self.max_drawdown <= 0:
            raise ValueError("Le drawdown maximum doit être positif")

        if self.max_holding_time.total_seconds() <= 0:
            raise ValueError("Le temps maximum de détention doit être positif")


class TradingSimulation:
    """Simulateur de trading pour tester les stratégies."""

    def __init__(self, parameters: TradingParameters):
        """
        Initialise le simulateur de trading.

        Args:
            parameters: Paramètres de trading
        """
        self.parameters = parameters
        self.parameters.validate()
        self.api = parameters.api

        # Récupérer le solde réel du compte
        self.current_balance = self.parameters.initial_balance

        self.backtesting_engine = BacktestManager(
            config={
                'initial_balance': self.current_balance,
                'risk_per_trade': parameters.risk_per_trade,
                'leverage': parameters.leverage,
                'stop_loss_percent': parameters.stop_loss_percent,
                'take_profit_percent': parameters.take_profit_percent,
                'max_positions': parameters.max_positions,
                'max_drawdown': parameters.max_drawdown,
                'max_holding_time': parameters.max_holding_time
            },
            predictor=None,  # À implémenter
            signal_generator=None,  # À implémenter
            strategy_manager=None  # À implémenter
        )

        self.positions = []
        self.closed_positions = []
        self.metrics = {
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'total_profit': 0.0,
            'max_drawdown': 0.0,
            'peak_balance': parameters.initial_balance,
            'current_drawdown': 0.0,
            'risk_exposure': 0.0,
            'current_positions': 0,
            'max_consecutive_wins': 0,
            'max_consecutive_losses': 0,
            'consecutive_wins': 0,
            'consecutive_losses': 0,
            'total_fees': 0.0
        }

    def calculate_position_size(
            self,
            entry_price: float,
            price_history: pd.Series) -> float:
        """
        Calcule la taille de position en fonction du levier adaptatif et des conditions du marché.

        Args:
            entry_price: Prix d'entrée
            price_history: Historique des prix pour calculer la volatilité

        Returns:
            Taille de position ajustée
        """
        try:
            # Calcul des indicateurs du marché
            volatility = self.parameters.adaptive_leverage.calculate_volatility(
                price_history)
            trend = self.parameters.adaptive_leverage.calculate_trend(
                price_history)
            market_sentiment = self.parameters.adaptive_leverage.detect_market_sentiment(
                price_history)

            self.logger.info(
                f"Conditions du marché: volatilité={volatility:.2%}, trend={trend:.2f}, sentiment={market_sentiment}")

            # Calcul de la taille de position avec le levier adaptatif
            position_size = self.parameters.adaptive_leverage.calculate_position_size(
                balance=self.current_balance,
                volatility=volatility,
                entry_price=entry_price,
                stop_loss_percent=self.parameters.stop_loss_percent,
                trend=trend,
                market_sentiment=market_sentiment
            )

            self.logger.info(
                f"Position size calculée: {position_size:.2f} avec levier adaptatif")

            return position_size

        except Exception as e:
            self.logger.error(
                f"Erreur lors du calcul de la position size: {str(e)}")
            # En cas d'erreur, utilise une taille de position par défaut
            return self.current_balance * self.parameters.risk_per_trade

    def open_position(
            self,
            pair: str,
            entry_price: float,
            is_long: bool = True,
            price_history: pd.Series = None) -> Position:
        """
        Ouvre une nouvelle position.

        Args:
            pair: Paire de trading
            entry_price: Prix d'entrée
            is_long: True pour une position longue, False pour une position courte

        Returns:
            Position ouverte
        """
        if len(self.positions) >= self.parameters.max_positions:
            raise ValueError("Nombre maximum de positions atteint")

        position_size = self.calculate_position_size(entry_price)

        if self.metrics['risk_exposure'] + \
                position_size > self.current_balance:
            raise ValueError("Exposition au risque trop élevée")

        position = Position(
            pair=pair,
            size=position_size if is_long else -position_size,
            entry_price=entry_price,
            entry_time=datetime.now(),
            leverage=self.parameters.leverage,
            stop_loss=entry_price * (1 - self.parameters.stop_loss_percent) if is_long else entry_price *
            (1 + self.parameters.stop_loss_percent),
            take_profit=entry_price *
            (1 + self.parameters.take_profit_percent) if is_long else entry_price * (1 - self.parameters.take_profit_percent),
            fee_rate=0.0026
        )

        self.positions.append(position)
        self.metrics['current_positions'] += 1
        self.metrics['risk_exposure'] += position_size

        logger.info(f"Position ouverte: {pair} - Taille: {position_size}")
        return position

    def close_position(self, position: Position) -> float:
        """
        Ferme une position et met à jour les métriques.

        Args:
            position: Position à fermer

        Returns:
            P&L de la position
        """
        pnl = position.close(position.entry_price *
                             (1 + self.parameters.take_profit_percent))
        self.metrics['total_profit'] += pnl
        self.metrics['current_positions'] -= 1
        self.metrics['risk_exposure'] -= abs(position.size)

        if pnl > 0:
            self.metrics['winning_trades'] += 1
            self.metrics['consecutive_wins'] += 1
            self.metrics['consecutive_losses'] = 0
        else:
            self.metrics['losing_trades'] += 1
            self.metrics['consecutive_losses'] += 1
            self.metrics['consecutive_wins'] = 0

        self.metrics['total_trades'] += 1
        self.metrics['total_fees'] += position.calculate_fees(
            position.entry_price)

        # Mise à jour du drawdown
        self.current_balance += pnl
        if self.current_balance > self.metrics['peak_balance']:
            self.metrics['peak_balance'] = self.current_balance

        drawdown = (self.metrics['peak_balance'] -
                    self.current_balance) / self.metrics['peak_balance']
        self.metrics['max_drawdown'] = max(
            self.metrics['max_drawdown'], drawdown)
        self.metrics['current_drawdown'] = drawdown

        return pnl

    def check_positions(self, current_price: float) -> None:
        """
        Vérifie toutes les positions ouvertes.

        Args:
            current_price: Prix actuel
        """
        positions_to_close = []

        for position in self.positions:
            if position.check_stop_loss(
                    current_price) or position.check_take_profit(current_price):
                positions_to_close.append(position)

            # Vérification du temps de détention
            holding_time = datetime.now() - position.entry_time
            if holding_time > self.parameters.max_holding_time:
                positions_to_close.append(position)

        for position in positions_to_close:
            self.close_position(position)
            self.positions.remove(position)

    def get_metrics(self) -> Dict:
        """Retourne les métriques de performance."""
        return {
            'total_trades': self.metrics['total_trades'],
            'winning_trades': self.metrics['winning_trades'],
            'losing_trades': self.metrics['losing_trades'],
            'total_profit': self.metrics['total_profit'],
            'current_balance': self.current_balance,
            'max_drawdown': self.metrics['max_drawdown'],
            'current_drawdown': self.metrics['current_drawdown'],
            'risk_exposure': self.metrics['risk_exposure'],
            'current_positions': self.metrics['current_positions'],
            'max_consecutive_wins': self.metrics['max_consecutive_wins'],
            'max_consecutive_losses': self.metrics['max_consecutive_losses'],
            'total_fees': self.metrics['total_fees']
        }

    def run_simulation(self, data: pd.DataFrame, strategy: Callable) -> Dict:
        """
        Exécute une simulation de trading.

        Args:
            data: Données historiques
            strategy: Fonction de stratégie

        Returns:
            Résultats de la simulation
        """
        for i in range(len(data)):
            data.index[i]
            current_price = data['close'].iloc[i]

            # Vérification des positions
            self.check_positions(current_price)

            # Exécution de la stratégie
            signals = strategy(data.iloc[:i + 1])

            if isinstance(signals, dict):
                for pair, signal in signals.items():
                    if signal['action'] == 'buy' and self.current_balance > 0:
                        try:
                            self.open_position(
                                pair, current_price, is_long=True)
                        except ValueError as e:
                            logger.warning(
                                f"Impossible d'ouvrir une position: {str(e)}")

                    elif signal['action'] == 'sell':
                        for position in self.positions:
                            if position.pair == pair:
                                self.close_position(position)
                                self.positions.remove(position)

        return self.get_metrics()
