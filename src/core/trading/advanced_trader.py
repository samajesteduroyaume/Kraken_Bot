"""
Trader avancé avec séparation des responsabilités.

Ce module implémente un trader avancé qui utilise les modules spécialisés
pour la génération de signaux, l'exécution des ordres, l'analyse de marché
et la gestion des risques.
"""
import asyncio
import logging
from typing import Dict, Optional, Any
from datetime import datetime, timedelta
from decimal import Decimal
from src.core.types.types import TradingConfig, TradeSignal, TradingMetrics
from src.core.types.market_data import MarketData

from src.core.trading.base_trader import BaseTrader
from src.core.trading.signals import SignalGenerator
from src.core.trading.execution import OrderExecutor
from src.core.trading.analysis import MarketAnalyzer
from src.core.trading.risk import RiskManager
from src.utils import helpers
import numpy as np

logger = logging.getLogger(__name__)


class AdvancedAITrader(BaseTrader):
    """
    Trader avancé qui intègre l'analyse technique, les signaux ML,
    la gestion des risques et l'exécution des ordres.
    """

    def __init__(self,
                 api,  # Type hinté dans base_trader
                 predictor=None,  # Type hinté dans base_trader
                 config: Optional[Dict[str, Any]] = None,
                 db_manager=None):
        """Initialise le trader avancé."""
        if config is None:
            raise ValueError("La configuration de trading (config) ne peut pas être None pour AdvancedAITrader.")
        super().__init__(api, predictor, config)

        # Initialisation des composants
        self._stop_event = asyncio.Event()
        self.signal_generator = SignalGenerator(predictor)
        self.order_executor = OrderExecutor(api, config)
        self.market_analyzer = MarketAnalyzer()
        # Initialiser le RiskManager avec les fonds réels (ici 0.0 par défaut)
        self.risk_manager = RiskManager(
            initial_balance=0.0,  # À remplacer par la vraie balance dès que possible
            risk_profile={
                'max_drawdown': float(config.get('max_slippage', 0.1)),
                'risk_per_trade': float(config.get('max_slippage', 0.01)),
                'max_leverage': float(config.get('max_leverage', 5.0)),
                'leverage_strategy': config.get('position_adjustment', 'moderate')
            }
        )
        # Initialiser le trader de base avec la configuration
        # super().__init__(api, predictor, config)  # déjà appelé plus haut

        # État du trader
        self.active_strategies = {}
        self.market_data: Dict[str, MarketData] = {}
        self.last_analysis: Dict[str, Any] = {}
        self.db_manager = db_manager

    async def get_account_balance(self, currency: str = None) -> float:
        """Récupère le solde du compte pour une devise spécifique."""
        try:
            if hasattr(self.api, 'get_balance') and callable(getattr(self.api, 'get_balance')):
                balance = await self.api.get_balance()
            else:
                raise NotImplementedError("La méthode get_balance n'est pas disponible sur l'API Kraken.")
            if currency:
                return float(balance.get(currency, 0))
            return float(balance.get('total', 0))
        except Exception as e:
            logger.error(f"Erreur lors de la récupération du solde: {e}")
            return 0.0

    def is_config_valid(self) -> bool:
        """
        Vérifie si la configuration est valide.

        Returns:
            bool: True si la configuration est valide, False sinon
        """
        if not self.config:
            return False

        required_keys = [
            'trading_pair',
            'trading_amount',
            'trading_fee',
            'risk_level',
            'quote_currency',
            'min_volume_btc',
            'max_spread_pct',
            'min_trades_24h',
            'max_daily_drawdown',
            'risk_percentage',
            'max_leverage']

        return all(key in self.config for key in required_keys)

    async def run(self) -> None:
        """Boucle principale du trader avancé."""
        logger.info("Démarrage du trader avancé...")
        await self._initialize()

        while not self._stop_event.is_set():
            try:
                # 1. Mise à jour des données
                await self._update_market_data()

                # 2. Analyse du marché
                for pair, data in self.market_data.items():
                    if hasattr(self.market_analyzer, 'analyze_market') and callable(getattr(self.market_analyzer, 'analyze_market')):
                        analysis = self.market_analyzer.analyze_market(data)
                        self.last_analysis[pair] = analysis

                # 3. Génération des signaux
                signals = await self.generate_signals()

                # 4. Gestion des risques
                if hasattr(self.risk_manager, 'process_signals') and callable(getattr(self.risk_manager, 'process_signals')):
                    risk_signals = self.risk_manager.process_signals(signals)
                else:
                    risk_signals = signals

                # 5. Exécution des trades
                await self.execute_trades(risk_signals)

                # 6. Mise à jour des positions
                await self.manage_positions()

                # 7. Journalisation du statut
                status = self.get_trading_status()
                self.logger.info(f"Statut du trading: {status}")

                # 8. Sauvegarde dans la base de données
                if self.db_manager:
                    await self.db_manager.save_trading_status(status)

                await asyncio.sleep(5)  # Mise à jour toutes les 5 secondes

            except asyncio.CancelledError:
                logger.info("Arrêt demandé...")
                break
            except Exception as e:
                self.logger.error(f"Erreur dans la boucle principale: {e}")
                await asyncio.sleep(10)  # Attendre avant de réessayer

    def _calculate_stop_loss(self,
                             signal: TradeSignal) -> Decimal:
        """Calcule le niveau de stop-loss pour un signal de trading."""
        # Stratégie de stop-loss basée sur l'ATR
        price = signal.get('price', Decimal('0'))
        atr = signal.get('atr', price * Decimal('0.02'))  # 2% par défaut
        # Ajuster le stop en fonction de la volatilité
        volatility = signal.get('volatility', Decimal('1.0'))
        volatility_factor = min(
            max(volatility * Decimal('1.5'), Decimal('1.0')), Decimal('3.0'))
        atr *= volatility_factor
        action = signal.get('action', None)
        if action == 'buy':
            return price - atr * Decimal('1.5')
        elif action == 'sell':
            return price + atr * Decimal('1.5')
        return price * Decimal('0.99')  # 1% de stop-loss par défaut

    async def _calculate_performance_metrics(self) -> TradingMetrics:
        """Calcule les métriques de performance du système."""
        trades = await self.get_trades()
        if not trades:
            return self.metrics
        pnls = [float(t.get('pnl', 0.0)) for t in trades]
        returns = pnls
        self.metrics['total_trades'] = len(trades)
        self.metrics['win_rate'] = helpers.calculate_win_rate(
            [{'pnl': p} for p in pnls])
        self.metrics['avg_profit'] = Decimal(str(np.mean([p for p in pnls if p > 0])) if any(p > 0 for p in pnls) else '0.0')
        self.metrics['avg_loss'] = Decimal(str(np.mean([p for p in pnls if p < 0])) if any(p < 0 for p in pnls) else '0.0')
        self.metrics['profit_factor'] = helpers.calculate_profit_factor(
            [{'pnl': p} for p in pnls])
        self.metrics['sharpe_ratio'] = helpers.calculate_sharpe_ratio(returns)
        self.metrics['sortino_ratio'] = helpers.calculate_sortino_ratio(
            returns)
        self.metrics['max_drawdown'] = Decimal(str(helpers.calculate_drawdown(pnls)))
        self.metrics['calmar_ratio'] = float(self.metrics['sharpe_ratio']) / float(self.metrics['max_drawdown']) if self.metrics['max_drawdown'] else 0.0
        self.metrics['roi'] = Decimal(str(sum(pnls) / float(self.config.get('initial_balance', 1.0)))) if self.config.get('initial_balance') else Decimal('0.0')
        return self.metrics

    async def _initialize(self) -> None:
        """Initialise le trader et charge les données nécessaires."""
        super().initialize(self.config)

        # Charger les stratégies actives
        self.active_strategies = self._load_active_strategies()

        # Charger les données historiques
        await self._load_historical_data()

        logger.info("Trader avancé initialisé avec succès")

    def _load_active_strategies(self) -> Dict[str, Dict]:
        """Charge les stratégies actives à partir de la configuration."""
        strategies = {}
        for name, params in self.config.get('strategies', {}).items():
            if params.get('enabled', False):
                strategies[name] = params
        return strategies

    async def _load_historical_data(self, days: int = 30) -> None:
        """Charge les données historiques pour l'analyse."""
        logger.info(f"Chargement des données historiques pour {days} jours...")

        for pair in self.config.get('trading_pairs', []):
            try:
                timeframes = ['1h', '4h', '1d']
                ohlcv_data = {}

                for tf in timeframes:
                    if hasattr(self.api, 'get_ohlc') and callable(getattr(self.api, 'get_ohlc')):
                        ohlcv = await self.api.get_ohlc(
                            pair=pair,
                            interval=1,  # Adapter selon tf si besoin
                        )
                    else:
                        raise NotImplementedError("La méthode get_ohlc n'est pas disponible sur l'API Kraken.")

                    if ohlcv and not ohlcv.get('error'):
                        ohlcv_data[tf] = ohlcv

                if ohlcv_data:
                    self.market_data[pair] = ohlcv_data
                    logger.debug(
                        f"Données chargées pour {pair}: {', '.join(ohlcv_data.keys())}")

            except Exception as e:
                logger.error(
                    f"Erreur lors du chargement des données pour {pair}: {e}")

        if self.market_data:
            self.market_analyzer = MarketAnalyzer(self.market_data)

    async def _update_market_data(self) -> None:
        """Met à jour les données de marché pour toutes les paires de trading."""
        try:
            for pair in self.config.get('trading_pairs', []):
                try:
                    if hasattr(self.api, 'get_ticker') and callable(getattr(self.api, 'get_ticker')):
                        ticker = await self.api.get_ticker(pair=pair)
                    else:
                        raise NotImplementedError("La méthode get_ticker n'est pas disponible sur l'API Kraken.")

                    if ticker:
                        self.prices[pair] = {
                            'last': float(ticker.get('c', {}).get('price', 0)),
                            'bid': float(ticker.get('b', {}).get('price', 0)),
                            'ask': float(ticker.get('a', {}).get('price', 0))
                        }

                        logger.debug(
                            f"Mise à jour des prix pour {pair}: {self.prices[pair]}")

                except Exception as e:
                    logger.error(
                        f"Erreur lors de la mise à jour des données pour {pair}: {e}")

        except Exception as e:
            logger.error(
                f"Erreur lors de la mise à jour des données de marché: {e}")

    async def _log_trading_status(self) -> None:
        """Log l'état actuel du trading."""
        status = self.get_trading_status()

        logger.info(f"État du trading:")
        logger.info(
            f"- Portefeuille total: {status['portfolio']['total_value']} USD")
        logger.info(f"- Cash: {status['portfolio']['cash']} USD")
        logger.info(
            f"- Positions ouvertes: {len(status['portfolio']['positions'])}")
        logger.info(
            f"- PNL réalisé: {status['portfolio']['realized_pnl']} USD")
        logger.info(
            f"- PNL non réalisé: {status['portfolio']['unrealized_pnl']} USD")
        logger.info(f"- Nombre de trades: {status['metrics']['total_trades']}")
        logger.info(f"- Taux de réussite: {status['metrics']['win_rate']:.2%}")
        logger.info(f"- Ratio Sharpe: {status['metrics']['sharpe_ratio']:.2f}")
        logger.info(
            f"- Drawdown maximum: {status['metrics']['max_drawdown']} USD")

    async def run(self) -> None:
        """Boucle principale du trader."""
        logger.info("Démarrage du trader avancé...")
        await self.initialize()

        while not self._stop_event.is_set():
            try:
                # 1. Mettre à jour les données de marché
                await self.update_market_data()

                # 2. Analyser le marché
                await self.analyze_market()

                # 3. Générer des signaux de trading
                signals = await self.generate_signals()

                # 4. Prendre des décisions de trading
                await self.execute_trading_decisions(signals)

                # 5. Mettre à jour l'état des ordres et positions
                await self.update_orders_and_positions()

                # 6. Journaliser l'état actuel
                await self._log_trading_status()

                # Attente avant la prochaine itération
                await asyncio.sleep(self.config.get('tick_interval', 60))

            except asyncio.CancelledError:
                logger.info("Arrêt du trader demandé...")
                break
            except Exception as e:
                logger.error(
                    f"Erreur dans la boucle principale: {e}",
                    exc_info=True)
                await asyncio.sleep(10)  # Attendre avant de réessayer

    async def analyze_market(self) -> None:
        """Analyse le marché et met à jour les indicateurs."""
        try:
            # Récupérer les données de marché pour toutes les paires
            # configurées
            for pair in self.config.get('trading_pairs', []):
                try:
                    # Récupérer le ticker et les données OHLCV
                    ticker = await self.api.get_ticker(pair=pair)
                    ohlcv_data = await self.api.get_ohlc_data(pair=pair)

                    # Analyser les données pour cette paire
                    self.market_analyzer.update_data(ticker, ohlcv_data)
                    self.market_data[pair] = self.market_analyzer.get_analysis()

                    logger.debug(
                        f"Analyse du marché pour {pair}: {self.market_data[pair]}")

                except Exception as e:
                    logger.error(
                        f"Erreur lors de l'analyse du marché pour {pair}: {e}")

        except Exception as e:
            logger.error(f"Erreur lors de l'analyse du marché: {e}")

    async def generate_signals(self) -> Dict[str, Dict]:
        """Génère des signaux de trading pour toutes les paires."""
        signals = {}

        for pair, ohlcv_data in self.market_data.items():
            try:
                # Générer les signaux pour cette paire
                signals[pair] = await self.signal_generator.generate_signals(
                    ohlc_data=ohlcv_data,
                    pair=pair,
                    # Plus petit timeframe disponible
                    timeframe=min(ohlcv_data.keys())
                )

                # Journaliser les signaux
                if 'final_signal' in signals[pair]:
                    logger.info(
                        f"Signal pour {pair}: {signals[pair]['final_signal']}")

            except Exception as e:
                logger.error(
                    f"Erreur lors de la génération des signaux pour {pair}: {e}")

        return signals

    async def execute_trading_decisions(
            self, signals: Dict[str, Dict]) -> None:
        """Prend des décisions de trading basées sur les signaux."""
        if not signals:
            return

        for pair, signal_data in signals.items():
            try:
                if 'final_signal' not in signal_data:
                    continue

                # Récupérer le prix actuel
                current_price = self.prices.get(pair, {}).get('last')
                if not current_price:
                    continue

                # Calculer la taille de position
                stop_loss = self._calculate_stop_loss(
                    signal_data)
                position_size = self.risk_manager.calculate_position_size(
                    entry_price=current_price,
                    stop_loss=stop_loss,
                    risk_percent=self.config.get('risk_per_trade', 0.01)
                )

                # Calculer le levier
                leverage = self.risk_manager.get_leverage(
                    market_volatility=signal_data.get(
                        'atr',
                        0.0) / current_price if 'atr' in signal_data else 0.01,
                    signal_confidence=signal_data.get(
                        'ml_confidence',
                        0.5),
                    recent_performance=self._calculate_recent_performance())

                # Prendre une décision de trading
                if signal_data['final_signal'] == 'buy':
                    await self._execute_buy(pair, position_size, current_price, stop_loss, leverage)
                elif signal_data['final_signal'] == 'sell':
                    await self._execute_sell(pair, position_size, current_price, stop_loss, leverage)

            except Exception as e:
                logger.error(
                    f"Erreur lors de la prise de décision pour {pair}: {e}")

    async def _execute_buy(self,
                           pair: str,
                           amount: float,
                           price: float,
                           stop_loss: float,
                           leverage: float = 1.0) -> None:
        """Exécute un ordre d'achat."""
        try:
            # Calculer le take-profit (par exemple, 2:1 ratio risk:reward)
            take_profit = price + 2 * (price - stop_loss)

            # Placer l'ordre
            order = await self.order_executor.execute_order(
                pair=pair,
                order_type='market',
                side='buy',
                amount=amount,
                price=price,
                leverage=leverage
            )

            # Enregistrer le trade
            self.risk_manager.log_trade(
                pair=pair,
                side='buy',
                amount=amount,
                entry_price=price,
                stop_loss=stop_loss,
                take_profit=take_profit
            )

            logger.info(f"Ordre d'achat exécuté: {amount} {pair} à {price}")

        except Exception as e:
            logger.error(f"Erreur lors de l'exécution de l'ordre d'achat: {e}")

    async def _execute_sell(self,
                            pair: str,
                            amount: float,
                            price: float,
                            stop_loss: float,
                            leverage: float = 1.0) -> None:
        """Exécute un ordre de vente."""
        try:
            # Calculer le take-profit (par exemple, 2:1 ratio risk:reward)
            take_profit = price - 2 * (stop_loss - price)

            # Placer l'ordre
            order = await self.order_executor.execute_order(
                pair=pair,
                order_type='market',
                side='sell',
                amount=amount,
                price=price,
                leverage=leverage
            )

            # Enregistrer le trade
            self.risk_manager.log_trade(
                pair=pair,
                side='sell',
                amount=amount,
                entry_price=price,
                stop_loss=stop_loss,
                take_profit=take_profit
            )

            logger.info(f"Ordre de vente exécuté: {amount} {pair} à {price}")

        except Exception as e:
            logger.error(
                f"Erreur lors de l'exécution de l'ordre de vente: {e}")

    async def update_orders_and_positions(self) -> None:
        """Met à jour l'état des ordres et des positions."""
        try:
            # Mettre à jour les statuts des ordres
            await self.order_executor.update_orders_status()

            # Mettre à jour les positions
            positions = await self.api.get_open_positions()
            if positions:
                for pos_id, position in positions.items():
                    # Mettre à jour la position dans le gestionnaire de risques
                    self.risk_manager.update_position(
                        position_id=pos_id,
                        pair=position.get('pair'),
                        side=position.get('type'),
                        amount=float(
                            position.get(
                                'vol',
                                0)),
                        entry_price=float(
                            position.get(
                                'cost',
                                0)) /
                        float(
                            position.get(
                                'vol',
                                1)),
                        current_price=float(
                            position.get(
                                'value',
                                0)) /
                        float(
                            position.get(
                                'vol',
                                1)),
                        leverage=float(
                            position.get(
                                'leverage',
                                1.0)))
        except Exception as e:
            logger.error(
                f"Erreur lors de la mise à jour des ordres et positions: {e}")
            raise

    async def get_open_positions(self) -> list:
        """Retourne la liste des positions ouvertes réelles."""
        # Supposons que self.portfolio['positions'] contient les positions
        return self.portfolio.get('positions', [])

    async def get_account_performance(self) -> float:
        """Retourne la performance réelle du compte (ROI en %)."""
        # Supposons que self.metrics['roi'] contient le ROI sous forme de Decimal
        roi = self.metrics.get('roi', Decimal('0.0'))
        return float(roi) * 100 if roi else 0.0

    @property
    def pair_configs(self) -> dict:
        """Retourne la configuration réelle des paires de trading."""
        # Supposons que self.active_strategies contient la config des paires
        return self.active_strategies

    @property
    def active_pairs(self) -> list:
        """Retourne la liste réelle des paires actives."""
        # Supposons que les clés de self.active_strategies sont les paires actives
        return list(self.active_strategies.keys())

    @property
    def market_trend_detector(self):
        """Retourne l'analyseur de tendance du marché réel."""
        # Supposons que self.market_analyzer a la méthode get_market_metrics
        return self.market_analyzer

    def get_global_metrics(self) -> dict:
        """Retourne les métriques globales réelles du bot."""
        # Supposons que self.metrics contient les métriques globales
        return {k: float(v) if isinstance(v, Decimal) else v for k, v in self.metrics.items()}

    def get_detailed_metrics(self) -> dict:
        """Retourne les métriques détaillées réelles par paire."""
        # Supposons que self.last_analysis contient les métriques par paire
        return self.last_analysis
