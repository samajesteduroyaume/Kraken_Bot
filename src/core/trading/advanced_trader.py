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
from src.utils import pair_utils

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

    async def is_config_valid(self) -> bool:
        """
        Vérifie si la configuration est valide.
        
        Note:
            Cette méthode est maintenant asynchrone car elle nécessite l'initialisation
            des paires de trading.

        Returns:
            bool: True si la configuration est valide, False sinon
        """
        # Import relatif pour éviter les problèmes de chemin
        from ..market.available_pairs import initialize_available_pairs
        
        if not self.config:
            return False
            
        try:
            # Initialiser les paires de trading
            available_pairs = await initialize_available_pairs()
            
            required_keys = [
                'trading_pair',
                'trading_amount',
                'trading_fee',
                'risk_level',
                'quote_currency',
                'min_volume_btc',
                'max_spread_pct',
                'min_trades_24h',
                'custom_pairs',
                'max_daily_drawdown',
                'risk_percentage',
                'max_leverage'
            ]
            
            # Vérifier que toutes les clés requises sont présentes
            for key in required_keys:
                if key not in self.config:
                    logger.error(f"Clé de configuration manquante: {key}")
            trading_pair = self.config.get('trading_pair')
            if not trading_pair:
                logger.error("Aucune paire de trading spécifiée dans la configuration")
                return False
                
            # Vérifier si la paire est supportée par Kraken
            if not available_pairs.is_pair_supported(trading_pair):
                normalized = available_pairs.normalize_pair(trading_pair)
                if normalized and available_pairs.is_pair_supported(normalized):
                    # Mise à jour de la configuration avec la paire normalisée
                    self.config['trading_pair'] = normalized
                    logger.info(f"Paire normalisée : {trading_pair} -> {normalized}")
                else:
                    logger.error(f"Paire non supportée par Kraken : {trading_pair}")
                    return False
                    
            # Vérification de la devise de cotation
            quote_currency = self.config.get('quote_currency', '').upper()
            if quote_currency and not available_pairs.is_quote_currency_supported(quote_currency):
                logger.error(f"Devise de cotation non supportée : {quote_currency}")
                return False
                
            return True
            
        except Exception as e:
            logger.error(f"Erreur lors de la validation de la configuration : {e}")
            return False

    async def run(self) -> None:
        """Boucle principale du trader avancé.
        
        Cette méthode gère le cycle de vie complet du trader, y compris :
        1. L'initialisation des composants
        2. La mise à jour des données de marché
        3. L'analyse du marché et génération de signaux
        4. La gestion des risques et exécution des trades
        5. La mise à jour des positions et journalisation
        """
        logger.info("Démarrage du trader avancé...")
        
        # Initialisation du trader (appelle super().initialize() en interne)
        await self._initialize()
        
        # Intervalle entre les itérations (en secondes)
        tick_interval = self.config.get('tick_interval', 60)

        while not self._stop_event.is_set():
            try:
                # 1. Mise à jour des données de marché
                await self._update_market_data()

                # 2. Analyse du marché
                await self.analyze_market()
                
                # 2.1 Analyse supplémentaire pour chaque paire (si nécessaire)
                for pair, data in self.market_data.items():
                    if hasattr(self.market_analyzer, 'analyze_market') and callable(getattr(self.market_analyzer, 'analyze_market')):
                        analysis = self.market_analyzer.analyze_market(data)
                        self.last_analysis[pair] = analysis

                # 3. Génération des signaux
                signals = await self.generate_signals()

                # 4. Gestion des risques
                risk_signals = signals
                if hasattr(self.risk_manager, 'process_signals') and callable(getattr(self.risk_manager, 'process_signals')):
                    risk_signals = self.risk_manager.process_signals(signals)

                # 5. Exécution des décisions de trading
                await self.execute_trading_decisions(risk_signals)
                
                # 5.1 Mise à jour des positions
                await self.manage_positions()

                # 6. Mise à jour de l'état des ordres et positions
                await self.update_orders_and_positions()

                # 7. Journalisation de l'état actuel
                status = self.get_trading_status()
                logger.info(f"Statut du trading: {status}")
                await self._log_trading_status()
                
                # 8. Sauvegarde dans la base de données (si disponible)
                if self.db_manager and hasattr(self.db_manager, 'save_trading_status'):
                    await self.db_manager.save_trading_status(status)

                # Attente avant la prochaine itération
                await asyncio.sleep(tick_interval)

            except asyncio.CancelledError:
                logger.info("Arrêt du trader demandé...")
                break
                
            except Exception as e:
                logger.error(
                    f"Erreur dans la boucle principale: {e}",
                    exc_info=True
                )
                # Attente plus longue en cas d'erreur
                await asyncio.sleep(min(tick_interval * 2, 300))  # Maximum 5 minutes

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
        logger.info("🔄 Début de la mise à jour des données de marché...")
        try:
            trading_pairs = self.config.get('trading_pairs', [])
            logger.info(f"📊 Nombre de paires à traiter: {len(trading_pairs)}")
            
            if not trading_pairs:
                logger.warning("⚠️ Aucune paire de trading configurée dans config['trading_pairs']")
                return
                
            for i, pair in enumerate(trading_pairs, 1):
                try:
                    logger.debug(f"[{i}/{len(trading_pairs)}] Traitement de la paire: {pair}")
                    
                    if not hasattr(self.api, 'get_ticker') or not callable(getattr(self.api, 'get_ticker', None)):
                        logger.error("❌ L'API ne possède pas de méthode get_ticker valide")
                        continue
                        
                    # Extraire le symbole de la paire (peut être un dict ou une chaîne)
                    pair_symbol = pair['pair'] if isinstance(pair, dict) and 'pair' in pair else pair
                    
                    # S'assurer que pair_symbol est bien une chaîne
                    if not isinstance(pair_symbol, str):
                        logger.error(f"❌ Le symbole de paire doit être une chaîne, reçu: {type(pair_symbol).__name__} - {pair_symbol}")
                        continue
                        
                    # Normaliser la paire au format wsname Kraken avant l'appel API
                    try:
                        pair_symbol = pair_utils.normalize_pair_input(pair_symbol)
                        logger.debug(f"Paire normalisée: {pair_symbol}")
                    except Exception as e:
                        logger.error(f"❌ Erreur lors de la normalisation de la paire {pair_symbol}: {e}")
                        continue
                        
                    logger.debug(f"Appel de get_ticker pour la paire: '{pair_symbol}' (type: {type(pair_symbol).__name__})")
                    try:
                        ticker = await self.api.get_ticker(pair=pair_symbol)
                        logger.debug(f"Réponse get_ticker pour {pair_symbol}: {ticker}")
                    except Exception as e:
                        logger.error(f"❌ Erreur lors de l'appel à get_ticker pour {pair_symbol}: {str(e)}")
                        continue

                    if not ticker or 'result' not in ticker or not ticker['result']:
                        logger.warning(f"⚠️ Réponse de get_ticker vide ou invalide pour {pair_symbol}")
                        continue

                    # Extraire les données du ticker
                    ticker_data = ticker['result'].get(pair_symbol, {}) if isinstance(ticker['result'], dict) else {}
                    
                    if not ticker_data:
                        logger.warning(f"⚠� Aucune donnée de ticker pour {pair} dans la réponse")
                        continue

                    # Mettre à jour les prix avec gestion des erreurs
                    try:
                        self.prices[pair] = {
                            'bid': float(ticker_data.get('b', [0])[0] if isinstance(ticker_data.get('b'), list) else 0),
                            'ask': float(ticker_data.get('a', [0])[0] if isinstance(ticker_data.get('a'), list) else 0)
                        }
                        logger.info(f"✅ Prix mis à jour pour {pair}: {self.prices[pair]}")
                    except (IndexError, ValueError, TypeError) as e:
                        logger.error(f"❌ Erreur lors de l'extraction des prix pour {pair}: {e}")
                        continue

                except asyncio.TimeoutError:
                    logger.error(f"⏱️ Timeout lors de la récupération du ticker pour {pair}")
                except Exception as e:
                    logger.error(f"❌ Erreur lors de la mise à jour des données pour {pair}: {e}", exc_info=True)

            logger.info("✅ Mise à jour des données de marché terminée")

        except Exception as e:
            logger.error(f"❌ Erreur critique dans _update_market_data: {e}", exc_info=True)
            raise

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
        
        # Initialisation du trader (appelle super().initialize() en interne)
        await self._initialize()

        while not self._stop_event.is_set():
            try:
                # 1. Mettre à jour les données de marché
                await self._update_market_data()

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
        """
        Génère des signaux de trading pour toutes les paires.
        
        Returns:
            Dict[str, Dict]: Dictionnaire des signaux par paire
        """
        signals = {}

        for pair, market_data in self.market_data.items():
            try:
                # Vérifier si les données OHLCV sont disponibles
                if not hasattr(market_data, 'candles') or not market_data.candles:
                    logger.warning(f"Aucune donnée OHLCV disponible pour {pair}")
                    continue
                    
                # Préparer les données OHLCV au format attendu par le SignalGenerator
                ohlcv_data = {}
                for tf, candles in market_data.candles.items():
                    # Convertir les bougies en DataFrame pandas
                    df = pd.DataFrame([{
                        'open': c.open,
                        'high': c.high,
                        'low': c.low,
                        'close': c.close,
                        'volume': c.volume,
                        'timestamp': pd.to_datetime(c.timestamp, unit='ms')
                    } for c in candles])
                    df.set_index('timestamp', inplace=True)
                    ohlcv_data[tf] = df
                
                # Extraire le carnet d'ordres s'il est disponible
                order_book = market_data.order_book if hasattr(market_data, 'order_book') else None
                
                # Générer les signaux pour cette paire
                signals[pair] = await self.signal_generator.generate_signals(
                    ohlc_data=ohlcv_data,
                    pair=pair,
                    timeframe=min(ohlcv_data.keys()),  # Plus petit timeframe disponible
                    order_book=order_book  # Passer le carnet d'ordres
                )

                # Journaliser les signaux
                if 'final_signal' in signals[pair]:
                    logger.info(
                        f"Signal pour {pair}: {signals[pair]['final_signal']}")

            except Exception as e:
                logger.error(
                    f"Erreur lors de la génération des signaux pour {pair}: {e}",
                    exc_info=True
                )

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
