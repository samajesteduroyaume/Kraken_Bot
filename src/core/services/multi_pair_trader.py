import pandas as pd
from typing import Dict, Optional, List, Any
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor

# Importer la fonction de normalisation robuste
from src.utils.pair_utils import normalize_pair_input

from src.core.simulation_mode import SimulationConfig
from src.core.api.kraken import KrakenAPI
from src.core.ml_predictor import MLPredictor
from src.core.technical_analyzer import TechnicalAnalyzer
from src.core.services.pair_selector.core import PairSelector
from src.core.pair_rotation import PairRotationStrategy
from src.core.market_trend_detector import MarketTrendDetector
from src.core.pair_cluster_analyzer import PairClusterAnalyzer
from src.core.risk_manager import RiskManager
from src.core.logging import LoggerManager
from src.core.trading_simulation import TradingSimulation, TradingParameters
from src.core.simulation_mode import SimulationMode


class PairConfig:
    """Configuration spécifique à une paire de trading."""

    def __init__(
        self,
        pair: str,
        risk_per_trade: float = 0.02,
        stop_loss_percent: float = 0.02,
        take_profit_percent: float = 0.04,
        max_positions: int = 5,
        max_drawdown: float = 0.1,
        max_holding_time: int = 1,
        ml_predictor: Optional[MLPredictor] = None,
        technical_analyzer: Optional[TechnicalAnalyzer] = None
    ):
        self.pair = pair
        self.risk_per_trade = risk_per_trade
        self.stop_loss_percent = stop_loss_percent
        self.take_profit_percent = take_profit_percent
        self.max_positions = max_positions
        self.max_drawdown = max_drawdown
        self.max_holding_time = max_holding_time
        self.ml_predictor = ml_predictor or MLPredictor()
        self.technical_analyzer = technical_analyzer or TechnicalAnalyzer()


class MultiPairTrader:
    """Gère plusieurs traders pour différentes paires de trading."""

    def __init__(
        self,
        api: KrakenAPI,
        config: SimulationConfig,
        max_concurrent_trades: int = 5,
        max_trading_pairs: int = 50,
        min_pair_score: float = 0.5,
        risk_per_pair: float = 0.1,
        rotation_frequency: timedelta = timedelta(hours=4),
        performance_threshold: float = 0.01,
        volatility_threshold: float = 0.05,
        max_global_risk: float = 0.05,
        stop_loss_threshold: float = 0.02,
        take_profit_threshold: float = 0.03,
        logger: Optional[LoggerManager] = None
    ):
        self.api = api
        self.config = config
        self.max_concurrent_trades = max_concurrent_trades
        self.max_trading_pairs = max_trading_pairs
        self.min_pair_score = min_pair_score
        self.risk_per_pair = risk_per_pair
        self.logger = logger or LoggerManager()
        self.traders: Dict[str, TradingSimulation] = {}
        self.metrics: Dict[str, Dict] = {}
        self.pair_metrics: Dict[str, Dict] = {}
        self.executor = ThreadPoolExecutor(max_workers=max_concurrent_trades)

        # Initialiser le sélecteur de paires avec la configuration
        self.pair_selector = PairSelector(
            kraken_api=api,  # Correction: utiliser kraken_api au lieu de api
            config={
                'max_pairs': max_trading_pairs,
                'min_volume': 100000,
                'volatility_window': 20,
                'momentum_window': 14,
                'executor': self.executor,
                **config
            }
        )

        # Initialiser la stratégie de rotation
        self.rotation_strategy = PairRotationStrategy(
            pair_selector=self.pair_selector,
            rotation_frequency=rotation_frequency,
            performance_threshold=performance_threshold,
            volatility_threshold=volatility_threshold,
            logger=self.logger
        )

        # Initialiser le détecteur de tendance
        self.market_trend_detector = MarketTrendDetector(
            technical_analyzer=config.technical_analyzer,
            logger=self.logger
        )

        # Initialiser le clusteriseur de paires
        self.pair_cluster = PairClusterAnalyzer(
            n_clusters=5,
            min_pairs_per_cluster=3,
            max_pairs_per_cluster=10,
            logger=self.logger
        )

        # Initialiser le gestionnaire de risques
        self.risk_manager = RiskManager(
            max_global_risk=max_global_risk,
            max_pair_risk=risk_per_pair,
            stop_loss_threshold=stop_loss_threshold,
            take_profit_threshold=take_profit_threshold,
            logger=self.logger
        )

        # Les traders seront initialisés lors du premier appel à run()
        self._initialized = False

    async def update_status(self):
        """
        Met à jour l'état du trader avec les dernières données de marché.
        
        Returns:
            Dict contenant les métriques mises à jour
        """
        try:
            # Récupérer les paires valides (cela charge et analyse automatiquement les paires)
            valid_pairs = await self.pair_selector.get_valid_pairs(min_score=self.min_pair_score)
            
            # Construire le dictionnaire de métriques pour les paires sélectionnées
            pair_metrics = {}
            for pair_info in valid_pairs:
                try:
                    pair_name = pair_info['pair']  # Accès au nom de la paire normalisée
                    metrics = self.pair_selector.get_pair_metrics(pair_name)
                    if metrics:
                        pair_metrics[pair_name] = metrics
                except (KeyError, TypeError) as e:
                    self.logger.error(f"Erreur lors de l'extraction des métriques pour la paire {pair_info}: {e}")
            self.pair_metrics = pair_metrics

            # Mettre à jour l'analyse des clusters
            clusters = self.pair_cluster.cluster_pairs(self.pair_metrics)

            # Mettre à jour la détection des tendances
            trends = {}
            for pair in self.pair_metrics.keys():
                # Ici, il faudrait idéalement passer les données de marché (DataFrame) de chaque paire
                # Mais si non disponible, on met 'neutral' par défaut
                try:
                    # TODO: récupérer les vraies données de marché pour chaque
                    # paire
                    trends[pair] = self.market_trend_detector.detect_trend(
                        pd.DataFrame())
                except Exception:
                    trends[pair] = 'neutral'

            # Mettre à jour la rotation des paires
            rotated_pairs = await self.rotation_strategy.rotate_pairs(list(self.traders.keys()))

            # Mettre à jour les métriques
            self.metrics = {
                'pairs_count': len(pairs),
                'clusters_count': len(clusters),
                'trends': trends,
                'rotated_pairs': rotated_pairs
            }

            return self.metrics

        except Exception as e:
            error_msg = f"Erreur lors de la mise à jour du statut: {str(e)}"
            self.logger.log(error_msg, "error")
            print(error_msg)  # Pour débogage
            raise

    async def initialize(self) -> None:
        """Initialise les traders de manière asynchrone."""
        if not self._initialized:
            await self._initialize_traders()
            self._initialized = True

    async def _initialize_traders(self) -> None:
        """
        Initialise les traders pour les paires sélectionnées.
        
        Cette méthode récupère les paires valides, les filtre selon le score minimum,
        puis initialise un trader pour chaque paire sélectionnée.
        """
        try:
            # Initialiser le logger
            logger = self.logger.get_logger('multi_pair_trader')
            logger.info("Démarrage de l'initialisation des traders...")

            # Récupérer les paires valides (cela charge et analyse automatiquement les paires)
            selected_pairs = await self.pair_selector.get_valid_pairs(min_score=self.min_pair_score)
            
            if not selected_pairs:
                logger.warning("Aucune paire valide trouvée pour l'initialisation des traders")
                return
            
            # Extraire les noms de paires normalisés
            pair_names = []
            for pair_info in selected_pairs:
                try:
                    pair_name = pair_info['pair']
                    pair_names.append(pair_name)
                except (KeyError, TypeError) as e:
                    logger.warning(f"Format de paire invalide ignoré: {pair_info} - {e}")
            
            if not pair_names:
                logger.warning("Aucun nom de paire valide n'a pu être extrait")
                return
                
            logger.info(f"Initialisation de {len(pair_names)} traders pour les paires: {', '.join(pair_names)}")

            # Initialiser les traders pour les paires sélectionnées
            for pair in pair_names:
                # Calculer le risque par trade pour cette paire
                risk_per_trade = self.risk_per_pair / len(selected_pairs)

                # Créer la configuration spécifique à la paire
                pair_config = PairConfig(
                    pair=pair,
                    risk_per_trade=risk_per_trade,
                    ml_predictor=self.config.ml_predictor,
                    technical_analyzer=self.config.technical_analyzer
                )

                # Créer et initialiser les paramètres de trading
                trading_params = await TradingParameters.initialize(
                    api=self.api,
                    risk_per_trade=self.risk_per_pair,
                    stop_loss_percent=pair_config.stop_loss_percent,
                    take_profit_percent=pair_config.take_profit_percent,
                    max_positions=pair_config.max_positions,
                    max_drawdown=pair_config.max_drawdown,
                    max_holding_time=timedelta(days=pair_config.max_holding_time),
                    adaptive_leverage=self.config.adaptive_leverage,
                    backtest_manager=self.config.backtest_manager,
                    ml_predictor=self.config.ml_predictor,
                    technical_analyzer=self.config.technical_analyzer,
                    sentiment_analyzer=self.config.sentiment_analyzer,
                    signal_generator=self.config.signal_generator,
                    database_manager=self.config.database_manager,
                    metrics_manager=self.config.metrics_manager,
                    logger=self.logger,
                    config=self.config,
                    utils=self.config.utils
                )

                # Initialiser le trader
                trader = TradingSimulation(trading_params)
                self.traders[pair] = trader

                # Stocker les métriques de la paire
                self.pair_metrics[pair] = self.pair_selector.get_pair_metrics(
                    pair)

                logger.info(
                    f"Trader initialisé pour {pair} avec score {self.pair_metrics[pair]['score']:.2f}")

        except Exception as e:
            logger.error(
                f"Erreur lors de l'initialisation des traders: {str(e)}")
            raise
            # Calculer le risque par trade pour cette paire
            risk_per_trade = self.risk_per_pair / len(self.pairs)

            # Créer la configuration spécifique à la paire
            pair_config = PairConfig(
                pair=pair,
                risk_per_trade=risk_per_trade,
                ml_predictor=self.config.ml_predictor,
                technical_analyzer=self.config.technical_analyzer
            )

            # Créer les paramètres de trading
            trading_params = TradingParameters(
                api=self.api,
                risk_per_trade=risk_per_trade,
                stop_loss_percent=pair_config.stop_loss_percent,
                take_profit_percent=pair_config.take_profit_percent,
                max_positions=pair_config.max_positions,
                max_drawdown=pair_config.max_drawdown,
                max_holding_time=timedelta(days=pair_config.max_holding_time),
                ml_predictor=pair_config.ml_predictor,
                technical_analyzer=pair_config.technical_analyzer
            )

            # Initialiser le trader
            trader = TradingSimulation(trading_params)
            self.traders[pair] = trader
            self.metrics[pair] = {
                'total_trades': 0,
                'total_profit': 0,
                'max_drawdown': 0,
                'current_balance': 0
            }

    async def _run_trader(self, pair: str) -> None:
        """Exécute un trader pour une paire spécifique avec gestion des risques."""
        from src.simulation.monthly_simulation import get_price_history, process_trade
        try:
            logger = self.logger.get_logger(f'trader_{pair}')
            logger.info(f"Démarrage du trader pour la paire {pair}")

            # Récupérer les données pour la paire
            ohlc_data = await get_price_history(self.api, self.config, pair)

            # Détecter la tendance du marché
            market_trend = self.market_trend_detector.detect_trend(ohlc_data)
            logger.info(f"Tendance détectée: {market_trend}")

            # Initialiser le trader
            trader = self.traders[pair]

            # Gérer les trades selon le mode de simulation
            if self.config.mode == SimulationMode.HISTORICAL:
                # Mode historique
                price_history = ohlc_data['close']
                for i in range(len(price_history) - 1):
                    current_time = price_history.index[i]
                    current_price = float(price_history.iloc[i])
                    next_price = float(price_history.iloc[i + 1])
                    price_hist_slice = price_history.iloc[:i + 1]

                    # Gérer le risque
                    pair_metrics = self.pair_metrics[pair]
                    risk_allocation = self.risk_manager.calculate_risk_allocation(
                        {pair: pair_metrics},
                        trader.current_balance,
                        market_trend
                    )

                    position_size = self.risk_manager.adjust_position_size(
                        current_price,
                        risk_allocation[pair],
                        market_trend,
                        pair_metrics
                    )

                    # Calculer les niveaux de stop loss et take profit
                    stop_loss = self.risk_manager.calculate_stop_loss(
                        current_price,
                        pair_metrics
                    )
                    take_profit = self.risk_manager.calculate_take_profit(
                        current_price,
                        pair_metrics
                    )

                    await process_trade(
                        trader,
                        current_time,
                        current_price,
                        next_price,
                        price_hist_slice
                    )

            elif self.config.mode == SimulationMode.REAL_TIME:
                # Mode temps réel
                now = datetime.now()
                real_time_data = ohlc_data[ohlc_data.index >=
                                           now - self.config.real_time_window]

                # S'assurer que real_time_data['close'] est une Series avec un
                # DatetimeIndex
                if not isinstance(real_time_data['close'], pd.Series):
                    if hasattr(
                            real_time_data,
                            'index') and isinstance(
                            real_time_data.index,
                            pd.DatetimeIndex):
                        real_time_data['close'] = pd.Series(
                            real_time_data['close'], index=real_time_data.index)
                    else:
                        # Générer un index temporel factice
                        real_time_data['close'] = pd.Series(
                            real_time_data['close'], index=pd.date_range(
                                start=datetime.now(), periods=len(
                                    real_time_data['close']), freq='T'))
                for i in range(len(real_time_data['close']) - 1):
                    current_time = real_time_data['close'].index[i]
                    current_price = float(real_time_data['close'].iloc[i])
                    next_price = float(real_time_data['close'].iloc[i + 1])
                    price_hist_slice = real_time_data['close'].iloc[:i + 1]

                    # Gérer le risque
                    pair_metrics = self.pair_metrics[pair]
                    risk_allocation = self.risk_manager.calculate_risk_allocation(
                        {pair: pair_metrics},
                        trader.current_balance,
                        market_trend
                    )

                    position_size = self.risk_manager.adjust_position_size(
                        current_price,
                        risk_allocation[pair],
                        market_trend,
                        pair_metrics
                    )

                    # Calculer les niveaux de stop loss et take profit
                    stop_loss = self.risk_manager.calculate_stop_loss(
                        current_price,
                        pair_metrics
                    )
                    take_profit = self.risk_manager.calculate_take_profit(
                        current_price,
                        pair_metrics
                    )

                    await process_trade(
                        trader,
                        current_time,
                        current_price,
                        next_price,
                        price_hist_slice
                    )

            elif self.config.mode == SimulationMode.BACKTEST:
                # Mode backtest avec prédiction adaptative
                train_data, val_data, test_data = self.config.get_data_splits(
                    ohlc_data)

                # Initialiser le système de prédiction adaptative
                adaptive_prediction = AdaptivePrediction(
                    base_predictor=self.config.ml_predictor,
                    technical_analyzer=self.config.technical_analyzer,
                    lookback_window=30,
                    update_frequency=10,
                    confidence_threshold=0.7
                )

                # Faire les prédictions adaptatives
                predictions = []

                for i in range(len(test_data) - 1):
                    context_data = test_data[:i + 1]
                    pred, _ = adaptive_prediction.predict(context_data)
                    predictions.append(pred)

                    if (i + 1) % adaptive_prediction.update_frequency == 0:
                        adaptive_prediction._update_model(context_data)

                # Trading avec les prédictions
                # S'assurer que test_data['close'] est une Series avec un
                # DatetimeIndex
                if not isinstance(test_data['close'], pd.Series):
                    if hasattr(
                            test_data,
                            'index') and isinstance(
                            test_data.index,
                            pd.DatetimeIndex):
                        test_data['close'] = pd.Series(
                            test_data['close'], index=test_data.index)
                    else:
                        test_data['close'] = pd.Series(
                            test_data['close'], index=pd.date_range(
                                start=datetime.now(), periods=len(
                                    test_data['close']), freq='T'))
                for i in range(len(test_data['close']) - 1):
                    current_time = test_data['close'].index[i]
                    current_price = float(test_data['close'].iloc[i])
                    next_price = float(predictions[i])
                    price_hist_slice = test_data['close'].iloc[:i + 1]

                    # Gérer le risque
                    pair_metrics = self.pair_metrics[pair]
                    risk_allocation = self.risk_manager.calculate_risk_allocation(
                        {pair: pair_metrics},
                        trader.current_balance,
                        market_trend
                    )

                    position_size = self.risk_manager.adjust_position_size(
                        current_price,
                        risk_allocation[pair],
                        market_trend,
                        pair_metrics
                    )

                    # Calculer les niveaux de stop loss et take profit
                    stop_loss = self.risk_manager.calculate_stop_loss(
                        current_price,
                        pair_metrics
                    )
                    take_profit = self.risk_manager.calculate_take_profit(
                        current_price,
                        pair_metrics
                    )

                    await process_trade(
                        trader,
                        current_time,
                        current_price,
                        next_price,
                        price_hist_slice
                    )

            # Mettre à jour les métriques
            self.metrics[pair] = {
                'total_trades': trader.metrics['total_trades'],
                'total_profit': trader.metrics['total_profit'],
                'max_drawdown': trader.metrics['max_drawdown'],
                'current_balance': trader.current_balance
            }

            logger.info(
                f"Trader {pair} terminé. Profit total: {trader.metrics['total_profit']:.2f} USD"
            )

        except Exception as e:
            logger.error(f"Erreur dans le trader {pair}: {str(e)}")
            raise

    async def _run_trader(self, pair_input: Any):
        """
        Exécute un trader pour une paire spécifique avec gestion des risques.
        
        Args:
            pair_input: Peut être une chaîne (nom de paire) ou un dictionnaire avec une clé 'pair'
        """
        try:
            # Normaliser l'entrée de la paire avec gestion des erreurs améliorée
            try:
                if isinstance(pair_input, dict) and 'pair' in pair_input:
                    pair = pair_input['pair']
                elif isinstance(pair_input, str):
                    # Utiliser raise_on_error=False pour gérer les erreurs nous-mêmes
                    pair = normalize_pair_input(pair_input, raise_on_error=False)
                    
                    if not pair:
                        # Si la paire n'est pas reconnue, essayer de trouver des alternatives
                        try:
                            # Cette fois avec raise_on_error=True pour obtenir les suggestions
                            normalize_pair_input(pair_input, raise_on_error=True)
                        except UnsupportedTradingPairError as e:
                            # Afficher les suggestions dans les logs
                            if e.alternatives:
                                self.logger.warning(
                                    f"Paire non reconnue: {pair_input}. "
                                    f"Suggestions: {', '.join(e.alternatives[:3])}"
                                )
                            else:
                                self.logger.warning(f"Paire non reconnue: {pair_input}. Aucune suggestion disponible.")
                        except Exception as e:
                            self.logger.error(f"Erreur lors de la recherche d'alternatives pour {pair_input}: {e}")
                        return
                else:
                    self.logger.error(f"Format de paire non supporté: {pair_input}")
                    return
                    
            except Exception as e:
                self.logger.error(f"Erreur de normalisation de la paire {pair_input}: {e}")
                return

            logger = self.logger.get_logger(f'trader_{pair}')
            logger.info(f"Démarrage du trader pour la paire {pair}")

            # Récupérer les données pour la paire
            ohlc_data = await get_price_history(self.api, self.config, pair)
            
            # Détecter la tendance du marché
            market_trend = self.market_trend_detector.detect_trend(ohlc_data)
            logger.info(f"Tendance détectée: {market_trend}")

            # Initialiser le trader
            trader = self.traders[pair]

            # Gérer les trades selon le mode de simulation
            if self.config.mode == SimulationMode.HISTORICAL:
                # Mode historique
                price_history = ohlc_data['close']
                for i in range(len(price_history) - 1):
                    current_time = price_history.index[i]
                    current_price = price_history[i]
                    next_price = price_history[i + 1]

                    # Ajuster la taille de position selon la tendance
                    position_size = self._adjust_position_size(
                        current_price,
                        market_trend,
                        trader.current_balance
                    )

                    await process_trade(
                        trader,
                        current_time,
                        float(current_price),
                        float(next_price),
                        price_history[:i + 1]
                    )

            elif self.config.mode == SimulationMode.REAL_TIME:
                # Mode temps réel
                now = datetime.now()
                real_time_data = ohlc_data[ohlc_data.index >=
                                           now - self.config.real_time_window]

                for i in range(len(real_time_data) - 1):
                    current_time = real_time_data.index[i]
                    current_price = real_time_data['close'].iloc[i]
                    next_price = real_time_data['close'].iloc[i + 1]

                    # Ajuster la taille de position selon la tendance
                    position_size = self._adjust_position_size(
                        current_price,
                        market_trend,
                        trader.current_balance
                    )

                    await process_trade(
                        trader,
                        current_time,
                        float(current_price),
                        float(next_price),
                        real_time_data[:i + 1]
                    )

            elif self.config.mode == SimulationMode.BACKTEST:
                # Mode backtest avec prédiction adaptative
                train_data, val_data, test_data = self.config.get_data_splits(
                    ohlc_data)

                # Initialiser le système de prédiction adaptative
                adaptive_prediction = AdaptivePrediction(
                    base_predictor=self.config.ml_predictor,
                    technical_analyzer=self.config.technical_analyzer,
                    lookback_window=30,
                    update_frequency=10,
                    confidence_threshold=0.7
                )

                # Faire les prédictions adaptatives
                predictions = []

                for i in range(len(test_data) - 1):
                    context_data = test_data[:i + 1]
                    pred, _ = adaptive_prediction.predict(context_data)
                    predictions.append(pred)

                    if (i + 1) % adaptive_prediction.update_frequency == 0:
                        adaptive_prediction._update_model(context_data)

                # Trading avec les prédictions
                for i in range(len(test_data) - 1):
                    current_time = test_data.index[i]
                    current_price = test_data['close'].iloc[i]
                    next_price = predictions[i]

                    # Ajuster la taille de position selon la tendance
                    position_size = self._adjust_position_size(
                        current_price,
                        market_trend,
                        trader.current_balance
                    )

                    await process_trade(
                        trader,
                        current_time,
                        float(current_price),
                        float(next_price),
                        test_data[:i + 1]
                    )

            # Mettre à jour les métriques
            self.metrics[pair] = {
                'total_trades': trader.metrics['total_trades'],
                'total_profit': trader.metrics['total_profit'],
                'max_drawdown': trader.metrics['max_drawdown'],
                'current_balance': trader.current_balance
            }

            logger.info(
                f"Trader {pair} terminé. Profit total: {trader.metrics['total_profit']:.2f} USD")

        except Exception as e:
            logger.error(f"Erreur dans le trader {pair}: {str(e)}")
            raise
        """Exécute un trader pour une paire spécifique."""
        try:
            logger = self.logger.get_logger(f'trader_{pair}')
            logger.info(f"Démarrage du trader pour la paire {pair}")

            # Récupérer les données pour la paire
            ohlc_data = await get_price_history(self.api, self.config, pair)

            # Initialiser le trader
            trader = self.traders[pair]

            # Gérer les trades selon le mode de simulation
            if self.config.mode == SimulationMode.HISTORICAL:
                # Mode historique
                price_history = ohlc_data['close']
                for i in range(len(price_history) - 1):
                    current_time = price_history.index[i]
                    current_price = price_history[i]
                    next_price = price_history[i + 1]

                    await process_trade(
                        trader,
                        current_time,
                        float(current_price),
                        float(next_price),
                        price_history[:i + 1]
                    )

            elif self.config.mode == SimulationMode.REAL_TIME:
                # Mode temps réel
                now = datetime.now()
                real_time_data = ohlc_data[ohlc_data.index >=
                                           now - self.config.real_time_window]

                for i in range(len(real_time_data) - 1):
                    current_time = real_time_data.index[i]
                    current_price = real_time_data['close'].iloc[i]
                    next_price = real_time_data['close'].iloc[i + 1]

                    await process_trade(
                        trader,
                        current_time,
                        float(current_price),
                        float(next_price),
                        real_time_data[:i + 1]
                    )

            elif self.config.mode == SimulationMode.BACKTEST:
                # Mode backtest avec prédiction adaptative
                train_data, val_data, test_data = self.config.get_data_splits(
                    ohlc_data)

                # Initialiser le système de prédiction adaptative
                adaptive_prediction = AdaptivePrediction(
                    base_predictor=self.config.ml_predictor,
                    technical_analyzer=self.config.technical_analyzer,
                    lookback_window=30,
                    update_frequency=10,
                    confidence_threshold=0.7
                )

                # Faire les prédictions adaptatives
                predictions = []

                for i in range(len(test_data) - 1):
                    context_data = test_data[:i + 1]
                    pred, _ = adaptive_prediction.predict(context_data)
                    predictions.append(pred)

                    if (i + 1) % adaptive_prediction.update_frequency == 0:
                        adaptive_prediction._update_model(context_data)

                # Trading avec les prédictions
                for i in range(len(test_data) - 1):
                    current_time = test_data.index[i]
                    current_price = test_data['close'].iloc[i]
                    next_price = predictions[i]

                    await process_trade(
                        trader,
                        current_time,
                        float(current_price),
                        float(next_price),
                        test_data[:i + 1]
                    )

            # Mettre à jour les métriques
            self.metrics[pair] = {
                'total_trades': trader.metrics['total_trades'],
                'total_profit': trader.metrics['total_profit'],
                'max_drawdown': trader.metrics['max_drawdown'],
                'current_balance': trader.current_balance
            }

            logger.info(
                f"Trader {pair} terminé. Profit total: {trader.metrics['total_profit']:.2f} USD")

        except Exception as e:
            logger.error(f"Erreur dans le trader {pair}: {str(e)}")
            raise
        """Exécute un trader pour une paire spécifique."""
        try:
            logger = self.logger.get_logger(f'trader_{pair}')
            logger.info(f"Démarrage du trader pour la paire {pair}")

            # Récupérer les données pour la paire
            ohlc_data = await get_price_history(self.api, self.config, pair)

            # Initialiser le trader
            trader = self.traders[pair]

            if self.config.mode == SimulationMode.HISTORICAL:
                # Mode historique
                price_history = ohlc_data['close']
                for i in range(len(price_history) - 1):
                    current_time = price_history.index[i]
                    current_price = price_history[i]
                    next_price = price_history[i + 1]

                    await process_trade(
                        trader,
                        current_time,
                        float(current_price),
                        float(next_price),
                        price_history[:i + 1]
                    )

            elif self.config.mode == SimulationMode.REAL_TIME:
                # Mode temps réel
                now = datetime.now()
                real_time_data = ohlc_data[ohlc_data.index >=
                                           now - self.config.real_time_window]

                for i in range(len(real_time_data) - 1):
                    current_time = real_time_data.index[i]
                    current_price = real_time_data['close'].iloc[i]
                    next_price = real_time_data['close'].iloc[i + 1]

                    await process_trade(
                        trader,
                        current_time,
                        float(current_price),
                        float(next_price),
                        real_time_data[:i + 1]
                    )

            elif self.config.mode == SimulationMode.BACKTEST:
                # Mode backtest avec prédiction adaptative
                train_data, val_data, test_data = self.config.get_data_splits(
                    ohlc_data)

                # Initialiser le système de prédiction adaptative
                adaptive_prediction = AdaptivePrediction(
                    base_predictor=self.config.ml_predictor,
                    technical_analyzer=self.config.technical_analyzer,
                    lookback_window=30,
                    update_frequency=10,
                    confidence_threshold=0.7
                )

                # Faire les prédictions adaptatives
                predictions = []

                for i in range(len(test_data) - 1):
                    context_data = test_data[:i + 1]
                    pred, _ = adaptive_prediction.predict(context_data)
                    predictions.append(pred)

                    if (i + 1) % adaptive_prediction.update_frequency == 0:
                        adaptive_prediction._update_model(context_data)

                # Trading avec les prédictions
                for i in range(len(test_data) - 1):
                    current_time = test_data.index[i]
                    current_price = test_data['close'].iloc[i]
                    next_price = predictions[i]

                    await process_trade(
                        trader,
                        current_time,
                        float(current_price),
                        float(next_price),
                        test_data[:i + 1]
                    )

            # Mettre à jour les métriques
            self.metrics[pair]['total_trades'] = trader.metrics['total_trades']
            self.metrics[pair]['total_profit'] = trader.metrics['total_profit']
            self.metrics[pair]['max_drawdown'] = trader.metrics['max_drawdown']
            self.metrics[pair]['current_balance'] = trader.current_balance

            logger.info(
                f"Trader {pair} terminé. Profit total: {trader.metrics['total_profit']:.2f} USD")

        except Exception as e:
            logger.error(f"Erreur dans le trader {pair}: {str(e)}")
            raise

    async def run(self) -> None:
        """Exécute tous les traders en parallèle avec rotation, clustering et gestion des risques."""
        try:
            logger = self.logger.get_logger('multi_pair_trader')
            logger.info(
                f"Démarrage des traders pour {len(self.traders)} paires")

            # Initialiser les traders
            await self._initialize_traders()

            # Analyser et clusteriser les paires
            pair_metrics = {
                pair: self.pair_metrics[pair] for pair in self.traders}
            self.pair_cluster.cluster_pairs(pair_metrics)

            # Vérifier si une rotation est nécessaire
            if self.rotation_strategy.should_rotate(self.metrics):
                logger.info("Rotation des paires nécessaire")
                new_pairs = await self.rotation_strategy.rotate_pairs(list(self.traders.keys()))

                # Mettre à jour les traders
                self._update_traders(new_pairs)

            # Exécuter les traders en parallèle
            futures = [
                self.executor.submit(
                    self._run_trader,
                    pair) for pair in self.traders.keys()]

            # Attendre la fin de tous les traders
            for future in futures:
                await future.result()

            # Calculer les métriques globales
            total_profit = sum(m['total_profit']
                               for m in self.metrics.values())
            total_trades = sum(m['total_trades']
                               for m in self.metrics.values())
            avg_drawdown = sum(m['max_drawdown']
                               for m in self.metrics.values()) / len(self.metrics)

            logger.info("\nRésultats globaux:")
            logger.info(f"Nombre total de trades: {total_trades}")
            logger.info(f"Profit total: {total_profit:.2f} USD")
            logger.info(f"Drawdown moyen: {avg_drawdown:.2%}")

            # Afficher les résultats par cluster
            logger.info("\nRésultats par cluster:")
            cluster_metrics = self.pair_cluster_analyzer.get_cluster_metrics()
            for cluster_id, metrics in cluster_metrics.items():
                logger.info(f"\nCluster {cluster_id}:")
                logger.info(f"Nombre de paires: {metrics['num_pairs']}")
                logger.info(
                    f"Volatilité moyenne: {metrics['avg_volatility']:.2%}")
                logger.info(f"Momentum moyen: {metrics['avg_momentum']:.2%}")
                logger.info(
                    f"Diversification: {metrics['diversification_score']:.2%}")

            # Afficher les résultats par paire
            logger.info("\nRésultats par paire:")
            for pair, metrics in self.metrics.items():
                logger.info(f"\n{pair}:")
                logger.info(f"Score: {self.pair_metrics[pair]['score']:.2f}")
                logger.info(f"Trades: {metrics['total_trades']}")
                logger.info(f"Profit: {metrics['total_profit']:.2f} USD")
                logger.info(f"Drawdown: {metrics['max_drawdown']:.2%}")
                logger.info(
                    f"Solde final: {metrics['current_balance']:.2f} USD")

        except Exception as e:
            logger.error(f"Erreur dans MultiPairTrader: {str(e)}")
            raise
        """Exécute tous les traders en parallèle avec rotation et gestion des tendances."""
        try:
            logger = self.logger.get_logger('multi_pair_trader')
            logger.info(
                f"Démarrage des traders pour {len(self.traders)} paires")

            # Initialiser les traders
            await self._initialize_traders()

            # Vérifier si une rotation est nécessaire
            if self.rotation_strategy.should_rotate(self.metrics):
                logger.info("Rotation des paires nécessaire")
                new_pairs = await self.rotation_strategy.rotate_pairs(list(self.traders.keys()))

                # Mettre à jour les traders
                self._update_traders(new_pairs)

            # Exécuter les traders en parallèle
            futures = [
                self.executor.submit(
                    self._run_trader,
                    pair) for pair in self.traders.keys()]

            # Attendre la fin de tous les traders
            for future in futures:
                await future.result()

            # Calculer les métriques globales
            total_profit = sum(m['total_profit']
                               for m in self.metrics.values())
            total_trades = sum(m['total_trades']
                               for m in self.metrics.values())
            avg_drawdown = sum(m['max_drawdown']
                               for m in self.metrics.values()) / len(self.metrics)

            logger.info("\nRésultats globaux:")
            logger.info(f"Nombre total de trades: {total_trades}")
            logger.info(f"Profit total: {total_profit:.2f} USD")
            logger.info(f"Drawdown moyen: {avg_drawdown:.2%}")

            # Afficher les résultats par paire
            logger.info("\nRésultats par paire:")
            for pair, metrics in self.metrics.items():
                logger.info(f"\n{pair}:")
                logger.info(f"Score: {self.pair_metrics[pair]['score']:.2f}")
                logger.info(f"Trades: {metrics['total_trades']}")
                logger.info(f"Profit: {metrics['total_profit']:.2f} USD")
                logger.info(f"Drawdown: {metrics['max_drawdown']:.2%}")
                logger.info(
                    f"Solde final: {metrics['current_balance']:.2f} USD")

        except Exception as e:
            logger.error(f"Erreur dans MultiPairTrader: {str(e)}")
            raise
        """Exécute tous les traders en parallèle avec rotation des paires."""
        try:
            logger = self.logger.get_logger('multi_pair_trader')
            logger.info(
                f"Démarrage des traders pour {len(self.traders)} paires")

            # Initialiser les traders
            await self._initialize_traders()

            # Exécuter les traders en parallèle
            futures = [
                self.executor.submit(
                    self._run_trader,
                    pair) for pair in self.traders.keys()]

            # Attendre la fin de tous les traders
            for future in futures:
                await future.result()

            # Calculer les métriques globales
            total_profit = sum(m['total_profit']
                               for m in self.metrics.values())
            total_trades = sum(m['total_trades']
                               for m in self.metrics.values())
            avg_drawdown = sum(m['max_drawdown']
                               for m in self.metrics.values()) / len(self.metrics)

            logger.info("\nRésultats globaux:")
            logger.info(f"Nombre total de trades: {total_trades}")
            logger.info(f"Profit total: {total_profit:.2f} USD")
            logger.info(f"Drawdown moyen: {avg_drawdown:.2%}")

            # Afficher les résultats par paire
            logger.info("\nRésultats par paire:")
            for pair, metrics in self.metrics.items():
                logger.info(f"\n{pair}:")
                logger.info(f"Score: {self.pair_metrics[pair]['score']:.2f}")
                logger.info(f"Trades: {metrics['total_trades']}")
                logger.info(f"Profit: {metrics['total_profit']:.2f} USD")
                logger.info(f"Drawdown: {metrics['max_drawdown']:.2%}")
                logger.info(
                    f"Solde final: {metrics['current_balance']:.2f} USD")

        except Exception as e:
            logger.error(f"Erreur dans MultiPairTrader: {str(e)}")
            raise
        try:
            logger = self.logger.get_logger('multi_pair_trader')
            logger.info(f"Démarrage des traders pour {len(self.pairs)} paires")

            # Exécuter les traders en parallèle
            futures = [
                self.executor.submit(
                    self._run_trader,
                    pair) for pair in self.pairs]

            # Attendre la fin de tous les traders
            for future in futures:
                await future.result()

            # Calculer les métriques globales
            total_profit = sum(m['total_profit']
                               for m in self.metrics.values())
            total_trades = sum(m['total_trades']
                               for m in self.metrics.values())
            avg_drawdown = sum(m['max_drawdown']
                               for m in self.metrics.values()) / len(self.metrics)

            logger.info("\nRésultats globaux:")
            logger.info(f"Nombre total de trades: {total_trades}")
            logger.info(f"Profit total: {total_profit:.2f} USD")
            logger.info(f"Drawdown moyen: {avg_drawdown:.2%}")

            # Afficher les résultats par paire
            logger.info("\nRésultats par paire:")
            for pair, metrics in self.metrics.items():
                logger.info(f"\n{pair}:")
                logger.info(f"Trades: {metrics['total_trades']}")
                logger.info(f"Profit: {metrics['total_profit']:.2f} USD")
                logger.info(f"Drawdown: {metrics['max_drawdown']:.2%}")
                logger.info(
                    f"Solde final: {metrics['current_balance']:.2f} USD")

        except Exception as e:
            logger.error(f"Erreur dans MultiPairTrader: {str(e)}")
            raise
