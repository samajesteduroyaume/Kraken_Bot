import os
import sys
from datetime import datetime, timedelta
import pandas as pd
from src.core.api.kraken import KrakenAPI
from src.core.trading_simulation import TradingSimulation, TradingParameters
from src.core.adaptive_leverage import AdaptiveLeverage
from src.core.ml_predictor import MLPredictor
from src.core.technical_analyzer import TechnicalAnalyzer
from src.core.simulation_mode import SimulationMode, SimulationConfig
from src.core.adaptive_prediction import AdaptivePrediction
from src.core.services.multi_pair_trader import MultiPairTrader
import logging

logger = logging.getLogger('monthly_simulation')


async def get_price_history(
    api: KrakenAPI,
    config: SimulationConfig,
    pair: str = 'BTC/USD'
) -> pd.DataFrame:
    """
    Récupère les données OHLC selon le mode de simulation.

    Args:
        api: Instance de l'API Kraken
        config: Configuration de la simulation
        pair: Paire de trading

    Returns:
        DataFrame avec les données OHLC
    """
    try:
        async with api:
            start_time, end_time = config.get_time_range()

            # Convertir les dates en timestamp
            start_timestamp = int(start_time.timestamp())
            int(end_time.timestamp())

            # Récupérer les données OHLC
            ohlc_data = await api.get_ohlc_data(
                pair=pair,
                interval=1440,  # 1 jour
                since=start_timestamp
            )

            # Convertir en DataFrame
            result = ohlc_data['result']
            if 'XXBTZUSD' in result:
                df = pd.DataFrame(result['XXBTZUSD'])
            elif 'XBT/USD' in result:
                df = pd.DataFrame(result['XBT/USD'])
            elif 'BTC/USD' in result:
                df = pd.DataFrame(result['BTC/USD'])
            else:
                raise KeyError(
                    f"Paire de trading non trouvée dans les résultats: {result.keys()}")

            df.columns = [
                'time',
                'open',
                'high',
                'low',
                'close',
                'vwap',
                'volume',
                'count']
            df['time'] = pd.to_datetime(df['time'], unit='s')
            df.set_index('time', inplace=True)

            return df

    except Exception as e:
        logger.error(
            f"Erreur lors de la récupération des données OHLC: {str(e)}")
        raise


async def run_simulation(
    api: KrakenAPI,
    config: SimulationConfig
):
    """
    Exécute une simulation sur le mois en cours.

    Args:
        api: Instance de l'API Kraken
    """
    try:
        logger.info(
            f"Démarrage de la simulation en mode {config.mode.value}...")

        # Récupérer les données OHLC
        ohlc_data = await get_price_history(api, config, pair)

        # Initialiser les paramètres de trading
        trading_params = TradingParameters(
            api=api,
            risk_per_trade=0.05,  # 5% de risque
            adaptive_leverage=AdaptiveLeverage(
                min_leverage=1.0,
                max_leverage=5.0,
                volatility_threshold=0.02,
                risk_factor=0.05,
                volatility_window=20,
                trend_threshold=0.01,
                market_sentiment='neutral'
            ),
            stop_loss_percent=0.03,
            take_profit_percent=0.06,
            max_positions=10,
            max_drawdown=0.2,
            max_holding_time=timedelta(days=3),
            ml_predictor=config.ml_predictor,
            technical_analyzer=config.technical_analyzer
        )
        await trading_params.__post_init__()

        # Initialiser la simulation
        simulation = TradingSimulation(trading_params)

        # Initialiser le solde
        initial_balance = api.get_account_balance().get('ZUSD', 0)
        simulation.current_balance = float(initial_balance)

        logger.info(f"Solde initial: {initial_balance:.2f} USD")

        # Gérer les données selon le mode de simulation
        if config.mode == SimulationMode.HISTORICAL:
            # Mode historique - utilise uniquement les données passées
            price_history = ohlc_data['close']
            for i in range(len(price_history) - 1):
                current_time = price_history.index[i]
                current_price = price_history[i]
                next_price = price_history[i + 1]

                await process_trade(
                    simulation,
                    current_time,
                    current_price,
                    next_price,
                    price_history[:i + 1]
                )

        elif config.mode == SimulationMode.REAL_TIME:
            # Mode temps réel - utilise les données récentes
            now = datetime.now()
            real_time_data = ohlc_data[ohlc_data.index >=
                                       now - config.real_time_window]

            for i in range(len(real_time_data) - 1):
                current_time = real_time_data.index[i]
                current_price = real_time_data['close'].iloc[i]
                next_price = real_time_data['close'].iloc[i + 1]

                await process_trade(
                    simulation,
                    current_time,
                    current_price,
                    next_price,
                    real_time_data[:i + 1]
                )

        elif config.mode == SimulationMode.BACKTEST:
            # Mode backtest - utilise données historiques et prédictions
            train_data, val_data, test_data = config.get_data_splits(ohlc_data)

            # Initialiser le système de prédiction adaptative
            adaptive_prediction = AdaptivePrediction(
                base_predictor=config.ml_predictor,
                technical_analyzer=config.technical_analyzer,
                lookback_window=30,
                update_frequency=10,
                confidence_threshold=0.7
            )

            # Faire les prédictions adaptatives
            predictions = []
            confidences = []

            # Pour chaque point de données
            for i in range(len(test_data) - 1):
                # Préparer les données de contexte
                context_data = test_data[:i + 1]

                # Faire la prédiction adaptative
                pred, conf = adaptive_prediction.predict(context_data)
                predictions.append(pred)
                confidences.append(conf)

                # Mettre à jour le modèle si nécessaire
                if (i + 1) % adaptive_prediction.update_frequency == 0:
                    adaptive_prediction._update_model(context_data)

            # Ajouter les métriques de confiance
            config.ml_predictor.metrics = adaptive_prediction.get_metrics()

            for i in range(len(test_data) - 1):
                current_time = test_data.index[i]
                current_price = test_data['close'].iloc[i]
                next_price = predictions[i]

                await process_trade(
                    simulation,
                    current_time,
                    current_price,
                    next_price,
                    test_data[:i + 1]
                )

        elif config.mode == SimulationMode.HYBRID:
            # Mode hybride - mélange de données historiques et temps réel
            train_data, val_data, test_data = config.get_data_splits(ohlc_data)

            # Entraîner le modèle ML
            config.ml_predictor.train(train_data)

            # Prédictions futures adaptatives
            future_predictions = []
            future_confidences = []

            # Pour chaque point futur
            for i in range(config.prediction_window):
                # Préparer les données de contexte
                # Utiliser les 30 derniers jours
                context_data = ohlc_data.tail(30)

                # Faire la prédiction adaptative
                pred, conf = adaptive_prediction.predict(context_data)
                future_predictions.append(pred)
                future_confidences.append(conf)

                # Ajouter la prédiction aux données pour la prochaine itération
                next_time = context_data.index[-1] + timedelta(days=1)
                new_data = pd.DataFrame({
                    'close': [pred],
                    'open': [context_data['close'].iloc[-1]],
                    'high': [pred],
                    'low': [pred],
                    'volume': [context_data['volume'].iloc[-1]],
                    'vwap': [pred]
                }, index=[next_time])
                context_data = pd.concat([context_data, new_data])

            # Trading sur les données historiques
            for i in range(len(test_data) - 1):
                current_time = test_data.index[i]
                current_price = test_data['close'].iloc[i]
                next_price = test_data['close'].iloc[i + 1]

                await process_trade(
                    simulation,
                    current_time,
                    current_price,
                    next_price,
                    test_data[:i + 1]
                )

            # Trading sur les données futures
            future_data = pd.DataFrame(future_predictions, columns=['close'])
            future_data.index = pd.date_range(
                start=ohlc_data.index[-1],
                periods=len(future_predictions),
                freq='D'
            )

            for i in range(len(future_data) - 1):
                current_time = future_data.index[i]
                current_price = future_data['close'].iloc[i]
                next_price = future_data['close'].iloc[i + 1]

                await process_trade(
                    simulation,
                    current_time,
                    current_price,
                    next_price,
                    future_data[:i + 1]
                )

        # Afficher les résultats finaux
        logger.info("\nRésultats de la simulation:")
        logger.info(
            f"Nombre total de trades: {simulation.metrics['total_trades']}")
        logger.info(
            f"Profit total: {simulation.metrics['total_profit']:.2f} USD")
        logger.info(
            f"Drawdown maximum: {simulation.metrics['max_drawdown']:.2%}")
        logger.info(f"Solde final: {simulation.current_balance:.2f} USD")

    except Exception as e:
        logger.error(f"Erreur lors de la simulation: {str(e)}")
        raise


async def process_trade(
    simulation: TradingSimulation,
    current_time: datetime,
    current_price: float,
    next_price: float,
    price_history: pd.Series
) -> None:
    """Traite un trade individuel."""
    # Calculer la taille de position
    position_size = await simulation.calculate_position_size(
        entry_price=current_price,
        price_history=price_history
    )

    # Simuler l'ouverture d'une position
    position = await simulation.open_position(
        pair="BTC/USD",
        entry_price=current_price,
        is_long=True,
        price_history=price_history
    )

    # Simuler la fermeture de la position
    pnl = await simulation.close_position(position, next_price)

    # Mettre à jour les métriques
    simulation.metrics['total_trades'] += 1
    simulation.metrics['total_profit'] += pnl

    # Afficher les résultats
    logger.info(
        f"{current_time}: Price={current_price:.2f}, Position={position_size:.2f}, PNL={pnl:.2f}")


async def main():
    """Point d'entrée principal."""
    try:
        # Configurer le logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )

        # Initialiser l'API Kraken
        api_key = os.getenv('KRAKEN_API_KEY')
        api_secret = os.getenv('KRAKEN_API_SECRET')

        if not api_key or not api_secret:
            logger.error("Les clés API Kraken ne sont pas configurées")
            sys.exit(1)

        api = KrakenAPI(api_key=api_key, private_key=api_secret)

        # Gérer la session async
        async with api:
            # Exécuter la simulation
            await run_monthly_simulation(api)

    except Exception as e:
        logger.error(f"Erreur lors de l'exécution: {str(e)}")
        raise

if __name__ == "__main__":
    import asyncio

    # Configuration de la simulation
    config = SimulationConfig(
        mode=SimulationMode.HYBRID,
        historical_days=90,
        real_time_window=timedelta(hours=24),
        prediction_window=7,
        ml_predictor=MLPredictor(),
        technical_analyzer=TechnicalAnalyzer()
    )

    # Initialiser l'API Kraken
    api = KrakenAPI()

    # Initialiser et lancer le MultiPairTrader
    trader = MultiPairTrader(
        api=api,
        config=config,
        max_concurrent_trades=3,  # Limite le nombre de trades simultanés
        max_trading_pairs=50,     # Nombre maximum de paires à trader
        min_pair_score=0.5,       # Score minimum pour trader une paire
        risk_per_pair=0.1         # 10% du capital par paire
    )

    # Initialiser le trader et lancer la simulation
    async def main():
        try:
            # Initialiser le trader
            await trader.initialize()

            # Vérifier l'initialisation
            if not trader._initialized:
                logger.error("Échec de l'initialisation du trader")
                return

            # Vérifier les paires analysées
            if not trader.pair_selector.analysis_results:
                logger.error("Aucune paire n'a été analysée")
                return

            # Vérifier les paires sélectionnées
            selected_pairs = trader.pair_selector.get_valid_pairs(
                min_score=trader.min_pair_score)
            if not selected_pairs:
                logger.error(
                    f"Aucune paire n'a été sélectionnée parmi {len(trader.pair_selector.analysis_results)} paires analysées")
                return

            logger.info(f"Paires sélectionnées: {selected_pairs}")

            # Lancer la simulation
            await trader.run()

        except Exception as e:
            logger.error(f"Erreur lors de la simulation: {str(e)}")

    asyncio.run(main())
