import asyncio
import logging
import os
import sys
from datetime import datetime, timedelta
from typing import Dict
from dotenv import load_dotenv
import pandas as pd

from src.core.types.types import TradingParameters

# Charger les variables d'environnement depuis .env
load_dotenv()

from src.core.api.kraken import KrakenAPI
from src.core.simulation_mode import SimulationMode, SimulationConfig
from src.core.trading_simulation import TradingSimulation
from src.core.ml_predictor import MLPredictor
from src.core.technical_analyzer import TechnicalAnalyzer
from src.core.adaptive_leverage import AdaptiveLeverage
from src.core.adaptive_prediction import AdaptivePrediction

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
        pair: Paire de trading (par défaut: 'BTC/USD')

    Returns:
        DataFrame avec les données OHLC contenant les colonnes:
        - open: Prix d'ouverture
        - high: Prix le plus haut
        - low: Prix le plus bas
        - close: Prix de clôture
        - vwap: Volume Weighted Average Price
        - volume: Volume échangé
        - count: Nombre de trades

    Raises:
        KeyError: Si la paire de trading n'est pas trouvée dans les résultats
        Exception: Pour toute autre erreur lors de la récupération des données
    """
    try:
        async with api:
            start_time, _ = config.get_time_range()
            start_timestamp = int(start_time.timestamp())

            # Récupérer les données OHLC
            ohlc_data = await api.get_ohlc_data(
                pair=pair,
                interval=1440,  # 1 jour
                since=start_timestamp
            )

            if not ohlc_data or 'result' not in ohlc_data:
                raise ValueError("Aucune donnée OHLC reçue de l'API")

            # Convertir en DataFrame
            result = ohlc_data['result']
            pair_key = next((k for k in ['XXBTZUSD', 'XBT/USD', 'BTC/USD'] 
                           if k in result), None)
            
            if not pair_key:
                available_pairs = list(result.keys())
                raise KeyError(
                    f"Paire de trading non trouvée. Paires disponibles: {available_pairs}")

            df = pd.DataFrame(result[pair_key], columns=[
                'time', 'open', 'high', 'low', 'close', 'vwap', 'volume', 'count'
            ])
            
            # Convertir les types
            numeric_cols = ['open', 'high', 'low', 'close', 'vwap', 'volume', 'count']
            for col in numeric_cols:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            df['time'] = pd.to_datetime(df['time'], unit='s')
            df.set_index('time', inplace=True)
            
            # Vérifier les données manquantes
            if df.isnull().values.any():
                logger.warning("Données manquantes détectées dans les données OHLC")
                df = df.ffill()  # Remplir les valeurs manquantes avec la dernière valeur valide

            return df

    except Exception as e:
        logger.error(
            f"Erreur lors de la récupération des données OHLC pour {pair}: {str(e)}")
        logger.debug("Détails de l'erreur:", exc_info=True)
        raise


async def run_simulation(
    api: KrakenAPI,
    config: SimulationConfig,
    pair: str = 'BTC/USD'
) -> Dict[str, float]:
    """
    Exécute une simulation de trading selon le mode spécifié.

    Args:
        api: Instance de l'API Kraken
        config: Configuration de la simulation
        pair: Paire de trading à utiliser (par défaut: 'BTC/USD')

    Returns:
        Dictionnaire contenant les métriques de performance de la simulation

    Raises:
        ValueError: Si les paramètres de configuration sont invalides
        RuntimeError: Si une erreur survient pendant l'exécution de la simulation
    """
    try:
        logger.info(f"Démarrage de la simulation en mode {config.mode.value}...")

        # Validation des paramètres
        if not config.ml_predictor or not config.technical_analyzer:
            raise ValueError("Predictor et technical_analyzer doivent être configurés")

        # Récupérer les données OHLC
        ohlc_data = await get_price_history(api, config, pair)
        
        if ohlc_data.empty:
            raise ValueError("Aucune donnée disponible pour la simulation")

        # Initialiser les paramètres de trading
        trading_params = await _initialize_trading_parameters(api, config)
        simulation = TradingSimulation(trading_params)
        
        # Initialiser le solde
        await _initialize_simulation_balance(simulation, api)

        # Exécuter la stratégie selon le mode
        if config.mode == SimulationMode.HISTORICAL:
            await _run_historical_simulation(simulation, ohlc_data)
            
        elif config.mode == SimulationMode.REAL_TIME:
            await _run_real_time_simulation(simulation, ohlc_data, config.real_time_window)
            
        elif config.mode == SimulationMode.BACKTEST:
            await _run_backtest_simulation(simulation, ohlc_data, config)
            
        elif config.mode == SimulationMode.HYBRID:
            await _run_hybrid_simulation(simulation, ohlc_data, config)
            
        else:
            raise ValueError(f"Mode de simulation non supporté: {config.mode}")

        # Afficher et retourner les résultats
        return _log_and_return_results(simulation)

    except Exception as e:
        logger.error(f"Erreur lors de la simulation: {str(e)}")
        logger.debug("Détails de l'erreur:", exc_info=True)
        raise RuntimeError(f"Échec de la simulation: {str(e)}") from e


async def _initialize_trading_parameters(api: KrakenAPI, config: SimulationConfig) -> TradingParameters:
    """Initialise les paramètres de trading."""
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
    return trading_params


async def _initialize_simulation_balance(simulation: TradingSimulation, api: KrakenAPI) -> None:
    """Initialise le solde de la simulation."""
    try:
        balance_data = await api.get_account_balance()
        initial_balance = float(balance_data.get('ZUSD', 10000))  # 10,000 USD par défaut si non trouvé
        simulation.current_balance = initial_balance
        logger.info(f"Solde initial: {initial_balance:.2f} USD")
    except Exception as e:
        logger.warning(f"Impossible de récupérer le solde: {str(e)}. Utilisation de 10,000 USD par défaut")
        simulation.current_balance = 10000.0


async def _run_historical_simulation(simulation: TradingSimulation, ohlc_data: pd.DataFrame) -> None:
    """Exécute une simulation en mode historique."""
    price_history = ohlc_data['close']
    for i in range(len(price_history) - 1):
        current_time = price_history.index[i]
        current_price = price_history.iloc[i]
        next_price = price_history.iloc[i + 1]

        await process_trade(
            simulation,
            current_time,
            current_price,
            next_price,
            price_history.iloc[:i + 1]
        )


async def _run_real_time_simulation(
    simulation: TradingSimulation, 
    ohlc_data: pd.DataFrame, 
    real_time_window: timedelta
) -> None:
    """Exécute une simulation en temps réel."""
    now = datetime.now()
    real_time_data = ohlc_data[ohlc_data.index >= (now - real_time_window)]

    for i in range(len(real_time_data) - 1):
        current_time = real_time_data.index[i]
        current_price = real_time_data['close'].iloc[i]
        next_price = real_time_data['close'].iloc[i + 1]

        await process_trade(
            simulation,
            current_time,
            current_price,
            next_price,
            real_time_data.iloc[:i + 1]
        )


async def _run_backtest_simulation(
    simulation: TradingSimulation, 
    ohlc_data: pd.DataFrame,
    config: SimulationConfig
) -> None:
    """Exécute une simulation en mode backtest avec prédictions."""
    train_data, _, test_data = config.get_data_splits(ohlc_data)
    
    # Entraîner le modèle sur les données d'entraînement
    config.ml_predictor.train(train_data)
    
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
    for i in range(len(test_data) - 1):
        context_data = test_data.iloc[:i + 1]
        pred, _ = adaptive_prediction.predict(context_data)
        predictions.append(pred)

        # Mettre à jour le modèle périodiquement
        if (i + 1) % adaptive_prediction.update_frequency == 0:
            adaptive_prediction._update_model(context_data)

    # Exécuter les trades basés sur les prédictions
    for i in range(len(test_data) - 1):
        current_time = test_data.index[i]
        current_price = test_data['close'].iloc[i]
        next_price = predictions[i]

        await process_trade(
            simulation,
            current_time,
            current_price,
            next_price,
            test_data.iloc[:i + 1]
        )


async def _run_hybrid_simulation(
    simulation: TradingSimulation, 
    ohlc_data: pd.DataFrame,
    config: SimulationConfig
) -> None:
    """Exécute une simulation hybride combinant historique et prédictions."""
    train_data, _, test_data = config.get_data_splits(ohlc_data)
    
    # Entraîner le modèle sur les données d'entraînement
    config.ml_predictor.train(train_data)
    
    # Exécuter le backtest sur les données de test
    await _run_backtest_simulation(simulation, ohlc_data, config)
    
    # Générer des prédictions futures
    future_predictions = []
    context_data = ohlc_data.tail(30).copy()  # Utiliser les 30 derniers jours comme contexte
    
    for _ in range(config.prediction_window):
        pred, _ = config.ml_predictor.predict(context_data)
        future_predictions.append(pred)
        
        # Mettre à jour les données de contexte avec la prédiction
        next_time = context_data.index[-1] + pd.Timedelta(days=1)
        new_row = {
            'open': context_data['close'].iloc[-1],
            'high': pred,
            'low': pred,
            'close': pred,
            'volume': context_data['volume'].iloc[-1],
            'vwap': pred
        }
        context_data = pd.concat([
            context_data, 
            pd.DataFrame([new_row], index=[next_time])
        ])
    
    # Exécuter les trades sur les prédictions futures
    future_data = pd.DataFrame({
        'close': future_predictions,
        'open': [future_predictions[0]] + future_predictions[:-1],
        'high': future_predictions,
        'low': future_predictions,
        'volume': [0] * len(future_predictions),  # Volume inconnu pour les prédictions
        'vwap': future_predictions
    }, index=pd.date_range(
        start=ohlc_data.index[-1] + pd.Timedelta(days=1),
        periods=len(future_predictions)
    ))
    
    for i in range(len(future_data) - 1):
        current_time = future_data.index[i]
        current_price = future_data['close'].iloc[i]
        next_price = future_data['close'].iloc[i + 1]

        await process_trade(
            simulation,
            current_time,
            current_price,
            next_price,
            future_data.iloc[:i + 1]
        )


def _log_and_return_results(simulation: TradingSimulation) -> Dict[str, float]:
    """Affiche et retourne les résultats de la simulation."""
    results = {
        'total_trades': simulation.metrics.get('total_trades', 0),
        'total_profit': simulation.metrics.get('total_profit', 0.0),
        'max_drawdown': simulation.metrics.get('max_drawdown', 0.0),
        'final_balance': simulation.current_balance
    }
    
    logger.info("\nRésultats de la simulation:")
    logger.info(f"Nombre total de trades: {results['total_trades']}")
    logger.info(f"Profit total: {results['total_profit']:.2f} USD")
    logger.info(f"Drawdown maximum: {results['max_drawdown']:.2%}")
    logger.info(f"Solde final: {results['final_balance']:.2f} USD")
    
    return results


async def process_trade(
    simulation: TradingSimulation,
    current_time: datetime,
    current_price: float,
    next_price: float,
    price_history: pd.DataFrame
) -> None:
    """
    Traite un trade individuel en ouvrant et en fermant une position.

    Args:
        simulation: Instance de la simulation de trading
        current_time: Horodatage du trade
        current_price: Prix actuel pour ouvrir la position
        next_price: Prix suivant pour fermer la position
        price_history: Données historiques des prix utilisées pour l'analyse

    Raises:
        ValueError: Si les paramètres sont invalides
        RuntimeError: Si une erreur survient pendant le traitement du trade
    """
    if not simulation or not isinstance(simulation, TradingSimulation):
        raise ValueError("Instance de simulation invalide")
        
    if current_price <= 0 or next_price <= 0:
        raise ValueError(f"Prix invalide: current={current_price}, next={next_price}")
    
    if price_history.empty or not isinstance(price_history, (pd.DataFrame, pd.Series)):
        raise ValueError("Historique des prix invalide ou vide")
    
    try:
        # Calculer la taille de position
        position_size = await simulation.calculate_position_size(
            entry_price=current_price,
            price_history=price_history
        )
        
        if position_size <= 0:
            logger.warning(f"Taille de position non valide: {position_size}. Trade ignoré.")
            return

        # Simuler l'ouverture d'une position
        position = await simulation.open_position(
            pair="BTC/USD",
            entry_price=current_price,
            is_long=True,
            price_history=price_history
        )
        
        if not position:
            logger.warning("Échec de l'ouverture de la position. Trade ignoré.")
            return

        # Simuler la fermeture de la position
        pnl = await simulation.close_position(position, next_price)
        
        # Mettre à jour les métriques
        simulation.metrics['total_trades'] = simulation.metrics.get('total_trades', 0) + 1
        simulation.metrics['total_profit'] = simulation.metrics.get('total_profit', 0.0) + pnl
        
        # Mettre à jour le drawdown maximum
        current_balance = simulation.current_balance
        peak_balance = simulation.metrics.get('peak_balance', current_balance)
        drawdown = (peak_balance - current_balance) / peak_balance if peak_balance > 0 else 0.0
        
        simulation.metrics['peak_balance'] = max(peak_balance, current_balance)
        simulation.metrics['max_drawdown'] = max(
            simulation.metrics.get('max_drawdown', 0.0),
            drawdown
        )

        # Journaliser les résultats
        logger.debug(
            f"Trade {simulation.metrics['total_trades']} à {current_time}: "
            f"Prix={current_price:.2f}, Position={position_size:.4f}, PNL={pnl:.2f} USD, "
            f"Drawdown={drawdown:.2%}, Solde={current_balance:.2f} USD"
        )
        
    except Exception as e:
        logger.error(f"Erreur lors du traitement du trade à {current_time}: {str(e)}")
        logger.debug("Détails de l'erreur:", exc_info=True)
        raise RuntimeError(f"Échec du traitement du trade: {str(e)}") from e


def get_simulation_mode() -> SimulationMode:
    """
    Récupère le mode de simulation depuis les variables d'environnement.
    
    Returns:
        SimulationMode: Le mode de simulation configuré (REAL_TIME par défaut)
    """
    mode_str = os.getenv('TRADING_MODE', 'real').upper()
    
    # Mappage des valeurs de TRADING_MODE vers SimulationMode
    mode_mapping = {
        'REAL': SimulationMode.REAL_TIME,
        'REAL_TIME': SimulationMode.REAL_TIME,
        'HISTORICAL': SimulationMode.HISTORICAL,
        'BACKTEST': SimulationMode.BACKTEST,
        'HYBRID': SimulationMode.HYBRID
    }
    
    # Retourner le mode correspondant ou REAL_TIME par défaut
    return mode_mapping.get(mode_str, SimulationMode.REAL_TIME)


async def main() -> None:
    """
    Point d'entrée principal de l'application de simulation de trading.
    
    Configure et exécute la simulation en fonction des paramètres d'environnement.
    """
    try:
        # Configuration du logging
        logging.basicConfig(
            level=os.getenv('LOG_LEVEL', 'INFO'),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler('trading_simulation.log')
            ]
        )
        
        # Désactiver les logs verbeux des bibliothèques tierces
        logging.getLogger('urllib3').setLevel(logging.WARNING)
        logging.getLogger('asyncio').setLevel(logging.WARNING)
        
        logger.info("Démarrage de l'application de simulation de trading...")

        # Vérification des variables d'environnement requises
        api_key = os.getenv('KRAKEN_API_KEY')
        api_secret = os.getenv('KRAKEN_API_SECRET')
        
        if not api_key or not api_secret:
            logger.error("Les variables d'environnement KRAKEN_API_KEY et KRAKEN_API_SECRET sont requises")
            sys.exit(1)

        # Configuration de la simulation
        simulation_mode = get_simulation_mode()
        logger.info(f"Mode de simulation: {simulation_mode.value}")
        
        config = SimulationConfig(
            mode=simulation_mode,
            historical_days=int(os.getenv('HISTORICAL_DAYS', '90')),
            real_time_window=timedelta(hours=int(os.getenv('REAL_TIME_WINDOW_HOURS', '24'))),
            prediction_window=int(os.getenv('PREDICTION_WINDOW', '7')),
            ml_predictor=MLPredictor(),
            technical_analyzer=TechnicalAnalyzer()
        )

        # Initialisation de l'API Kraken
        logger.info("Initialisation de l'API Kraken...")
        api = KrakenAPI(api_key=api_key, private_key=api_secret)

        # Exécution de la simulation
        async with api:
            logger.info("Démarrage de la simulation...")
            results = await run_simulation(api, config)
            logger.info("Simulation terminée avec succès")
            return results
            
    except Exception as e:
        logger.critical(f"Erreur critique lors de l'exécution: {str(e)}", exc_info=True)
        sys.exit(1)
    finally:
        logging.shutdown()


if __name__ == "__main__":
    import asyncio
    
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Arrêt demandé par l'utilisateur")
        sys.exit(0)
    except Exception as e:
        logger.critical(f"Erreur inattendue: {str(e)}", exc_info=True)
        sys.exit(1)
