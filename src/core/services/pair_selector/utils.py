"""
Fonctions utilitaires pour le module PairSelector.

Ce module contient des fonctions utilitaires utilisées dans tout le module PairSelector,
y compris la gestion des erreurs, la journalisation, et d'autres fonctions d'aide.
"""
import asyncio
import functools
import hashlib
import inspect
import json
import logging
import os
import random
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Type, TypeVar, Union

import numpy as np
import pandas as pd
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
    before_sleep_log,
)

from . import constants
from .models import PairAnalysis

# Type variable générique pour les fonctions décorées
T = TypeVar('T')

# Configuration du logger
logger = logging.getLogger(__name__)


def setup_logging(log_level: str = 'INFO', log_file: Optional[str] = None) -> None:
    """Configure la journalisation pour le module PairSelector.
    
    Args:
        log_level: Niveau de journalisation (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Chemin vers le fichier de log (optionnel)
    """
    # Définir le format des logs
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    date_format = '%Y-%m-%d %H:%M:%S'
    
    # Configurer le niveau de log de base
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format=log_format,
        datefmt=date_format,
        handlers=[logging.StreamHandler()]
    )
    
    # Si un fichier de log est spécifié, ajouter un gestionnaire de fichier
    if log_file:
        # Créer le répertoire des logs s'il n'existe pas
        log_dir = os.path.dirname(log_file)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
            
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(getattr(logging, log_level.upper()))
        file_handler.setFormatter(logging.Formatter(log_format, datefmt=date_format))
        
        # Ajouter le gestionnaire au logger racine
        logging.getLogger('').addHandler(file_handler)
    
    logger.info("Journalisation configurée avec succès (niveau: %s)", log_level)


def retry_on_exception(
    max_retries: int = 3,
    initial_delay: float = 1.0,
    max_delay: float = 30.0,
    exponential_backoff: bool = True,
    retry_on: Optional[Tuple[Type[Exception], ...]] = None,
    logger: Optional[logging.Logger] = None
) -> Callable:
    """Décorateur pour réessayer une fonction en cas d'échec.
    
    Args:
        max_retries: Nombre maximum de tentatives
        initial_delay: Délai initial en secondes
        max_delay: Délai maximum en secondes
        exponential_backoff: Si True, utilise un backoff exponentiel
        retry_on: Tuple d'exceptions à intercepter (par défaut: Exception)
        logger: Logger pour enregistrer les tentatives
        
    Returns:
        La fonction décorée
    """
    if retry_on is None:
        retry_on = (Exception,)
    
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs) -> T:
            last_exception = None
            
            for attempt in range(max_retries + 1):
                try:
                    if asyncio.iscoroutinefunction(func):
                        return await func(*args, **kwargs)
                    else:
                        return func(*args, **kwargs)
                        
                except retry_on as e:
                    last_exception = e
                    
                    if attempt == max_retries:
                        break
                        
                    # Calculer le délai d'attente
                    if exponential_backoff:
                        delay = min(initial_delay * (2 ** attempt), max_delay)
                    else:
                        delay = initial_delay
                    
                    # Ajouter un peu de jitter
                    delay = random.uniform(0.8 * delay, 1.2 * delay)
                    
                    # Logger l'erreur
                    if logger is not None:
                        logger.warning(
                            "Tentative %s/%s échouée pour %s: %s. Nouvelle tentative dans %.1fs...",
                            attempt + 1,
                            max_retries,
                            func.__name__,
                            str(e),
                            delay,
                            exc_info=True
                        )
                    
                    # Attendre avant de réessayer
                    await asyncio.sleep(delay)
            
            # Si on arrive ici, toutes les tentatives ont échoué
            if logger is not None:
                logger.error(
                    "Toutes les %s tentatives pour %s ont échoué. Dernière erreur: %s",
                    max_retries + 1,
                    func.__name__,
                    str(last_exception) if last_exception else "Inconnue",
                    exc_info=last_exception
                )
            
            raise last_exception if last_exception else Exception("Échec inconnu")
            
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs) -> T:
            last_exception = None
            
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except retry_on as e:
                    last_exception = e
                    
                    if attempt == max_retries:
                        break
                        
                    # Calculer le délai d'attente
                    if exponential_backoff:
                        delay = min(initial_delay * (2 ** attempt), max_delay)
                    else:
                        delay = initial_delay
                    
                    # Ajouter un peu de jitter
                    delay = random.uniform(0.8 * delay, 1.2 * delay)
                    
                    # Logger l'erreur
                    if logger is not None:
                        logger.warning(
                            "Tentative %s/%s échouée pour %s: %s. Nouvelle tentative dans %.1fs...",
                            attempt + 1,
                            max_retries,
                            func.__name__,
                            str(e),
                            delay,
                            exc_info=True
                        )
                    
                    # Attendre avant de réessayer
                    time.sleep(delay)
            
            # Si on arrive ici, toutes les tentatives ont échoué
            if logger is not None:
                logger.error(
                    "Toutes les %s tentatives pour %s ont échoué. Dernière erreur: %s",
                    max_retries + 1,
                    func.__name__,
                    str(last_exception) if last_exception else "Inconnue",
                    exc_info=last_exception
                )
            
            raise last_exception if last_exception else Exception("Échec inconnu")
        
        # Retourner le wrapper approprié selon que la fonction est asynchrone ou non
        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
    
    return decorator


def time_execution(func: Callable[..., T]) -> Callable[..., T]:
    """Décorateur pour mesurer le temps d'exécution d'une fonction.
    
    Args:
        func: La fonction à décorer
        
    Returns:
        La fonction décorée
    """
    @functools.wraps(func)
    async def async_wrapper(*args, **kwargs):
        start_time = time.monotonic()
        try:
            return await func(*args, **kwargs)
        finally:
            duration = time.monotonic() - start_time
            logger.debug("%s a mis %.3f secondes à s'exécuter", func.__name__, duration)
    
    @functools.wraps(func)
    def sync_wrapper(*args, **kwargs):
        start_time = time.monotonic()
        try:
            return func(*args, **kwargs)
        finally:
            duration = time.monotonic() - start_time
            logger.debug("%s a mis %.3f secondes à s'exécuter", func.__name__, duration)
    
    return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper


def normalize_pair(pair: str) -> str:
    """Normalise le format d'une paire de trading.
    
    Args:
        pair: La paire à normaliser
        
    Returns:
        La paire normalisée
    """
    if not pair:
        return ""
    
    # Supprimer les espaces et convertir en majuscules
    pair = pair.strip().upper()
    
    # Remplacer les séparateurs par un slash
    for sep in ['-', '_', ' ']:
        pair = pair.replace(sep, '/')
    
    # Normaliser les formats courants
    if '/' in pair:
        base, quote = pair.split('/', 1)
        
        # Normaliser les préfixes/suffixes courants
        for prefix in ['X', 'Z', 'XX', 'ZUSD', 'ZUSDT', 'ZUSDC', 'ZDAI', 'ZEUR', 'ZJPY', 'ZGBP']:
            if base.startswith(prefix):
                base = base[len(prefix):]
            if quote.startswith(prefix):
                quote = quote[len(prefix):]
        
        # Normaliser les alias de devises
        base = constants.CURRENCY_ALIASES.get(base, base)
        quote = constants.CURRENCY_ALIASES.get(quote, quote)
        
        return f"{base}/{quote}"
    
    # Pour les paires sans séparateur, essayer de trouver une correspondance
    for common_pair in constants.POPULAR_PAIRS:
        base, quote = common_pair
        if pair == f"{base}{quote}" or pair == f"X{base}Z{quote}" or pair == f"{base}Z{quote}":
            return f"{base}/{quote}"
    
    # Si aucune correspondance, retourner la paire telle quelle
    return pair


def generate_cache_key(*args, **kwargs) -> str:
    """Génère une clé de cache à partir des arguments d'une fonction.
    
    Args:
        *args: Arguments positionnels
        **kwargs: Arguments nommés
        
    Returns:
        Une chaîne de caractères représentant la clé de cache
    """
    # Convertir les arguments en une chaîne
    args_str = ",".join(str(arg) for arg in args)
    kwargs_str = ",".join(f"{k}={v}" for k, v in sorted(kwargs.items()))
    
    # Combiner les arguments
    combined = f"{args_str}|{kwargs_str}"
    
    # Créer un hachage MD5 de la chaîne combinée
    return hashlib.md5(combined.encode('utf-8')).hexdigest()


def parse_kraken_ohlc(data: Dict) -> pd.DataFrame:
    """Analyse les données OHLC de l'API Kraken et les convertit en DataFrame.
    
    Args:
        data: Données brutes de l'API Kraken
        
    Returns:
        Un DataFrame pandas avec les données OHLC
    """
    if not data or 'result' not in data or not data['result']:
        return pd.DataFrame()
    
    # Extraire les données de la première paire trouvée
    pair_data = next(iter(data['result'].values()))
    
    # Vérifier si les données sont valides
    if not pair_data or not isinstance(pair_data, list):
        return pd.DataFrame()
    
    # Convertir en DataFrame
    df = pd.DataFrame(
        pair_data,
        columns=['time', 'open', 'high', 'low', 'close', 'vwap', 'volume', 'count']
    )
    
    # Convertir les types de données
    numeric_cols = ['open', 'high', 'low', 'close', 'vwap', 'volume']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Convertir le timestamp en datetime
    df['time'] = pd.to_datetime(df['time'], unit='s')
    df.set_index('time', inplace=True)
    
    # Supprimer les doublons et les valeurs manquantes
    df = df[~df.index.duplicated(keep='first')]
    df = df.dropna()
    
    return df


def calculate_technical_indicators(df: pd.DataFrame) -> Dict[str, float]:
    """Calcule des indicateurs techniques à partir d'un DataFrame OHLC.
    
    Args:
        df: DataFrame avec les colonnes 'open', 'high', 'low', 'close', 'volume'
        
    Returns:
        Un dictionnaire avec les indicateurs calculés
    """
    if df.empty or len(df) < 2:
        return {}
    
    indicators = {}
    
    try:
        close = df['close'].values
        high = df['high'].values
        low = df['low'].values
        volume = df['volume'].values
        
        # Calculer les rendements
        returns = np.diff(np.log(close))
        
        # Volatilité (écart-type des rendements annualisé)
        if len(returns) > 1:
            indicators['volatility'] = float(np.std(returns, ddof=1) * np.sqrt(252))
        else:
            indicators['volatility'] = 0.0
        
        # Momentum (variation sur 10 périodes)
        lookback = min(10, len(close) - 1)
        if lookback > 0:
            momentum = ((close[-1] - close[-lookback-1]) / close[-lookback-1]) * 100
            indicators['momentum'] = float(momentum)
        else:
            indicators['momentum'] = 0.0
        
        # Volume moyen (sur 20 périodes)
        volume_window = min(20, len(volume))
        if volume_window > 0:
            indicators['volume_ma'] = float(np.mean(volume[-volume_window:]))
        else:
            indicators['volume_ma'] = 0.0
        
        # Spread moyen (écart relatif haut/bas en %)
        spread_pct = (high - low) / ((high + low) / 2) * 100
        indicators['avg_spread'] = float(np.mean(spread_pct))
        
        # RSI (14 périodes)
        if len(close) >= 15:  # Besoin d'au moins 15 points pour RSI(14)
            deltas = np.diff(close)
            gains = np.where(deltas > 0, deltas, 0)
            pertes = np.where(deltas < 0, -deltas, 0)
            
            # Moyenne mobile des gains et pertes
            avg_gain = np.mean(gains[:14])
            avg_loss = np.mean(pertes[:14])
            
            # Calcul du RS et RSI
            if avg_loss != 0:
                rs = avg_gain / avg_loss
                indicators['rsi'] = float(100 - (100 / (1 + rs)))
            else:
                indicators['rsi'] = 100.0  # Si pas de perte, RSI = 100
        else:
            indicators['rsi'] = 50.0  # Valeur neutre si pas assez de données
        
        # ATR (Average True Range) - 14 périodes
        if len(close) >= 15:
            high = np.asarray(high, dtype=np.float64)
            low = np.asarray(low, dtype=np.float64)
            close = np.asarray(close, dtype=np.float64)
            
            # Calcul du True Range
            tr1 = high[1:] - low[1:]
            tr2 = np.abs(high[1:] - close[:-1])
            tr3 = np.abs(low[1:] - close[:-1])
            
            true_range = np.maximum(np.maximum(tr1, tr2), tr3)
            
            # Moyenne mobile simple sur 14 périodes
            atr = np.mean(true_range[-14:]) if len(true_range) >= 14 else np.mean(true_range)
            indicators['atr'] = float(atr)
        else:
            indicators['atr'] = 0.0
        
        # ADX (Average Directional Index) - 14 périodes
        if len(close) >= 28:  # Besoin de plus de données pour ADX
            # Calcul des mouvements directionnels
            up = high[1:] - high[:-1]
            down = low[:-1] - low[1:]
            
            plus_dm = np.where((up > down) & (up > 0), up, 0.0)
            minus_dm = np.where((down > up) & (down > 0), down, 0.0)
            
            # Lissage des DM avec une moyenne mobile
            def smooth(series, period):
                return np.convolve(
                    series, 
                    np.ones(period) / period, 
                    mode='valid'
                )
            
            # Calcul de l'ADX
            tr = true_range[-28:]
            plus_di = 100 * smooth(plus_dm[-28:], 14) / smooth(tr, 14)
            minus_di = 100 * smooth(minus_dm[-28:], 14) / smooth(tr, 14)
            
            dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)  # Éviter la division par zéro
            adx = np.mean(dx[-14:]) if len(dx) >= 14 else np.mean(dx)
            
            indicators['adx'] = float(adx)
            indicators['plus_di'] = float(plus_di[-1] if len(plus_di) > 0 else 0)
            indicators['minus_di'] = float(minus_di[-1] if len(minus_di) > 0 else 0)
        else:
            indicators['adx'] = 0.0
            indicators['plus_di'] = 0.0
            indicators['minus_di'] = 0.0
        
    except Exception as e:
        logger.error("Erreur lors du calcul des indicateurs techniques: %s", str(e), exc_info=True)
    
    return indicators


def calculate_market_metrics(df: pd.DataFrame, indicators: Dict[str, float]) -> Dict[str, float]:
    """Calcule des métriques de marché à partir d'un DataFrame OHLC et d'indicateurs.
    
    Args:
        df: DataFrame avec les colonnes 'open', 'high', 'low', 'close', 'volume'
        indicators: Dictionnaire d'indicateurs techniques précalculés
        
    Returns:
        Un dictionnaire avec les métriques calculées
    """
    metrics = {}
    
    try:
        close = df['close'].values
        high = df['high'].values
        low = df['low'].values
        volume = df['volume'].values
        
        # Dernier prix
        metrics['last_price'] = float(close[-1]) if len(close) > 0 else 0.0
        
        # Variation sur différentes périodes
        periods = [1, 7, 30]  # 1 jour, 1 semaine, 1 mois
        for period in periods:
            if len(close) > period:
                change = ((close[-1] - close[-period-1]) / close[-period-1]) * 100
                metrics[f'change_{period}d'] = float(change)
            else:
                metrics[f'change_{period}d'] = 0.0
        
        # Volume sur 24h
        metrics['volume_24h'] = float(np.sum(volume[-24:])) if len(volume) >= 24 else float(np.sum(volume))
        
        # Prix moyen pondéré par le volume (VWAP)
        if len(close) > 0 and np.sum(volume) > 0:
            vwap = np.sum(close * volume) / np.sum(volume)
            metrics['vwap'] = float(vwap)
        else:
            metrics['vwap'] = metrics.get('last_price', 0.0)
        
        # Spread moyen
        metrics['avg_spread'] = indicators.get('avg_spread', 0.0)
        
        # Volatilité historique
        metrics['volatility'] = indicators.get('volatility', 0.0)
        
        # Liquidité (volume * prix)
        metrics['liquidity'] = metrics['volume_24h'] * metrics['last_price']
        
        # Ratio de Sharpe (simplifié, sans taux sans risque)
        returns = np.diff(np.log(close))
        if len(returns) > 1 and np.std(returns) > 0:
            sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252)
            metrics['sharpe_ratio'] = float(sharpe)
        else:
            metrics['sharpe_ratio'] = 0.0
        
        # Drawdown maximum
        cum_returns = np.cumsum(returns)
        peak = np.maximum.accumulate(cum_returns)
        drawdown = (peak - cum_returns) / (peak + 1e-10)  # Éviter la division par zéro
        metrics['max_drawdown'] = float(np.max(drawdown)) if len(drawdown) > 0 else 0.0
        
    except Exception as e:
        logger.error("Erreur lors du calcul des métriques de marché: %s", str(e), exc_info=True)
    
    return metrics


def calculate_score(metrics: Dict[str, float], indicators: Dict[str, float]) -> float:
    """Calcule un score global pour une paire basé sur les métriques et indicateurs.
    
    Args:
        metrics: Dictionnaire des métriques de marché
        indicators: Dictionnaire des indicateurs techniques
        
    Returns:
        Un score entre 0 et 1
    """
    score = 0.5  # Score neutre par défaut
    
    try:
        # Utiliser les poids de la configuration ou les valeurs par défaut
        weights = constants.SCORE_WEIGHTS
        
        # Facteurs basés sur les métriques
        factors = {}
        
        # Liquidité (log pour réduire l'échelle)
        liquidity = metrics.get('liquidity', 0)
        factors['liquidity'] = min(np.log10(liquidity + 1) / 6.0, 1.0) if liquidity > 0 else 0.0
        
        # Volatilité (optimale autour de 0.5-1.5)
        volatility = metrics.get('volatility', 0)
        factors['volatility'] = 1.0 - min(abs(volatility - 1.0), 1.0)
        
        # Volume 24h (log pour réduire l'échelle)
        volume = metrics.get('volume_24h', 0)
        factors['volume_24h'] = min(np.log10(volume + 1) / 5.0, 1.0) if volume > 0 else 0.0
        
        # Spread moyen (moins c'est mieux, max à 2%)
        spread = metrics.get('avg_spread', 0)
        factors['avg_spread'] = max(0, 1.0 - (spread / 2.0)) if spread > 0 else 1.0
        
        # Ratio de Sharpe (normalisé entre 0 et 1)
        sharpe = metrics.get('sharpe_ratio', 0)
        factors['sharpe_ratio'] = min(max((sharpe + 2.0) / 4.0, 0.0), 1.0)
        
        # Drawdown maximum (moins c'est mieux)
        drawdown = metrics.get('max_drawdown', 0)
        factors['max_drawdown'] = 1.0 - min(drawdown, 1.0)
        
        # RSI (proche de 50 est préféré)
        rsi = indicators.get('rsi', 50)
        factors['rsi'] = 1.0 - (abs(rsi - 50) / 50.0)
        
        # ADX (plus c'est élevé, mieux c'est, max à 50)
        adx = indicators.get('adx', 0)
        factors['adx'] = min(adx / 50.0, 1.0)
        
        # Momentum (normalisé)
        momentum = indicators.get('momentum', 0)
        factors['momentum'] = min(max((momentum + 20.0) / 40.0, 0.0), 1.0)
        
        # Calcul du score pondéré
        total_weight = sum(abs(w) for w in weights.values())
        if total_weight > 0:
            score = sum(factors.get(k, 0.5) * weights[k] for k in weights) / total_weight
            score = max(0.0, min(1.0, score))  # Borne entre 0 et 1
        
    except Exception as e:
        logger.error("Erreur lors du calcul du score: %s", str(e), exc_info=True)
    
    return score


def format_currency(value: float, currency: str = 'USD') -> str:
    """Formate une valeur monétaire avec le symbole de la devise.
    
    Args:
        value: Valeur à formater
        currency: Code de la devise (par défaut: 'USD')
        
    Returns:
        Chaîne formatée (ex: "$1,234.56")
    """
    currency_symbols = {
        'USD': '$',
        'EUR': '€',
        'GBP': '£',
        'JPY': '¥',
        'BTC': '₿',
        'ETH': 'Ξ',
    }
    
    symbol = currency_symbols.get(currency, currency)
    
    # Formater le nombre avec des séparateurs de milliers et 2 décimales
    formatted_value = f"{value:,.2f}"
    
    # Supprimer les décimales pour les valeurs entières
    if formatted_value.endswith('.00'):
        formatted_value = formatted_value[:-3]
    
    return f"{symbol}{formatted_value}"
