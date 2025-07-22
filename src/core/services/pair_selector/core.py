"""
Module principal du PairSelector.

Ce module contient la classe principale PairSelector qui gère la sélection,
l'analyse et le filtrage des paires de trading.
"""
import asyncio
import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union, cast

import numpy as np
import pandas as pd

from . import constants, validators, normalizers, utils
from .validators import validate_pair_format as validate_pair_format_util
from .analyzers import PairAnalyzer
from .cache import CacheManager
from .models import PairAnalysis, CacheStats
from .types import TradingPair, TradingPairMetrics, PairInput
from src.utils.pair_utils import normalize_pair_input as robust_normalize_pair_input

# Configuration du logger
logger = logging.getLogger(__name__)

class PairSelector:
    """Classe principale pour la sélection et l'analyse des paires de trading.
    
    Cette classe fournit des méthodes pour récupérer, filtrer et analyser les paires
    de trading disponibles sur l'API Kraken, avec mise en cache des résultats.
    """
    
    def __init__(
        self,
        kraken_api: Any,
        config: Optional[Dict[str, Any]] = None,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        """Initialise le PairSelector avec une instance de l'API Kraken.
        
        Args:
            kraken_api: Instance de l'API Kraken (doit implémenter les méthodes nécessaires)
            config: Dictionnaire de configuration optionnel
            logger: Instance de logger optionnelle
        """
        self.kraken_api = kraken_api
        self.logger = logger or logging.getLogger(__name__)
        
        # Fusionner la configuration fournie avec les valeurs par défaut
        self.config = {**constants.DEFAULT_CONFIG, **(config or {})}
        
        # Initialiser le gestionnaire de cache
        cache_dir = Path(self.config.get('cache_dir', 'data/cache/pair_selector'))
        cache_ttl = self.config.get('cache_ttl', 3600)
        max_cache_size = self.config.get('max_cache_size', 100)
        
        self.cache = CacheManager(
            cache_dir=cache_dir,
            ttl=cache_ttl,
            max_size=max_cache_size
        )
        
        # Initialiser l'analyseur de paires
        self.analyzer = PairAnalyzer(config=self.config)
        
        # Utiliser l'instance globale de AvailablePairs
        from src.core.market.pair_initializer import get_available_pairs as get_available_pairs_instance
        self._available_pairs_instance = get_available_pairs_instance()
        
        # Initialiser le cache des paires disponibles
        self._available_pairs: Optional[Dict[str, Dict]] = None
        
        # Cache local pour les analyses de paires
        self._analysis_cache: Dict[str, PairAnalysis] = {}
        
        # Statistiques
        self.stats = {
            'api_calls': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'pairs_analyzed': 0,
            'errors': 0,
            'alternative_match': 0,  # Nombre de paires avec correspondance alternative
            'no_match': 0,          # Nombre de paires sans correspondance
            'start_time': time.time(),
        }
    
    async def initialize(self) -> None:
        """Initialise le PairSelector en chargeant les paires disponibles."""
        try:
            await self._load_available_pairs()
            self.logger.info("PairSelector initialisé avec succès")
        except Exception as e:
            self.logger.error("Erreur lors de l'initialisation du PairSelector: %s", str(e), exc_info=True)
            raise
            
    def _validate_pair_format(self, pair: str) -> bool:
        """Valide le format d'une paire de trading.
        
        Args:
            pair: La paire à valider (ex: 'XBT/USDT', 'XBT-USD', 'XBTUSDT')
            
        Returns:
            bool: True si le format est valide, False sinon
        """
        try:
            # Utiliser la fonction utilitaire du module validators
            return validate_pair_format_util(pair)
        except Exception as e:
            self.logger.warning(
                "Erreur lors de la validation du format de la paire '%s': %s",
                pair, str(e), exc_info=True
            )
            return False
    
    async def get_valid_pairs(
        self,
        min_volume: Optional[float] = None,
        min_liquidity: Optional[float] = None,
        max_spread: Optional[float] = None,
        min_volatility: Optional[float] = None,
        max_volatility: Optional[float] = None,
        min_score: Optional[float] = None,
        quote_currencies: Optional[List[str]] = None,
        exclude: Optional[List[str]] = None,
        limit: Optional[int] = None,
        force_refresh: bool = False,
    ) -> List[TradingPair]:
        """Récupère et filtre les paires de trading valides selon les critères spécifiés.
        
        Args:
            min_volume: Volume minimal sur 24h (en USD)
            min_liquidity: Liquidité minimale (volume * prix)
            max_spread: Spread maximal accepté (en %)
            min_volatility: Volatilité minimale (en %)
            max_volatility: Volatilité maximale (en %)
            min_score: Score minimal (0-1)
            quote_currencies: Liste des devises de cotation à inclure
            exclude: Liste des paires à exclure (peut être des noms de paires ou des objets avec clé 'pair')
            limit: Nombre maximum de paires à retourner
            force_refresh: Si True, force le rafraîchissement du cache
            
        Returns:
            Liste des paires valides avec leurs métriques, triées par score décroissant
            
        Raises:
            ValueError: Si une paire dans la liste d'exclusion est invalide
        """
        start_time = time.time()
        self.logger.info(
            "Récupération des paires valides avec les critères: min_volume=%s, min_liquidity=%s, "
            "max_spread=%s, min_volatility=%s, max_volatility=%s, min_score=%s, quote_currencies=%s",
            min_volume, min_liquidity, max_spread, min_volatility, max_volatility, 
            min_score, quote_currencies
        )
        
        try:
            # S'assurer que les paires sont chargées
            await self._load_available_pairs(force_refresh=force_refresh)
            
            if not hasattr(self, '_available_pairs') or not self._available_pairs:
                self.logger.error("Impossible de charger les paires disponibles")
                return []
            
            # Normaliser les devises de cotation en majuscules
            if quote_currencies:
                quote_currencies = [c.upper().replace('-', '/') for c in quote_currencies]
            
            # Normaliser la liste d'exclusion
            normalized_exclude = set()
            if exclude:
                for item in exclude:
                    try:
                        normalized = robust_normalize_pair_input(item)
                        normalized_exclude.add(normalized)
                    except ValueError as e:
                        self.logger.warning(f"Paire d'exclusion invalide ignorée: {item} - {e}")
            
            # Appliquer les filtres
            valid_pairs: List[TradingPair] = []
            
            for pair, pair_info in self._available_pairs.items():
                try:
                    # Normaliser le nom de la paire
                    normalized_pair = robust_normalize_pair_input(pair)
                    
                    # Exclure les paires dans la liste d'exclusion
                    if normalized_exclude and normalized_pair in normalized_exclude:
                        continue
                    
                    # Filtrer par devise de cotation si spécifié
                    quote = pair_info.get('quote', '').upper()
                    if quote_currencies and quote not in quote_currencies:
                        continue
                    
                    # Vérifier les conditions de filtrage
                    volume_24h = float(pair_info.get('volume_24h', 0))
                    if min_volume is not None and volume_24h < min_volume:
                        continue
                        
                    liquidity = float(pair_info.get('liquidity', 0))
                    if min_liquidity is not None and liquidity < min_liquidity:
                        continue
                        
                    spread = float(pair_info.get('avg_spread', float('inf')))
                    if max_spread is not None and spread > max_spread:
                        continue
                        
                    volatility = float(pair_info.get('volatility', 0))
                    if min_volatility is not None and volatility < min_volatility:
                        continue
                        
                    if max_volatility is not None and volatility > max_volatility:
                        continue
                        
                    score = float(pair_info.get('score', 0))
                    if min_score is not None and score < min_score:
                        continue
                    
                    # Créer un objet TradingPair typé
                    valid_pair: TradingPair = {
                        'pair': normalized_pair,
                        'base_currency': pair_info.get('base', ''),
                        'quote_currency': quote,
                        'min_size': float(pair_info.get('min_size', 0)),
                        'max_size': float(pair_info.get('max_size', 0)),
                        'step_size': float(pair_info.get('step_size', 0)),
                        'price_precision': int(pair_info.get('price_precision', 8)),
                        'min_volume_btc': float(pair_info.get('min_volume_btc', 0)),
                        'max_spread': spread,
                        'risk_level': pair_info.get('risk_level', 'medium'),
                        'enabled': bool(pair_info.get('enabled', True)),
                        'volume_24h': volume_24h,
                        'liquidity': liquidity,
                        'spread': spread,
                        'volatility': volatility,
                        'score': score,
                        'last_updated': time.time()
                    }
                    
                    valid_pairs.append(valid_pair)
                    
                except Exception as e:
                    self.logger.error(
                        "Erreur lors du filtrage de la paire %s: %s", 
                        pair, str(e), 
                        exc_info=True
                    )
                    self.stats['errors'] += 1
            
            # Trier par score décroissant
            valid_pairs.sort(key=lambda x: x.get('score', 0), reverse=True)
            
            # Limiter le nombre de résultats si nécessaire
            if limit is not None and limit > 0:
                valid_pairs = valid_pairs[:limit]
            
            self.logger.info(
                "%s paires valides trouvées sur %s (en %.2fs)", 
                len(valid_pairs), 
                len(self._available_pairs),
                time.time() - start_time
            )
            
            # S'assurer que toutes les paires sont dans le bon format
            return [{
                'pair': p['pair'],
                'base_currency': p['base_currency'],
                'quote_currency': p['quote_currency'],
                'min_size': p['min_size'],
                'max_size': p['max_size'],
                'step_size': p['step_size'],
                'price_precision': p['price_precision'],
                'min_volume_btc': p['min_volume_btc'],
                'max_spread': p['max_spread'],
                'risk_level': p['risk_level'],
                'enabled': p['enabled'],
                'volume_24h': p['volume_24h'],
                'liquidity': p['liquidity'],
                'spread': p['spread'],
                'volatility': p['volatility'],
                'score': p['score'],
                'last_updated': p['last_updated']
            } for p in valid_pairs]
            
        except Exception as e:
            self.logger.error(
                "Erreur lors de la récupération des paires valides: %s", 
                str(e), 
                exc_info=True
            )
            self.stats['errors'] += 1
            raise
    
    async def analyze_pair(self, pair: str, force_refresh: bool = False) -> Optional[PairAnalysis]:
        """Analyse une paire de trading spécifique.
        
        Args:
            pair: La paire à analyser (ex: 'XBT/USD')
            force_refresh: Si True, force une nouvelle analyse sans utiliser le cache
            
        Returns:
            Objet PairAnalysis avec les résultats de l'analyse, ou None en cas d'erreur
        """
        start_time = time.time()
        self.logger.debug("Analyse de la paire: %s", pair)
        
        # Valider le format de la paire
        if not self._validate_pair_format(pair):
            self.logger.warning("Format de paire invalide: %s", pair)
            return None
            
        try:
            # Vérifier si la paire est dans le cache et si on ne force pas le rafraîchissement
            if not force_refresh and pair in self._analysis_cache:
                cached_analysis = self._analysis_cache[pair]
                
                # Vérifier si l'analyse en cache est toujours valide
                if (time.time() - cached_analysis.timestamp) < self.config.get('cache_ttl', 3600):
                    self.stats['cache_hits'] += 1
                    self.logger.debug("Utilisation de l'analyse en cache pour %s", pair)
                    return cached_analysis
            
            # Récupérer les données OHLC pour la paire
            ohlc_data = await self._get_ohlc_data(pair)
            
            if not ohlc_data:
                self.logger.warning("Aucune donnée OHLC pour la paire %s", pair)
                return None
            
            # Analyser la paire
            analysis = await self.analyzer.analyze_pair(pair, ohlc_data)
            
            # Mettre à jour le cache
            self._analysis_cache[pair] = analysis
            
            # Mettre à jour les statistiques
            self.stats['pairs_analyzed'] += 1
            self.stats['cache_misses'] += 1
            
            self.logger.debug(
                "Analyse terminée pour %s (score: %.2f, en %.2fs)", 
                pair, 
                analysis.score,
                time.time() - start_time
            )
            
            return analysis
            
        except Exception as e:
            self.logger.error(
                "Erreur lors de l'analyse de la paire %s: %s", 
                pair, 
                str(e), 
                exc_info=self.logger.isEnabledFor(logging.DEBUG)
            )
            self.stats['errors'] += 1
            return None
    
    async def get_pair_info(self, pair: str) -> Optional[Dict[str, Any]]:
        """Récupère les informations détaillées pour une paire spécifique.
        
        Args:
            pair: La paire à récupérer (ex: 'XBT/USD')
            
        Returns:
            Dictionnaire avec les informations de la paire, ou None si non trouvée ou invalide
        """
        # Valider le format de la paire
        if not self._validate_pair_format(pair):
            self.logger.warning("Format de paire invalide: %s", pair)
            return None
            
        if not self._available_pairs:
            await self._load_available_pairs()
        
        return self._available_pairs.get(pair)
    
    def get_stats(self) -> Dict[str, Any]:
        """Retourne les statistiques d'utilisation du PairSelector.
        
        Returns:
            Dictionnaire avec les statistiques
        """
        return {
            **self.stats,
            'uptime': time.time() - self.stats['start_time'],
            'available_pairs': len(self._available_pairs) if self._available_pairs else 0,
            'cached_analyses': len(self._analysis_cache),
            'cache_hit_ratio': (
                self.stats['cache_hits'] / (self.stats['cache_hits'] + self.stats['cache_misses'])
                if (self.stats['cache_hits'] + self.stats['cache_misses']) > 0
                else 0
            ),
            'last_updated': self._last_updated,
        }
    
    async def clear_cache(self) -> Dict[str, Any]:
        """Vide le cache du PairSelector.
        
        Returns:
            Dictionnaire avec les statistiques du cache avant la suppression
        """
        cache_stats = self.cache.clear()
        self._analysis_cache.clear()
        self._available_pairs = None
        self._last_updated = None
        
        self.logger.info("Cache vidé avec succès")
        return cache_stats
    
    async def _load_available_pairs(self, force_refresh: bool = False) -> None:
        """Charge la liste des paires disponibles depuis l'instance globale AvailablePairs.
        
        Args:
            force_refresh: Si True, force le rechargement des paires
            
        Raises:
            Exception: Si le chargement des paires échoue
        """
        # Vérifier si le cache est encore valide
        if not force_refresh and self._available_pairs is not None:
            return
            
        try:
            # Utiliser l'instance globale de AvailablePairs
            if force_refresh:
                from src.core.market.pair_initializer import refresh_available_pairs
                await refresh_available_pairs()
            
            # Initialiser le dictionnaire des paires
            self._available_pairs = {}
            
            # Récupérer toutes les paires disponibles
            all_pairs = self._available_pairs_instance.get_available_pairs()
            if not all_pairs:
                self.logger.warning("Aucune paire disponible depuis AvailablePairs")
                return
            
            # Convertir au format attendu par le PairSelector
            for pair_name in all_pairs:
                try:
                    # Récupérer les informations détaillées de la paire
                    pair_info = self._available_pairs_instance.get_pair_info(pair_name)
                    if not pair_info:
                        self.logger.warning("Aucune information pour la paire: %s", pair_name)
                        continue
                        
                    # Ajouter la paire avec ses métadonnées
                    self._available_pairs[pair_name] = {
                        'base': pair_info.get('base', ''),
                        'quote': pair_info.get('quote', ''),
                        'pair_id': pair_info.get('altname', pair_name.replace('/', '')),
                        'altname': pair_info.get('altname', ''),
                        'wsname': pair_info.get('wsname', ''),
                        'status': 'online',  # Par défaut, considérer comme en ligne
                        'fees': pair_info.get('fees', []),
                        'fees_maker': pair_info.get('fees_maker', []),
                        'leverage_buy': pair_info.get('leverage_buy', []),
                        'leverage_sell': pair_info.get('leverage_sell', []),
                        'lot': pair_info.get('lot', ''),
                        'pair_decimals': pair_info.get('pair_decimals', 8),
                        'lot_decimals': pair_info.get('lot_decimals', 8),
                        'lot_multiplier': pair_info.get('lot_multiplier', 1),
                        'margin_call': pair_info.get('margin_call', 0),
                        'margin_stop': pair_info.get('margin_stop', 0),
                    }
                    
                except Exception as e:
                    self.logger.error(
                        "Erreur lors du traitement de la paire %s: %s", 
                        pair_name, 
                        str(e),
                        exc_info=True
                    )
            
            self.logger.info(
                "%s paires chargées avec succès depuis AvailablePairs", 
                len(self._available_pairs) if self._available_pairs else 0
            )
            
        except Exception as e:
            self.logger.error(
                "Erreur lors du chargement des paires depuis AvailablePairs: %s", 
                str(e), 
                exc_info=True
            )
            self.stats['errors'] += 1
            raise
    
    async def _get_ohlc_data(self, pair: str, interval: int = 1440) -> Optional[Dict]:
        """Récupère les données OHLC pour une paire spécifique.
        
        Args:
            pair: La paire à récupérer (ex: 'XBT/USD')
            interval: Intervalle en minutes (par défaut: 1440 = 1 jour)
            
        Returns:
            Dictionnaire avec les données OHLC, ou None en cas d'erreur
        """
        cache_key = f"ohlc_{pair}_{interval}"
        
        # Vérifier le cache
        cached_data = await self.cache.get(cache_key)
        if cached_data:
            self.stats['cache_hits'] += 1
            return cached_data
        
        self.logger.debug("Récupération des données OHLC pour %s (intervalle: %d)", pair, interval)
        
        try:
            # Récupérer l'identifiant de la paire
            pair_info = await self.get_pair_info(pair)
            if not pair_info:
                self.logger.warning("Paire non trouvée: %s", pair)
                return None
            
            # Appeler l'API pour récupérer les données OHLC
            ohlc_data = await self._make_api_call(
                'OHLC',
                pair=pair_info['pair_id'],
                interval=interval
            )
            
            if not ohlc_data or 'result' not in ohlc_data:
                self.logger.warning("Aucune donnée OHLC pour la paire %s", pair)
                return None
            
            # Extraire les données de la réponse
            result = next(iter(ohlc_data['result'].values()))
            
            # Mettre en cache les données
            await self.cache.set(cache_key, result)
            self.stats['cache_misses'] += 1
            
            return result
            
        except Exception as e:
            self.logger.error(
                "Erreur lors de la récupération des données OHLC pour %s: %s", 
                pair, 
                str(e),
                exc_info=True
            )
            self.stats['errors'] += 1
            return None
    
    @utils.retry_on_exception(max_retries=3, initial_delay=1.0, exponential_backoff=True)
    async def _make_api_call(self, endpoint: str, **params) -> Optional[Dict]:
        """Effectue un appel à l'API Kraken avec gestion des erreurs et des réessais.
        
        Args:
            endpoint: Nom du point de terminaison de l'API
            **params: Paramètres à passer à l'API
            
        Returns:
            Réponse de l'API sous forme de dictionnaire, ou None en cas d'erreur
        """
        self.stats['api_calls'] += 1
        
        try:
            # Gérer le cas spécial pour AssetPairs qui utilise get_asset_pairs
            if endpoint == 'AssetPairs':
                # Essayer d'abord avec l'attribut privé _endpoints
                if hasattr(self.kraken_api, '_endpoints') and hasattr(self.kraken_api._endpoints, 'get_asset_pairs'):
                    response = await self.kraken_api._endpoints.get_asset_pairs(**params)
                # Ensuite essayer avec l'attribut public (pour compatibilité)
                elif hasattr(self.kraken_api, 'endpoints') and hasattr(self.kraken_api.endpoints, 'get_asset_pairs'):
                    response = await self.kraken_api.endpoints.get_asset_pairs(**params)
                # En dernier recours, essayer d'appeler directement sur kraken_api
                else:
                    method = getattr(self.kraken_api, 'get_asset_pairs', None)
                    if not method or not callable(method):
                        raise ValueError("Méthode get_asset_pairs non trouvée dans l'API Kraken")
                    response = await method(**params)
            else:
                # Appeler la méthode appropriée de l'API
                method = getattr(self.kraken_api, endpoint.lower(), None)
                if not method or not callable(method):
                    raise ValueError(f"Méthode d'API non trouvée: {endpoint}")
                
                # Appeler la méthode avec les paramètres
                response = await method(**params)
            
            # Vérifier les erreurs dans la réponse
            if isinstance(response, dict) and 'error' in response and response['error']:
                error_msg = response['error']
                if isinstance(error_msg, list):
                    error_msg = ", ".join(str(e) for e in error_msg if e)
                
                # Vérifier les codes d'erreur connus
                for code, description in constants.KRAKEN_ERROR_CODES.items():
                    if code in error_msg:
                        error_msg = f"{description} ({code})"
                        break
                
                raise Exception(f"Erreur API: {error_msg}")
            
            return response
            
        except Exception as e:
            self.logger.error(
                "Erreur lors de l'appel à l'API %s: %s", 
                endpoint, 
                str(e),
                exc_info=True
            )
            self.stats['errors'] += 1
            raise
    
    async def close(self) -> None:
        """Ferme les ressources utilisées par le PairSelector."""
        # Nettoyer les ressources si nécessaire
        self.logger.info("Fermeture du PairSelector")
        
        # Sauvegarder l'état si nécessaire
        # ...
        
        self.logger.info("PairSelector fermé avec succès")
