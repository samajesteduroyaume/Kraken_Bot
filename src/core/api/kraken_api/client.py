"""
Classe principale pour l'API Kraken.
"""

import os
import time
import json
import logging
import aiohttp
import asyncio
import hashlib
import hmac
import base64
from datetime import datetime, timezone
from typing import Dict, Any, Optional, List, TypeVar, Generic, Callable
from collections import defaultdict
import uuid

from .exceptions import KrakenAPIError, APIError, ValidationError, APIConnectionError, AuthenticationError
from .ratelimiter import RateLimiter
from .validators import Validator
from .orders import KrakenOrders
from .positions import KrakenPositions
from .websocket import KrakenWebSocket
from .cache import KrakenCache
from .metrics import KrakenMetrics
from .config import KrakenConfig
from .session import KrakenSession
from .endpoints import KrakenEndpoints

T = TypeVar('T')


class KrakenAPI:
    def __init__(self,
                 env: Optional[str] = None,
                 config_file: Optional[str] = None,
                 api_key: Optional[str] = None,
                 api_secret: Optional[str] = None,
                 loop: Optional[asyncio.AbstractEventLoop] = None,
                 session_dir: Optional[str] = None,
                 timeout: Optional[float] = 30.0,  # Timeout par défaut de 30 secondes
                 max_retries: int = 3,  # Nombre maximum de tentatives
                 retry_delay: float = 1.0  # Délai entre les tentatives en secondes
                 ):
        """
        Initialise l'API Kraken avec gestion des timeouts et des réessais.
        
        Args:
            env: Environnement (production, sandbox, etc.)
            config_file: Chemin vers le fichier de configuration
            api_key: Clé API Kraken
            api_secret: Secret API Kraken
            loop: Boucle d'événements asyncio
            session_dir: Répertoire pour stocker les sessions
            timeout: Timeout par défaut pour les requêtes (en secondes)
            max_retries: Nombre maximum de tentatives en cas d'échec
            retry_delay: Délai entre les tentatives (en secondes)
        """
        # Initialiser le gestionnaire de configuration
        self.config = KrakenConfig(env=env, config_file=config_file)

        # Configuration avec valeurs par défaut si non spécifiées
        self.base_url = self.config.get('base_url', 'https://api.kraken.com')
        self.api_version = self.config.get('version', '0')
        self.timeout = float(self.config.get('timeout', timeout))
        self.max_retries = int(self.config.get('max_retries', max_retries))
        self.retry_delay = float(self.config.get('retry_delay', retry_delay))
        self.cache_ttl = int(self.config.get('cache_ttl', 300))  # 5 minutes par défaut

        # Configuration du logging
        self.logger = logging.getLogger(__name__ + '.KrakenAPI')
        self.logger.setLevel(self.config.get('log_level', 'INFO').upper())

        # Gestion de la boucle d'événements
        self._loop = loop
        self._loop_owner = loop is None
        self._shutting_down = asyncio.Event()
        self._pending_requests = set()

        # Configuration des timeouts pour la session HTTP
        timeout_settings = aiohttp.ClientTimeout(
            total=self.timeout * 2,  # Timeout total plus long pour permettre plusieurs tentatives
            connect=self.timeout,    # Timeout de connexion
            sock_connect=self.timeout,  # Timeout de connexion socket
            sock_read=self.timeout      # Timeout de lecture socket
        )
        
        # Configuration du connecteur TCP
        connector = aiohttp.TCPConnector(
            ssl=True,
            limit=100,  # Nombre maximum de connexions simultanées
            limit_per_host=10,  # Nombre maximum de connexions par hôte
            ttl_dns_cache=300,  # Cache DNS pendant 5 minutes
            enable_cleanup_closed=True,  # Nettoyage automatique des connexions fermées
            force_close=False,  # Ne pas forcer la fermeture des connexions
        )
        
        # En-têtes HTTP
        headers = {
            'User-Agent': f'KrakenAPI/1.0 (Python/{os.getenv("PYTHON_VERSION", "3.10")})',
            'Accept': 'application/json',
            'Connection': 'keep-alive',
            'Accept-Encoding': 'gzip, deflate',
        }
        
        # Initialiser la session HTTP avec les paramètres configurés
        self.session = aiohttp.ClientSession(
            timeout=timeout_settings,
            connector=connector,
            headers=headers,
            raise_for_status=True,  # Lève une exception pour les codes HTTP d'erreur
            auto_decompress=True,   # Décompression automatique des réponses
            trust_env=True,         # Utiliser les paramètres de proxy de l'environnement
        )

        # Authentification
        try:
            credentials = self.config.get_credentials()
            self.logger.debug(f"[DEBUG] Authentification - Configuration chargée: {bool(credentials)}")
            
            # Récupération des clés avec priorité aux paramètres du constructeur
            self.api_key = api_key if api_key is not None else credentials.get('api_key')
            self.api_secret = api_secret if api_secret is not None else credentials.get('api_secret')
            
            # Log des clés (masquées pour la sécurité)
            if self.api_key:
                masked_key = f"{self.api_key[:4]}...{self.api_key[-4:] if len(self.api_key) > 8 else ''}"
                self.logger.debug(f"[DEBUG] Authentification - Clé API chargée: {masked_key}")
            if self.api_secret:
                self.logger.debug("[DEBUG] Authentification - Clé secrète chargée: [PROTECTED]")
            
            # Validation des clés
            if not isinstance(self.api_key, str) or not isinstance(self.api_secret, str):
                error_msg = "Les clés API doivent être des chaînes de caractères"
                self.logger.error(f"[ERREUR] {error_msg}")
                raise AuthenticationError(error_msg)

            if not self.api_key.strip() or not self.api_secret.strip():
                error_msg = "Les clés API ne peuvent pas être vides ou contenir uniquement des espaces"
                self.logger.error(f"[ERREUR] {error_msg}")
                raise AuthenticationError(error_msg)
                
            # Vérification du format de la clé API (doit commencer par 'API-')
            if not self.api_key.startswith('API-'):
                self.logger.warning("[ATTENTION] La clé API ne semble pas être au format attendu (doit commencer par 'API-')")
                
            # Vérification du format de la clé secrète (doit être en base64 valide)
            try:
                base64.b64decode(self.api_secret)
            except Exception as e:
                self.logger.error(f"[ERREUR] La clé secrète n'est pas un format base64 valide: {str(e)}")
                raise AuthenticationError("Format de clé secrète invalide (attendu: base64)")
                
            self.logger.info("Authentification - Clés API chargées avec succès")
            
        except Exception as e:
            self.logger.error(f"[ERREUR] Échec du chargement des clés API: {str(e)}")
            raise AuthenticationError(f"Échec du chargement des clés API: {str(e)}")

        # Rate limiting
        rate_limit = self.config.get_rate_limit()
        self.rate_limit_enabled = rate_limit.get('enabled', True)
        self.rate_limit_window = rate_limit.get('window', 30)
        self.rate_limit_limit = rate_limit.get('limit', 50)
        self.requests = defaultdict(int)
        self.last_reset = time.time()
        
        # Cache
        self.cache = KrakenCache(ttl=self.cache_ttl)

        # Session
        self.session_manager = KrakenSession(session_dir)
        
        # Initialisation des métriques
        self.metrics = KrakenMetrics(self)
        
        # Initialisation des endpoints
        self._endpoints = KrakenEndpoints(self)
        
        # Initialisation des gestionnaires de positions et d'ordres
        self.positions = KrakenPositions(self)
        self.orders = KrakenOrders(self)
        
    def _is_rate_limited(self) -> bool:
        """
        Vérifie si le taux de requêtes a été dépassé.
        
        Returns:
            bool: True si le taux est dépassé, False sinon
        """
        if not self.rate_limit_enabled:
            return False
            
        # Réinitialiser le compteur si la fenêtre de temps est écoulée
        current_time = time.time()
        if current_time - self.last_reset > self.rate_limit_window:
            self.requests.clear()
            self.last_reset = current_time
            return False
            
        # Vérifier si le nombre de requêtes dépasse la limite
        current_requests = sum(self.requests.values())
        return current_requests >= self.rate_limit_limit
        
    def _get_rate_limit_reset_time(self) -> float:
        """
        Calcule le temps avant la réinitialisation du compteur de taux.
        
        Returns:
            float: Temps en secondes avant la réinitialisation
        """
        current_time = time.time()
        time_since_reset = current_time - self.last_reset
        return max(0, self.rate_limit_window - time_since_reset)

    async def _make_request(
            self,
            method: str,
            endpoint: str,
            data: Optional[Dict[str, Any]] = None,
            private: bool = False
    ) -> Dict[str, Any]:
        """
        Effectue une requête à l'API Kraken avec gestion des retries, métriques et cache.

        Args:
            method: Méthode HTTP (GET, POST, PUT, DELETE)
            endpoint: Point de terminaison de l'API
            data: Données à envoyer
            private: Si True, utilise l'authentification privée

        Returns:
            Réponse de l'API

        Raises:
            ValueError: Si la méthode HTTP est invalide
            APIConnectionError: Si la connexion échoue
            KrakenAPIError: Si l'API retourne une erreur
            asyncio.CancelledError: Si la tâche est annulée
        """
        if method.upper() not in ('GET', 'POST', 'PUT', 'DELETE'):
            raise ValueError(f"Méthode HTTP non supportée: {method}")

        # Vérifier si nous sommes en train de nous arrêter
        if hasattr(self, '_shutting_down') and self._shutting_down.is_set():
            raise asyncio.CancelledError("L'API est en cours d'arrêt")

        # Générer un ID unique pour le suivi de la requête
        request_id = str(uuid.uuid4())[:8]
        self.logger.debug(f"[{request_id}] Requête {method} vers {endpoint}")

        # Préparer l'URL complète
        url = f"{self.base_url}/{self.api_version}{endpoint}"
        params = {k: v for k, v in (data or {}).items() if v is not None}

        # Vérifier si la réponse est en cache (uniquement pour les requêtes GET non privées)
        cache_key = None
        if method.upper() == 'GET' and not private and hasattr(self, 'cache'):
            cache_key = f"{method}:{endpoint}:{json.dumps(params, sort_keys=True)}"
            try:
                cached_response = self.cache.get(cache_key)
                if cached_response is not None:
                    self.logger.debug(f"[{request_id}] Réponse récupérée depuis le cache")
                    return cached_response
            except Exception as e:
                self.logger.warning(f"[{request_id}] Erreur lors de l'accès au cache: {e}")

        # Préparer les en-têtes et les données pour les requêtes privées
        headers = {}
        if private:
            nonce = self._get_nonce()
            params['nonce'] = nonce
            headers.update({
                'API-Key': self.api_key,
                'API-Sign': self._sign_message(endpoint, nonce, params)
            })

        # Gestion des réessais
        last_exception = None
        task = None
        
        try:
            for attempt in range(1, self.max_retries + 1):
                try:
                    # Vérifier si nous sommes en train de nous arrêter
                    if hasattr(self, '_shutting_down') and self._shutting_down.is_set():
                        raise asyncio.CancelledError("L'API est en cours d'arrêt")
                    
                    # Vérifier le rate limiting
                    if hasattr(self, 'rate_limit_enabled') and self.rate_limit_enabled and self._is_rate_limited():
                        wait_time = self._get_rate_limit_reset_time() - time.time()
                        if wait_time > 0:
                            self.logger.warning(
                                f"[{request_id}] Limite de taux atteinte, attente de {wait_time:.1f}s...")
                            await asyncio.sleep(wait_time)
                    
                    # Vérifier si la boucle d'événements est toujours en cours d'exécution
                    try:
                        asyncio.get_running_loop()
                    except RuntimeError as e:
                        if 'no running event loop' in str(e) or 'Event loop is closed' in str(e):
                            self.logger.warning(f"[{request_id}] La boucle d'événements n'est plus disponible, annulation de la requête")
                            raise asyncio.CancelledError("La boucle d'événements n'est plus disponible") from e
                        raise
                    
                    # Créer une tâche pour la requête afin de pouvoir l'annuler si nécessaire
                    task = asyncio.create_task(
                        self._execute_http_request(
                            method=method,
                            url=url,
                            params=params,
                            headers=headers,
                            request_id=request_id
                        )
                    )
                    
                    # Ajouter la tâche à la liste des requêtes en cours
                    if hasattr(self, '_pending_requests'):
                        self._pending_requests.add(task)
                        task.add_done_callback(self._pending_requests.discard)
                    
                    # Attendre la réponse avec un timeout
                    response_data = await asyncio.wait_for(task, timeout=self.timeout * 1.5)
                    
                    # Mettre en cache la réponse si nécessaire
                    if cache_key and hasattr(self, 'cache'):
                        try:
                            self.cache.set(cache_key, response_data, ttl=self.cache_ttl)
                        except Exception as e:
                            self.logger.warning(f"[{request_id}] Échec de la mise en cache: {e}")
                    
                    return response_data
                    
                except asyncio.TimeoutError as e:
                    last_exception = e
                    self.logger.warning(f"[{request_id}] Timeout de la requête (tentative {attempt}/{self.max_retries})")
                    if attempt == self.max_retries:
                        raise APIConnectionError(f"Timeout après {self.max_retries} tentatives") from e
                    
                    # Attendre avant de réessayer
                    await asyncio.sleep(self.retry_delay * attempt)
                    
                except asyncio.CancelledError:
                    self.logger.debug(f"[{request_id}] Requête annulée")
                    raise
                    
                except Exception as e:
                    last_exception = e
                    self.logger.error(f"[{request_id}] Erreur lors de la requête (tentative {attempt}/{self.max_retries}): {e}")
                    if attempt == self.max_retries:
                        if isinstance(e, KrakenAPIError):
                            raise
                        raise APIConnectionError(f"Échec après {self.max_retries} tentatives: {e}") from e
                    
                    # Attendre avant de réessayer
                    await asyncio.sleep(self.retry_delay * attempt)
            
            # Si nous arrivons ici, toutes les tentatives ont échoué
            raise APIConnectionError("Échec de la requête après plusieurs tentatives") from last_exception
            
        except asyncio.CancelledError:
            # Annuler la tâche si elle existe
            if task and not task.done():
                task.cancel()
            raise
            
        except Exception as e:
            # S'assurer que la tâche est annulée en cas d'erreur
            if task and not task.done():
                task.cancel()
            raise
            
        finally:
            # Nettoyer la référence à la tâche
            if task and hasattr(self, '_pending_requests') and task in self._pending_requests:
                self._pending_requests.discard(task)
    
    async def _execute_http_request(self, method: str, url: str, params: Dict[str, Any], 
                                  headers: Dict[str, str], request_id: str) -> Dict[str, Any]:
        """
        Exécute une requête HTTP et gère la réponse.
        
        Args:
            method: Méthode HTTP (GET, POST, PUT, DELETE)
            url: URL complète de la requête
            params: Paramètres de la requête
            headers: En-têtes de la requête
            request_id: ID unique de la requête pour le suivi
            
        Returns:
            Données de la réponse JSON
            
        Raises:
            KrakenAPIError: Si l'API retourne une erreur
            APIConnectionError: Si la connexion échoue
        """
        try:
            async with self.session.request(
                method=method,
                url=url,
                params=params,
                headers=headers,
                timeout=self.timeout
            ) as response:
                # Vérifier le code de statut HTTP
                if response.status == 429:  # Too Many Requests
                    retry_after = int(response.headers.get('Retry-After', self.retry_delay))
                    self.logger.warning(
                        f"[{request_id}] Trop de requêtes. Réessai dans {retry_after}s...")
                    await asyncio.sleep(retry_after)
                    raise APIConnectionError("Trop de requêtes, veuillez réessayer plus tard")
                    
                # Vérifier les autres erreurs HTTP
                response.raise_for_status()
                
                # Traiter la réponse
                data = await response.json()
                
                # Vérifier les erreurs de l'API Kraken
                if 'error' in data and data['error']:
                    error_msg = f"Erreur API Kraken: {data['error']}"
                    self.logger.error(f"[{request_id}] {error_msg}")
                    raise KrakenAPIError(error_msg)
                
                self.logger.debug(f"[{request_id}] Réponse reçue avec succès")
                return data
                
        except aiohttp.ClientError as e:
            self.logger.error(f"[{request_id}] Erreur de connexion: {e}")
            raise APIConnectionError(f"Erreur de connexion: {e}") from e
            
        except json.JSONDecodeError as e:
            self.logger.error(f"[{request_id}] Erreur de décodage JSON: {e}")
            raise APIConnectionError("Réponse invalide du serveur") from e
            
        except asyncio.TimeoutError as e:
            self.logger.error(f"[{request_id}] Timeout de la requête après {self.timeout} secondes")
            raise APIConnectionError("Délai d'attente dépassé") from e
            
        except Exception as e:
            self.logger.error(f"[{request_id}] Erreur inattendue: {e}")
            raise APIConnectionError(f"Erreur inattendue: {e}") from e
    
    async def __aenter__(self):
        """Context manager pour la session HTTP et la session de données."""
        await self.session.__aenter__()
        await self.session_manager.cleanup_periodic()
        return self

    async def _ensure_pair_altname_map(self, timeout: float = 15.0) -> None:
        """
        S'assure que le mapping des paires de trading est chargé.
        
        Args:
            timeout: Délai maximal d'attente en secondes pour le chargement des paires
            
        Raises:
            KrakenAPIError: Si le chargement des paires échoue
            asyncio.TimeoutError: Si le délai d'attente est dépassé
        """
        request_id = str(uuid.uuid4())[:8]
        self.logger.info(f"🔍 [4/5] [{request_id}] Vérification du mapping des paires...")
        
        if self.pair_altname_map is not None:
            self.logger.info(f"✅ [4/5] [{request_id}] Mapping déjà chargé")
            return
            
        try:
            self.logger.info(f"🔍 [4/5] [{request_id}] Chargement du mapping des paires de trading (timeout: {timeout}s)...")
            
            # Utiliser wait_for pour ajouter un timeout global
            pairs = await asyncio.wait_for(self.get_asset_pairs(), timeout=timeout)
            
            if not isinstance(pairs, dict):
                error_msg = f"❌ [4/5] [{request_id}] Format de réponse inattendu pour les paires d'actifs: {type(pairs).__name__}"
                self.logger.error(error_msg)
                raise KrakenAPIError(error_msg)
                
            self.pair_altname_map = {}
            processed = 0
            skipped = 0
            
            # Limiter le nombre de paires à traiter pour éviter les performances médiocres
            max_pairs = 100  # Limite arbitraire pour éviter les problèmes de performances
            
            for pair_name, pair_info in list(pairs.items())[:max_pairs]:
                if not isinstance(pair_info, dict):
                    skipped += 1
                    continue
                    
                altname = pair_info.get('altname')
                if altname and altname != pair_name:
                    self.pair_altname_map[pair_name] = altname
                    processed += 1
                    
                    # Journalisation périodique pour éviter de surcharger les logs
                    if processed % 20 == 0:
                        self.logger.info(f"🔍 [4/5] [{request_id}] Traitement en cours... {processed} paires traitées")
            
            self.logger.info(f"✅ [4/5] [{request_id}] Mapping des paires terminé: {processed} paires traitées, {skipped} ignorées")
            self.logger.debug(f"✅ [4/5] [{request_id}] Mapping des paires chargé: {len(self.pair_altname_map)} entrées")
            
            # Vérifier que le mapping n'est pas vide
            if not self.pair_altname_map:
                error_msg = f"❌ [4/5] [{request_id}] Aucune paire valide n'a pu être chargée"
                self.logger.error(error_msg)
                raise KrakenAPIError(error_msg)
                
        except asyncio.TimeoutError as e:
            error_msg = f"❌ [4/5] [{request_id}] Délai d'attente dépassé lors du chargement des paires"
            self.logger.error(error_msg)
            raise asyncio.TimeoutError(error_msg) from e
            
        except Exception as e:
            error_msg = f"❌ [4/5] [{request_id}] Erreur lors du chargement des paires: {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            raise KrakenAPIError(error_msg) from e
                    
    async def get_asset_pairs(self, pair: Optional[str] = None) -> Dict[str, Any]:
        """
        Récupère les informations sur les paires d'actifs disponibles sur Kraken.

        Args:
            pair: Symbole de la paire de trading (optionnel). Si non spécifié, toutes les paires sont retournées.

        Returns:
            Dictionnaire avec les informations pour chaque paire, où les clés sont les symboles des paires
            et les valeurs sont des dictionnaires contenant les détails de chaque paire.

        Raises:
            ValueError: Si les paramètres sont invalides
            KrakenAPIError: Si l'API retourne une erreur
        """
        return await self._endpoints.get_asset_pairs(pair)
        
    async def get_ticker(self, pair: str, timeout: float = 10.0) -> Dict[str, Any]:
        """
        Récupère les informations de prix en temps réel pour une paire de trading.

        Args:
            pair: Symbole de la paire de trading (ex: 'XXBTZUSD')
            timeout: Délai maximal d'attente en secondes (10 secondes par défaut)

        Returns:
            Dictionnaire contenant les informations de prix avec la structure suivante :
            {
                'result': {
                    pair: {
                        'a': [ask_price, ask_lot, timestamp],
                        'b': [bid_price, bid_lot, timestamp],
                        'c': [last_price, last_lot],
                        'v': [volume_today, volume_24h],
                        'p': [vwap_today, vwap_24h],
                        't': [trades_today, trades_24h],
                        'l': [low_today, low_24h],
                        'h': [high_today, high_24h],
                        'o': open_price
                    }
                }
            }

        Raises:
            ValueError: Si les paramètres sont invalides
            KrakenAPIError: Si l'API retourne une erreur
            asyncio.TimeoutError: Si le délai d'attente est dépassé
        """
        ticker_data = await self._endpoints.get_ticker(pair, timeout=timeout)
        return {'result': {pair: ticker_data}}
        
    async def get_open_orders(self) -> Dict[str, Any]:
        """
        Récupère les ordres ouverts sur le compte.

        Returns:
            Dictionnaire contenant les ordres ouverts avec leurs détails
            
        Raises:
            KrakenAPIError: Si l'API retourne une erreur
            APIConnectionError: Si la connexion échoue
        """
        try:
            # Utiliser le gestionnaire d'ordres pour récupérer les ordres ouverts
            response = await self.orders.get_open_orders()
            
            # Vérifier la réponse
            if not isinstance(response, dict) or 'result' not in response:
                self.logger.error(f"Réponse inattendue de l'API OpenOrders: {response}")
                return {}
                
            return response.get('result', {})
            
        except Exception as e:
            self.logger.error(f"Erreur lors de la récupération des ordres ouverts: {str(e)}")
            raise
            
    async def get_open_positions(self, txids: Optional[List[str]] = None, docalcs: bool = False) -> Dict[str, Any]:
        """
        Récupère les positions ouvertes sur le compte.

        Args:
            txids: Liste des identifiants de transaction à filtrer (optionnel)
            docalcs: Si True, calcule les valeurs des positions (peut être plus lent)

        Returns:
            Dictionnaire contenant les positions ouvertes avec leurs détails
            
        Raises:
            KrakenAPIError: Si l'API retourne une erreur
            APIConnectionError: Si la connexion échoue
        """
        try:
            # Utiliser le gestionnaire de positions pour récupérer les positions ouvertes
            return await self.positions.get_open_positions(txids=txids, docalcs=docalcs)
            
        except Exception as e:
            self.logger.error(f"Erreur lors de la récupération des positions ouvertes: {str(e)}")
            raise

    async def get_ohlc_data(self, pair: str, interval: int = 1, since: Optional[int] = None, timeout: float = 15.0) -> Dict[str, Any]:
        """
        Récupère les données OHLC (Open, High, Low, Close) pour une paire de trading.

        Args:
            pair: Symbole de la paire de trading (ex: 'XXBTZUSD')
            interval: Intervalle de temps en minutes (1, 5, 15, 30, 60, 240, 1440, 10080, 21600)
            since: Timestamp UNIX pour récupérer les données depuis une date spécifique (optionnel)
            timeout: Délai maximal d'attente en secondes (15 secondes par défaut)

        Returns:
            Dictionnaire contenant les données OHLC et le dernier timestamp

        Raises:
            ValueError: Si les paramètres sont invalides
            KrakenAPIError: Si l'API retourne une erreur
            asyncio.TimeoutError: Si le délai d'attente est dépassé
        """
        return await self._endpoints.get_ohlc_data(pair, interval, since, timeout)

    async def get_time(self, timeout: float = 10.0) -> Dict[str, Any]:
        """
        Récupère l'heure actuelle du serveur Kraken.
        
        Args:
            timeout: Délai maximal d'attente en secondes
            
        Returns:
            Dictionnaire contenant l'heure du serveur au format Unix
            
        Raises:
            APIConnectionError: Si la connexion échoue
            KrakenAPIError: Si l'API retourne une erreur
            asyncio.TimeoutError: Si le délai d'attente est dépassé
        """
        try:
            # Créer une tâche avec timeout
            response = await asyncio.wait_for(
                self._request('GET', 'public/Time'),
                timeout=timeout
            )
            self.logger.debug(f"Réponse de l'API Time: {response}")
            
            if not isinstance(response, dict) or 'result' not in response:
                self.logger.error(f"Réponse inattendue de l'API Time: {response}")
                return {}
                
            return response.get('result', {})
            
        except asyncio.TimeoutError:
            self.logger.error(f"Timeout de {timeout} secondes dépassé pour la requête Time")
            raise
        except Exception as e:
            self.logger.error(f"Erreur lors de l'appel à l'API Time: {str(e)}")
            raise
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """
        Nettoyage de la session HTTP et de la session de données.
        
        Args:
            exc_type: Type de l'exception si une exception a été levée, sinon None
            exc_val: L'instance d'exception si une exception a été levée, sinon None
            exc_tb: L'objet traceback si une exception a été levée, sinon None
        """
        self._shutting_down.set()
        
        # Annuler toutes les requêtes en cours
        if hasattr(self, '_pending_requests') and self._pending_requests:
            self.logger.debug(f"Annulation de {len(self._pending_requests)} requêtes en cours...")
            for task in self._pending_requests:
                if not task.done():
                    task.cancel()
            
            # Attendre que toutes les tâches soient annulées ou terminées
            if self._pending_requests:
                done, pending = await asyncio.wait(
                    self._pending_requests,
                    timeout=5.0,
                    return_when=asyncio.ALL_COMPLETED
                )
                
                # Vérifier s'il reste des tâches en attente
                if pending:
                    self.logger.warning(f"{len(pending)} tâches n'ont pas pu être annulées correctement")
        
        try:
            # Nettoyage de la session HTTP
            if hasattr(self, 'session') and not self.session.closed:
                try:
                    # Fermer le connecteur pour libérer les connexions
                    if hasattr(self.session, '_connector'):
                        self.logger.debug("Fermeture du connecteur HTTP...")
                        await self.session._connector.close()
                    
                    # Fermer la session
                    self.logger.debug("Fermeture de la session HTTP...")
                    if exc_type is not None:
                        # En cas d'erreur, on ferme immédiatement
                        await self.session.close()
                    else:
                        # Sinon, on utilise __aexit__ qui gère proprement la fermeture
                        await self.session.__aexit__(exc_type, exc_val, exc_tb)
                    self.logger.debug("Session HTTP fermée avec succès")
                except Exception as e:
                    self.logger.error(f"Erreur lors de la fermeture de la session HTTP: {e}")
            
            # Nettoyage du gestionnaire de session
            if hasattr(self, 'session_manager'):
                try:
                    self.logger.debug("Nettoyage du gestionnaire de session...")
                    self.session_manager.cleanup()
                    self.logger.debug("Gestionnaire de session nettoyé avec succès")
                except Exception as e:
                    self.logger.error(f"Erreur lors du nettoyage du gestionnaire de session: {e}")
                
            # Nettoyage du cache
            if hasattr(self, 'cache'):
                try:
                    self.logger.debug("Nettoyage du cache...")
                    self.cache.clear()
                    self.logger.debug("Cache nettoyé avec succès")
                except Exception as e:
                    self.logger.error(f"Erreur lors du nettoyage du cache: {e}")
            
            # Si nous sommes propriétaires de la boucle d'événements, on la ferme
            if hasattr(self, '_loop_owner') and self._loop_owner and self._loop is not None:
                self.logger.debug("Fermeture de la boucle d'événements...")
                if self._loop.is_running():
                    self._loop.stop()
                self._loop.close()
                self.logger.debug("Boucle d'événements fermée avec succès")
                    
        except Exception as e:
            self.logger.error(f"Erreur inattendue lors du nettoyage des ressources: {e}")
            raise
        finally:
            # Nettoyer les références
            self._pending_requests.clear()
            self._shutting_down.clear()


    async def _request(
            self,
            method: str,
            endpoint: str,
            data: Optional[Dict[str, Any]] = None,
            private: bool = False
    ) -> Dict[str, Any]:
        """
        Effectue une requête à l'API Kraken avec gestion des retries, métriques et cache.

        Args:
            method: Méthode HTTP (GET, POST, PUT, DELETE)
            endpoint: Point de terminaison de l'API
            data: Données à envoyer
            private: Si True, utilise l'authentification privée

        Returns:
            Réponse de l'API

        Raises:
            ValueError: Si la méthode HTTP est invalide
            APIConnectionError: Si la connexion échoue
            KrakenAPIError: Si l'API retourne une erreur
            asyncio.CancelledError: Si la tâche est annulée
        """
        request_id = str(uuid.uuid4())[:8]  # ID unique pour le suivi de la requête
        self.logger.info(f"🔍 [{request_id}] Début de la requête {method} vers {endpoint} (privée: {private})")
        
        # Vérifier que la session est toujours ouverte
        if not hasattr(self, 'session') or self.session.closed:
            raise APIConnectionError("La session HTTP est fermée")
            
        # Validation de la méthode HTTP
        if method not in ['GET', 'POST', 'PUT', 'DELETE']:
            error_msg = f"Méthode HTTP invalide: {method}"
            self.logger.error(f"❌ [{request_id}] {error_msg}")
            raise ValueError("La méthode HTTP doit être GET, POST, PUT ou DELETE")

        # Préparation de l'URL
        url = f"{self.base_url}/{self.api_version}/{endpoint}"
        self.logger.info(f"🔍 [6/5] [{request_id}] URL complète: {url}")
        
        # Essai de cache pour les requêtes GET non privées
        if method == 'GET' and not private:
            try:
                # Création d'une clé de cache basée sur l'endpoint et les paramètres
                cache_params = None
                if data:
                    # Si data est un dictionnaire, l'utiliser tel quel
                    if isinstance(data, dict):
                        cache_params = data
                    # Si c'est un FormData, le convertir en dictionnaire
                    elif hasattr(data, '_fields'):  # Vérifie si c'est un FormData
                        cache_params = {k: v for k, v in data._fields.items()}
                
                # Créer la clé de cache en sérialisant les paramètres si nécessaire
                cache_key = f"{endpoint}:{json.dumps(cache_params, sort_keys=True) if cache_params else ''}"
                
                # Vérifier le cache
                cached_result = self.cache.get(cache_key)

                if cached_result is not None:
                    self.metrics.record_cache_hit(endpoint)
                    self.logger.info(f"✅ [6/5] [{request_id}] Résultat trouvé dans le cache")
                    return cached_result

                self.metrics.record_cache_miss(endpoint)
                self.logger.info(f"🔍 [6/5] [{request_id}] Aucun résultat dans le cache, requête vers l'API nécessaire")
                
            except Exception as e:
                self.logger.warning(f"⚠️ [6/5] [{request_id}] Erreur lors de l'accès au cache: {str(e)}")
                # En cas d'erreur de cache, on continue avec une requête normale

        # Essai de cache pour les requêtes GET non privées
        if method == 'GET' and not private:
            # Création d'une clé de cache basée sur l'endpoint et les paramètres
            cache_params = None
            if data:
                # Si data est un dictionnaire, l'utiliser tel quel
                if isinstance(data, dict):
                    cache_params = data
                # Si c'est un FormData, le convertir en dictionnaire
                elif hasattr(data, '_fields'):  # Vérifie si c'est un FormData
                    cache_params = {k: v for k, v in data._fields.items()}
            
            # Créer la clé de cache en sérialisant les paramètres si nécessaire
            cache_key = f"{endpoint}:{json.dumps(cache_params, sort_keys=True) if cache_params else ''}"
            
            # Vérifier le cache
            cached_result = self.cache.get(cache_key)

            if cached_result is not None:
                self.metrics.record_cache_hit(endpoint)
                return cached_result

            self.metrics.record_cache_miss(endpoint)

        # Gestion du rate limiting et des tentatives
        last_exception = None
        for attempt in range(self.max_retries + 1):
            try:
                # Point de contrôle pour l'annulation
                await asyncio.sleep(0)  # Point de contrôle pour l'annulation
                    
                # Gestion du rate limiting
                if self.rate_limit_enabled and self._is_rate_limited():
                    wait_time = self._get_rate_limit_reset_time() - time.time()
                    if wait_time > 0:
                        self.logger.warning(
                            f"[{request_id}] Limite de taux atteinte, attente de {wait_time:.1f}s...")
                        try:
                            await asyncio.sleep(wait_time)
                        except asyncio.CancelledError:
                            self.logger.warning(f"[{request_id}] Attente du rate limit annulée")
                            raise
                            
                # Préparation des headers
                headers = {
                    'Content-Type': 'application/x-www-form-urlencoded'
                }

                # Ajout de l'authentification pour les appels privés
                if private:
                    nonce = self._get_nonce()
                    if data is None:
                        data = {}
                    data['nonce'] = nonce

                    self.logger.debug(f"[DEBUG] Requête privée - Nonce: {nonce}")
                    self.logger.debug(f"[DEBUG] Requête privée - Données avant signature: {data}")
                    
                    signature = self._sign_message(endpoint, nonce, data)
                    
                    headers['API-Key'] = self.api_key
                    headers['API-Sign'] = signature
                    
                    self.logger.debug(f"[DEBUG] Requête privée - En-têtes: {headers}")
                    self.logger.debug(f"[DEBUG] Requête privée - Clé API utilisée: {self.api_key}")
                    self.logger.debug(f"[DEBUG] Requête privée - Signature: {signature}")
                    
                    # Masquer partiellement la clé API pour les logs
                    if self.api_key and len(self.api_key) > 8:
                        masked_key = f"{self.api_key[:4]}...{self.api_key[-4:]}"
                        self.logger.info(f"Utilisation de la clé API: {masked_key}")
                    else:
                        self.logger.warning("Clé API non valide ou trop courte")

                # Préparation des données
                request_data = None
                if data is not None:
                    if method == 'GET':
                        # Pour les requêtes GET, on passe les paramètres dans l'URL
                        # Création d'une nouvelle URL avec les paramètres
                        from urllib.parse import urlencode
                        if '?' in url:
                            url = f"{url}&{urlencode(data)}"
                        else:
                            url = f"{url}?{urlencode(data)}"
                    else:
                        # Pour les autres méthodes, on utilise FormData
                        request_data = aiohttp.FormData(data)

                # Envoi de la requête
                start_time = time.time()
                async with self.session.request(
                    method=method,
                    url=url,
                    data=request_data,
                    headers=headers
                ) as response:
                    duration = time.time() - start_time

                    # Mise à jour des métriques
                    self.metrics.record_request(
                        method, endpoint, True, duration)

                    # Vérification du statut
                    if response.status != 200:
                        self.logger.error(
                            f"Erreur HTTP {response.status} lors de la requête à {url}")
                        self.metrics.record_error(
                            'HTTP', endpoint, f"Status {response.status}")
                        raise APIConnectionError(
                            f"Erreur HTTP {response.status}")

                    # Lecture de la réponse
                    result = await response.json()

                    # Vérification de l'erreur Kraken
                    if 'error' in result and result['error']:
                        self.logger.error(
                            f"Erreur Kraken: {result['error']}")
                        self.metrics.record_error(
                            'Kraken', endpoint, result['error'])
                        raise KrakenAPIError(
                            f"Erreur Kraken: {result['error']}")

                    # Stockage dans le cache pour les requêtes GET non privées
                    if method == 'GET' and not private:
                        cache_key = f"{endpoint}:{json.dumps(data, sort_keys=True) if data else endpoint}"
                        self.cache.set(cache_key, result)

                    return result

            except asyncio.CancelledError:
                self.logger.warning(f"[{request_id}] Requête annulée pendant la tentative {attempt + 1}")
                raise
            except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                last_exception = e
                self.logger.error(f"[{request_id}] Erreur lors de la tentative {attempt + 1}/{self.max_retries + 1}: {str(e)}")
                self.metrics.record_error('Connection', endpoint, str(e))

                if attempt == self.max_retries:
                    raise APIConnectionError(
                        f"Erreur de connexion après {self.max_retries + 1} tentatives: {str(e)}")

                # Attente exponentielle avec jitter
                delay = min(self.retry_delay * (2 ** attempt) * (0.5 + random.random()), 30)
                try:
                    await asyncio.sleep(delay)
                except asyncio.CancelledError:
                    self.logger.warning(f"[{request_id}] Attente de réessai annulée")
                    raise



    def _get_nonce(self) -> int:
        """Génère un nonce unique pour les appels privés."""
        return int(time.time() * 1000000)

    def _sign_message(self, endpoint: str, nonce: int, data: Dict) -> str:
        """
        Signe un message pour une requête privée selon la documentation Kraken.
        Utilise HMAC-SHA512 de (URI path + SHA256(nonce + POST data))
        avec la clé secrète décodée en base64.

        Args:
            endpoint: Point de terminaison de l'API (doit commencer par "/0/private/")
            nonce: Nonce généré (entier)
            data: Données de la requête (dictionnaire)

        Returns:
            Signature encodée en base64
        """
            # S'assurer que l'endpoint a le format correct
        # Si l'endpoint commence déjà par "/0/private/", on le laisse tel quel
        # Sinon, on retire tout slash initial et on ajoute le préfixe "/0/private/"
        if endpoint.startswith('0/private/'):
            endpoint = f'/{endpoint}'
        elif not endpoint.startswith('/0/private/'):
            endpoint = f'/0/private/{endpoint.lstrip("/")}'
        
        self.logger.debug(f"[DEBUG] Signature - Endpoint: {endpoint}")
        self.logger.debug(f"[DEBUG] Signature - Nonce: {nonce}")
        self.logger.debug(f"[DEBUG] Signature - Données brutes: {data}")
            
        # Préparation des données pour le hachage
        post_data = ''
        if data:
            # Trier les clés pour assurer un ordre cohérent
            sorted_data = dict(sorted(data.items()))
            # Convertir les données en chaîne de requête URL-encodée
            # Utiliser urlencode pour un encodage conforme à la spécification Kraken
            from urllib.parse import urlencode
            post_data = urlencode(sorted_data)
        
        self.logger.debug(f"[DEBUG] Signature - Données encodées: {post_data}")
        
        # Calculer SHA256(nonce + post_data)
        # Note: L'API Kraken attend nonce + données URL encodées
        message = f"{nonce}{post_data}".encode()
        sha256_hash = hashlib.sha256(message).digest()
        
        # Préparer le message final (endpoint + hash_sha256)
        message = endpoint.encode() + sha256_hash
        
        try:
            # Décoder la clé secrète en base64
            self.logger.debug(f"[DEBUG] Signature - Clé secrète avant décodage: {self.api_secret}")
            api_secret = base64.b64decode(self.api_secret)
            self.logger.debug(f"[DEBUG] Signature - Clé secrète après décodage: {api_secret.hex()}")
            
            # Calculer la signature HMAC-SHA512
            hmac_sha512 = hmac.new(
                api_secret,
                message,
                hashlib.sha512
            )
            
            # Log de débogage pour la signature
            signature = base64.b64encode(hmac_sha512.digest()).decode()
            self.logger.debug(f"[DEBUG] Signature - Signature générée: {signature}")
            self.logger.debug(f"[DEBUG] Signature - Message signé (hex): {message.hex()}")
            
            # Retourner la signature encodée en base64
            return base64.b64encode(hmac_sha512.digest()).decode()
            
        except Exception as e:
            self.logger.error(f"Erreur lors de la génération de la signature: {str(e)}")
            raise AuthenticationError(f"Erreur de signature: {str(e)}")
