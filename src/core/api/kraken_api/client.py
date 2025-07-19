"""
Classe principale pour l'API Kraken.
"""

import os
import time
import json
import logging
import aiohttp
import asyncio
from datetime import datetime, timezone
from typing import Dict, Any, Optional, List, TypeVar, Generic, Callable
from collections import defaultdict

from .exceptions import KrakenAPIError, APIError, ValidationError, APIConnectionError, AuthenticationError
from .ratelimiter import RateLimiter
from .validators import Validator
from .orders import KrakenOrders
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
                 session_dir: Optional[str] = None):
        """Initialise l'API Kraken."""
        # Initialiser le gestionnaire de configuration
        self.config = KrakenConfig(env=env, config_file=config_file)

        # Configuration
        self.base_url = self.config.get('base_url')
        self.api_version = self.config.get('version')
        self.timeout = self.config.get('timeout')
        self.max_retries = self.config.get('max_retries')
        self.retry_delay = self.config.get('retry_delay')
        self.cache_ttl = self.config.get('cache_ttl')

        # Configuration du logging
        self.logger = logging.getLogger(__name__ + '.KrakenAPI')
        self.logger.setLevel(self.config.get('log_level', 'INFO').upper())

        # Initialiser l'event loop
        self.loop = loop or asyncio.get_event_loop()

        # Initialiser la session HTTP
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(
                total=self.timeout),
            connector=aiohttp.TCPConnector(
                ssl=True,
                loop=self.loop),
            headers={
                'User-Agent': f'KrakenAPI/1.0 (Python/{os.getenv("PYTHON_VERSION", "3.10")})',
                'Accept': 'application/json'})

        # Authentification
        credentials = self.config.get_credentials()
        self.api_key = api_key if api_key is not None else credentials['api_key']
        self.api_secret = api_secret if api_secret is not None else credentials['api_secret']

        if not isinstance(
                self.api_key,
                str) or not isinstance(
                self.api_secret,
                str):
            self.logger.error(
                "Les clés API doivent être des chaînes de caractères")
            raise AuthenticationError(
                "Les clés API doivent être des chaînes de caractères")

        if not self.api_key.strip() or not self.api_secret.strip():
            self.logger.error(
                "Les clés API ne peuvent pas être vides ou contenir uniquement des espaces")
            raise AuthenticationError(
                "Les clés API ne peuvent pas être vides ou contenir uniquement des espaces")

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
        
        # Initialiser les endpoints
        self._endpoints = KrakenEndpoints(self)

        # Métriques
        self.metrics = KrakenMetrics(self)

        # Mapping interne pair Kraken -> altname
        self.pair_altname_map = None  # Rempli à la demande

        # Initialisation des modules
        self.orders = KrakenOrders(self)
        self.websocket = KrakenWebSocket(self)

        # Démarrer le nettoyage périodique des sessions
        # La méthode cleanup_periodic sera appelée via le context manager __aenter__
        # lors de l'initialisation du bot

    async def __aenter__(self):
        """Context manager pour la session HTTP et la session de données."""
        await self.session.__aenter__()
        await self.session_manager.cleanup_periodic()
        return self

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

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Nettoyage de la session HTTP et de la session de données."""
        await self.session.__aexit__(exc_type, exc_val, exc_tb)
        self.session_manager.cleanup()
        await self.session.close()

    async def _request(self,
                       method: str,
                       endpoint: str,
                       data: Optional[Dict[str,
                                           Any]] = None,
                       private: bool = False) -> Dict[str,
                                                      Any]:
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
        """
        # Validation de la méthode HTTP
        if method not in ['GET', 'POST', 'PUT', 'DELETE']:
            self.logger.error(f"Méthode HTTP invalide: {method}")
            raise ValueError(
                "La méthode HTTP doit être GET, POST, PUT ou DELETE")

        # Préparation de l'URL
        url = f"{self.base_url}/{self.api_version}/{endpoint}"

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

        # Gestion du rate limiting
        async with RateLimiter(max_requests=self.rate_limit_limit,
                               window_seconds=self.rate_limit_window):
            # Tentatives de requête
            for attempt in range(self.max_retries + 1):
                try:
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

                        signature = self._sign_message(endpoint, nonce, data)
                        headers['API-Key'] = self.api_key
                        headers['API-Sign'] = signature

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

                        # Stockage dans le cache pour les requêtes GET non
                        # privées
                        if method == 'GET' and not private:
                            cache_key = f"{endpoint}:{json.dumps(data, sort_keys=True)}" if data else endpoint
                            self.cache.set(cache_key, result)

                        return result

                except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                    self.logger.error(f"Erreur lors de la requête: {str(e)}")
                    self.metrics.record_error('Connection', endpoint, str(e))

                    if attempt == self.max_retries:
                        raise APIConnectionError(
                            f"Erreur de connexion après {self.max_retries} tentatives: {str(e)}")

                    await asyncio.sleep(self.retry_delay * (attempt + 1))

    def _get_nonce(self) -> int:
        """Génère un nonce unique pour les appels privés."""
        return int(time.time() * 1000000)

    def _sign_message(self, endpoint: str, nonce: int, data: Dict) -> str:
        """
        Signe un message pour une requête privée selon la documentation Kraken.

        Args:
            endpoint: Point de terminaison de l'API
            nonce: Nonce généré
            data: Données de la requête

        Returns:
            Signature encodée en base64
        """
        # Préparation du message à signer
        message = endpoint.encode() + hashlib.sha256(str(nonce).encode() +
                                                     str(data).encode()).digest()
        hmac_sha256 = hmac.new(
            base64.b64decode(
                self.api_secret),
            message,
            hashlib.sha256)
        return base64.b64encode(hmac_sha256.digest()).decode()
