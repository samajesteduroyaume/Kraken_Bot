"""
Module contenant les endpoints de l'API Kraken.
"""

import asyncio
import socket
import aiohttp
from typing import Dict, Any, Optional, Tuple
import logging
import uuid
from datetime import datetime, timezone

from .validators import Validator
from .exceptions import KrakenAPIError, APIConnectionError
from src.core.market.available_pairs_refactored import available_pairs


class KrakenEndpoints:
    def __init__(self, client):
        self.client = client
        self.validator = Validator()
        self.logger = logging.getLogger(__name__ + '.KrakenEndpoints')
        # Cache pour stocker le mapping des paires de trading
        self._pair_mapping = {}
        self._pair_mapping_loaded = False

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
        request_id = str(uuid.uuid4())[:8]
        self.logger.info(f"🔍 [5/5] [{request_id}] Début de get_ohlc_data pour la paire: {pair}, intervalle: {interval}, since: {since}")
        
        try:
            # Vérification simplifiée de la connectivité Internet
            try:
                # Vérification directe avec asyncio.open_connection
                reader, writer = await asyncio.wait_for(
                    asyncio.open_connection("kraken.com", 443),
                    timeout=5.0
                )
                writer.close()
                await writer.wait_closed()
                self.logger.info(f"✅ [5/5] [{request_id}] Connectivité Internet vérifiée")
            except (asyncio.TimeoutError, OSError) as e:
                error_msg = f"❌ [5/5] [{request_id}] Pas de connectivité Internet ou impossible de joindre Kraken: {str(e)}"
                self.logger.error(error_msg)
                raise APIConnectionError(error_msg) from e
            
            # Le reste de la méthode reste inchangé
            self.logger.info(f"🔍 [5/5] [{request_id}] Validation des paramètres...")
            self.validator.validate_pair(pair)
            self.validator.validate_ohlc_interval(interval)
            
            if since is not None and not isinstance(since, int):
                error_msg = f"❌ [5/5] [{request_id}] Le paramètre 'since' doit être un timestamp UNIX (entier)"
                self.logger.error(error_msg)
                raise ValueError(error_msg)

            params = {
                'pair': pair,
                'interval': interval
            }
            
            if since is not None:
                params['since'] = since

            self.logger.info(f"🔍 [5/5] [{request_id}] Préparation de la requête OHLC avec params: {params}")
            
            # Utiliser wait_for pour ajouter un timeout à la requête
            request_task = self.client._request('GET', 'public/OHLC', params)
            self.logger.info(f"🔍 [5/5] [{request_id}] Envoi de la requête OHLC (timeout: {timeout}s)...")
            
            try:
                response = await asyncio.wait_for(request_task, timeout=timeout)
                
                if not isinstance(response, dict):
                    error_msg = f"❌ [5/5] [{request_id}] Format de réponse inattendu pour les données OHLC: {type(response).__name__}"
                    self.logger.error(error_msg)
                    raise KrakenAPIError(error_msg)
                
                candles = response.get(pair, [])
                self.logger.info(f"✅ [5/5] [{request_id}] Réponse OHLC reçue: {len(candles)} bougies")
                
                if not candles:
                    self.logger.warning(f"⚠️ [5/5] [{request_id}] Aucune donnée OHLC disponible pour la paire {pair}")
                
                return response
                
            except asyncio.TimeoutError:
                error_msg = f"❌ [5/5] [{request_id}] Timeout de {timeout} secondes dépassé pour get_ohlc_data"
                self.logger.error(error_msg)
                raise asyncio.TimeoutError(error_msg) from None
                
        except Exception as e:
            if not isinstance(e, (KrakenAPIError, asyncio.TimeoutError, APIConnectionError, ValueError)):
                error_msg = f"❌ [5/5] [{request_id}] Erreur inattendue lors de la récupération des données OHLC: {str(e)}"
                self.logger.error(error_msg, exc_info=True)
                raise KrakenAPIError(error_msg) from e
            raise  # Relancer les erreurs déjà gérées

    async def get_asset_pairs(self, pair: Optional[str] = None, timeout: float = 10.0) -> Dict[str, Any]:
        """
        Récupère les informations sur les paires d'actifs disponibles sur Kraken.

        Args:
            pair: Symbole de la paire de trading (optionnel). Si non spécifié, toutes les paires sont retournées.
            timeout: Délai maximal d'attente en secondes (10 secondes par défaut)

        Returns:
            Dictionnaire avec les informations pour chaque paire, où les clés sont les symboles des paires
            et les valeurs sont des dictionnaires contenant les détails de chaque paire.

        Raises:
            ValueError: Si les paramètres sont invalides
            KrakenAPIError: Si l'API retourne une erreur
            asyncio.TimeoutError: Si le délai d'attente est dépassé
        """
        request_id = str(uuid.uuid4())[:8]
        self.logger.info(f"🔍 [5/5] [{request_id}] Début de get_asset_pairs pour la paire: {pair if pair else 'toutes'}")
        
        try:
            # Vérifier la connexion Internet avant de continuer
            try:
                await asyncio.wait_for(asyncio.get_event_loop().getaddrinfo("kraken.com", 443), timeout=5.0)
                self.logger.info(f"✅ [5/5] [{request_id}] Connectivité Internet vérifiée")
            except (asyncio.TimeoutError, OSError) as e:
                error_msg = f"❌ [5/5] [{request_id}] Pas de connectivité Internet ou impossible de joindre Kraken: {str(e)}"
                self.logger.error(error_msg)
                raise APIConnectionError(error_msg) from e
            
            # Préparer les paramètres de la requête
            params = {}
            if pair is not None:
                self.logger.info(f"🔍 [5/5] [{request_id}] Validation de la paire: {pair}")
                self.validator.validate_pair(pair)
                params['pair'] = pair
                self.logger.info(f"✅ [5/5] [{request_id}] Paire validée: {pair}")

            self.logger.info(f"🔍 [5/5] [{request_id}] Préparation de la requête API avec params: {params}")
            
            # Utiliser wait_for pour ajouter un timeout à la requête
            request_task = self.client._request('GET', 'public/AssetPairs', params)
            self.logger.info(f"🔍 [5/5] [{request_id}] Requête API créée, attente de la réponse (timeout: {timeout}s)...")
            
            try:
                # Réduire le timeout pour éviter les blocages prolongés
                response = await asyncio.wait_for(request_task, timeout=timeout)
                
                if not isinstance(response, dict):
                    error_msg = f"❌ [5/5] [{request_id}] Format de réponse inattendu: {type(response).__name__}"
                    self.logger.error(error_msg)
                    raise KrakenAPIError(error_msg)
                    
                self.logger.info(f"✅ [5/5] [{request_id}] Réponse API reçue, {len(response)} paires")
                return response
                
            except asyncio.TimeoutError:
                error_msg = f"❌ [5/5] [{request_id}] Timeout de {timeout} secondes dépassé pour get_asset_pairs"
                self.logger.error(error_msg)
                raise asyncio.TimeoutError(error_msg) from None
                
            except Exception as e:
                error_msg = f"❌ [5/5] [{request_id}] Erreur lors de l'attente de la réponse: {str(e)}"
                self.logger.error(error_msg, exc_info=True)
                raise KrakenAPIError(error_msg) from e
            
        except Exception as e:
            if not isinstance(e, (KrakenAPIError, asyncio.TimeoutError, APIConnectionError)):
                error_msg = f"❌ [5/5] [{request_id}] Erreur inattendue lors de la récupération des paires d'actifs: {str(e)}"
                self.logger.error(error_msg, exc_info=True)
                raise KrakenAPIError(error_msg) from e
            raise  # Relancer les erreurs déjà gérées

    async def get_ticker(self, pair: str, timeout: float = 10.0) -> Dict[str, Any]:
        """
        Récupère les informations de prix en temps réel pour une paire de trading.

        Args:
            pair: Symbole de la paire de trading (ex: 'XXBTZUSD')
            timeout: Délai maximal d'attente en secondes (10 secondes par défaut)

        Returns:
            Dictionnaire contenant les informations de prix avec la structure suivante :
            {
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

        Raises:
            ValueError: Si les paramètres sont invalides
            KrakenAPIError: Si l'API retourne une erreur
            asyncio.TimeoutError: Si le délai d'attente est dépassé
        """
        request_id = str(uuid.uuid4())[:8]
        self.logger.info(f"🔍 [5/5] [{request_id}] Début de get_ticker pour la paire: {pair}")
        
        try:
            # Vérification de la connectivité Internet
            try:
                reader, writer = await asyncio.wait_for(
                    asyncio.open_connection("kraken.com", 443),
                    timeout=5.0
                )
                writer.close()
                await writer.wait_closed()
                self.logger.info(f"✅ [5/5] [{request_id}] Connectivité Internet vérifiée")
            except (asyncio.TimeoutError, OSError) as e:
                error_msg = f"❌ [5/5] [{request_id}] Pas de connectivité Internet ou impossible de joindre Kraken: {str(e)}"
                self.logger.error(error_msg)
                raise APIConnectionError(error_msg) from e
            
            # Validation de la paire
            self.logger.info(f"🔍 [5/5] [{request_id}] Validation de la paire: {pair}")
            self.validator.validate_pair(pair)
            self.logger.info(f"✅ [5/5] [{request_id}] Paire validée: {pair}")

            # Préparation des paramètres de la requête
            params = {'pair': pair}
            self.logger.info(f"🔍 [5/5] [{request_id}] Préparation de la requête Ticker avec params: {params}")
            
            # Utiliser wait_for pour ajouter un timeout à la requête
            request_task = self.client._request('GET', 'public/Ticker', params)
            self.logger.info(f"🔍 [5/5] [{request_id}] Envoi de la requête Ticker (timeout: {timeout}s)...")
            
            try:
                # Attendre la réponse avec un timeout
                response = await asyncio.wait_for(request_task, timeout=timeout)
                
                if not isinstance(response, dict):
                    error_msg = f"❌ [5/5] [{request_id}] Format de réponse inattendu pour le ticker: {type(response).__name__}"
                    self.logger.error(error_msg)
                    raise KrakenAPIError(error_msg)
                
                # Extraire les données de la paire demandée
                if pair not in response:
                    error_msg = f"❌ [5/5] [{request_id}] Paire {pair} non trouvée dans la réponse"
                    self.logger.error(error_msg)
                    raise KrakenAPIError(error_msg)
                
                ticker_data = response[pair]
                self.logger.info(f"✅ [5/5] [{request_id}] Données Ticker reçues pour {pair}")
                return ticker_data
                
            except asyncio.TimeoutError:
                error_msg = f"❌ [5/5] [{request_id}] Timeout de {timeout} secondes dépassé pour get_ticker"
                self.logger.error(error_msg)
                raise asyncio.TimeoutError(error_msg) from None
                
            except Exception as e:
                error_msg = f"❌ [5/5] [{request_id}] Erreur lors de la récupération du ticker: {str(e)}"
                self.logger.error(error_msg, exc_info=True)
                raise KrakenAPIError(error_msg) from e
            
        except Exception as e:
            if not isinstance(e, (KrakenAPIError, asyncio.TimeoutError, APIConnectionError)):
                error_msg = f"❌ [5/5] [{request_id}] Erreur inattendue lors de la récupération du ticker: {str(e)}"
                self.logger.error(error_msg, exc_info=True)
                raise KrakenAPIError(error_msg) from e
            raise  # Relancer les erreurs déjà gérées

    async def get_orderbook(self, pair: str, count: int = 100, timeout: float = 10.0) -> Dict[str, Any]:
        """
        Récupère le carnet d'ordres pour une paire de trading.

        Args:
            pair: Symbole de la paire de trading (ex: 'XXBTZUSD')
            count: Nombre d'ordres à récupérer (max 500)
            timeout: Délai maximal d'attente en secondes (10 secondes par défaut)

        Returns:
            Dictionnaire contenant les offres (asks) et les demandes (bids) avec la structure suivante :
            {
                'pair_name': {
                    'asks': [ [price, volume, timestamp], ... ],
                    'bids': [ [price, volume, timestamp], ... ]
                }
            }

        Raises:
            ValueError: Si les paramètres sont invalides
            KrakenAPIError: Si l'API retourne une erreur
            asyncio.TimeoutError: Si le délai d'attente est dépassé
            APIConnectionError: Si la connexion à l'API échoue
        """
        request_id = str(uuid.uuid4())[:8]
        self.logger.info(f"🔍 [{request_id}] Début de get_orderbook pour la paire: {pair}, count: {count}")
        
        try:
            # Vérification de la connectivité Internet
            try:
                reader, writer = await asyncio.wait_for(
                    asyncio.open_connection("kraken.com", 443),
                    timeout=5.0
                )
                writer.close()
                await writer.wait_closed()
                self.logger.info(f"✅ [{request_id}] Connectivité Internet vérifiée")
            except (asyncio.TimeoutError, OSError) as e:
                error_msg = f"❌ [{request_id}] Pas de connectivité Internet ou impossible de joindre Kraken: {str(e)}"
                self.logger.error(error_msg)
                raise APIConnectionError(error_msg) from e
            
            # Validation des paramètres
            self.validator.validate_pair(pair)
            if not isinstance(count, int) or count < 1 or count > 500:
                error_msg = f"❌ [{request_id}] Le paramètre 'count' doit être un entier entre 1 et 500"
                self.logger.error(error_msg)
                raise ValueError(error_msg)
            
            # Préparation des paramètres de la requête
            params = {
                'pair': pair,
                'count': count
            }
            
            self.logger.info(f"🔍 [{request_id}] Préparation de la requête OrderBook avec params: {params}")
            
            # Envoi de la requête avec timeout
            request_task = self.client._request('GET', 'public/Depth', params)
            self.logger.info(f"🔍 [{request_id}] Envoi de la requête OrderBook (timeout: {timeout}s)...")
            
            try:
                response = await asyncio.wait_for(request_task, timeout=timeout)
                
                if not isinstance(response, dict):
                    error_msg = f"❌ [{request_id}] Format de réponse inattendu pour l'orderbook: {type(response).__name__}"
                    self.logger.error(error_msg)
                    raise KrakenAPIError(error_msg)
                
                # Vérification de la présence de la paire dans la réponse
                if not response:
                    error_msg = f"❌ [{request_id}] Réponse vide de l'API pour l'orderbook"
                    self.logger.error(error_msg)
                    raise KrakenAPIError(error_msg)
                
                # Le nom de la paire dans la réponse peut être différent (ex: 'XBTUSD' au lieu de 'XXBTZUSD')
                pair_key = next(iter(response.keys()), None)
                if not pair_key:
                    error_msg = f"❌ [{request_id}] Aucune donnée d'orderbook trouvée pour la paire {pair}"
                    self.logger.error(error_msg)
                    raise KrakenAPIError(error_msg)
                
                orderbook_data = response[pair_key]
                
                # Validation de la structure des données
                if 'asks' not in orderbook_data or 'bids' not in orderbook_data:
                    error_msg = f"❌ [{request_id}] Format de données d'orderbook invalide: {orderbook_data.keys()}"
                    self.logger.error(error_msg)
                    raise KrakenAPIError(error_msg)
                
                self.logger.info(f"✅ [{request_id}] OrderBook reçu avec {len(orderbook_data['asks'])} asks et {len(orderbook_data['bids'])} bids")
                return response
                
            except asyncio.TimeoutError:
                error_msg = f"❌ [{request_id}] Timeout de {timeout} secondes dépassé pour get_orderbook"
                self.logger.error(error_msg)
                raise asyncio.TimeoutError(error_msg) from None
                
            except Exception as e:
                error_msg = f"❌ [{request_id}] Erreur lors de la récupération de l'orderbook: {str(e)}"
                self.logger.error(error_msg, exc_info=True)
                raise KrakenAPIError(error_msg) from e
            
        except Exception as e:
            if not isinstance(e, (KrakenAPIError, asyncio.TimeoutError, APIConnectionError, ValueError)):
                error_msg = f"❌ [{request_id}] Erreur inattendue lors de la récupération de l'orderbook: {str(e)}"
                self.logger.error(error_msg, exc_info=True)
                raise KrakenAPIError(error_msg) from e
            raise  # Relancer les erreurs déjà gérées

    async def get_balance(self) -> Dict[str, float]:
        """
        Récupère le solde du compte.

        Returns:
            Dictionnaire avec les montants pour chaque actif

        Raises:
            KrakenAPIError: Si l'API retourne une erreur
            asyncio.TimeoutError: Si le délai d'attente est dépassé
            APIConnectionError: Si la connexion à l'API échoue
        """
        request_id = str(uuid.uuid4())[:8]
        self.logger.info(f"🔍 [{request_id}] Début de get_balance")
        
        try:
            # Vérification de la connectivité Internet
            try:
                reader, writer = await asyncio.wait_for(
                    asyncio.open_connection("kraken.com", 443),
                    timeout=5.0
                )
                writer.close()
                await writer.wait_closed()
                self.logger.info(f"✅ [{request_id}] Connectivité Internet vérifiée")
            except (asyncio.TimeoutError, OSError) as e:
                error_msg = f"❌ [{request_id}] Pas de connectivité Internet ou impossible de joindre Kraken: {str(e)}"
                self.logger.error(error_msg)
                raise APIConnectionError(error_msg) from e

            # Envoi de la requête
            self.logger.info(f"🔍 [{request_id}] Envoi de la requête de solde...")
            response = await self.client._request('POST', 'private/Balance', private=True)
            
            if not isinstance(response, dict):
                error_msg = f"❌ [{request_id}] Format de réponse inattendu pour le solde: {type(response).__name__}"
                self.logger.error(error_msg)
                raise KrakenAPIError(error_msg)
                
            self.logger.info(f"✅ [{request_id}] Solde récupéré avec succès: {len(response)} actifs")
            return response
            
        except KrakenAPIError:
            raise  # Relancer les erreurs déjà gérées
            
        except Exception as e:
            error_msg = f"❌ [{request_id}] Erreur inattendue lors de la récupération du solde: {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            raise KrakenAPIError(error_msg) from e
