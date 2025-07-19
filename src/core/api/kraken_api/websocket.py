"""
Module pour la gestion du websocket dans l'API Kraken.
"""

import asyncio
import aiohttp
import json
import logging
from typing import Any, Callable
from collections import defaultdict
from .exceptions import KrakenAPIError


class KrakenWebSocket:
    def __init__(self, client):
        self.client = client
        self.logger = logging.getLogger(__name__ + '.KrakenWebSocket')

        # Configuration
        self.ws_url = 'wss://ws.kraken.com'
        self.ws_session = None
        self.ws_connection = None
        self.ws_subscriptions = defaultdict(set)
        self.ws_token = None
        self.ws_token_expiration = 0
        self.ws_reconnect_attempts = 0
        self.ws_max_reconnect_attempts = 5
        self.ws_reconnect_delay = 5
        self.ws_ping_interval = 10
        self.ws_ping_timeout = 5
        self.ws_message_queue = asyncio.Queue()
        self.ws_tasks = set()
        self.ws_running = False

        # Handlers
        self.message_handlers = {}

    async def connect(self) -> None:
        """Établit la connexion websocket."""
        try:
            self.ws_session = aiohttp.ClientSession()
            self.ws_connection = await self.ws_session.ws_connect(self.ws_url)
            self.ws_running = True

            # Démarrage des tâches
            self.ws_tasks.add(asyncio.create_task(self._message_handler()))
            self.ws_tasks.add(asyncio.create_task(self._ping_handler()))

            self.logger.info("Connexion websocket établie")
        except Exception as e:
            self.logger.error(
                f"Erreur lors de la connexion websocket: {str(e)}")
            raise KrakenAPIError(
                f"Erreur lors de la connexion websocket: {str(e)}")

    async def disconnect(self) -> None:
        """Déconnecte le websocket."""
        try:
            if self.ws_connection:
                await self.ws_connection.close()
            if self.ws_session:
                await self.ws_session.close()
            self.ws_running = False
            self.ws_tasks.clear()
            self.logger.info("Déconnexion websocket réussie")
        except Exception as e:
            self.logger.error(
                f"Erreur lors de la déconnexion websocket: {str(e)}")
            raise KrakenAPIError(
                f"Erreur lors de la déconnexion websocket: {str(e)}")

    async def subscribe(self, channel: str, pair: str) -> None:
        """
        S'abonne à un canal pour une paire.

        Args:
            channel: Canal à suivre
            pair: Paire de trading
        """
        try:
            message = {
                'event': 'subscribe',
                'pair': [pair],
                'subscription': {'name': channel}
            }
            if self.ws_connection:
                await self.ws_connection.send_json(message)
            self.ws_subscriptions[channel].add(pair)
            self.logger.info(f"Souscription réussie à {channel} pour {pair}")
        except Exception as e:
            self.logger.error(f"Erreur lors de la souscription: {str(e)}")
            raise KrakenAPIError(f"Erreur lors de la souscription: {str(e)}")

    async def unsubscribe(self, channel: str, pair: str) -> None:
        """
        Se désabonne d'un canal pour une paire.

        Args:
            channel: Canal à quitter
            pair: Paire de trading
        """
        try:
            message = {
                'event': 'unsubscribe',
                'pair': [pair],
                'subscription': {'name': channel}
            }
            if self.ws_connection:
                await self.ws_connection.send_json(message)
            self.ws_subscriptions[channel].discard(pair)
            self.logger.info(f"Désabonnement réussi de {channel} pour {pair}")
        except Exception as e:
            self.logger.error(f"Erreur lors du désabonnement: {str(e)}")
            raise KrakenAPIError(f"Erreur lors du désabonnement: {str(e)}")

    async def register_handler(self, channel: str, handler: Callable) -> None:
        """
        Enregistre un handler pour un canal.

        Args:
            channel: Canal pour lequel enregistrer le handler
            handler: Fonction à appeler quand un message arrive
        """
        self.message_handlers[channel] = handler
        self.logger.info(f"Handler enregistré pour le canal {channel}")

    async def _message_handler(self) -> None:
        """Gère les messages entrants."""
        while self.ws_running:
            try:
                if self.ws_connection is not None:
                    msg = await self.ws_connection.receive()
                else:
                    # Gérer le cas où la connexion n'est pas encore établie
                    await asyncio.sleep(0.1)  # Attendre un peu
                    continue

                if msg.type == aiohttp.WSMsgType.TEXT:
                    data = json.loads(msg.data)
                    await self._process_message(data)
                elif msg.type == aiohttp.WSMsgType.CLOSED:
                    self.logger.info("Connexion websocket fermée")
                    break
                elif msg.type == aiohttp.WSMsgType.ERROR:
                    self.logger.error(
                        f"Erreur websocket: {self.ws_connection.exception()}")
                    break
            except Exception as e:
                self.logger.error(
                    f"Erreur dans le handler de message: {str(e)}")
                raise KrakenAPIError(
                    f"Erreur dans le handler de message: {str(e)}")

    async def _ping_handler(self) -> None:
        """Envoie des pings réguliers pour maintenir la connexion."""
        while self.ws_running:
            try:
                await asyncio.sleep(self.ws_ping_interval)
                if self.ws_connection:
                    await self.ws_connection.send_json({'event': 'ping'})
            except Exception as e:
                self.logger.error(f"Erreur lors de l'envoi du ping: {str(e)}")
                raise KrakenAPIError(
                    f"Erreur lors de l'envoi du ping: {str(e)}")

    async def _process_message(self, data: Any) -> None:
        """
        Traite un message websocket.

        Args:
            data: Message reçu
        """
        try:
            if isinstance(data, list):
                # Message de données
                data[0]
                channel_name = data[1]

                if channel_name in self.message_handlers:
                    await self.message_handlers[channel_name](data)
            elif isinstance(data, dict):
                # Message de contrôle
                event = data.get('event')

                if event == 'systemStatus':
                    self.logger.info(f"Statut système: {data.get('status')}")
                elif event == 'subscriptionStatus':
                    self.logger.info(
                        f"Statut souscription: {data.get('status')} pour {data.get('pair')}")
                elif event == 'pong':
                    self.logger.debug("Pong reçu")
                else:
                    self.logger.warning(f"Événement inconnu: {event}")
            else:
                self.logger.warning(f"Format de message inconnu: {type(data)}")
        except Exception as e:
            self.logger.error(
                f"Erreur lors du traitement du message: {str(e)}")
            raise KrakenAPIError(
                f"Erreur lors du traitement du message: {str(e)}")
