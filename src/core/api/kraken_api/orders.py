"""
Module pour la gestion des ordres et des positions dans l'API Kraken.
"""

from typing import Dict, Any
import logging

from .validators import Validator
from .exceptions import KrakenAPIError


class KrakenOrders:
    def __init__(self, client):
        self.client = client
        self.validator = Validator()
        self.logger = logging.getLogger(__name__ + '.KrakenOrders')

    async def create_order(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Crée un ordre sur Kraken.

        Args:
            params: Paramètres de l'ordre

        Returns:
            Dictionnaire avec:
            - descr: Description de l'ordre
            - txid: ID de la transaction
            - ordertxid: ID de l'ordre
            - trades: Liste des trades exécutés
            - status: Statut de l'ordre
            - reason: Raison de l'échec (si applicable)

        Raises:
            ValidationError: Si les paramètres sont invalides
            KrakenAPIError: Si l'API retourne une erreur
        """
        # Validation des paramètres
        self.validator.validate_order_params(params)

        # Préparation des données
        data = {
            'pair': params['pair'],
            'type': params['type'],
            'ordertype': params['ordertype'],
            'volume': params['volume']
        }

        # Paramètres optionnels
        if 'price' in params:
            data['price'] = params['price']
        if 'price2' in params:
            data['price2'] = params['price2']
        if 'leverage' in params:
            data['leverage'] = params['leverage']
        if 'oflags' in params:
            data['oflags'] = params['oflags']
        if 'starttm' in params:
            data['starttm'] = params['starttm']
        if 'expiretm' in params:
            data['expiretm'] = params['expiretm']
        if 'userref' in params:
            data['userref'] = params['userref']
        if 'validate' in params:
            data['validate'] = params['validate']
        if 'close' in params:
            data['close'] = params['close']

        # Appel à l'API
        response = await self.client._request('POST', 'private/AddOrder', data, private=True)

        # Validation de la réponse
        if 'error' in response and response['error']:
            self.logger.error(
                f"Erreur lors de la création de l'ordre: {response['error']}")
            raise KrakenAPIError(
                f"Erreur lors de la création de l'ordre: {response['error']}")

        return response

    async def cancel_order(self, txid: str) -> Dict[str, Any]:
        """
        Annule un ordre.

        Args:
            txid: ID de l'ordre à annuler

        Returns:
            Dictionnaire avec:
            - count: Nombre d'ordres annulés
            - pending: Liste des IDs d'ordres en attente

        Raises:
            ValidationError: Si le txid est invalide
            KrakenAPIError: Si l'API retourne une erreur
        """
        # Validation du txid
        self.validator.validate_txid(txid)

        # Préparation des données
        data = {'txid': txid}

        # Appel à l'API
        response = await self.client._request('POST', 'private/CancelOrder', data, private=True)

        # Validation de la réponse
        if 'error' in response and response['error']:
            self.logger.error(
                f"Erreur lors de l'annulation de l'ordre: {response['error']}")
            raise KrakenAPIError(
                f"Erreur lors de l'annulation de l'ordre: {response['error']}")

        return response

    async def get_open_orders(self) -> Dict[str, Any]:
        """
        Récupère les ordres ouverts.

        Returns:
            Dictionnaire avec:
            - open: Dictionnaire des ordres ouverts
            - closed: Dictionnaire des ordres fermés
            - pending: Liste des ordres en attente

        Raises:
            KrakenAPIError: Si l'API retourne une erreur
        """
        response = await self.client._request('POST', 'private/OpenOrders', {}, private=True)

        # Validation de la réponse
        if 'error' in response and response['error']:
            self.logger.error(
                f"Erreur lors de la récupération des ordres ouverts: {response['error']}")
            raise KrakenAPIError(
                f"Erreur lors de la récupération des ordres ouverts: {response['error']}")

        # Validation des données
        self._validate_open_orders_response(response)

        return response

    def _validate_open_orders_response(self, response: Dict[str, Any]) -> None:
        """
        Valide la structure de la réponse des ordres ouverts.

        Args:
            response: Réponse de l'API

        Raises:
            KrakenAPIError: Si la structure est invalide
        """
        required_fields = ['open', 'closed', 'pending']

        for field in required_fields:
            if field not in response:
                self.logger.error(f"Champ manquant dans la réponse: {field}")
                raise KrakenAPIError(
                    f"Champ manquant dans la réponse: {field}")

        # Validation des ordres ouverts
        if not isinstance(response['open'], dict):
            self.logger.error(
                "Les ordres ouverts doivent être un dictionnaire")
            raise KrakenAPIError(
                "Les ordres ouverts doivent être un dictionnaire")

        # Validation des ordres fermés
        if not isinstance(response['closed'], dict):
            self.logger.error("Les ordres fermés doivent être un dictionnaire")
            raise KrakenAPIError(
                "Les ordres fermés doivent être un dictionnaire")

        # Validation des ordres en attente
        if not isinstance(response['pending'], list):
            self.logger.error("Les ordres en attente doivent être une liste")
            raise KrakenAPIError(
                "Les ordres en attente doivent être une liste")
