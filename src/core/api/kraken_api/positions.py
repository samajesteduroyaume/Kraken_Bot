"""
Module pour la gestion des positions dans l'API Kraken.
"""

from typing import Dict, Any, List, Optional
import logging

from .validators import Validator
from .exceptions import KrakenAPIError


class KrakenPositions:
    def __init__(self, client):
        """
        Initialise le gestionnaire de positions.
        
        Args:
            client: Instance de KrakenAPI
        """
        self.client = client
        self.validator = Validator()
        self.logger = logging.getLogger(__name__ + '.KrakenPositions')

    async def get_open_positions(self, txids: Optional[List[str]] = None, docalcs: bool = False) -> Dict[str, Any]:
        """
        Récupère les positions ouvertes sur le compte.

        Args:
            txids: Liste des identifiants de transaction à filtrer (optionnel)
            docalcs: Si True, calcule les valeurs des positions (peut être plus lent)

        Returns:
            Dictionnaire avec:
            - {txid}: Dictionnaire contenant les détails de la position
              - ordertxid: ID de l'ordre d'origine
              - pair: Paire de trading (ex: 'XBTUSD')
              - time: Horodatage d'ouverture
              - type: Type de position ('buy' ou 'sell')
              - ordertype: Type d'ordre
              - price: Prix d'ouverture
              - cost: Coût de la position
              - fee: Frais payés
              - vol: Volume de la position
              - vol_closed: Volume fermé
              - margin: Marge initiale
              - value: Valeur actuelle si docalcs=True
              - net: Valeur nette si docalcs=True

        Raises:
            KrakenAPIError: Si l'API retourne une erreur
        """
        # Préparation des paramètres
        data = {}
        
        if txids:
            if not isinstance(txids, list):
                txids = [txids]
            data['txid'] = ','.join(txids)
            
        if docalcs:
            data['docalcs'] = True

        # Appel à l'API
        response = await self.client._request('POST', 'private/OpenPositions', data, private=True)

        # Validation de la réponse
        if 'error' in response and response['error']:
            error_msg = f"Erreur lors de la récupération des positions ouvertes: {response['error']}"
            self.logger.error(error_msg)
            raise KrakenAPIError(error_msg)

        # Validation de la structure de la réponse
        self._validate_positions_response(response)

        return response.get('result', {})

    def _validate_positions_response(self, response: Dict[str, Any]) -> None:
        """
        Valide la structure de la réponse des positions ouvertes.

        Args:
            response: Réponse de l'API

        Raises:
            KrakenAPIError: Si la structure est invalide
        """
        if not isinstance(response, dict):
            error_msg = "La réponse de l'API doit être un dictionnaire"
            self.logger.error(error_msg)
            raise KrakenAPIError(error_msg)

        if 'result' not in response:
            error_msg = "La réponse de l'API ne contient pas de champ 'result'"
            self.logger.error(error_msg)
            raise KrakenAPIError(error_msg)

        if not isinstance(response['result'], dict):
            error_msg = "Le champ 'result' doit être un dictionnaire"
            self.logger.error(error_msg)
            raise KrakenAPIError(error_msg)

        # Vérification des champs obligatoires pour chaque position
        required_fields = [
            'ordertxid', 'pair', 'time', 'type', 'ordertype',
            'price', 'cost', 'fee', 'vol', 'vol_closed', 'margin'
        ]

        for txid, position in response['result'].items():
            if not isinstance(position, dict):
                error_msg = f"La position {txid} doit être un dictionnaire"
                self.logger.error(error_msg)
                raise KrakenAPIError(error_msg)

            for field in required_fields:
                if field not in position:
                    error_msg = f"Champ manquant dans la position {txid}: {field}"
                    self.logger.error(error_msg)
                    raise KrakenAPIError(error_msg)

    async def close_position(self, txid: str) -> Dict[str, Any]:
        """
        Ferme une position ouverte.

        Args:
            txid: Identifiant de la position à fermer

        Returns:
            Dictionnaire avec le résultat de la fermeture

        Raises:
            KrakenAPIError: Si l'API retourne une erreur
        """
        # Validation du txid
        if not txid or not isinstance(txid, str):
            error_msg = "L'identifiant de transaction (txid) est invalide"
            self.logger.error(error_msg)
            raise ValueError(error_msg)

        # Préparation des données
        data = {
            'txid': txid
        }

        # Appel à l'API
        response = await self.client._request('POST', 'private/ClosePosition', data, private=True)

        # Validation de la réponse
        if 'error' in response and response['error']:
            error_msg = f"Erreur lors de la fermeture de la position {txid}: {response['error']}"
            self.logger.error(error_msg)
            raise KrakenAPIError(error_msg)

        return response
