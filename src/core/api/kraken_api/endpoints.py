"""
Module contenant les endpoints de l'API Kraken.
"""

from typing import Dict, Any, Optional
import logging

from .validators import Validator


class KrakenEndpoints:
    def __init__(self, client):
        self.client = client
        self.validator = Validator()
        self.logger = logging.getLogger(__name__ + '.KrakenEndpoints')

    async def get_recent_spread(
            self, pair: str, since: Optional[int] = None) -> Dict[str, Any]:
        """
        Récupère le spread récent pour une paire.

        Args:
            pair: Paire de trading
            since: Timestamp depuis lequel récupérer le spread (optionnel)

        Returns:
            Dictionnaire avec:
            - spread: Liste des spreads
            - last: Timestamp du dernier spread
            Chaque spread contient:
            - time: Timestamp
            - bid: Prix d'achat
            - ask: Prix de vente

        Raises:
            ValueError: Si les paramètres sont invalides
            KrakenAPIError: Si l'API retourne une erreur
        """
        self.validator.validate_pair(pair)
        if since is not None:
            self.validator.validate_timestamp(since)

        params = {'pair': pair}
        if since is not None:
            params['since'] = since

        response = await self.client._request('GET', 'public/Spread', params)
        return response

    async def get_assets(self,
                         asset: Optional[str] = None,
                         aclass: Optional[str] = None) -> Dict[str,
                                                               Any]:
        """
        Récupère les informations sur les actifs.

        Args:
            asset: Actif (optionnel)
            aclass: Classe d'actif (optionnel)

        Returns:
            Dictionnaire avec les informations sur les actifs:
            - altname: Nom alternatif
            - aclass: Classe d'actif
            - decimals: Nombre de décimales
            - display_decimals: Nombre de décimales à afficher

        Raises:
            ValueError: Si les paramètres sont invalides
            KrakenAPIError: Si l'API retourne une erreur
        """
        params = {}

        if asset is not None:
            self.validator.validate_pair(asset)
            params['asset'] = asset

        if aclass is not None:
            if not isinstance(aclass, str):
                self.logger.error(
                    f"Type invalide pour aclass: {type(aclass).__name__}")
                raise ValueError("La classe d'actif doit être une chaîne")
            params['aclass'] = aclass

        response = await self.client._request('GET', 'public/Assets', params)
        return response

    async def get_ohlc_data(self,
                            pair: str,
                            interval: int = 1,
                            since: Optional[int] = None) -> Dict[str,
                                                                 Any]:
        """
        Récupère les données OHLC (Open/High/Low/Close) pour une paire.

        Args:
            pair: Paire de trading
            interval: Intervalle en minutes
            since: Timestamp depuis lequel récupérer les données (optionnel)

        Returns:
            Dictionnaire avec:
            - OHLC: Liste des données OHLC
            - last: Timestamp du dernier OHLC

        Raises:
            ValueError: Si les paramètres sont invalides
            KrakenAPIError: Si l'API retourne une erreur
        """
        self.validator.validate_pair(pair)
        self.validator.validate_interval(interval)
        if since is not None:
            self.validator.validate_timestamp(since)

        params = {'pair': pair, 'interval': interval}
        if since is not None:
            params['since'] = since

        response = await self.client._request('GET', 'public/OHLC', params)
        return response

    async def get_tradable_asset_pairs(self, info: Optional[str] = None, pair: Optional[str] = None) -> Dict:
        """
        Récupère les informations sur les paires de trading disponibles sur Kraken.
        
        Args:
            info: Type d'information ('info', 'leverage', 'fees', 'margin')
            pair: Paire de trading spécifique (optionnel)
            
        Returns:
            Dictionnaire avec les informations sur les paires de trading
            
        Raises:
            ValueError: Si les paramètres sont invalides
            KrakenAPIError: Si l'API retourne une erreur
        """
        try:
            # Validation des paramètres
            params = {}
            
            # Validation de info
            if info is not None:
                if not isinstance(info, str):
                    self.logger.error(f"Type invalide pour info: {type(info).__name__}")
                    raise ValueError("info doit être une chaîne")
                if not info:
                    self.logger.error("info vide")
                    raise ValueError("info ne peut pas être vide")
                valid_infos = ['info', 'leverage', 'fees', 'margin']
                if info not in valid_infos:
                    self.logger.error(f"Type d'information invalide: {info}")
                    raise ValueError(f"info doit être l'un des suivants: {', '.join(valid_infos)}")
                params['info'] = info

            # Validation de pair
            if pair is not None:
                if not isinstance(pair, str):
                    self.logger.error(f"Type invalide pour pair: {type(pair).__name__}")
                    raise ValueError("pair doit être une chaîne")
                if not pair:
                    self.logger.error("pair vide")
                    raise ValueError("pair ne peut pas être vide")
                params['pair'] = pair

            # Récupération des paires de trading
            response = await self.client._request('GET', 'public/AssetPairs', params)
            
            # Validation de la réponse
            if not isinstance(response, dict):
                self.logger.error(f"Réponse invalide: {response}")
                raise KrakenAPIError("La réponse n'est pas un dictionnaire")

            # Validation des informations des paires
            for pair_name, pair_info in response.items():
                if not isinstance(pair_info, dict):
                    self.logger.error(f"Format invalide pour la paire: {pair_info}")
                    raise KrakenAPIError(f"Les informations de la paire {pair_name} doivent être un dictionnaire")

                # Validation des champs requis
                required_fields = ['wsname', 'base', 'quote', 'lot', 'pair_decimals', 'lot_decimals',
                                 'lot_multiplier', 'leverage_buy', 'leverage_sell', 'fees', 'fees_maker',
                                 'margin_call', 'margin_stop']
                for field in required_fields:
                    if field not in pair_info:
                        self.logger.error(f"Champ manquant dans la paire {pair_name}: {field}")
                        raise KrakenAPIError(f"Le champ {field} est manquant dans la paire {pair_name}")

                # Validation des types
                string_fields = ['wsname', 'base', 'quote', 'lot']
                for field in string_fields:
                    if not isinstance(pair_info[field], str):
                        self.logger.error(f"Type invalide pour {field}: {type(pair_info[field]).__name__}")
                        raise KrakenAPIError(f"{field} doit être une chaîne")

                numeric_fields = ['pair_decimals', 'lot_decimals', 'lot_multiplier',
                                 'margin_call', 'margin_stop']
                for field in numeric_fields:
                    if not isinstance(pair_info[field], (int, float)):
                        self.logger.error(f"Type invalide pour {field}: {type(pair_info[field]).__name__}")
                        raise KrakenAPIError(f"{field} doit être un nombre")
                    if pair_info[field] < 0:
                        self.logger.error(f"Valeur invalide pour {field}: {pair_info[field]}")
                        raise KrakenAPIError(f"{field} ne peut pas être négatif")

                # Validation des listes
                for field in ['leverage_buy', 'leverage_sell', 'fees', 'fees_maker']:
                    if not isinstance(pair_info[field], list):
                        self.logger.error(f"Type invalide pour {field}: {type(pair_info[field]).__name__}")
                        raise KrakenAPIError(f"{field} doit être une liste")
                    for value in pair_info[field]:
                        if not isinstance(value, (int, float)):
                            self.logger.error(f"Type invalide dans {field}: {type(value).__name__}")
                            raise KrakenAPIError(f"Les valeurs dans {field} doivent être des nombres")

            self.logger.info("Paires de trading récupérées avec succès")
            return response
            
        except Exception as e:
            self.logger.error(f"Erreur lors de la récupération des paires de trading: {str(e)}")
            raise

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
        params = {}
        
        if pair is not None:
            self.validator.validate_pair(pair)
            params['pair'] = pair

        try:
            response = await self.client._request('GET', 'public/AssetPairs', params)
            
            # Vérifier que la réponse est un dictionnaire
            if not isinstance(response, dict):
                self.logger.error(f"Réponse inattendue de l'API: {response}")
                return {}
                
            return response
            
        except Exception as e:
            self.logger.error(f"Erreur lors de la récupération des paires d'actifs: {str(e)}")
            raise
