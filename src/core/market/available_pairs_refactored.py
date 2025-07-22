"""
Module refactorisé pour gérer la liste des paires de trading disponibles sur Kraken.

Ce module fournit une instance unique et partagée de AvailablePairs pour toute l'application.
"""

import os
import json
import time
import logging
import aiohttp
import asyncio
from pathlib import Path
from typing import Dict, List, Set, Optional, Any, Union, TypeVar, Type
from dataclasses import dataclass

# Configuration du logger
logger = logging.getLogger(__name__)

class KrakenAPIError(Exception):
    """Exception pour les erreurs liées à l'API Kraken."""
    pass


class UnsupportedTradingPairError(Exception):
    """Exception levée lorsqu'une paire de trading n'est pas reconnue."""
    
    def __init__(self, pair: str, alternatives: List[str] = None, message: str = None):
        """
        Initialise l'exception avec la paire non reconnue et des alternatives possibles.
        
        Args:
            pair: La paire de trading qui n'a pas été reconnue
            alternatives: Liste des paires alternatives suggérées
            message: Message d'erreur personnalisé
        """
        self.pair = pair
        self.alternatives = alternatives or []
        
        if not message:
            if self.alternatives:
                message = (
                    f"La paire '{pair}' n'est pas reconnue. "
                    f"Suggestions : {', '.join(self.alternatives[:3])}"
                )
            else:
                message = (
                    f"La paire '{pair}' n'est pas reconnue et aucune alternative n'a été trouvée. "
                    f"Veuillez vérifier le format (ex: 'XBT/USD') ou la disponibilité sur Kraken."
                )
                
        super().__init__(message)

T = TypeVar('T', bound='AvailablePairs')

class AvailablePairs:
    """
    Classe singleton pour gérer les paires de trading disponibles sur Kraken.
    
    Cette classe fournit des méthodes pour récupérer, valider et formater les paires
    de trading disponibles sur Kraken, avec mise en cache des résultats.
    """
    
    # Chemin du fichier de cache
    CACHE_DIR = Path("data")
    CACHE_FILE = CACHE_DIR / "available_pairs_cache.json"
    CACHE_EXPIRY = 24 * 60 * 60  # 24 heures en secondes
    
    # Mappage des symboles pour la normalisation
    SYMBOL_MAPPING = {
        'BTC': 'XBT',  # Bitcoin
        'XBT': 'XBT',  # Bitcoin (officiel)
        'BCH': 'BCH',  # Bitcoin Cash
        'BSV': 'BSV',  # Bitcoin SV
        'XDG': 'XDG',  # Dogecoin (DOGE)
        'DOGE': 'XDG',
        'XRP': 'XRP',  # Ripple
        'XLM': 'XLM',  # Stellar
        'TRX': 'TRX',  # Tron
        'ADA': 'ADA',  # Cardano
        'DOT': 'DOT',  # Polkadot
        'SOL': 'SOL',  # Solana
        'ETH': 'ETH',  # Ethereum
        'ETC': 'ETC',  # Ethereum Classic
        'LTC': 'LTC',  # Litecoin
        'USDT': 'USDT',  # Tether
        'USDC': 'USDC',  # USD Coin
        'DAI': 'DAI',   # DAI
        'EUR': 'ZEUR',  # Euro (code interne Kraken)
        'USD': 'ZUSD',  # US Dollar (code interne Kraken)
        'GBP': 'ZGBP',  # British Pound (code interne Kraken)
        'JPY': 'ZJPY',  # Japanese Yen (code interne Kraken)
        'CHF': 'CHF',   # Swiss Franc
        'CAD': 'ZCAD',  # Canadian Dollar (code interne Kraken)
        'AUD': 'AUD',   # Australian Dollar
        # Mappage inverse pour la normalisation
        'ZEUR': 'EUR',
        'ZUSD': 'USD',
        'ZGBP': 'GBP',
        'ZJPY': 'JPY',
        'ZCAD': 'CAD'
    }
    
    # Instance unique de la classe
    _instance = None
    _initialized = False
    
    def __new__(cls: Type[T], *args: Any, **kwargs: Any) -> T:
        """Implémentation du pattern Singleton."""
        if cls._instance is None:
            cls._instance = super(AvailablePairs, cls).__new__(cls)
        return cls._instance
    
    def __init__(self, api: Optional[Any] = None) -> None:
        """Initialise le gestionnaire de paires disponibles."""
        # Ne s'exécute qu'une seule fois
        if self._initialized:
            return
            
        # Initialisation des attributs d'instance
        self._pairs_data: Dict[str, Dict] = {}
        self._wsname_to_pair_id: Dict[str, str] = {}
        self._base_currencies: Set[str] = set()
        self._quote_currencies: Set[str] = set()
        self._last_updated: Optional[float] = None
        self._api = api
        self._initialized = True
        self.KRAKEN_API_URL = "https://api.kraken.com/0/public/AssetPairs"
        
        # Créer le répertoire de cache s'il n'existe pas
        os.makedirs(self.CACHE_DIR, exist_ok=True)
    
    @classmethod
    async def create(cls: Type[T], api: Optional[Any] = None) -> T:
        """Méthode de fabrique asynchrone pour créer et initialiser l'instance."""
        instance = cls(api)
        await instance.initialize()
        return instance
    
    async def initialize(self, use_cache: bool = True) -> None:
        """Initialise les données des paires disponibles."""
        if not self._initialized:
            logger.warning("L'instance n'est pas correctement initialisée. Utilisez AvailablePairs.create()")
            return
            
        logger.info("Initialisation des paires disponibles...")
        
        # Essayer de charger depuis le cache si demandé
        if use_cache and self._load_from_cache():
            logger.info("Données des paires chargées depuis le cache")
            return
            
        # Sinon, récupérer depuis l'API
        logger.info("Récupération des paires depuis l'API Kraken...")
        await self._fetch_from_api()
    
    def _load_from_cache(self) -> bool:
        """Charge les données des paires depuis le cache local."""
        try:
            if not self.CACHE_FILE.exists():
                logger.debug("Aucun fichier de cache trouvé")
                return False
                
            # Vérifier l'âge du cache
            file_age = time.time() - self.CACHE_FILE.stat().st_mtime
            if file_age > self.CACHE_EXPIRY:
                logger.debug("Cache expiré, mise à jour nécessaire")
                return False
                
            # Charger les données
            logger.debug(f"Chargement du cache depuis {self.CACHE_FILE}")
            with open(self.CACHE_FILE, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            self._pairs_data = data.get('pairs', {})
            self._last_updated = data.get('last_updated')
            
            # Reconstruire les index
            self._build_indexes()
            
            logger.info(f"{len(self._pairs_data)} paires chargées depuis le cache")
            return True
            
        except (json.JSONDecodeError, OSError, KeyError) as e:
            logger.warning(f"Erreur lors du chargement du cache des paires: {e}")
            return False
    
    def _save_to_cache(self) -> None:
        """Sauvegarde les données des paires dans le cache local."""
        try:
            # Préparer les données à sauvegarder
            data = {
                'pairs': self._pairs_data,
                'last_updated': self._last_updated,
                'metadata': {
                    'version': '1.0',
                    'source': 'Kraken API',
                    'generated_at': time.strftime('%Y-%m-%d %H:%M:%S')
                }
            }
            
            # Sauvegarder dans un fichier temporaire d'abord
            temp_file = str(self.CACHE_FILE) + '.tmp'
            with open(temp_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
                
            # Remplacer l'ancien fichier par le nouveau (opération atomique)
            if os.path.exists(self.CACHE_FILE):
                os.remove(self.CACHE_FILE)
            os.rename(temp_file, self.CACHE_FILE)
            
            logger.debug(f"{len(self._pairs_data)} paires sauvegardées dans le cache")
            
        except (OSError, TypeError, json.JSONEncodeError) as e:
            logger.error(f"Erreur lors de la sauvegarde du cache des paires: {e}")
    
    async def _fetch_from_api(self) -> None:
        """Récupère les données des paires depuis l'API Kraken."""
        try:
            async with aiohttp.ClientSession() as session:
                logger.info(f"Récupération des paires depuis l'API Kraken: {self.KRAKEN_API_URL}")
                
                async with session.get(self.KRAKEN_API_URL) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        raise KrakenAPIError(
                            f"Erreur HTTP {response.status} lors de la récupération des paires: {error_text}"
                        )
                    
                    data = await response.json()
                    
                    if 'error' in data and data['error']:
                        error_msg = ", ".join(data['error']) if isinstance(data['error'], list) else str(data['error'])
                        raise KrakenAPIError(f"Erreur de l'API Kraken: {error_msg}")
                    
                    if not isinstance(data, dict) or 'result' not in data:
                        raise KrakenAPIError("Format de réponse invalide de l'API Kraken")
                    
                    self._pairs_data = data['result']
                    self._last_updated = time.time()
                    self._build_indexes()
                    self._save_to_cache()
                    
        except (aiohttp.ClientError, asyncio.TimeoutError) as e:
            logger.error(f"Erreur réseau lors de la récupération des paires: {e}")
            raise KrakenAPIError(f"Erreur réseau: {e}")
        except json.JSONDecodeError as e:
            logger.error(f"Erreur de décodage JSON de la réponse de l'API: {e}")
            raise KrakenAPIError("Réponse API invalide (JSON mal formé)")
        except Exception as e:
            logger.error(f"Erreur inattendue lors de la récupération des paires: {e}", exc_info=True)
            raise KrakenAPIError(f"Erreur inattendue: {e}")
    
    def _build_indexes(self) -> None:
        """Construit les index pour une recherche plus rapide des paires."""
        self._wsname_to_pair_id = {}
        self._base_currencies = set()
        self._quote_currencies = set()
        
        if not isinstance(self._pairs_data, dict):
            logger.warning("Aucune donnée de paire disponible pour l'indexation")
            return
        
        logger.debug(f"Construction des index pour {len(self._pairs_data)} paires...")
        
        # D'abord, indexer toutes les paires avec leurs codes internes
        internal_quote_currencies = set()
        
        for pair_id, pair_info in self._pairs_data.items():
            if not isinstance(pair_info, dict):
                logger.warning(f"Format de paire invalide pour l'ID: {pair_id}")
                continue
                
            try:
                # Index par wsname (format: BASE/QUOTE)
                wsname = pair_info.get('wsname')
                if wsname and isinstance(wsname, str):
                    # Normaliser la casse pour éviter les doublons
                    normalized_wsname = wsname.upper()
                    self._wsname_to_pair_id[normalized_wsname] = pair_id
                
                # Index par devise de base (ex: XBT, ETH, etc.)
                base = pair_info.get('base')
                if base and isinstance(base, str):
                    self._base_currencies.add(base.upper())
                
                # Index par devise de cotation (ex: ZUSD, ZEUR, etc.)
                quote = pair_info.get('quote')
                if quote and isinstance(quote, str):
                    quote_upper = quote.upper()
                    internal_quote_currencies.add(quote_upper)
                    
                    # Ajouter la version standard si c'est une devise interne Kraken
                    if quote_upper in self.SYMBOL_MAPPING:
                        standard_quote = self.SYMBOL_MAPPING[quote_upper]
                        self._quote_currencies.add(standard_quote)
                    else:
                        # Pour les devises sans code interne (comme USDT, USDC)
                        self._quote_currencies.add(quote_upper)
                    
            except Exception as e:
                logger.warning(f"Erreur lors de l'indexation de la paire {pair_id}: {e}", exc_info=True)
        
        # Maintenant, ajouter les codes standards pour les devises internes
        for internal_code, standard_code in self.SYMBOL_MAPPING.items():
            if internal_code in internal_quote_currencies and standard_code not in self._quote_currencies:
                self._quote_currencies.add(standard_code)
        
        logger.info(
            f"Indexation terminée: {len(self._wsname_to_pair_id)} paires, "
            f"{len(self._base_currencies)} devises de base, "
            f"{len(self._quote_currencies)} devises de cotation"
        )
    
    def is_pair_supported(self, pair: str, raise_on_error: bool = False) -> Union[bool, str]:
        """Vérifie si une paire est supportée par Kraken.
        
        Args:
            pair: La paire à vérifier (ex: 'XBT/USD', 'btc-eur')
            raise_on_error: Si True, lève une exception avec des détails si la paire n'est pas supportée
            
        Returns:
            bool: True si la paire est supportée, False sinon
            
        Raises:
            UnsupportedTradingPairError: Si la paire n'est pas supportée et que raise_on_error est True
        """
        if not isinstance(pair, str) or not pair.strip():
            if raise_on_error:
                raise UnsupportedTradingPairError(
                    pair=str(pair) if pair is not None else 'None',
                    message="Le format de la paire est invalide (doit être une chaîne non vide)"
                )
            return False
            
        if not self._initialized:
            error_msg = "Les données des paires ne sont pas encore initialisées"
            if raise_on_error:
                raise KrakenAPIError(error_msg)
            logger.warning(error_msg)
            return False
            
        # Vérifier d'abord le format wsname (ex: "XBT/USD")
        normalized_wsname = pair.upper()
        if normalized_wsname in self._wsname_to_pair_id:
            return True
            
        # Vérifier les formats alternatifs
        try:
            normalized = self.normalize_pair(pair, raise_on_error=raise_on_error)
            return bool(normalized)
        except UnsupportedTradingPairError:
            if raise_on_error:
                raise
            return False
        except Exception as e:
            if raise_on_error:
                raise UnsupportedTradingPairError(
                    pair=pair,
                    message=f"Erreur lors de la vérification de la paire: {str(e)}"
                )
            logger.debug(f"Erreur lors de la vérification de la paire {pair}: {e}", exc_info=True)
            return False
    
    def normalize_pair(self, pair: str, raise_on_error: bool = False) -> Optional[str]:
        """Normalise le nom d'une paire de trading.
        
        Args:
            pair: Nom de la paire à normaliser (ex: 'btc-usd', 'XBT/EUR', 'XXBTZUSD', 'dogeusd')
            raise_on_error: Si True, lève une exception si la paire n'est pas reconnue
            
        Returns:
            str: Nom normalisé de la paire (ex: 'XBT/USD'), ou None si non supportée
            
        Raises:
            UnsupportedTradingPairError: Si la paire n'est pas reconnue et que raise_on_error est True
        """
        if not pair or not isinstance(pair, str):
            if raise_on_error:
                raise UnsupportedTradingPairError(
                    pair=str(pair) if pair is not None else 'None',
                    message="Le format de la paire est invalide (doit être une chaîne non vide)"
                )
            return None
                
        if not self._initialized:
            error_msg = "Les données des paires ne sont pas encore initialisées"
            if raise_on_error:
                raise KrakenAPIError(error_msg)
            logger.warning(error_msg)
            return None
            
        # Nettoyer la chaîne d'entrée et normaliser les séparateurs
        clean_pair = pair.strip().upper()
        # Remplacer les tirets par des barres obliques pour la normalisation
        if '-' in clean_pair and '/' not in clean_pair:
            clean_pair = clean_pair.replace('-', '/')
        clean_pair_no_sep = clean_pair.replace('/', '').replace('-', '').replace('_', '').replace(' ', '')
        logger.debug(f"Recherche de la paire: {clean_pair} (nettoyée: {clean_pair_no_sep})")
        
        # 1. Vérifier si c'est déjà au format wsname (ex: "XBT/USD")
        if '/' in clean_pair and clean_pair.count('/') == 1:
            # Essayer d'abord avec la paire telle quelle
            if clean_pair in self._wsname_to_pair_id:
                return clean_pair
            
            # Séparer la base et la cotation
            base, quote = clean_pair.split('/')
            
            # Liste des variantes à essayer
            variants = []
            
            # 1. Gérer les cas spéciaux comme DOGE/XDG
            if base in ['DOGE', 'XDG']:
                variants.extend([
                    ('XDG', quote),  # Essayer avec XDG
                    ('XDG', self.SYMBOL_MAPPING.get(quote, quote))  # Essayer avec la cotation normalisée
                ])
            
            # 2. Gérer les cas spéciaux comme BTC/XBT
            if base in ['BTC', 'XBT']:
                other_base = 'XBT' if base == 'BTC' else 'BTC'
                variants.extend([
                    (other_base, quote),  # Essayer avec l'autre base
                    (other_base, self.SYMBOL_MAPPING.get(quote, quote))  # Essayer avec la cotation normalisée
                ])
            
            # 3. Essayer avec la base originale et la cotation normalisée
            variants.append((base, self.SYMBOL_MAPPING.get(quote, quote)))
            
            # 4. Essayer avec la base normalisée et la cotation originale
            normalized_base = self.SYMBOL_MAPPING.get(base, base)
            if normalized_base != base:
                variants.append((normalized_base, quote))
            
            # 5. Essayer avec la base et la cotation normalisées
            variants.append((normalized_base, self.SYMBOL_MAPPING.get(quote, quote)))
            
            # Essayer toutes les variantes
            for base_var, quote_var in variants:
                test_pair = f"{base_var}/{quote_var}"
                if test_pair in self._wsname_to_pair_id:
                    return test_pair
                    
            # Si on arrive ici, essayer avec des variantes de la cotation
            # en inversant le mappage pour les devises internes (ex: EUR -> ZEUR)
            for internal, standard in self.SYMBOL_MAPPING.items():
                if standard == quote and internal != standard:
                    test_pair = f"{base}/{internal}"
                    if test_pair in self._wsname_to_pair_id:
                        return test_pair
            
            # Essayer avec la paire originale (au cas où)
            if f"{base}/{quote}" in self._wsname_to_pair_id:
                return f"{base}/{quote}"
        
        # 2. Vérifier directement dans les identifiants de paires natifs (ex: 'XXBTZUSD')
        if clean_pair in self._pairs_data:
            return self._pairs_data[clean_pair].get('wsname')
        
        # 3. Essayer avec des formats sans séparateur (ex: 'BTCEUR')
        # Vérifier si la chaîne se termine par une devise de cotation connue
        all_currencies = set(self._quote_currencies)
        # Ajouter les devises internes (comme ZEUR, ZUSD, etc.)
        all_currencies.update([c for c in self.SYMBOL_MAPPING if c.startswith('Z')])
        
        # Trier par longueur décroissante pour gérer les cas comme 'USDT' avant 'USD'
        for quote_currency in sorted(all_currencies, key=len, reverse=True):
            if clean_pair.endswith(quote_currency):
                base = clean_pair[:-len(quote_currency)]
                if not base:
                    continue
                
                # Liste des variantes de base à essayer
                base_variants = [base]
                
                # Gérer les cas spéciaux comme BTC/XBT et DOGE/XDG
                if base == 'BTC':
                    base_variants.append('XBT')
                elif base == 'XBT':
                    base_variants.append('BTC')
                elif base in ['DOGE', 'XDG']:
                    base_variants.append('XDG' if base == 'DOGE' else 'DOGE')
                
                # Gérer les cas où la base pourrait être un symbole mappé
                mapped_base = self.SYMBOL_MAPPING.get(base, base)
                if mapped_base != base:
                    base_variants.append(mapped_base)
                
                # Liste des variantes de cotation à essayer
                quote_variants = [quote_currency]
                # Ajouter la version standard si c'est une devise interne (ex: ZEUR -> EUR)
                if quote_currency in self.SYMBOL_MAPPING and quote_currency != self.SYMBOL_MAPPING[quote_currency]:
                    quote_variants.append(self.SYMBOL_MAPPING[quote_currency])
                # Ajouter la version interne si c'est une devise standard (ex: EUR -> ZEUR)
                for internal, standard in self.SYMBOL_MAPPING.items():
                    if standard == quote_currency and internal != standard:
                        quote_variants.append(internal)
                        break
                
                # Essayer toutes les combinaisons de bases et de cotations
                for base_var in set(base_variants):  # Utiliser set pour éliminer les doublons
                    for quote_var in set(quote_variants):
                        test_pair = f"{base_var}/{quote_var}"
                        if test_pair in self._wsname_to_pair_id:
                            return test_pair
        
        # 4. Essayer avec différents séparateurs
        separators = ['-', '_', ' ']
        for sep in separators:
            if sep in clean_pair and clean_pair.count(sep) == 1:
                try:
                    base, quote = clean_pair.split(sep)
                    base = base.strip().upper()
                    quote = quote.strip().upper()
                    
                    # Liste des variantes à essayer
                    variants = []
                    
                    # Gérer les cas spéciaux comme DOGE/XDG
                    base_variants = [base]
                    if base in ['DOGE', 'XDG']:
                        base_variants.append('XDG' if base == 'DOGE' else 'DOGE')
                    # Gérer les cas spéciaux comme BTC/XBT
                    if base == 'BTC':
                        base_variants.append('XBT')
                    elif base == 'XBT':
                        base_variants.append('BTC')
                    
                    # Gérer les symboles mappés pour la base
                    for b in base_variants.copy():
                        mapped = self.SYMBOL_MAPPING.get(b, b)
                        if mapped != b and mapped not in base_variants:
                            base_variants.append(mapped)
                    
                    # Gérer les symboles mappés pour la cotation
                    quote_variants = [quote]
                    mapped_quote = self.SYMBOL_MAPPING.get(quote, quote)
                    if mapped_quote != quote and mapped_quote not in quote_variants:
                        quote_variants.append(mapped_quote)
                    
                    # Essayer toutes les combinaisons de bases et de cotations
                    for b in base_variants:
                        for q in quote_variants:
                            test_pair = f"{b}/{q}"
                            if test_pair in self._wsname_to_pair_id:
                                return test_pair
                except Exception as e:
                    logger.debug(f"Erreur lors du traitement de la paire {clean_pair} avec le séparateur '{sep}': {e}")
                    
        # 6. Si on arrive ici, essayer une correspondance partielle (contient)
        # C'est une solution de dernier recours
        logger.debug("Recherche de correspondance partielle...")
        for pair_id, pair_info in self._pairs_data.items():
            # Vérifier si la paire contient la chaîne recherchée (ou l'inverse)
            wsname = pair_info.get('wsname', '').upper()
            altname = pair_info.get('altname', '').upper()
            
            if (clean_pair_no_sep in wsname.replace('/', '') or 
                clean_pair_no_sep in altname.replace('-', '').replace('_', '').replace('/', '') or
                clean_pair_no_sep in pair_id.upper().replace('/', '')):
                logger.debug(f"Correspondance partielle trouvée: {pair_id} -> {wsname}")
                return pair_info.get('wsname')
        
        # Si on arrive ici, la paire n'est pas supportée directement
        logger.debug(f"Paire non reconnue directement: {pair}")
        
        # Essayer de trouver des alternatives
        logger.debug(f"Appel de _find_alternative_pairs avec clean_pair={clean_pair}, clean_pair_no_sep={clean_pair_no_sep}")
        alternatives = self._find_alternative_pairs(clean_pair, clean_pair_no_sep)
        logger.debug(f"Alternatives retournées par _find_alternative_pairs: {alternatives}")
        
        if raise_on_error and not alternatives:
            raise UnsupportedTradingPairError(pair=pair)
            
        if alternatives:
            logger.debug(f"Suggestions d'alternatives pour {pair}: {alternatives}")
            if raise_on_error:
                raise UnsupportedTradingPairError(pair=pair, alternatives=alternatives)
            # En mode silencieux, on retourne la première alternative
            return alternatives[0]
            
        logger.debug(f"Aucune alternative trouvée pour la paire: {pair}")
        if raise_on_error:
            raise UnsupportedTradingPairError(pair=pair)
        return None
    
    def get_pair_info(self, pair: str) -> Optional[Dict[str, Any]]:
        """Récupère les informations d'une paire de trading."""
        normalized = self.normalize_pair(pair)
        if not normalized:
            return None
            
        pair_id = self._wsname_to_pair_id.get(normalized)
        if not pair_id:
            return None
            
        return self._pairs_data.get(pair_id)
    
    def get_available_pairs(self, quote_currency: Optional[str] = None) -> List[str]:
        """Retourne la liste des paires disponibles, éventuellement filtrées par devise de cotation."""
        if not quote_currency:
            return sorted(self._wsname_to_pair_id.keys())
            
        # Normaliser la devise de cotation
        quote_upper = quote_currency.upper()
        if quote_upper in self.SYMBOL_MAPPING:
            quote_upper = self.SYMBOL_MAPPING[quote_upper]
            
        # Filtrer les paires par devise de cotation
        result = []
        for wsname, pair_id in self._wsname_to_pair_id.items():
            pair_info = self._pairs_data.get(pair_id, {})
            quote = pair_info.get('quote', '').upper()
            
            # Vérifier si la devise de cotation correspond
            if quote == quote_upper or (quote in self.SYMBOL_MAPPING and 
                                      self.SYMBOL_MAPPING[quote] == quote_upper):
                result.append(wsname)
                
        return sorted(result)
    
    def get_base_currencies(self) -> Set[str]:
        """Retourne l'ensemble des devises de base disponibles."""
        return self._base_currencies
    def get_quote_currencies(self) -> Set[str]:
        """Retourne l'ensemble des devises de cotation disponibles."""
        return self._quote_currencies
    
    def _find_alternative_pairs(self, pair: str, pair_no_sep: str) -> List[str]:
        """Trouve des paires alternatives pour une paire non disponible.
        
        Cette méthode tente de trouver des paires alternatives en plusieurs étapes :
        1. Cherche des paires avec la même devise de base
        2. Cherche des paires avec la même devise de cotation
        3. Propose des paires intermédiaires via des devises communes (ex: BTC, ETH, USDT)
        4. Propose des paires similaires basées sur la distance de Levenshtein
        
        Args:
            pair: La paire originale (ex: 'ZRX/ETH')
            pair_no_sep: La paire sans séparateur (ex: 'ZRXETH')
            
        Returns:
            Liste des paires alternatives disponibles, triées par pertinence. 
            Retourne une liste vide en cas d'erreur.
            
        Note:
            L'ordre de retour est important : plus une alternative est pertinente,
            plus elle apparaît tôt dans la liste.
        """
        logger.debug(f"\n=== DÉBUT DE _find_alternative_pairs ===")
        logger.debug(f"Paire d'entrée: {pair}, pair_no_sep: {pair_no_sep}")
        
        # Vérifier que les données sont initialisées
        if not self._initialized:
            logger.warning("Les données des paires ne sont pas encore initialisées")
            return []
            
        # Initialisation des variables
        alternatives = []
        base, quote = None, None
        
        # 1. Essayer de séparer la paire en base et cotation
        logger.debug("Tentative de séparation de la paire en base et cotation...")
        for sep in ['/', '-', '_', ' ']:
            if sep in pair and pair.count(sep) == 1:
                try:
                    parts = pair.upper().split(sep)
                    if len(parts) == 2:
                        base, quote = parts
                        logger.debug(f"Paire séparée avec succès: base='{base}', quote='{quote}' (séparateur: '{sep}')")
                        break
                except (ValueError, AttributeError) as e:
                    logger.debug(f"Erreur lors de la séparation de la paire {pair} avec le séparateur '{sep}': {e}")
                    continue
        
        # Si on n'a pas de base ou de cotation, essayer de trouver des paires similaires
        if not base or not quote:
            logger.debug("Impossible de séparer la paire, recherche de paires similaires...")
            
            try:
                # Vérifier que les données sont initialisées
                if not hasattr(self, '_wsname_to_pair_id') or not self._wsname_to_pair_id:
                    logger.warning("Les données des paires ne sont pas encore initialisées")
                    return []
                    
                # Convertir en liste pour éviter les problèmes de concurrence
                all_pairs = list(self._wsname_to_pair_id.keys())
                
                # Calculer la similarité avec toutes les paires
                scored_pairs = []
                for p in all_pairs:
                    try:
                        if not isinstance(p, str):
                            continue
                            
                        # Utiliser la distance de Levenshtein pour la similarité
                        score = self.similarity_score(pair_no_sep, p.replace('/', ''))
                        if score > 0.5:  # Seuil de similarité minimum
                            scored_pairs.append((p, score))
                    except Exception as e:
                        logger.debug(f"Erreur lors du calcul de similarité pour la paire {p}: {e}")
                        continue
                
                # Trier par score décroissant
                scored_pairs.sort(key=lambda x: x[1], reverse=True)
                
                # Retourner les meilleures correspondances (max 5)
                alternatives = [p[0] for p in scored_pairs[:5] if isinstance(p[0], str)]
                logger.debug(f"Paires similaires trouvées: {alternatives}")
                
                return alternatives
                
            except Exception as e:
                logger.error(f"Erreur lors de la recherche de paires similaires: {e}", exc_info=True)
                return []
        
        # Si on n'a toujours pas de base ou de cotation, essayer avec les codes de devises connus
        if not base or not quote:
            logger.debug("Impossible de séparer la paire avec les séparateurs standards, tentative avec les devises connues...")
            
            try:
                # Trier les devises par longueur décroissante pour éviter les faux positifs
                # (ex: 'USDT' avant 'USD')
                for curr in sorted(self._quote_currencies, key=len, reverse=True):
                    if not isinstance(curr, str):
                        continue
                        
                    curr_upper = curr.upper()
                    pair_upper = pair_no_sep.upper()
                    
                    # Vérifier si la paire se termine par la devise de cotation
                    if pair_upper.endswith(curr_upper):
                        base = pair_upper[:-len(curr_upper)]
                        quote = curr_upper
                        
                        # Vérifier que la base n'est pas vide
                        if base:
                            logger.debug(f"Paire déduite en cherchant les devises connues: base='{base}', quote='{quote}'")
                            break
            except Exception as e:
                logger.error(f"Erreur lors de la recherche de devises connues: {e}", exc_info=True)
            
            # Si on n'a toujours pas trouvé, essayer avec les symboles mappés
            if not base or not quote:
                try:
                    for symbol, mapped in self.SYMBOL_MAPPING.items():
                        if not isinstance(symbol, str) or not isinstance(mapped, str):
                            continue
                            
                        symbol_upper = symbol.upper()
                        pair_upper = pair_no_sep.upper()
                        
                        # Vérifier si la paire se termine par le symbole
                        if pair_upper.endswith(symbol_upper):
                            base = pair_upper[:-len(symbol_upper)]
                            quote = mapped.upper()
                            
                            # Vérifier que la base n'est pas vide
                            if base:
                                logger.debug(f"Paire déduite avec le mappage de symboles: base='{base}', quote='{quote}' (mappage: {symbol} -> {mapped})")
                                break
                except Exception as e:
                    logger.error(f"Erreur lors du traitement des symboles mappés: {e}", exc_info=True)
        
        # Si on n'a toujours pas de base ou de cotation, essayer avec des correspondances partielles
        if not base or not quote:
            logger.debug("Tentative de correspondance partielle avec les paires disponibles...")
            
            # Trier les paires disponibles par popularité (les plus communes en premier)
            common_pairs = [
                'XBT/USD', 'ETH/USD', 'XRP/USD', 'LTC/USD', 'BCH/USD',
                'XBT/EUR', 'ETH/EUR', 'XRP/EUR', 'LTC/EUR', 'BCH/EUR',
                'XBT/USDT', 'ETH/USDT', 'XRP/USDT', 'LTC/USDT', 'BCH/USDT'
            ]
            
            # Ajouter les paires disponibles qui ne sont pas déjà dans la liste
            all_pairs = common_pairs + [p for p in self._wsname_to_pair_id.keys() if p not in common_pairs]
            
            # Trier par similarité avec la paire d'entrée
            def similarity_score(a: str, b: str) -> float:
                """Calcule un score de similarité entre deux chaînes."""
                a = a.upper().replace('/', '').replace('-', '').replace('_', '')
                b = b.upper().replace('/', '').replace('-', '').replace('_', '')
                
                # Score basé sur la sous-chaîne commune la plus longue
                max_len = 0
                for i in range(len(a)):
                    for j in range(i + 1, len(a) + 1):
                        if a[i:j] in b and j - i > max_len:
                            max_len = j - i
                
                return max_len / max(len(a), len(b), 1)
            
            # Trier les paires par similarité
            scored_pairs = [(p, similarity_score(pair_no_sep, p)) for p in all_pairs]
            scored_pairs.sort(key=lambda x: (-x[1], x[0]))  # Trier par score décroissant
            
            # Prendre les 3 meilleures correspondances
            best_matches = [p for p, score in scored_pairs[:3] if score > 0.3]
            
            if best_matches:
                logger.debug(f"Meilleures correspondances trouvées: {best_matches}")
                return best_matches
        
        # Si on n'a toujours pas de base ou de cotation, on ne peut rien faire
        if not base or not quote:
            logger.debug("Impossible de déterminer la base et la cotation, retour d'une liste vide")
            return []
            
        logger.debug(f"\n=== RECHERCHE D'ALTERNATIVES POUR {base}/{quote} ===")
        logger.debug(f"Devises de base disponibles: {sorted(list(self._base_currencies))[:10]}...")
        logger.debug(f"Devises de cotation disponibles: {sorted(list(self._quote_currencies))[:10]}...")
        
        # 2. Chercher des paires avec la même base ou la même cotation
        base_pairs = [p for p in self._wsname_to_pair_id.keys() 
                     if p.split('/')[0] == base and p.split('/')[1] in self._quote_currencies]
        quote_pairs = [p for p in self._wsname_to_pair_id.keys() 
                      if p.split('/')[1] == quote and p.split('/')[0] in self._base_currencies]
        
        logger.debug(f"Paires avec la même base ({base}): {base_pairs}")
        logger.debug(f"Paires avec la même cotation ({quote}): {quote_pairs}")
        
        # 3. Ajouter les paires alternatives à la liste des alternatives
        alternatives = list(set(base_pairs + quote_pairs))
        
        # 4. Si on n'a pas trouvé d'alternatives, essayer de trouver des paires intermédiaires
        if not alternatives:
            logger.debug("Aucune alternative directe trouvée, recherche de paires intermédiaires...")
            alternatives = self._find_intermediate_pairs(base, quote)
        
        # 5. Trier les alternatives par pertinence
        if alternatives:
            # Donner la priorité aux paires avec la même base, puis à celles avec la même cotation
            alternatives.sort(key=lambda x: (not x.startswith(f"{base}/"), x))
            logger.debug(f"Alternatives triées: {alternatives}")
        
        logger.debug(f"=== FIN DE _find_alternative_pairs - {len(alternatives)} alternatives trouvées ===\n")
        return alternatives
    
    def _find_intermediate_pairs(self, base: str, quote: str) -> List[str]:
        """Trouve des paires intermédiaires pour convertir de la base à la cotation.
        
        Cette méthode tente de trouver un chemin de conversion entre deux devises en passant par une ou plusieurs
        paires intermédiaires. Par exemple, pour convertir ZRX en ETH, on peut passer par XBT (ZRX/XBT -> XBT/ETH).
        
        L'algorithme procède en plusieurs étapes :
        1. Vérifie d'abord si une paire directe existe
        2. Cherche des paires avec la même base ou la même cotation
        3. Cherche des chemins à une étape via des devises intermédiaires courantes
        4. En dernier recours, cherche des paires contenant la base ou la cotation
        
        Args:
            base: Code de la devise de base (ex: 'ZRX')
            quote: Code de la devise de cotation (ex: 'ETH')
            
        Returns:
            Liste des paires à utiliser pour la conversion, dans l'ordre d'utilisation
            
        Example:
            >>> pairs = _find_intermediate_pairs('ZRX', 'ETH')
            >>> print(pairs)
            ['ZRX/XBT', 'XBT/ETH']
        """
        logger.debug(f"\n=== DÉBUT DE _find_intermediate_pairs ===")
        logger.debug(f"Recherche d'un chemin de {base} à {quote}")
        
        try:
            # Vérifier que les données sont initialisées
            if not hasattr(self, '_wsname_to_pair_id') or not self._wsname_to_pair_id:
                logger.warning("Les données des paires ne sont pas encore initialisées")
                return []
                
            # 1. Vérifier si une paire directe existe
            direct_pair = f"{base}/{quote}"
            if direct_pair in self._wsname_to_pair_id:
                logger.debug(f"Paire directe trouvée: {direct_pair}")
                return [direct_pair]
                
            # 2. Chercher des paires avec la même base ou la même cotation
            base_pairs = []
            quote_pairs = []
            
            try:
                base_pairs = [p for p in self._wsname_to_pair_id.keys() 
                           if isinstance(p, str) and 
                           len(p.split('/')) == 2 and 
                           p.split('/')[0] == base and 
                           p.split('/')[1] in getattr(self, '_quote_currencies', set())]
                
                quote_pairs = [p for p in self._wsname_to_pair_id.keys() 
                            if isinstance(p, str) and 
                            len(p.split('/')) == 2 and 
                            p.split('/')[1] == quote and 
                            p.split('/')[0] in getattr(self, '_base_currencies', set())]
            except Exception as e:
                logger.error(f"Erreur lors de la recherche des paires de base/quote: {e}", exc_info=True)
            
            # 3. Si on a des paires avec la même base et la même cotation, 
            # essayer de trouver un chemin via une devise intermédiaire
            if base_pairs and quote_pairs:
                # Essayer d'abord avec des devises intermédiaires courantes
                common_intermediates = ['XBT', 'ETH', 'USDT', 'USD', 'EUR']
                
                for intermediate in common_intermediates:
                    try:
                        # Vérifier si on peut convertir de la base à l'intermédiaire
                        base_to_intermediate = f"{base}/{intermediate}"
                        intermediate_to_quote = f"{intermediate}/{quote}"
                        
                        if (base_to_intermediate in self._wsname_to_pair_id and 
                            intermediate_to_quote in self._wsname_to_pair_id):
                            logger.debug(f"Chemin trouvé via {intermediate}: {base_to_intermediate} -> {intermediate_to_quote}")
                            return [base_to_intermediate, intermediate_to_quote]
                    except Exception as e:
                        logger.debug(f"Erreur lors de la vérification de la paire intermédiaire {intermediate}: {e}")
                        continue
            
            # 4. Si on n'a pas trouvé, essayer avec n'importe quelle devise intermédiaire
            # qui permet de faire la conversion
            if base_pairs and quote_pairs:
                try:
                    # Pour chaque paire de base, essayer de trouver une paire de cotation compatible
                    for bp in base_pairs:
                        if not isinstance(bp, str) or '/' not in bp:
                            continue
                            
                        try:
                            _, base_quote = bp.split('/')
                            
                            for qp in quote_pairs:
                                if not isinstance(qp, str) or '/' not in qp:
                                    continue
                                    
                                try:
                                    qp_base, _ = qp.split('/')
                                    if base_quote == qp_base:
                                        # On a un chemin en deux étapes
                                        logger.debug(f"Chemin en deux étapes trouvé: {bp} -> {qp}")
                                        return [bp, qp]
                                except Exception as e:
                                    logger.debug(f"Erreur lors du traitement de la paire de cotation {qp}: {e}")
                                    continue
                        except Exception as e:
                            logger.debug(f"Erreur lors du traitement de la paire de base {bp}: {e}")
                            continue
                except Exception as e:
                    logger.error(f"Erreur lors de la recherche de chemins en deux étapes: {e}", exc_info=True)
            
            # 5. En dernier recours, retourner les paires avec la même base ou la même cotation
            if base_pairs:
                logger.debug(f"Retour des paires avec la même base: {base_pairs}")
                return base_pairs
                
            if quote_pairs:
                logger.debug(f"Retour des paires avec la même cotation: {quote_pairs}")
                return quote_pairs
                
            # 6. Si on n'a rien trouvé, essayer de trouver des paires qui contiennent la base ou la cotation
            try:
                # Chercher des paires similaires contenant la base ou la cotation
                similar_base = [p for p in self._wsname_to_pair_id.keys() 
                             if isinstance(p, str) and 
                             '/' in p and  # S'assurer que c'est une paire valide
                             base in p and 
                             p.split('/')[1] in getattr(self, '_quote_currencies', set())]
                
                similar_quote = [p for p in self._wsname_to_pair_id.keys() 
                              if isinstance(p, str) and 
                              '/' in p and  # S'assurer que c'est une paire valide
                              quote in p and
                              p.split('/')[0] in getattr(self, '_base_currencies', set())]
                
                if similar_base:
                    logger.debug(f"Paires similaires trouvées pour la base {base}: {similar_base}")
                    return similar_base
                    
                if similar_quote:
                    logger.debug(f"Paires similaires trouvées pour la cotation {quote}: {similar_quote}")
                    return similar_quote
                    
            except Exception as e:
                logger.error(f"Erreur lors de la recherche de paires similaires: {e}", exc_info=True)
                
            logger.debug("Aucun chemin trouvé via les paires intermédiaires")
            return []
            
        except Exception as e:
            logger.error(f"Erreur critique dans _find_intermediate_pairs: {e}", exc_info=True)
            return []
            
        # Cette partie du code n'est plus nécessaire car elle était redondante avec la logique précédente
        # et contenait des références à des variables non définies (pair1_exists, pair2_exists, etc.)
        # Elle a été remplacée par une logique plus robuste plus haut dans la fonction
        
        # Liste des devises communes par ordre de priorité (du plus au moins liquide)
        common_currencies = ['XBT', 'ETH', 'USDT', 'USDC', 'EUR', 'USD', 'DAI', 'GBP', 'JPY', 'CAD', 'AUD', 'CHF']
        priority_currencies = {c: i for i, c in enumerate(common_currencies)}
        
        def path_priority(path):
            """Calcule un score de priorité pour un chemin de conversion.
            
            Un score plus bas signifie une meilleure priorité.
            """
            # Moins il y a d'étapes, mieux c'est
            priority = len(path) * 1000
            
            # Priorité basée sur les devises utilisées (les plus liquides en premier)
            for p in path:
                try:
                    if not isinstance(p, str) or '/' not in p:
                        continue
                        
                    p_parts = p.split('/')
                    if len(p_parts) != 2:
                        continue
                        
                    p_base, p_quote = p_parts
                    
                    # Priorité pour la devise de cotation
                    priority += priority_currencies.get(p_quote, 100)
                    # Priorité plus faible pour la devise de base
                    priority += priority_currencies.get(p_base, 100) * 0.1
                except Exception as e:
                    logger.debug(f"Erreur lors du calcul de la priorité pour le chemin {p}: {e}")
                    continue
            
            return priority
            
            # Trier les chemins par priorité
            intermediate_paths.sort(key=path_priority)
            
            # Aplatir la liste des chemins et éliminer les doublons
            result = []
            seen = set()
            for path in intermediate_paths:
                for pair in path:
                    if pair not in seen:
                        result.append(pair)
                        seen.add(pair)
            
            logger.debug(f"Chemins intermédiaires triés: {result}")
            return result
        
        # 5. Si on arrive ici, essayer avec des paires inverses
        inverse_pair = f"{quote}/{base}"
        if inverse_pair in self._wsname_to_pair_id:
            logger.debug(f"Paire inverse trouvée: {inverse_pair}")
            return [inverse_pair]
        
        # 6. Dernier recours: essayer de trouver n'importe quelle paire qui contient la base ou la cotation
        fallback_pairs = []
        base_matches = []
        quote_matches = []
        
        try:
            # Parcourir toutes les paires disponibles
            for pair in self._wsname_to_pair_id.keys():
                try:
                    if not isinstance(pair, str) or '/' not in pair:
                        continue
                        
                    pair_parts = pair.split('/')
                    if len(pair_parts) != 2:
                        continue
                        
                    pair_base, pair_quote = pair_parts
                    
                    # Vérifier si la paire contient la base ou la cotation
                    if base in [pair_base, pair_quote]:
                        base_matches.append(pair)
                    elif quote in [pair_base, pair_quote]:
                        quote_matches.append(pair)
                    
                    # Limiter le nombre de résultats pour éviter de surcharger
                    if len(base_matches) >= 3 and len(quote_matches) >= 2:
                        break
                except Exception as e:
                    logger.debug(f"Erreur lors du traitement de la paire {pair}: {e}")
                    continue
            
            # Combiner les résultats en donnant la priorité aux correspondances avec la base
            fallback_pairs = base_matches[:3] + quote_matches[:2]
            
            if fallback_pairs:
                logger.debug(f"Aucun chemin direct trouvé, voici quelques paires contenant {base} ou {quote}: {fallback_pairs}")
                return fallback_pairs
            
            logger.debug("Aucune paire intermédiaire ou alternative trouvée")
            return []
            
        except Exception as e:
            logger.error(f"Erreur lors de la recherche de paires de repli: {e}", exc_info=True)
            return []
            
        # Le code suivant ne sera jamais atteint en raison du return précédent
        # Il est conservé pour référence mais n'est pas exécuté
        # intermediate_pairs = []
        
        # 3.1 Chercher des paires intermédiaires via des devises communes
        # Par exemple, pour ZRX/ETH, on cherche ZRX/XBT et XBT/ETH ou ETH/XBT
        logger.debug("\n=== RECHERCHE DE PAIRES INTERMÉDIAIRES ===")
        logger.debug(f"Devises communes testées: {common_currencies}")
        logger.debug(f"Base: {base}, Quote: {quote}")
        
        # Afficher toutes les paires disponibles pour ZRX et ETH
        logger.debug(f"\n=== TOUTES LES PAIRES DISPONIBLES POUR {base} (commençant par {base}/) ===")
        base_pairs_list = sorted([p for p in self._wsname_to_pair_id.keys() if p.startswith(f"{base}/")])
        for p in base_pairs_list:
            logger.debug(f"  - {p}")
        
        # Vérifier si ZRX/XBT est dans la liste des paires disponibles
        zrx_xbt = f"{base}/XBT"
        if zrx_xbt in base_pairs_list:
            logger.debug(f"  - {zrx_xbt} TROUVÉ DANS LES PAIRES DISPONIBLES")
        else:
            logger.debug(f"  - {zrx_xbt} ABSENT DES PAIRES DISPONIBLES")
            
        # Vérifier aussi avec XXBT au lieu de XBT
        zrx_xxbt = f"{base}/XXBT"
        if zrx_xxbt in base_pairs_list:
            logger.debug(f"  - {zrx_xxbt} TROUVÉ DANS LES PAIRES DISPONIBLES")
        else:
            logger.debug(f"  - {zrx_xxbt} ABSENT DES PAIRES DISPONIBLES")
            
        logger.debug(f"\n=== TOUTES LES PAIRES DISPONIBLES POUR {quote} (se terminant par /{quote}) ===")
        quote_pairs_list = sorted([p for p in self._wsname_to_pair_id.keys() if p.endswith(f"/{quote}")])
        for p in quote_pairs_list:
            logger.debug(f"  - {p}")
            
        # Vérifier si XBT/ETH ou XXBT/ETH est dans la liste des paires disponibles
        xbt_eth = f"XBT/{quote}"
        xxbt_eth = f"XXBT/{quote}"
        
        # Vérifier XBT/ETH
        xbt_eth_exists = xbt_eth in [p for p in self._wsname_to_pair_id.keys() if p.endswith(f"/{quote}")]
        logger.debug(f"  - {xbt_eth} {'TROUVÉ' if xbt_eth_exists else 'ABSENT'} DANS LES PAIRES DISPONIBLES")
        
        # Vérifier XXBT/ETH
        xxbt_eth_exists = xxbt_eth in [p for p in self._wsname_to_pair_id.keys() if p.endswith(f"/{quote}")]
        logger.debug(f"  - {xxbt_eth} {'TROUVÉ' if xxbt_eth_exists else 'ABSENT'} DANS LES PAIRES DISPONIBLES")
            
        logger.debug(f"\n=== TOUTES LES PAIRES CONTENANT XBT OU XXBT ===")
        xbt_pairs = [p for p in self._wsname_to_pair_id.keys() if 'XBT' in p or 'XXBT' in p]
        for p in sorted(xbt_pairs):
            logger.debug(f"  - {p}")
            
        # Vérifier si XBT/ETH ou XXBT/ETH existe dans les paires disponibles
        if xbt_eth in xbt_pairs or xxbt_eth in xbt_pairs:
            logger.debug(f"  - {xbt_eth} ou {xxbt_eth} TROUVÉ DANS LES PAIRES XBT/XXBT")
        else:
            logger.debug(f"  - {xbt_eth} et {xxbt_eth} ABSENTS DES PAIRES XBT/XXBT")
            
        logger.debug("\n=== RECHERCHE DE CHEMINS INTERMÉDIAIRES ===")
        logger.debug(f"Contenu de _wsname_to_pair_id (premiers 10 éléments): {list(self._wsname_to_pair_id.items())[:10]}")
        
        # Vérifier si XBT ou XXBT sont dans les devises communes
        logger.debug(f"\n=== VÉRIFICATION DES DEVISES COMMUNES ===")
        logger.debug(f"Devises communes: {common_curries}")
        logger.debug(f"XBT dans les devises communes: {'XBT' in common_currencies}")
        logger.debug(f"XXBT dans les devises communes: 'XXBT' in common_currencies")
        
        for curr in common_currencies:
            # Essayer base/curr -> curr/quote (ex: ZRX/XBT -> XBT/ETH)
            pair1 = f"{base}/{curr}"
            pair2 = f"{curr}/{quote}"
            pair1_exists = pair1 in self._wsname_to_pair_id
            pair2_exists = pair2 in self._wsname_to_pair_id
            
            logger.debug(f"\n=== TEST DE CHEMIN VIA {curr} ===")
            pair1_status = "existe" if pair1_exists else "n'existe pas"
            pair2_status = "existe" if pair2_exists else "n'existe pas"
            logger.debug(f"  {pair1} ({pair1_status}) -> {pair2} ({pair2_status})")
            
            # Afficher les paires disponibles pour la devise intermédiaire
            if curr in self._base_currencies or curr in self._quote_currencies:
                curr_pairs = [p for p in self._wsname_to_pair_id.keys() if f"{curr}/" in p or f"/{curr}" in p]
                logger.debug(f"  Paires disponibles pour {curr} ({len(curr_pairs)} trouvées):")
                for p in sorted(curr_pairs):
                    logger.debug(f"    - {p}")
                
                # Vérifier spécifiquement pour XBT/ETH et XXBT/ETH
                if curr in ['XBT', 'XXBT']:
                    logger.debug(f"\n  === VÉRIFICATION SPÉCIFIQUE POUR {curr}/ETH ===")
                    logger.debug(f"  Recherche de '{curr}/ETH' dans les paires: {f'{curr}/ETH' in curr_pairs}")
                    logger.debug(f"  Paires commençant par '{curr}/': {[p for p in curr_pairs if p.startswith(f'{curr}/')]}")
                    logger.debug(f"  Paires se terminant par '/ETH': {[p for p in curr_pairs if p.endswith('/ETH')]}")
                    
                    # Vérifier si ETH/XBT ou ETH/XXBT existe aussi (paire inverse)
                    eth_xbt = f"{quote}/{curr}"
                    logger.debug(f"  Vérification de la paire inverse {eth_xbt}: {eth_xbt in curr_pairs}")
                    
                    # Vérifier si la paire existe avec XETH au lieu de ETH
                    xeth_curr = f"XETH/{curr}" if 'XETH' in [c for c in self._quote_currencies if c.startswith('X')] else None
                    if xeth_curr:
                        logger.debug(f"  Vérification de la paire alternative {xeth_curr}: {xeth_curr in curr_pairs}")
                    
                    # Vérifier si la paire existe avec une autre variante de ETH
                    eth_variants = [c for c in self._quote_currencies if 'ETH' in c]
                    logger.debug(f"  Variantes de ETH trouvées: {eth_variants}")
                    for eth_var in eth_variants:
                        pair_var = f"{curr}/{eth_var}"
                        logger.debug(f"  Vérification de la paire {pair_var}: {pair_var in curr_pairs}")
            logger.debug(f"Test de chemin: {pair1} + {pair2}")
            logger.debug(f"  {pair1} existe: {pair1 in self._wsname_to_pair_id}")
            logger.debug(f"  {pair2} existe: {pair2 in self._wsname_to_pair_id}")
            
            if pair1 in self._wsname_to_pair_id and pair2 in self._wsname_to_pair_id:
                logger.debug(f"Paire intermédiaire trouvée via {curr}: {pair1} + {pair2}")
                intermediate_pairs.append([pair1, pair2])
            
            # Essayer aussi avec la paire inverse pour la deuxième partie
            pair2_inv = f"{quote}/{curr}"
            if pair1 in self._wsname_to_pair_id and pair2_inv in self._wsname_to_pair_id:
                logger.debug(f"Paire intermédiaire trouvée via {curr}: {pair1} + {pair2_inv}")
                intermediate_pairs.append([pair1, pair2_inv])
                
            # Essayer aussi avec la première paire inversée
            pair1_inv = f"{curr}/{base}"
            if pair1_inv in self._wsname_to_pair_id and pair2 in self._wsname_to_pair_id:
                logger.debug(f"Paire intermédiaire trouvée via {curr} (inverse): {pair1_inv} + {pair2}")
                intermediate_pairs.append([pair1_inv, pair2])
                
            # Et avec les deux paires inversées
            if pair1_inv in self._wsname_to_pair_id and pair2_inv in self._wsname_to_pair_id:
                logger.debug(f"Paire intermédiaire trouvée via {curr} (inverse): {pair1_inv} + {pair2_inv}")
                intermediate_pairs.append([pair1_inv, pair2_inv])
        
        # 3.2 Si on n'a pas trouvé de paires intermédiaires directes, essayer avec deux conversions
        if not intermediate_pairs:
            for curr1 in common_currencies:
                for curr2 in common_currencies:
                    if curr1 == curr2:
                        continue
                        
                    # Essayer base/curr1 -> curr1/curr2 -> curr2/quote
                    pair1 = f"{base}/{curr1}"
                    pair2 = f"{curr1}/{curr2}"
                    pair3 = f"{curr2}/{quote}"
                    
                    if (pair1 in self._wsname_to_pair_id and 
                        pair2 in self._wsname_to_pair_id and 
                        pair3 in self._wsname_to_pair_id):
                        logger.debug(f"Paire intermédiaire en deux étapes trouvée: {pair1} + {pair2} + {pair3}")
                        intermediate_pairs.append([pair1, pair2, pair3])
        
        # 3.3 Si on a des paires intermédiaires, les trier par pertinence
        if intermediate_pairs:
            logger.debug(f"\n=== PAIRES INTERMÉDIAIRES TROUVÉES ===")
            for i, path in enumerate(intermediate_pairs, 1):
                logger.debug(f"Chemin {i}: {' -> '.join(path)}")
            
            # Donner la priorité aux paires avec le moins d'étapes
            intermediate_pairs.sort(key=len)
            logger.debug("\nAprès tri par nombre d'étapes:")
            for i, path in enumerate(intermediate_pairs, 1):
                logger.debug(f"Chemin {i} ({len(path)} étapes): {' -> '.join(path)}")
            
            # Puis par priorité des devises (BTC/ETH d'abord, puis stablecoins, puis autres)
            priority_currencies = {'XBT': 0, 'BTC': 0, 'ETH': 1, 'USDT': 2, 'USDC': 2, 'USD': 2, 'EUR': 2}
            
            def get_priority(pair_list):
                # Moins il y a d'étapes, mieux c'est
                priority = len(pair_list) * 100
                # Priorité basée sur les devises utilisées
                for p in pair_list:
                    for c in p.split('/'):
                        priority += priority_currencies.get(c, 10)
                return priority
            
            # Trier d'abord par longueur, puis par priorité des devises
            intermediate_pairs.sort(key=get_priority)
            
            logger.debug("\nAprès tri par priorité des devises:")
            for i, path in enumerate(intermediate_pairs, 1):
                logger.debug(f"Chemin {i} (priorité {get_priority(path)}): {' -> '.join(path)}")
            
            # Retourner le meilleur chemin trouvé
            best_path = intermediate_pairs[0]
            logger.debug(f"\n=== MEILLEUR CHEMIN SÉLECTIONNÉ ===")
            logger.debug(f"Chemin final: {' -> '.join(best_path)}")
            return best_path
            
        logger.debug("Aucune paire intermédiaire trouvée")
            
        # 4. Essayer avec des paires inverses
        inverse_pair = f"{quote}/{base}"
        if inverse_pair in self._wsname_to_pair_id:
            return [inverse_pair]
        
        # 5. Essayer de trouver des paires avec la même base ou la même cotation
        # même si ce n'est pas dans les devises de cotation standards
        all_base_pairs = [p for p in self._wsname_to_pair_id.keys() 
                         if p.startswith(f"{base}/") and p != f"{base}/{quote}"]
        all_quote_pairs = [p for p in self._wsname_to_pair_id.keys() 
                          if p.endswith(f"/{quote}") and not p.startswith(f"{base}/")]
        
        # Combiner les résultats et limiter le nombre de suggestions
        all_alternatives = all_base_pairs + all_quote_pairs
        
        # Si on a des alternatives, les trier par pertinence
        if all_alternatives:
            # Donner la priorité aux paires avec des devises majeures
            priority_quotes = ['USDT', 'USDC', 'USD', 'EUR', 'BTC', 'XBT', 'ETH']
            
            def sort_key(p):
                q = p.split('/')[1] if not p.endswith(f"/{quote}") else quote
                try:
                    return (priority_quotes.index(q), p)
                except ValueError:
                    return (len(priority_quotes), p)
            
            return sorted(all_alternatives, key=sort_key)[:5]
        
        # Si on arrive ici, on n'a pas trouvé d'alternative
        return []
    
    def is_quote_currency_supported(self, currency: str) -> bool:
        """Vérifie si une devise de cotation est supportée."""
        if not isinstance(currency, str) or not currency:
            return False
            
        currency_upper = currency.upper()
        
        # Vérifier directement dans les devises de cotation
        if currency_upper in self._quote_currencies:
            return True
            
        # Vérifier les codes internes
        if currency_upper in self.SYMBOL_MAPPING:
            standard_currency = self.SYMBOL_MAPPING[currency_upper]
            return standard_currency in self._quote_currencies
            
        return False
    
    async def refresh_cache(self) -> None:
        """Force le rafraîchissement du cache des paires depuis l'API Kraken."""
        logger.info("Mise à jour forcée du cache des paires depuis l'API Kraken")
        await self._fetch_from_api()
        logger.info("Cache des paires mis à jour avec succès")

# Instance unique pour une utilisation facile
available_pairs = AvailablePairs()

# Alias pour compatibilité ascendante
initialize_available_pairs = AvailablePairs.create
