"""
Module pour gérer la liste des paires de trading disponibles sur Kraken.
Utilise l'API Kraken pour récupérer les paires supportées et fournit des méthodes
pour les valider et les formater.
"""
import asyncio
import json
import logging
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Set, Any, Union
import aiohttp

# Configuration du logger
logger = logging.getLogger(__name__)

class KrakenAPIError(Exception):
    """Exception pour les erreurs liées à l'API Kraken."""
    pass

class AvailablePairs:
    """
    Classe pour gérer les paires de trading disponibles sur Kraken.
    
    Cette classe fournit des méthodes pour récupérer, valider et formater les paires
    de trading disponibles sur Kraken, avec mise en cache des résultats.
    """
    
    # Fichier de cache pour stocker les paires entre les sessions
    CACHE_FILE = Path("data/available_pairs_cache.json")
    
    # Délai d'expiration du cache (en secondes) - 1 jour par défaut
    CACHE_EXPIRY = 24 * 60 * 60
    
    # Mappage des symboles alternatifs vers les symboles officiels Kraken
    SYMBOL_MAPPING = {
        # Cryptocurrencies
        'BTC': 'XBT',   # Bitcoin
        'XBT': 'XBT',   # Bitcoin (officiel)
        'BCH': 'BCH',   # Bitcoin Cash
        'BSV': 'BSV',   # Bitcoin SV
        'XDG': 'XDG',   # Dogecoin (DOGE)
        'DOGE': 'XDG',
        'XRP': 'XRP',   # Ripple
        'XLM': 'XLM',   # Stellar
        'TRX': 'TRX',   # Tron
        'ADA': 'ADA',   # Cardano
        'DOT': 'DOT',   # Polkadot
        'SOL': 'SOL',   # Solana
        'ETH': 'ETH',   # Ethereum
        'ETC': 'ETC',   # Ethereum Classic
        'LTC': 'LTC',   # Litecoin
        'USDT': 'USDT', # Tether
        'USDC': 'USDC', # USD Coin
        'DAI': 'DAI',   # DAI
        'ZRO': 'ZRO',   # 0x Protocol
        'ZRX': 'ZRX',   # 0x
        
        # Fiat currencies
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
        'ZCAD': 'CAD',
        
        # Additional altcoins and tokens
        '1INCH': '1INCH',
        'AAVE': 'AAVE',
        'ALGO': 'ALGO',
        'ATOM': 'ATOM',
        'BAL': 'BAL',
        'BAT': 'BAT',
        'COMP': 'COMP',
        'CRV': 'CRV',
        'ENJ': 'ENJ',
        'FIL': 'FIL',
        'GNO': 'GNO',
        'GRT': 'GRT',
        'ICX': 'ICX',
        'KAVA': 'KAVA',
        'KEEP': 'KEEP',
        'KNC': 'KNC',
        'KSM': 'KSM',
        'LINK': 'LINK',
        'LSK': 'LSK',
        'MANA': 'MANA',
        'MATIC': 'MATIC',
        'NANO': 'NANO',
        'OMG': 'OMG',
        'OXT': 'OXT',
        'PAXG': 'PAXG',
        'QTUM': 'QTUM',
        'REP': 'REP',
        'SC': 'SC',
        'SNX': 'SNX',
        'STORJ': 'STORJ',
        'SUSHI': 'SUSHI',
        'TBTC': 'TBTC',
        'UNI': 'UNI',
        'WAVES': 'WAVES',
        'XMR': 'XMR',
        'XRP': 'XRP',
        'XTZ': 'XTZ',
        'YFI': 'YFI',
        'ZEC': 'ZEC'
    }

    def __init__(self, api: Optional[Any] = None):
        """Initialise le gestionnaire de paires disponibles.
        
        Args:
            api: Instance de l'API Kraken (optionnel, conservé pour compatibilité)
        """
        self._pairs_data: Dict[str, Dict] = {}
        self._wsname_to_pair_id: Dict[str, str] = {}
        self._base_currencies: Set[str] = set()
        self._quote_currencies: Set[str] = set()
        self._initialized = False
        self._last_updated: Optional[float] = None
        self.KRAKEN_API_URL = "https://api.kraken.com/0/public/AssetPairs"
    
    async def initialize(self, use_cache: bool = True) -> None:
        """Initialise les données des paires disponibles.
        
        Args:
            use_cache: Si True, utilise le cache local si disponible
        """
        if self._initialized:
            return
            
        # Créer le répertoire de données si nécessaire
        self.CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
        
        # Essayer de charger depuis le cache si demandé
        if use_cache and self._load_from_cache():
            logger.info("Données des paires chargées depuis le cache")
            self._initialized = True
            return
            
        # Sinon, récupérer depuis l'API
        await self._fetch_from_api()
        self._initialized = True
    
    def _load_from_cache(self) -> bool:
        """Charge les données des paires depuis le cache local.
        
        Returns:
            bool: True si le chargement a réussi, False sinon
        """
        try:
            if not self.CACHE_FILE.exists():
                return False
                
            # Vérifier l'âge du cache
            file_age = time.time() - self.CACHE_FILE.stat().st_mtime
            if file_age > self.CACHE_EXPIRY:
                logger.debug("Cache expiré, mise à jour nécessaire")
                return False
                
            # Charger les données
            with open(self.CACHE_FILE, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            self._pairs_data = data.get('pairs', {})
            self._last_updated = data.get('last_updated')
            
            # Reconstruire les index
            self._build_indexes()
            
            return True
            
        except (json.JSONDecodeError, OSError, KeyError) as e:
            logger.warning(f"Erreur lors du chargement du cache des paires: {e}")
            return False
    
    def _save_to_cache(self) -> None:
        """Sauvegarde les données des paires dans le cache local."""
        try:
            # Créer le répertoire parent si nécessaire
            self.CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
            
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
        """
        Construit les index pour une recherche plus rapide des paires.
        
        Crée les index suivants :
        - _wsname_to_pair_id : mappage des noms de paires (format BASE/QUOTE) vers les IDs de paires
        - _base_currencies : ensemble des devises de base disponibles
        - _quote_currencies : ensemble des devises de cotation disponibles (codes standards)
        """
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
                    
                    # Log pour les paires AVAAI
                    if 'AVAAI' in normalized_wsname:
                        logger.debug(f"Indexation de la paire AVAAI: {normalized_wsname} -> {pair_id}")
                        logger.debug(f"Détails de la paire: base={pair_info.get('base')}, quote={pair_info.get('quote')}, altname={pair_info.get('altname')}")
                
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
        
        logger.debug(
            f"Indexation terminée: {len(self._wsname_to_pair_id)} paires, "
            f"{len(self._base_currencies)} devises de base, "
            f"{len(self._quote_currencies)} devises de cotation"
        )
    
    def is_pair_supported(self, pair: str) -> Union[bool, str, None]:
        """
        Vérifie si une paire est supportée par Kraken et retourne sa forme normalisée.
        
        Args:
            pair: Paire à vérifier (peut être dans différents formats :
                  "BASE/QUOTE", "BASEEUR", "BTC-USD", "btc_usdt", etc.)
            
        Returns:
            Union[bool, str, None]: 
                - None si la paire n'est pas supportée
                - str: La forme normalisée de la paire si elle est supportée (ex: 'XBT/USD')
                - True si la paire est supportée mais pas de forme normalisée disponible
            
        Raises:
            ValueError: Si le paramètre 'pair' n'est pas une chaîne non vide
        """
        if not isinstance(pair, str) or not pair.strip():
            raise ValueError("Le paramètre 'pair' doit être une chaîne non vide")
            
        if not self._initialized:
            logger.warning("Les données des paires ne sont pas encore initialisées")
            return None
            
        # 1. Essayer de normaliser la paire d'abord (cela couvre la plupart des cas)
        normalized = self.normalize_pair(pair)
        if normalized:
            return normalized
            
        # 2. Si la normalisation échoue, essayer des approches alternatives
        clean_pair = pair.strip().upper()
        clean_pair_no_sep = clean_pair.replace('-', '').replace('_', '').replace('/', '')
        
        # Vérifier si c'est un format sans séparateur (ex: BTCEUR)
        if len(clean_pair_no_sep) >= 6:  # Au moins 3 caractères pour chaque devise
            # Essayer avec les devises de cotation connues
            for quote_currency in sorted(self._quote_currencies, key=len, reverse=True):
                if clean_pair_no_sep.endswith(quote_currency):
                    base = clean_pair_no_sep[:-len(quote_currency)]
                    if not base:
                        continue
                        
                    # Essayer avec le format normalisé
                    normalized = self.normalize_pair(f"{base}/{quote_currency}")
                    if normalized:
                        return normalized
                    
                    # Essayer avec XBT/BTC alternance
                    if base == 'BTC':
                        normalized = self.normalize_pair(f"XBT/{quote_currency}")
                        if normalized:
                            return normalized
                    elif base == 'XBT':
                        normalized = self.normalize_pair(f"BTC/{quote_currency}")
                        if normalized:
                            return normalized
        
        # 3. Vérifier dans les altnames directement (pour les cas spéciaux)
        for pair_id, pair_info in self._pairs_data.items():
            # Vérifier dans les noms alternatifs
            altname = pair_info.get('altname', '').upper()
            if altname == clean_pair or altname == clean_pair_no_sep:
                return pair_info.get('wsname')
            
            # Vérifier dans le wsname (sans le séparateur)
            wsname = pair_info.get('wsname', '').upper().replace('/', '')
            if wsname == clean_pair or wsname == clean_pair_no_sep:
                return pair_info.get('wsname')
        
        # 4. Essayer avec des variantes de symboles
        variants = [
            clean_pair.replace('XBT', 'BTC'),
            clean_pair.replace('BTC', 'XBT'),
            clean_pair.replace('XDG', 'DOGE'),
            clean_pair.replace('DOGE', 'XDG'),
            clean_pair.replace('USD', 'ZUSD'),
            clean_pair.replace('EUR', 'ZEUR'),
            clean_pair.replace('GBP', 'ZGBP'),
            clean_pair.replace('JPY', 'ZJPY'),
            clean_pair.replace('CAD', 'ZCAD'),
            clean_pair.replace('AUD', 'ZAUD'),
            clean_pair.replace('ZUSD', 'USD'),
            clean_pair.replace('ZEUR', 'EUR'),
            clean_pair.replace('ZGBP', 'GBP'),
            clean_pair.replace('ZJPY', 'JPY'),
            clean_pair.replace('ZCAD', 'CAD'),
            clean_pair.replace('ZAUD', 'AUD')
        ]
        
        # Essayer toutes les variantes
        for variant in variants:
            normalized = self.normalize_pair(variant)
            if normalized:
                return normalized
        
        # Aucune correspondance trouvée
        logger.debug(f"Paire non supportée: {pair}")
        return None

    def normalize_pair(self, pair: str) -> Optional[str]:
        """
        Normalise le nom d'une paire de trading pour correspondre au format wsname de Kraken.
        
        Args:
            pair: Nom de la paire à normaliser (ex: 'btc-usd', 'XBT/EUR', 'XXBTZUSD', 'dogeusd')
            
        Returns:
            str: Nom normalisé de la paire au format 'XBT/USD', ou None si non supportée
        """
        if not pair or not isinstance(pair, str):
            return None
                
        if not self._initialized:
            logger.warning("Les données des paires ne sont pas encore initialisées")
            return None
            
        # Nettoyer la chaîne d'entrée et supprimer les espaces
        clean_pair = pair.strip().upper()
        
        # 1. Vérifier d'abord dans wsname_to_pair_id (format standard 'XBT/USD')
        if '/' in clean_pair and clean_pair.count('/') == 1:
            if clean_pair in self._wsname_to_pair_id:
                return clean_pair
        
        # 2. Vérifier dans les paires avec différents séparateurs
        separators = ['-', '_', ' ']
        for sep in separators:
            if sep in clean_pair and clean_pair.count(sep) == 1:
                base, quote = clean_pair.split(sep)
                # Essayer avec le format standard
                test_pair = f"{base}/{quote}"
                if test_pair in self._wsname_to_pair_id:
                    return test_pair
                
                # Essayer avec XBT/BTC alternance
                if base == 'BTC':
                    test_pair = f"XBT/{quote}"
                    if test_pair in self._wsname_to_pair_id:
                        return test_pair
                elif base == 'XBT':
                    test_pair = f"BTC/{quote}"
                    if test_pair in self._wsname_to_pair_id:
                        return test_pair
        
        # 3. Vérifier dans les altnames (format compact comme 'XXBTZUSD')
        clean_pair_no_sep = clean_pair.replace('-', '').replace('_', '').replace('/', '')
        
        # D'abord vérifier directement dans les paires
        if clean_pair_no_sep in self._pairs_data:
            return self._pairs_data[clean_pair_no_sep].get('wsname')
        
        # Ensuite vérifier dans les altnames
        for pair_id, pair_info in self._pairs_data.items():
            # Vérifier dans les noms alternatifs
            altname = pair_info.get('altname', '').upper()
            if altname == clean_pair_no_sep:
                return pair_info.get('wsname')
            
            # Vérifier dans le wsname (sans le séparateur)
            wsname = pair_info.get('wsname', '').upper().replace('/', '')
            if wsname == clean_pair_no_sep:
                return pair_info.get('wsname')
        
        # 4. Essayer avec les devises de cotation connues
        for quote_currency in sorted(self._quote_currencies, key=len, reverse=True):
            if clean_pair_no_sep.endswith(quote_currency):
                base = clean_pair_no_sep[:-len(quote_currency)]
                if not base:
                    continue
                
                # Essayer avec le format BASE/QUOTE
                test_pair = f"{base}/{quote_currency}"
                if test_pair in self._wsname_to_pair_id:
                    return test_pair
                
                # Essayer avec XBT/BTC alternance
                if base == 'BTC':
                    test_pair = f"XBT/{quote_currency}"
                    if test_pair in self._wsname_to_pair_id:
                        return test_pair
                elif base == 'XBT':
                    test_pair = f"BTC/{quote_currency}"
                    if test_pair in self._wsname_to_pair_id:
                        return test_pair
        
        # 5. Essayer avec les variantes de symboles
        variants = [
            clean_pair.replace('XBT', 'BTC'),
            clean_pair.replace('BTC', 'XBT'),
            clean_pair.replace('XDG', 'DOGE'),
            clean_pair.replace('DOGE', 'XDG'),
            clean_pair.replace('USD', 'ZUSD'),
            clean_pair.replace('EUR', 'ZEUR'),
            clean_pair.replace('GBP', 'ZGBP'),
            clean_pair.replace('JPY', 'ZJPY'),
            clean_pair.replace('CAD', 'ZCAD'),
            clean_pair.replace('AUD', 'ZAUD'),
            clean_pair.replace('ZUSD', 'USD'),
            clean_pair.replace('ZEUR', 'EUR'),
            clean_pair.replace('ZGBP', 'GBP'),
            clean_pair.replace('ZJPY', 'JPY'),
            clean_pair.replace('ZCAD', 'CAD'),
            clean_pair.replace('ZAUD', 'AUD')
        ]
        
        # Essayer toutes les variantes
        for variant in variants:
            # Essayer directement
            if variant in self._wsname_to_pair_id:
                return variant
                
            # Essayer avec séparateur slash
            if '/' not in variant and len(variant) >= 6:  # Au moins 3 caractères par devise
                for i in range(3, len(variant) - 2):
                    base = variant[:i]
                    quote = variant[i:]
                    test_pair = f"{base}/{quote}"
                    if test_pair in self._wsname_to_pair_id:
                        return test_pair
        
        # Aucune correspondance trouvée
        logger.debug(f"Aucune correspondance trouvée pour la paire: {pair}")
        return None
    
    def get_pair_info(self, pair: str) -> Optional[Dict[str, Any]]:
        """
        Récupère les informations d'une paire de trading.
        
        Args:
            pair: Nom de la paire (peut être dans n'importe quel format : 'BTC-USD', 'XBT/USD', 'XXBTZUSD', 'btc_usd')
            
        Returns:
            dict: Dictionnaire contenant les informations de la paire, ou None si non trouvée
            
        Note:
            - Gère les différents formats de paires (avec séparateurs, sans séparateur, codes internes, etc.)
            - Retourne les informations complètes de la paire telles que fournies par l'API Kraken
        """
        if not pair or not isinstance(pair, str):
            return None
            
        # Nettoyer la chaîne d'entrée
        clean_pair = pair.strip().upper()
        
        # 1. Vérifier directement avec l'ID de paire Kraken (ex: 'XXBTZUSD')
        if clean_pair in self._pairs_data:
            return self._pairs_data[clean_pair]
            
        # 2. Vérifier avec le nom normalisé (format 'XBT/USD')
        if '/' in clean_pair and clean_pair in self._wsname_to_pair_id:
            pair_id = self._wsname_to_pair_id[clean_pair]
            return self._pairs_data.get(pair_id)
            
        # 3. Essayer de normaliser la paire
        normalized = self.normalize_pair(clean_pair)
        if normalized:
            if normalized in self._wsname_to_pair_id:
                pair_id = self._wsname_to_pair_id[normalized]
                return self._pairs_data.get(pair_id)
            elif normalized in self._pairs_data:
                return self._pairs_data[normalized]
        
        # 4. Essayer avec différents formats de séparateurs
        separators = ['-', '_', ' ']
        for sep in separators:
            if sep in clean_pair and clean_pair.count(sep) == 1:
                base, quote = clean_pair.split(sep)
                # Essayer avec les codes standards
                base = self.SYMBOL_MAPPING.get(base, base)
                quote = self.SYMBOL_MAPPING.get(quote, quote)
                test_pair = f"{base}/{quote}"
                if test_pair in self._wsname_to_pair_id:
                    pair_id = self._wsname_to_pair_id[test_pair]
                    return self._pairs_data.get(pair_id)
        
        # 5. Vérifier dans les altnames (sans séparateur, ex: 'XBTUSD')
        clean_pair_no_sep = clean_pair.replace('-', '').replace('_', '').replace('/', '')
        for pair_id, pair_info in self._pairs_data.items():
            # Vérifier l'altname (sans séparateur)
            altname = pair_info.get('altname', '').upper()
            if altname == clean_pair_no_sep:
                return pair_info
            # Vérifier le wsname (sans séparateur)
            wsname = pair_info.get('wsname', '').upper().replace('/', '')
            if wsname == clean_pair_no_sep:
                return pair_info
        
        # 6. Dernier essai : vérifier si c'est une paire concaténée (ex: 'BTCEUR')
        if len(clean_pair) >= 6:  # Au moins 3 caractères pour chaque devise
            for i in range(3, len(clean_pair) - 2):
                base = clean_pair[:i]
                quote = clean_pair[i:]
                # Essayer avec les codes standards
                base = self.SYMBOL_MAPPING.get(base, base)
                quote = self.SYMBOL_MAPPING.get(quote, quote)
                test_pair = f"{base}/{quote}"
                if test_pair in self._wsname_to_pair_id:
                    pair_id = self._wsname_to_pair_id[test_pair]
                    return self._pairs_data.get(pair_id)
        
        # Aucune correspondance trouvée
        logger.debug(f"Aucune information trouvée pour la paire: {pair}")
        return None
    
    def get_available_pairs(self, quote_currency: Optional[str] = None) -> List[str]:
        """
        Retourne la liste des paires disponibles, éventuellement filtrées par devise de cotation.
        
        Args:
            quote_currency: Code de la devise de cotation (ex: 'USD', 'EUR', 'ZUSD', 'ZEUR'). 
                          Si None, retourne toutes les paires.
            
        Returns:
            list: Liste des paires disponibles (format: 'BASE/QUOTE')
            
        Note:
            - Gère à la fois les codes standards (USD, EUR) et les codes internes Kraken (ZUSD, ZEUR)
            - Retourne toujours les paires dans le format 'BASE/QUOTE' avec les codes standards
        """
        if not self._initialized:
            logger.warning("AvailablePairs n'est pas initialisé. Appelez initialize() d'abord.")
            return []
            
        # Si pas de filtre, retourner toutes les paires
        if not quote_currency:
            return sorted(self._wsname_to_pair_id.keys())
            
        # Normaliser la devise de cotation
        quote_currency = quote_currency.upper()
        
        # Déterminer si c'est un code interne (ZUSD, ZEUR) ou standard (USD, EUR)
        is_internal_code = quote_currency in self.SYMBOL_MAPPING and \
                         quote_currency not in self.SYMBOL_MAPPING.values()
        
        # Si c'est un code standard, on peut avoir besoin de le convertir en code interne
        # pour la comparaison avec les paires réelles
        target_quotes = set()
        
        if is_internal_code:
            # C'est déjà un code interne (ex: ZUSD)
            target_quotes.add(quote_currency)
            # Ajouter aussi le code standard s'il existe
            if quote_currency in self.SYMBOL_MAPPING:
                target_quotes.add(self.SYMBOL_MAPPING[quote_currency])
        else:
            # C'est un code standard (ex: USD), on doit vérifier à la fois le standard et l'interne
            target_quotes.add(quote_currency)
            # Ajouter le code interne s'il existe
            for internal, standard in self.SYMBOL_MAPPING.items():
                if standard == quote_currency:
                    target_quotes.add(internal)
        
        # Filtrer les paires par devise de cotation
        filtered_pairs = []
        
        for wsname in self._wsname_to_pair_id:
            parts = wsname.split('/')
            if len(parts) == 2 and parts[1] in target_quotes:
                # Si la devise cible est un code interne, on la convertit en code standard
                if is_internal_code and parts[1] in self.SYMBOL_MAPPING:
                    standard_quote = self.SYMBOL_MAPPING[parts[1]]
                    wsname = f"{parts[0]}/{standard_quote}"
                filtered_pairs.append(wsname)
        
        return sorted(filtered_pairs)
    
    def get_base_currencies(self) -> Set[str]:
        """Retourne l'ensemble des devises de base disponibles."""
        return self._base_currencies
    
    def get_quote_currencies(self) -> Set[str]:
        """Retourne l'ensemble des devises de cotation disponibles."""
        return self._quote_currencies
    
    def is_quote_currency_supported(self, currency: str) -> bool:
        """
        Vérifie si une devise de cotation est supportée.
        
        Args:
            currency: Code de la devise à vérifier (ex: 'USD', 'EUR', 'ZUSD', 'ZEUR')
            
        Returns:
            bool: True si la devise est supportée, False sinon
            
        Note:
            - Gère les codes internes Kraken (ex: ZUSD, ZEUR)
            - Gère les codes standards (ex: USD, EUR)
            - Vérifie que la devise est effectivement utilisée dans les paires disponibles
            - Certaines devises comme CHF et AUD sont explicitement rejetées car non supportées
        """
        if not currency or not isinstance(currency, str):
            return False
            
        # Normaliser la devise
        currency = currency.upper()
        
        # Liste des devises explicitement non supportées
        unsupported_currencies = {'CHF', 'AUD'}
        if currency in unsupported_currencies:
            return False
            
        # Vérifier d'abord dans les devises supportées (le plus rapide)
        if currency in self._quote_currencies:
            # Vérifier que la devise est effectivement utilisée dans les paires
            return any(
                wsname.split('/')[-1] == currency 
                for wsname in self._wsname_to_pair_id
            )
            
        # Vérifier les alias de devises connus
        alt_currencies = {
            'XBT': 'BTC',
            'BTC': 'XBT',
            'XDG': 'DOGE',
            'DOGE': 'XDG',
            # Ne pas inclure CHF ici car non supporté
        }
        
        # Vérifier si c'est un alias connu
        if currency in alt_currencies:
            return alt_currencies[currency] in self._quote_currencies
            
        # Vérifier les codes internes Kraken
        if currency in self.SYMBOL_MAPPING:
            standard_currency = self.SYMBOL_MAPPING[currency]
            if standard_currency in self._quote_currencies:
                return any(
                    wsname.split('/')[-1] == standard_currency 
                    for wsname in self._wsname_to_pair_id
                )
                
        # Vérifier dans le mapping inverse (ex: USD -> ZUSD)
        for internal_code, standard_code in self.SYMBOL_MAPPING.items():
            if standard_code == currency and internal_code in self._quote_currencies:
                return any(
                    wsname.split('/')[-1] == internal_code 
                    for wsname in self._wsname_to_pair_id
                )
        return False
        
    def refresh_cache(self, force: bool = False) -> bool:
        """
        Force le rafraîchissement du cache des paires depuis l'API Kraken.
        
        Args:
            force: Si True, force le rafraîchissement même si le cache est récent
            
        Returns:
            bool: True si le rafraîchissement a réussi, False sinon
            
        Note:
            Cette méthode est utile pour forcer une mise à jour des données
            sans attendre l'expiration du cache.
        """
        try:
            logger.info("⏳ Rafraîchissement du cache des paires disponibles...")
            success = self._fetch_from_api()
            if success:
                logger.info("✅ Cache des paires rafraîchi avec succès")
                # Reconstruire les index après le rafraîchissement
                self._build_indexes()
                return True
            else:
                logger.error("❌ Échec du rafraîchissement du cache des paires")
                return False
        except Exception as e:
            logger.error(f"❌ Erreur lors du rafraîchissement du cache: {str(e)}")
            return False

# Suppression de l'instance globale pour forcer l'initialisation explicite
# via la fonction asynchrone initialize_available_pairs

# Variable pour stocker l'instance initialisée
_available_pairs_instance = None

# Fonction utilitaire pour initialiser et obtenir l'instance de AvailablePairs
async def initialize_available_pairs(api: Optional[Any] = None) -> 'AvailablePairs':
    """
    Initialise et retourne une instance de AvailablePairs.
    
    Args:
        api: Instance de l'API Kraken (optionnel)
        
    Returns:
        AvailablePairs: Instance initialisée
        
    Raises:
        RuntimeError: Si l'initialisation échoue
    """
    global _available_pairs_instance
    
    # Si l'instance existe déjà et est initialisée, la retourner
    if _available_pairs_instance is not None and _available_pairs_instance._initialized:
        return _available_pairs_instance
    
    # Sinon, en créer une nouvelle et l'initialiser
    _available_pairs_instance = AvailablePairs(api)
    await _available_pairs_instance.initialize()
    
    if not _available_pairs_instance._initialized:
        raise RuntimeError("Échec de l'initialisation de AvailablePairs")
        
    return _available_pairs_instance

# Fonction pour obtenir l'instance initialisée (à utiliser avec précaution)
async def get_available_pairs() -> 'AvailablePairs':
    """
    Récupère l'instance de AvailablePairs si elle est initialisée.
    
    Returns:
        AvailablePairs: Instance initialisée
        
    Raises:
        RuntimeError: Si l'instance n'est pas initialisée
    """
    if _available_pairs_instance is None or not _available_pairs_instance._initialized:
        raise RuntimeError(
            "AvailablePairs n'a pas été initialisé. "
            "Appelez d'abord initialize_available_pairs()"
        )
    return _available_pairs_instance
