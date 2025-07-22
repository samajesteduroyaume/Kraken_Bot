"""
Tests for the pair_utils module.
"""
import unittest
import asyncio
import time
from unittest.mock import patch, MagicMock, AsyncMock
import pytest
from src.core.market.available_pairs_refactored import UnsupportedTradingPairError

from src.utils.pair_utils import (
    extract_pair_name, 
    extract_pair_names, 
    normalize_pair_input,
    initialize_pair_utils,
    available_pairs
)

# Mock data for testing
MOCK_PAIRS = {
    "XXBTZUSD": {
        "wsname": "XBT/USD",
        "base": "XBT",
        "quote": "USD"
    },
    "XETHZUSD": {
        "wsname": "ETH/USD",
        "base": "ETH",
        "quote": "USD"
    },
    "XLTCXXBT": {
        "wsname": "LTC/XBT",
        "base": "LTC",
        "quote": "XBT"
    },
    "XXRPZEUR": {
        "wsname": "XRP/EUR",
        "base": "XRP",
        "quote": "EUR"
    }
}

# Fixture pour initialiser les paires disponibles
@pytest.fixture(autouse=True)
def setup_available_pairs(monkeypatch):
    """Fixture to set up the available pairs for testing."""
    from src.core.market.available_pairs_refactored import AvailablePairs
    from src.core.market.pair_initializer import _available_pairs_instance, _initialized, get_available_pairs
    from unittest.mock import MagicMock, AsyncMock, patch
    
    # Créer une sous-classe pour les tests qui hérite d'AvailablePairs
    class TestAvailablePairs(AvailablePairs):
        def __init__(self, *args, **kwargs):
            # Initialiser les attributs nécessaires avant d'appeler le parent
            self._pairs_data = MOCK_PAIRS.copy()
            
            # Initialiser les attributs de base
            self._initialized = True
            self._last_updated = time.time()
            self._api = None
            
            # Construire les index nécessaires
            self._build_indexes()
            
            # Initialiser les attributs de la classe parente
            super().__init__(*args, **kwargs)
            
            # S'assurer que l'instance est marquée comme initialisée
            self._initialized = True
        
        def _build_indexes(self):
            """Construit les index nécessaires pour la recherche de paires."""
            self._wsname_to_pair_id = {}
            self._base_currencies = set()
            self._quote_currencies = set()
            
            for pair_id, pair_info in self._pairs_data.items():
                wsname = pair_info["wsname"]
                base = pair_info["base"]
                quote = pair_info["quote"]
                
                # Ajouter la paire avec son wsname
                self._wsname_to_pair_id[wsname] = pair_id
                
                # Ajouter des variantes de format
                for sep in ['-', '_', ' ']:
                    variant = f"{base}{sep}{quote}"
                    self._wsname_to_pair_id[variant] = pair_id
                
                # Ajouter les devises de base et de cotation
                self._base_currencies.add(base)
                self._quote_currencies.add(quote)
            
            # Ajouter les paires en majuscules pour la recherche insensible à la casse
            for wsname, pair_id in list(self._wsname_to_pair_id.items()):
                self._wsname_to_pair_id[wsname.upper()] = pair_id
            
        def is_initialized(self):
            return self._initialized
            
        def get_available_pairs(self, quote_currency=None):
            if quote_currency:
                quote_currency = quote_currency.upper()
                return {k: v for k, v in self._pairs_data.items() 
                        if v.get('quote') == quote_currency}
            return self._pairs_data
            
        def is_pair_supported(self, pair, raise_on_error=False):
            try:
                normalized = self.normalize_pair(pair, raise_on_error=raise_on_error)
                return normalized is not None
            except (ValueError, UnsupportedTradingPairError):
                if raise_on_error:
                    raise
                return False
        
        # Méthode de normalisation personnalisée pour les tests
        def normalize_pair(self, pair, raise_on_error=True):
            # Vérification des entrées invalides
            if not pair or not isinstance(pair, str):
                if raise_on_error:
                    raise ValueError("Pair must be a string")
                return None
                
            clean_pair = pair.strip()
            if not clean_pair:
                if raise_on_error:
                    raise ValueError("Empty pair string")
                return None
                
            # Convertir en majuscules pour la cohérence
            clean_pair = clean_pair.upper()
            
            # 1. Vérifier si c'est un wsname exact (ex: 'XBT/USD')
            if clean_pair in self._wsname_to_pair_id:
                return clean_pair
                
            # 2. Vérifier si c'est un ID de paire (ex: 'XXBTZUSD')
            if clean_pair in self._pairs_data:
                return self._pairs_data[clean_pair]['wsname']
                
            # 3. Essayer avec différents séparateurs
            for sep in ['-', '_', ' ']:
                if sep in clean_pair and clean_pair.count(sep) == 1:
                    base, quote = clean_pair.split(sep)
                    # Essayer différentes combinaisons de base/quote
                    for b in [base, self.SYMBOL_MAPPING.get(base, base)]:
                        for q in [quote, self.SYMBOL_MAPPING.get(quote, quote)]:
                            test_pair = f"{b}/{q}"
                            if test_pair in self._wsname_to_pair_id:
                                return test_pair
            
            # 4. Si on arrive ici, la paire n'est pas reconnue
            if raise_on_error:
                # Lever une UnsupportedTradingPairError pour correspondre au comportement de production
                raise UnsupportedTradingPairError(pair=pair)
            return None
        
        # Surcharger _find_alternative_pairs pour utiliser les données de test
        def _find_alternative_pairs(self, pair, pair_no_sep):
            # Implémentation simplifiée pour les tests
            alternatives = []
            
            # Essayer de trouver des paires avec la même base ou la même cotation
            for pair_id, pair_info in self._pairs_data.items():
                wsname = pair_info.get('wsname', '')
                base, quote = wsname.split('/') if '/' in wsname else ('', '')
                
                if pair_no_sep.startswith(base) or pair_no_sep.endswith(quote):
                    alternatives.append(wsname)
            
            # Limiter le nombre d'alternatives pour les tests
            return sorted(list(set(alternatives)))[:3]
    
    # Créer une instance de la sous-classe de test
    test_instance = TestAvailablePairs()
    
    # Sauvegarder l'état original
    original_instance = _available_pairs_instance
    original_initialized = _initialized
    original_get_available_pairs = get_available_pairs
    
    # Configurer l'instance globale pour les tests
    _available_pairs_instance = test_instance
    _initialized = True
    
    # S'assurer que l'instance est correctement initialisée
    test_instance._initialized = True
    test_instance._pairs_data = MOCK_PAIRS.copy()
    test_instance._build_indexes()
    
    # Créer un mock pour get_available_pairs
    def mock_get_available_pairs():
        if _available_pairs_instance is None or not _initialized:
            raise RuntimeError("AvailablePairs not initialized")
        return _available_pairs_instance
    
    # Utiliser monkeypatch pour les variables et fonctions globales
    monkeypatch.setattr('src.core.market.pair_initializer._available_pairs_instance', test_instance)
    monkeypatch.setattr('src.core.market.pair_initializer._initialized', True)
    
    # Créer un mock pour get_available_pairs qui retourne notre instance de test
    def mock_get_available_pairs():
        if not _initialized or _available_pairs_instance is None:
            raise RuntimeError("AvailablePairs not initialized")
        return test_instance
    
    monkeypatch.setattr('src.core.market.pair_initializer.get_available_pairs', mock_get_available_pairs)
    
    # Importer le module pair_utils après avoir configuré les mocks
    import sys
    import importlib
    
    # Forcer le rechargement du module pair_utils pour s'assurer qu'il utilise nos mocks
    if 'src.utils.pair_utils' in sys.modules:
        importlib.reload(sys.modules['src.utils.pair_utils'])
    
    # Importer le module après le rechargement
    import src.utils.pair_utils as pair_utils_module
    
    # S'assurer que la variable globale est définie
    pair_utils_module.available_pairs = test_instance
    
    # Mettre à jour la référence dans sys.modules
    sys.modules['src.utils.pair_utils'] = pair_utils_module
    
    # Appeler la méthode initialize de l'instance de test
    test_instance.initialize = AsyncMock(return_value=test_instance)
    
    # Donner accès à l'instance de test si nécessaire
    yield test_instance
    
    # Nettoyage après les tests
    import sys
    import importlib
    
    # Restaurer l'état original dans pair_initializer
    import src.core.market.pair_initializer as pair_initializer
    pair_initializer._available_pairs_instance = original_instance
    pair_initializer._initialized = original_initialized
    
    # Restaurer la variable globale dans pair_utils
    if 'src.utils.pair_utils' in sys.modules:
        import src.utils.pair_utils as pair_utils_module
        pair_utils_module.available_pairs = None
        
        # Recharger le module pour restaurer l'état original
        importlib.reload(pair_utils_module)
        sys.modules['src.utils.pair_utils'] = pair_utils_module
    
    # Recharger le module pair_initializer
    if 'src.core.market.pair_initializer' in sys.modules:
        importlib.reload(sys.modules['src.core.market.pair_initializer'])

class TestPairUtils:
    """Test cases for pair utilities with AvailablePairs integration."""
    
    @pytest.mark.asyncio
    @pytest.mark.asyncio
    async def test_initialize_pair_utils(self, setup_available_pairs):
        """Test the initialize_pair_utils function."""
        # Sauvegarder l'état original
        from src.utils.pair_utils import available_pairs, initialize_pair_utils
        from src.core.market.pair_initializer import initialize_available_pairs as original_initialize
        
        # Créer un mock pour la fonction initialize_available_pairs
        async def mock_initialize_available_pairs(*args, **kwargs):
            return setup_available_pairs
            
        # Remplacer temporairement la fonction d'initialisation
        import src.core.market.pair_initializer
        original_func = src.core.market.pair_initializer.initialize_available_pairs
        src.core.market.pair_initializer.initialize_available_pairs = mock_initialize_available_pairs
        
        try:
            # Réinitialiser available_pairs à None pour forcer l'initialisation
            from src.utils import pair_utils
            pair_utils.available_pairs = None
            
            # Appeler la fonction à tester
            await initialize_pair_utils()
            
            # Vérifier que available_pairs a été correctement initialisé
            assert pair_utils.available_pairs is not None
            assert pair_utils.available_pairs == setup_available_pairs
            
        finally:
            # Restaurer la fonction d'origine
            src.core.market.pair_initializer.initialize_available_pairs = original_func
            
            # Restaurer l'état original de available_pairs
            pair_utils.available_pairs = available_pairs
    
    def test_normalize_pair_input(self):
        """Test the normalize_pair_input function with various inputs."""
        # Test basic normalization with different separators
        assert normalize_pair_input("xbt/usd") == "XBT/USD"
        assert normalize_pair_input("XBT_USD") == "XBT/USD"
        assert normalize_pair_input("xbt-usd") == "XBT/USD"
        assert normalize_pair_input(" xbt usd ") == "XBT/USD"
        
        # Test with Kraken pair IDs
        assert normalize_pair_input("XXBTZUSD") == "XBT/USD"
        
        # Test with alt names
        assert normalize_pair_input("btc/usd") == "XBT/USD"
        
        # Test invalid inputs
        with pytest.raises(UnsupportedTradingPairError):
            normalize_pair_input("INVALID/PAIR")
            
        with pytest.raises(ValueError, match="Pair must be a string"):
            normalize_pair_input(123)  # type: ignore
            
        with pytest.raises(ValueError, match="Empty pair string"):
            normalize_pair_input("")

    def test_extract_pair_name(self):
        """Test the extract_pair_name function with various inputs."""
        # Test with string input
        assert extract_pair_name("xbt/usd") == "XBT/USD"
        assert extract_pair_name("XXBTZUSD") == "XBT/USD"
        
        # Test with dictionary input
        assert extract_pair_name({"pair": "xbt/usd"}) == "XBT/USD"
        assert extract_pair_name({"pair": "XXBTZUSD"}) == "XBT/USD"
        assert extract_pair_name({"pair": "eth/usd", "other": "data"}) == "ETH/USD"
        
        # Test with alt names
        assert extract_pair_name("btc/usd") == "XBT/USD"
        
        # Test invalid inputs
        with pytest.raises(ValueError, match="Invalid pair format"):
            extract_pair_name({"no_pair": "xbt/usd"})
            
        with pytest.raises(ValueError, match="Invalid pair format"):
            extract_pair_name(123)  # type: ignore

    def test_extract_pair_names(self):
        """Test the extract_pair_names function with various inputs."""
        # Test with mixed input types
        pairs = [
            "xbt/usd",
            {"pair": "eth/usd"},
            "XXBTZUSD",  # ID Kraken
            "invalidpair",  # Invalide
            {"no_pair": "xrp/btc"},  # Invalide
            "LTC/XBT",
            "XRP-EUR"  # Valide avec séparateur différent
        ]
        
        result = extract_pair_names(pairs)
        # Vérifier que nous avons le bon nombre de résultats (5 entrées valides)
        assert len(result) == 5  # 5 entrées valides: xbt/usd, eth/usd, XXBTZUSD, LTC/XBT, XRP-EUR
        # Vérifier que toutes les paires attendues sont présentes
        assert "XBT/USD" in result  # Correspond à xbt/usd
        assert "XBT/USD" in result  # Correspond à XXBTZUSD
        assert "ETH/USD" in result  # Correspond à eth/usd
        assert "LTC/XBT" in result  # Correspond à LTC/XBT
        assert "XRP/EUR" in result  # Correspond à XRP-EUR
        
        # Test with empty list
        assert extract_pair_names([]) == []
        
        # Test with all invalid pairs
        # Note: extract_pair_names devrait gérer silencieusement les paires invalides
        assert extract_pair_names(["invalid", {"no_pair": "xrp/btc"}]) == []

    def test_edge_cases(self):
        """Test edge cases and error handling."""
        # Test with None
        with pytest.raises(ValueError, match="Invalid pair format"):
            extract_pair_name(None)  # type: ignore
            
        # Test with empty string
        with pytest.raises(ValueError, match="Empty pair string"):
            extract_pair_name("")
            
            
        # Test with empty dict
        with pytest.raises(ValueError, match="Invalid pair format"):
            extract_pair_name({})
            
        # Test with unsupported pair
        with pytest.raises(UnsupportedTradingPairError):
            normalize_pair_input("UNSUPPORTED/PAIR")

# This file uses pytest for testing
# Run with: pytest tests/test_pair_utils.py -v
