"""
Tests pour le module available_pairs.py
"""
import asyncio
import os
import sys
from pathlib import Path

# Ajouter le répertoire parent au PYTHONPATH pour les imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.market.available_pairs import AvailablePairs, KrakenAPIError
import pytest

@pytest.fixture
def available_pairs():
    """Fixture pour initialiser AvailablePairs pour les tests."""
    return AvailablePairs()

@pytest.mark.asyncio
async def test_initialize(available_pairs):
    """Teste l'initialisation de AvailablePairs."""
    # Test avec utilisation du cache
    await available_pairs.initialize(use_cache=True)
    assert available_pairs._initialized is True
    assert len(available_pairs._pairs_data) > 0
    assert len(available_pairs._wsname_to_pair_id) > 0
    assert len(available_pairs._base_currencies) > 0
    assert len(available_pairs._quote_currencies) > 0
    
    # Vérifier que le cache a été créé
    cache_file = Path("data/available_pairs_cache.json")
    assert cache_file.exists()
    
    # Vérifier la date de mise à jour
    assert available_pairs._last_updated is not None

@pytest.mark.asyncio
async def test_is_pair_supported(available_pairs):
    """Teste la vérification des paires supportées."""
    await available_pairs.initialize()
    
    # Paires connues à tester
    test_pairs = [
        "XBT/USD",  # Format standard
        "btc/usd",  # Minuscules
        "BTC-USD",  # Tiret
        "BTC_USD",  # Underscore
        "XXBTZUSD",  # ID Kraken
        "XXBTZEUR",  # ID Kraken EUR
        "ETH/USD",   # Autre crypto
        "XBT/EUR",   # Autre devise
    ]
    
    for pair in test_pairs:
        assert available_pairs.is_pair_supported(pair), f"La paire {pair} devrait être supportée"
    
    # Paires qui ne devraient pas être supportées
    invalid_pairs = ["", "INVALID/PAIR", "BTC", "12345", "XBT/USD/EXTRA"]
    for pair in invalid_pairs:
        if pair:  # Ignorer la chaîne vide pour ce test
            assert not available_pairs.is_pair_supported(pair), f"La paire {pair} ne devrait pas être supportée"

@pytest.mark.asyncio
async def test_normalize_pair(available_pairs):
    """Teste la normalisation des noms de paires."""
    await available_pairs.initialize()
    
    # Test des formats de paires
    test_cases = [
        ("XBT/USD", "XBT/USD"),      # Déjà normalisé
        ("btc/usd", "XBT/USD"),      # Conversion BTC -> XBT
        ("BTC-USD", "XBT/USD"),      # Tiret
        ("BTC_USD", "XBT/USD"),      # Underscore
        ("btceur", "XBT/EUR"),       # Sans séparateur
        ("XXBTZUSD", "XBT/USD"),     # ID Kraken
        ("XBT/ZUSD", "XBT/USD"),     # Code devise interne Kraken
        ("XBT/ZEUR", "XBT/EUR"),     # Code devise interne Kraken
        ("XBT/ZGBP", "XBT/GBP"),     # Code devise interne Kraken
        ("XBT/ZJPY", "XBT/JPY"),     # Code devise interne Kraken
        ("XBT/ZCAD", "XBT/CAD"),     # Code devise interne Kraken
        ("BTC/ZUSD", "XBT/USD"),     # Code devise interne avec BTC
        ("BTC/ZEUR", "XBT/EUR"),     # Code devise interne avec BTC
        ("XXBTZEUR", "XBT/EUR"),     # ID Kraken avec EUR
        ("XXBTZGBP", "XBT/GBP"),     # ID Kraken avec GBP
        ("XXBTZJPY", "XBT/JPY"),     # ID Kraken avec JPY
        ("XXBTZCAD", "XBT/CAD"),     # ID Kraken avec CAD
    ]
    
    for input_pair, expected in test_cases:
        normalized = available_pairs.normalize_pair(input_pair)
        assert normalized == expected, f"La normalisation de {input_pair} devrait donner {expected}, a donné {normalized}"

@pytest.mark.asyncio
async def test_is_quote_currency_supported(available_pairs):
    """Teste la vérification des devises de cotation supportées."""
    await available_pairs.initialize()
    
    # Devises supportées (codes standards)
    supported_currencies = ["USD", "EUR", "GBP", "JPY", "CAD"]
    for currency in supported_currencies:
        assert available_pairs.is_quote_currency_supported(currency), f"La devise {currency} devrait être supportée"
    
    # Codes internes Kraken
    kraken_codes = ["ZUSD", "ZEUR", "ZGBP", "ZJPY", "ZCAD"]
    for code in kraken_codes:
        assert available_pairs.is_quote_currency_supported(code), f"Le code interne {code} devrait être supporté"
    
    # Devises non supportées
    unsupported_currencies = ["AUD", "CHF", "CNY", "RUB"]
    for currency in unsupported_currencies:
        assert not available_pairs.is_quote_currency_supported(currency), f"La devise {currency} ne devrait pas être supportée"
    
    # Entrées invalides
    invalid_inputs = ["", " ", None, 123, "123"]
    for invalid in invalid_inputs:
        assert not available_pairs.is_quote_currency_supported(invalid), f"L'entivée {invalid} ne devrait pas être valide"

@pytest.mark.asyncio
async def test_altname_handling(available_pairs):
    """Teste la gestion des altnames comme DOGE/XDG."""
    await available_pairs.initialize()
    
    # Tester avec DOGE/XDG (même crypto, noms différents)
    assert available_pairs.is_pair_supported("DOGE/USD") == available_pairs.is_pair_supported("XDG/USD")
    assert available_pairs.normalize_pair("DOGE/USD") == available_pairs.normalize_pair("XDG/USD")
    
    # Vérifier que les deux noms pointent vers la même paire
    doge_info = available_pairs.get_pair_info("DOGE/USD")
    xdg_info = available_pairs.get_pair_info("XDG/USD")
    assert doge_info is not None and xdg_info is not None
    assert doge_info['wsname'] == xdg_info['wsname']
    
    # Tester avec des formats non standard
    assert available_pairs.normalize_pair("dogeusd") == available_pairs.normalize_pair("XDG/USD")
    assert available_pairs.normalize_pair("dogexdg") is None  # Ne devrait pas être valide

@pytest.mark.asyncio
async def test_edge_cases(available_pairs):
    """Teste les cas limites et les entrées non standard."""
    await available_pairs.initialize()
    
    # Chaînes très longues
    long_string = "A" * 1000
    assert not available_pairs.is_pair_supported(long_string)
    assert available_pairs.normalize_pair(long_string) is None
    
    # Caractères spéciaux
    special_chars = "!@#$%^&*()_+-=[]{}|;:,.<>?"
    for char in special_chars:
        test_pair = f"XBT{char}USD"
        if char in "/-_":  # Séparateurs valides
            assert available_pairs.is_pair_supported(test_pair)
        else:
            assert not available_pairs.is_pair_supported(test_pair)
    
    # Nombres et symboles
    assert not available_pairs.is_pair_supported("123/456")
    assert not available_pairs.is_pair_supported("XBT/USD!")
    assert not available_pairs.is_pair_supported("XBT/USD/EXTRA")

@pytest.mark.asyncio
async def test_performance(available_pairs):
    """Teste les performances des opérations fréquentes."""
    await available_pairs.initialize()
    
    # Test de performance pour la normalisation
    pairs = ["btc-usd", "eth/eur", "xrpusd", "XXBTZUSD", "dogeusd"] * 100  # 500 appels
    
    # Mesurer le temps de normalisation
    start_time = asyncio.get_event_loop().time()
    for pair in pairs:
        available_pairs.normalize_pair(pair)
    end_time = asyncio.get_event_loop().time()
    
    # Vérifier que c'est raisonnablement rapide (moins de 0.5 secondes pour 500 appels)
    assert (end_time - start_time) < 0.5, "La normalisation est trop lente"
    
    # Vérifier que le cache fonctionne
    start_time = asyncio.get_event_loop().time()
    for _ in range(1000):
        available_pairs.is_pair_supported("XBT/USD")
    end_time = asyncio.get_event_loop().time()
    
    # Vérifier que c'est très rapide grâce au cache (moins de 0.1 seconde pour 1000 appels)
    assert (end_time - start_time) < 0.1, "Le cache ne fonctionne pas correctement"
    assert (end_time - start_time) < 0.1  # Moins de 100ms pour 1000 appels

@pytest.mark.asyncio
async def test_get_pair_info(available_pairs):
    """Teste la récupération des informations d'une paire."""
    await available_pairs.initialize()
    
    # Tester avec différentes entrées pour la même paire
    test_inputs = ["XBT/USD", "XXBTZUSD", "btc-usd", "btcusd"]
    
    for pair_input in test_inputs:
        pair_info = available_pairs.get_pair_info(pair_input)
        assert pair_info is not None, f"Aucune information trouvée pour {pair_input}"
        assert 'wsname' in pair_info
        assert 'base' in pair_info
        assert 'quote' in pair_info
        assert pair_info['wsname'] == 'XBT/USD'  # Doit correspondre à la paire normalisée
    
    # Tester avec une paire inexistante
    assert available_pairs.get_pair_info("INVALID/PAIR") is None

@pytest.mark.asyncio
async def test_get_available_pairs(available_pairs):
    """Teste la récupération des paires disponibles."""
    await available_pairs.initialize()
    
    # Toutes les paires
    all_pairs = available_pairs.get_available_pairs()
    assert len(all_pairs) > 0
    assert all('/' in p for p in all_pairs)  # Toutes les paires devraient être au format BASE/QUOTE
    
    # Filtrage par devise de cotation
    usd_pairs = available_pairs.get_available_pairs("USD")
    assert len(usd_pairs) > 0
    assert all(p.endswith('/USD') for p in usd_pairs)
    
    eur_pairs = available_pairs.get_available_pairs("EUR")
    assert len(eur_pairs) > 0
    assert all(p.endswith('/EUR') for p in eur_pairs)

@pytest.mark.asyncio
async def test_error_handling():
    """Teste la gestion des erreurs."""
    pairs = AvailablePairs()
    
    # Tester avec une URL d'API invalide
    old_url = pairs.KRAKEN_API_URL
    pairs.KRAKEN_API_URL = "https://api.kraken.com/invalid/endpoint"
    
    with pytest.raises(KrakenAPIError):
        await pairs.initialize(use_cache=False)  # Forcer le rechargement depuis l'API
    
    # Restaurer l'URL
    pairs.KRAKEN_API_URL = old_url

@pytest.mark.asyncio
async def test_find_alternative_pairs(available_pairs):
    """Teste la recherche de paires alternatives pour une paire non reconnue."""
    await available_pairs.initialize()
    
    # Cas de test avec une paire qui n'existe pas directement
    test_cases = [
        ("ZRX/ETH", ["ZRX/XBT", "ZRX/USD", "ZRX/EUR"]),  # Devrait trouver des alternatives avec ZRX
        ("UNKNOWN/USD", ["XBT/USD", "ETH/USD"]),        # Devrait trouver des alternatives avec USD
        ("BTC/UNKNOWN", ["XBT/USD", "XBT/EUR"]),        # Devrait trouver des alternatives avec XBT
    ]
    
    for pair, expected_alternatives in test_cases:
        # On utilise la méthode publique qui appelle _find_alternative_pairs en interne
        try:
            available_pairs.normalize_pair(pair, raise_on_error=True)
            assert False, f"La paire {pair} ne devrait pas être reconnue directement"
        except Exception as e:
            # Vérifier que l'exception contient des suggestions d'alternatives
            if hasattr(e, 'alternatives') and e.alternatives:
                # Vérifier que certaines des alternatives attendues sont présentes
                found = any(alt in e.alternatives for alt in expected_alternatives)
                assert found, f"Aucune alternative attendue trouvée pour {pair}. Alternatives: {e.alternatives}"
            else:
                assert False, f"Aucune alternative suggérée pour {pair}"

@pytest.mark.asyncio
async def test_find_intermediate_pairs(available_pairs):
    """Teste la recherche de paires intermédiaires pour la conversion."""
    await available_pairs.initialize()
    
    # Cas de test avec des paires qui nécessitent une conversion intermédiaire
    test_cases = [
        ("ZRX", "ETH", ["XBT"]),  # Devrait trouver XBT comme intermédiaire
        ("XLM", "LTC", ["XBT", "ETH"]),  # Devrait trouver XBT ou ETH comme intermédiaire
    ]
    
    for base, quote, expected_intermediates in test_cases:
        # Appel direct à la méthode protégée pour le test
        # Note: En production, cette méthode est appelée via _find_alternative_pairs
        try:
            # On utilise la réflexion pour accéder à la méthode protégée
            method = available_pairs._find_intermediate_pairs
            result = method(base, quote)
            
            # Vérifier que le résultat n'est pas vide
            assert result, f"Aucun chemin trouvé pour {base}/{quote}"
            
            # Vérifier que les chemins utilisent bien les devises intermédiaires attendues
            for path in result:
                # Un chemin est une liste de paires, comme ["ZRX/XBT", "XBT/ETH"]
                intermediates = set()
                for step in path:
                    # Extraire les devises de chaque étape
                    parts = step.split('/')
                    if len(parts) == 2:
                        intermediates.update(parts)
                
                # Vérifier qu'au moins une des devises intermédiaires attendues est présente
                assert any(intermediate in intermediates for intermediate in expected_intermediates), \
                    f"Aucune des devises intermédiaires attendues {expected_intermediates} trouvée dans le chemin {path}"
                    
        except Exception as e:
            assert False, f"Erreur lors de la recherche de paires intermédiaires pour {base}/{quote}: {e}"

@pytest.mark.asyncio
async def test_error_handling_in_alternative_pairs(available_pairs):
    """Teste la gestion des erreurs dans la recherche de paires alternatives."""
    await available_pairs.initialize()
    
    # Cas de test avec des entrées invalides
    invalid_inputs = [
        "",                  # Chaîne vide
        "INVALID",           # Format invalide
        "A/B/C",             # Trop de parties
        "/",                 # Aucune devise
        "A/", "/B", "//",  # Formats incorrects
    ]
    
    for invalid_input in invalid_inputs:
        try:
            # On s'attend à ce que ces entrées lèvent une exception
            result = available_pairs._find_alternative_pairs(invalid_input, invalid_input.replace('/', ''))
            # Si on arrive ici, c'est une erreur
            assert False, f"L'entrée invalide {invalid_input} n'a pas levé d'exception. Résultat: {result}"
        except (ValueError, AttributeError):
            # C'est le comportement attendu
            pass
        except Exception as e:
            # Toute autre exception est une erreur
            assert False, f"Exception inattendue pour l'entrée {invalid_input}: {e}"

if __name__ == "__main__":
    # Exécuter les tests avec pytest
    import pytest
    pytest.main([__file__, "-v"])
