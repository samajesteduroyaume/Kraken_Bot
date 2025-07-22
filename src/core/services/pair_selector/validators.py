"""
Validation des paires de trading.

Ce module fournit des fonctions pour valider et traiter les noms de paires
de trading selon différents formats et standards.
"""
import re
from typing import List, Optional, Tuple

# Liste des devises courantes pour la validation
COMMON_CURRENCIES = [
    # Cryptomonnaies majeures
    'XBT', 'BTC', 'ETH', 'XRP', 'ADA', 'DOT', 'SOL', 'AVAX', 'MATIC', 'LINK',
    'AAVE', 'ALGO', 'ATOM', 'AXS', 'BAT', 'BCH', 'COMP', 'CRV', 'DASH', 'DOGE',
    'EGLD', 'ENJ', 'EOS', 'ETC', 'FIL', 'FLOW', 'GNO', 'GRT', 'ICX', 'KAVA',
    'KEEP', 'KNC', 'KSM', 'LRC', 'LSK', 'MANA', 'MKR', 'NANO', 'OCEAN', 'OMG',
    'OXT', 'PAXG', 'QTUM', 'RARI', 'REN', 'REPV2', 'SAND', 'SC', 'SNX', 'STORJ',
    'SUSHI', 'TBTC', 'TRX', 'UNI', 'WAVES', 'XLM', 'XMR', 'XTZ', 'YFI', 'ZEC', 'ZRX',
    
    # Stablecoins et monnaies fiat
    'USDT', 'USDC', 'DAI', 'USDP', 'TUSD', 'BUSD', 'HUSD', 'USD', 'EUR', 'GBP',
    'JPY', 'AUD', 'CAD', 'CHF', 'SGD', 'KRW', 'CNY', 'TRY', 'NZD', 'RUB'
]

# Devises de cotation courantes
QUOTE_CURRENCIES = [
    'USDT', 'USDC', 'USD', 'EUR', 'GBP', 'JPY', 'XBT', 'BTC', 'ETH',
    'AUD', 'CAD', 'CHF', 'SGD', 'KRW', 'CNY', 'TRY', 'NZD', 'RUB',
    'DAI', 'USDP', 'TUSD', 'BUSD', 'HUSD'
]

# Expression régulière pour valider les caractères d'une paire
PAIR_PATTERN = r'^[A-Z0-9]+(?:[\-/]?[A-Z0-9]+)*$'


def validate_pair_format(pair: str) -> bool:
    """Valide le format d'une paire de trading.
    
    Args:
        pair: La paire à valider (ex: 'XBT/USDT', 'XBT-USD', 'XBTUSDT')
        
    Returns:
        bool: True si le format est valide, False sinon
    """
    if not isinstance(pair, str) or not pair:
        return False
        
    # Nettoyer la chaîne
    pair = pair.strip().upper()
    
    # Vérifier la longueur minimale (ex: "A/B")
    if len(pair) < 3:
        return False
        
    # Vérifier les caractères autorisés (lettres, chiffres, tirets, slash)
    if not re.match(PAIR_PATTERN, pair):
        return False
    
    # Essayer de séparer la paire
    base, quote = split_pair(pair)
    if not base or not quote:
        return False
    
    # Vérifier que la paire n'est pas composée de la même devise
    if base == quote:
        return False
    
    # Vérifier que les devises ont une longueur raisonnable
    if len(base) < 2 or len(quote) < 2:
        return False
        
    return True


def split_pair(pair: str) -> Tuple[Optional[str], Optional[str]]:
    """Sépare une paire en devise de base et de cotation.
    
    Args:
        pair: La paire à séparer (ex: 'XBT/USDT', 'XBT-USD', 'XBTUSDT')
        
    Returns:
        Tuple contenant (base, quote) si la séparation réussit, (None, None) sinon
    """
    if not pair:
        return None, None
        
    pair = pair.strip().upper()
    
    # Si la paire contient un séparateur
    if '/' in pair or '-' in pair:
        parts = re.split(r'[\/-]', pair, 1)
        if len(parts) == 2 and all(parts):
            return parts[0], parts[1]
        return None, None
    
    # Pour les paires sans séparateur, essayer de deviner la séparation
    return split_pair_without_separator(pair)


def split_pair_without_separator(pair: str) -> Tuple[Optional[str], Optional[str]]:
    """Tente de séparer une paire sans séparateur en base et quote.
    
    Args:
        pair: La paire à séparer (sans séparateur, ex: 'XBTUSDT')
        
    Returns:
        Tuple contenant (base, quote) si la séparation réussit, (None, None) sinon
    """
    if not pair or len(pair) < 4:  # Au moins 2 caractères pour chaque devise
        return None, None
    
    # Essayer avec les devises de cotation les plus courantes
    for quote in QUOTE_CURRENCIES:
        if pair.endswith(quote):
            base = pair[:-len(quote)]
            if base and len(base) >= 2:  # Au moins 2 caractères pour la devise de base
                return base, quote
    
    # Essayer avec les devises de base
    for base in COMMON_CURRENCIES:
        if pair.startswith(base):
            quote = pair[len(base):]
            if quote and len(quote) >= 2:  # Au moins 2 caractères pour la cotation
                return base, quote
    
    # Dernier recours : séparer au milieu
    if len(pair) >= 4:
        split_pos = len(pair) // 2
        return pair[:split_pos], pair[split_pos:]
    
    return None, None


def is_common_currency(currency: str) -> bool:
    """Vérifie si une devise est dans la liste des devises courantes.
    
    Args:
        currency: La devise à vérifier
        
    Returns:
        bool: True si la devise est connue, False sinon
    """
    return currency.upper() in COMMON_CURRENCIES if currency else False
