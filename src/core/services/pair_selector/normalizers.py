"""
Normalisation des noms de paires de trading.

Ce module fournit des fonctions pour normaliser et générer des variantes
de noms de paires selon les conventions de Kraken.
"""
from typing import List, Set
from .validators import COMMON_CURRENCIES, QUOTE_CURRENCIES


def normalize_pair_name(base: str, quote: str) -> List[str]:
    """Génère les formats de noms de paires normalisés pour une paire donnée.
    
    Args:
        base: La devise de base (ex: 'XBT', 'ETH', 'BTC')
        quote: La devise de cotation (ex: 'EUR', 'USD', 'USDT')
        
    Returns:
        Liste des formats normalisés possibles pour la paire, triés par pertinence
    """
    if not base or not quote:
        return []
        
    base = base.upper().strip()
    quote = quote.upper().strip()
    
    # Liste pour stocker tous les formats générés
    formats = []
    
    # 1. Formats standards avec préfixes X/Z (formats Kraken)
    formats.append(f"X{base}Z{quote}")  # Format avec X et Z (XBTZEUR)
    formats.append(f"X{base}{quote}")    # Format avec X sans Z (XBTUSD)
    
    # 2. Formats spéciaux pour les devises courantes
    # Bitcoin (BTC/XBT)
    if base in ('BTC', 'XBT'):
        formats.append(f"XXBTZ{quote}")  # Format XXBT pour BTC
        formats.append(f"XXBT{quote}")
        formats.append(f"BTC{quote}")    # Format alternatif sans X
        formats.append(f"XBT{quote}")    # Format alternatif avec XBT
    # Ethereum (ETH)
    elif base == 'ETH':
        formats.append(f"XETHZ{quote}")
        formats.append(f"XETH{quote}")
        formats.append(f"ETH{quote}")    # Format alternatif sans X
    
    # 3. Gestion spéciale pour les paires stables
    stable_pairs = {
        'USDT': ['USD', 'USDT'],
        'USDC': ['USD', 'USDC'],
        'DAI': ['USD', 'DAI'],
        'USDP': ['USD', 'USDP'],
        'TUSD': ['USD', 'TUSD'],
        'BUSD': ['USD', 'BUSD'],
        'HUSD': ['USD', 'HUSD']
    }
    
    # Si la base ou la quote est un stablecoin, ajouter des variantes
    for stable, variants in stable_pairs.items():
        if base == stable or quote == stable:
            for v in variants:
                if base == stable:
                    formats.append(f"X{v}Z{quote}")
                    formats.append(f"X{v}{quote}")
                if quote == stable:
                    formats.append(f"X{base}Z{v}")
                    formats.append(f"X{base}{v}")
    
    # 4. Formats inversés (pour les paires comme USDT-BTC)
    formats.append(f"X{quote}Z{base}")
    formats.append(f"X{quote}{base}")
    
    # 5. Pour les paires inversées avec stablecoins, ajouter des variantes
    for stable, variants in stable_pairs.items():
        if base == stable or quote == stable:
            for v in variants:
                if base == stable:
                    formats.append(f"X{quote}Z{v}")
                    formats.append(f"X{quote}{v}")
                if quote == stable:
                    formats.append(f"X{v}Z{base}")
                    formats.append(f"X{v}{base}")
    
    # 6. Formats avec séparateurs
    formats.append(f"{base}/{quote}")  # Format avec slash
    formats.append(f"{base}-{quote}")  # Format avec tiret
    formats.append(f"{base}_{quote}")  # Format avec underscore
    formats.append(f"{base}{quote}")   # Format sans séparateur
    
    # 7. Formats alternatifs pour les paires courantes
    common_pairs = {
        'BTC': {
            'USD': ['XBT/USD', 'XXBTZUSD', 'XBTUSD'],
            'EUR': ['XBT/EUR', 'XXBTZEUR', 'XBTEUR']
        },
        'ETH': {
            'USD': ['ETH/USD', 'XETHZUSD', 'ETHUSD'],
            'EUR': ['ETH/EUR', 'XETHZEUR', 'ETHEUR']
        },
        'XRP': {
            'USD': ['XRP/USD', 'XXRPZUSD', 'XRPUSD'],
            'EUR': ['XRP/EUR', 'XXRPZEUR', 'XRPEUR']
        }
    }
    
    # Ajouter les formats communs si la paire est dans la liste
    if base in common_pairs and quote in common_pairs[base]:
        formats.extend(common_pairs[base][quote])
    
    # Nettoyage et déduplication
    seen = set()
    unique_formats = []
    
    for fmt in formats:
        # Nettoyer les formats vides ou invalides
        if not fmt or 'None' in fmt:
            continue
            
        # Convertir en majuscules et supprimer les espaces
        fmt_normalized = fmt.upper().strip()
        
        # Ajouter au résultat si non vide et non déjà vu
        if fmt_normalized and fmt_normalized not in seen:
            seen.add(fmt_normalized)
            unique_formats.append(fmt_normalized)
    
    # Trier par longueur (les plus courts d'abord, souvent plus courants)
    unique_formats.sort(key=len)
    
    return unique_formats


def generate_pair_variants(base: str, quote: str) -> Set[str]:
    """Génère toutes les variantes possibles pour une paire donnée.
    
    Args:
        base: La devise de base
        quote: La devise de cotation
        
    Returns:
        Ensemble des variantes de la paire
    """
    variants = set()
    
    # Ajouter les formats standard
    variants.update([
        f"{base}/{quote}",
        f"{base}-{quote}",
        f"{base}_{quote}",
        f"{base}{quote}",
        f"X{base}Z{quote}",
        f"X{base}{quote}",
        f"{base}Z{quote}",
    ])
    
    # Ajouter les variantes inversées
    variants.update([
        f"{quote}/{base}",
        f"{quote}-{base}",
        f"{quote}_{base}",
        f"{quote}{base}",
        f"X{quote}Z{base}",
        f"X{quote}{base}",
        f"{quote}Z{base}",
    ])
    
    # Nettoyer et retourner
    return {v.upper().strip() for v in variants if v and 'None' not in v}
