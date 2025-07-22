"""
Constantes et configurations pour le module PairSelector.

Ce module contient des constantes, des configurations par défaut et des listes
utilisées dans tout le module PairSelector.
"""
from typing import Dict, List, Set, Tuple

# Configuration par défaut pour le PairSelector
DEFAULT_CONFIG = {
    # Cache
    'cache_ttl': 3600,  # Durée de vie du cache en secondes (1h)
    'max_cache_size': 100,  # Nombre maximum d'entrées dans le cache
    'cache_dir': 'data/cache/pair_selector',  # Répertoire de cache
    
    # Analyse des paires
    'min_volume_24h': 100000,  # Volume minimal sur 24h (en USD)
    'max_spread': 0.1,  # Spread maximal accepté (en %)
    'min_volatility': 0.01,  # Volatilité minimale (1%)
    'max_volatility': 2.0,  # Volatilité maximale (200%)
    'rsi_overbought': 70,  # Niveau RSI de surachat
    'rsi_oversold': 30,  # Niveau RSI de survente
    'min_liquidity': 10000,  # Liquidité minimale (volume * prix)
    
    # Seuils de scoring
    'score_threshold': 0.6,  # Score minimal pour qu'une paire soit considérée comme valide
    'min_confidence': 0.7,  # Confiance minimale pour les correspondances
    
    # Timeouts et retries
    'api_timeout': 30,  # Timeout des appels API en secondes
    'max_retries': 3,  # Nombre maximum de tentatives pour les appels API
    'retry_delay': 1,  # Délai initial entre les tentatives en secondes
    
    # Journalisation
    'log_level': 'INFO',  # Niveau de journalisation
    'log_file': 'logs/pair_selector.log',  # Fichier de log
    
    # Performance
    'max_concurrent_requests': 10,  # Nombre maximum de requêtes concurrentielles
    'batch_size': 50,  # Taille des lots pour le traitement par lots
}

# Liste des devises courantes
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

# Mappage des alias de devises (synonymes)
CURRENCY_ALIASES = {
    'XBT': 'BTC',  # Bitcoin
    'XBT.d': 'BTC',
    'XXBT': 'BTC',
    'XETH': 'ETH',  # Ethereum
    'XXRP': 'XRP',  # Ripple
    'XLTC': 'LTC',  # Litecoin
    'XXLM': 'XLM',  # Stellar
    'XZEC': 'ZEC',  # Zcash
    'XXMR': 'XMR',  # Monero
    'XREP': 'REP',  # Augur
    'XXDG': 'DOGE', # Dogecoin
    'XDG': 'DOGE',
    'ZAUD': 'AUD',  # Dollar australien
    'ZCAD': 'CAD',  # Dollar canadien
    'ZEUR': 'EUR',  # Euro
    'ZGBP': 'GBP',  # Livre sterling
    'ZJPY': 'JPY',  # Yen japonais
    'ZUSD': 'USD',  # Dollar américain
    'ZUSDT': 'USDT', # Tether
    'ZUSDC': 'USDC', # USD Coin
    'ZDAI': 'DAI',  # DAI
    'ZPAX': 'USDP', # Paxos Standard
    'ZUSDT': 'USDT',
    'ZUSDC': 'USDC',
    'ZDAI': 'DAI',
    'ZPAX': 'USDP',
    'ZUSD-M': 'ZUSD',  # Certaines paires utilisent ce format
    'ZUSD.d': 'USD',
    'ZUSD.HOLD': 'USD',
    'ZUSD.M': 'USD',
    'ZUSD.MARGIN': 'USD',
}

# Paires de trading populaires avec leurs formats préférés
POPULAR_PAIRS = {
    # Format: (base, quote, formats_préférés)
    ('BTC', 'USD'): ['XBT/USD', 'XXBTZUSD', 'XBTUSD'],
    ('BTC', 'USDT'): ['XBT/USDT', 'XBTUSDT', 'XXBTZUSDT'],
    ('BTC', 'EUR'): ['XBT/EUR', 'XXBTZEUR', 'XBTEUR'],
    ('ETH', 'USD'): ['ETH/USD', 'XETHZUSD', 'ETHUSD'],
    ('ETH', 'USDT'): ['ETH/USDT', 'ETHUSDT', 'XETHZUSDT'],
    ('ETH', 'BTC'): ['ETH/XBT', 'XETHXXBT', 'ETHXBT'],
    ('XRP', 'USD'): ['XRP/USD', 'XXRPZUSD', 'XRPUSD'],
    ('XRP', 'BTC'): ['XRP/XBT', 'XXRPXXBT', 'XRPXBT'],
    ('LTC', 'USD'): ['LTC/USD', 'XLTCZUSD', 'LTCUSD'],
    ('LTC', 'BTC'): ['LTC/XBT', 'XLTCXXBT', 'LTCXBT'],
    ('BCH', 'USD'): ['BCH/USD', 'BCHUSD', 'BCHUSD.d'],
    ('BCH', 'BTC'): ['BCH/XBT', 'BCHXBT', 'BCHXBT.d'],
    ('EOS', 'USD'): ['EOS/USD', 'EOSUSD', 'EOSEUR'],
    ('EOS', 'BTC'): ['EOS/XBT', 'EOSXBT'],
    ('XMR', 'USD'): ['XMR/USD', 'XXMRZUSD', 'XMRUSD'],
    ('XMR', 'BTC'): ['XMR/XBT', 'XXMRXXBT', 'XMRXBT'],
    ('DASH', 'USD'): ['DASH/USD', 'DASHUSD', 'DASHUSD.d'],
    ('DASH', 'BTC'): ['DASH/XBT', 'DASHXBT', 'DASHXBT.d'],
    ('ZEC', 'USD'): ['ZEC/USD', 'XZECZUSD', 'ZECUSD'],
    ('ZEC', 'BTC'): ['ZEC/XBT', 'XZECXXBT', 'ZECXBT'],
    ('ETC', 'USD'): ['ETC/USD', 'XETCZUSD', 'ETCUSD'],
    ('ETC', 'BTC'): ['ETC/XBT', 'XETCXXBT', 'ETCXBT'],
    ('USDT', 'USD'): ['USDT/USD', 'USDTZUSD', 'USDTUSD'],
    ('USDC', 'USD'): ['USDC/USD', 'USDCUSD', 'USDCUSD.d'],
    ('DAI', 'USD'): ['DAI/USD', 'DAIUSD', 'DAIUSD.d'],
}

# Configuration des indicateurs techniques
TECHNICAL_INDICATORS = {
    'rsi': {
        'period': 14,
        'overbought': 70,
        'oversold': 30,
        'weight': 0.15
    },
    'bbands': {
        'period': 20,
        'std_dev': 2.0,
        'weight': 0.10
    },
    'macd': {
        'fast_period': 12,
        'slow_period': 26,
        'signal_period': 9,
        'weight': 0.10
    },
    'stoch': {
        'k_period': 14,
        'd_period': 3,
        'smoothing': 3,
        'weight': 0.05
    },
    'adx': {
        'period': 14,
        'threshold': 25,
        'weight': 0.10
    },
    'atr': {
        'period': 14,
        'weight': 0.10
    },
    'ema': {
        'short_period': 9,
        'medium_period': 21,
        'long_period': 50,
        'weight': 0.15
    },
    'volume': {
        'ma_period': 20,
        'weight': 0.15
    },
    'ichimoku': {
        'conversion_periods': 9,
        'base_periods': 26,
        'lagging_span2_periods': 52,
        'displacement': 26,
        'weight': 0.10
    }
}

# Poids pour le calcul du score global
SCORE_WEIGHTS = {
    'liquidity': 0.25,
    'volatility': 0.20,
    'volume_24h': 0.15,
    'avg_spread': -0.10,
    'sharpe_ratio': 0.15,
    'max_drawdown': -0.15,
    'rsi': 0.10,
    'adx': 0.05,
    'momentum': 0.05
}

# Codes d'erreur courants de l'API Kraken
KRAKEN_ERROR_CODES = {
    'EGeneral': 'Erreur générale',
    'EAPI': 'Erreur d\'API invalide',
    'EAuth': 'Échec d\'authentification',
    'EOrder': 'Erreur de commande invalide',
    'EQuery': 'Requête invalide',
    'EInsufficient funds': 'Fonds insuffisants',
    'ERateLimit': 'Limite de débit dépassée',
    'EService': 'Service indisponible',
    'ETrade': 'Erreur de trading',
    'EInvalid': 'Arguments invalides',
    'EInvalid:Invalid arguments': 'Arguments invalides',
    'EAPI:Invalid key': 'Clé API invalide',
    'EAPI:Invalid signature': 'Signature API invalide',
    'EAPI:Invalid nonce': 'Nonce invalide',
    'EOrder:Unknown order': 'Commande inconnue',
    'EOrder:Cannot open position': 'Impossible d\'ouvrir une position',
    'EOrder:Cannot open opposing position': 'Impossible d\'ouvrir une position opposée',
    'EOrder:Insufficient margin': 'Marge insuffisante',
    'EOrder:Invalid price': 'Prix invalide',
    'EOrder:Invalid volume': 'Volume invalide',
    'EOrder:Orders limit exceeded': 'Limite de commandes dépassée',
    'EOrder:Rate limit exceeded': 'Limite de débit dépassée',
    'EOrder:Unknown asset pair': 'Paire d\'actifs inconnue',
    'EOrder:Permission denied': 'Permission refusée',
    'ECancel:Invalid request': 'Demande d\'annulation invalide',
    'ECancel:Invalid transaction': 'Transaction d\'annulation invalide',
    'EQuery:Unknown asset': 'Actif inconnu',
    'EQuery:Unknown asset pair': 'Paire d\'actifs inconnue',
    'EService:Market in cancel_only': 'Marché en mode annulation uniquement',
    'EService:Market in post_only': 'Marché en mode publication uniquement',
    'EService:Deadline elapsed': 'Délai dépassé',
    'ETrade:Too many requests': 'Trop de requêtes',
    'EFunding:Unknown withdraw key': 'Clé de retrait inconnue',
    'EFunding:Invalid amount': 'Montant de retrait invalide',
    'EFunding:Insufficient funds': 'Fonds insuffisants pour le retrait',
    'EFunding:Withdrawal limit exceeded': 'Limite de retrait dépassée',
    'EFunding:Rate limit exceeded': 'Limite de débit de retrait dépassée',
    'EFunding:Withdrawal fee too low': 'Frais de retrait trop bas',
    'EFunding:Withdrawal fee too high': 'Frais de retrait trop élevés',
    'EFunding:Invalid withdrawal address': 'Adresse de retrait invalide',
    'EFunding:Address not whitelisted': 'Adresse non sur liste blanche',
    'EFunding:Temporary error': 'Erreur temporaire de financement',
    'EFunding:Email confirmation required': 'Confirmation par email requise',
    'EFunding:Withdrawal in progress': 'Retrait déjà en cours',
    'EFunding:Withdrawal already in progress': 'Retrait déjà en cours',
    'EFunding:Withdrawal limit exceeded': 'Limite de retrait dépassée',
    'EFunding:Daily withdrawal limit exceeded': 'Limite de retrait quotidienne dépassée',
    'EFunding:Monthly withdrawal limit exceeded': 'Limite de retrait mensuelle dépassée',
    'EFunding:Withdrawal fee too low': 'Frais de retrait trop bas',
    'EFunding:Withdrawal fee too high': 'Frais de retrait trop élevés',
    'EFunding:Insufficient available balance': 'Solde disponible insuffisant',
    'EFunding:Address not whitelisted': 'Adresse non sur liste blanche',
    'EFunding:Invalid withdrawal address': 'Adresse de retrait invalide',
    'EFunding:Withdrawal address not verified': 'Adresse de retrait non vérifiée',
    'EFunding:Withdrawal address requires approval': 'Approbation requise pour l\'adresse de retrait',
    'EFunding:Withdrawal address not in address book': 'Adresse de retrait absente du carnet d\'adresses',
    'EFunding:Withdrawal address requires email confirmation': 'Confirmation par email requise pour l\'adresse de retrait',
    'EFunding:Withdrawal address requires 2FA': '2FA requise pour le retrait',
    'EFunding:Withdrawal address requires email and 2FA': 'Email et 2FA requis pour le retrait',
    'EFunding:Withdrawal address requires confirmation email': 'Email de confirmation requis pour l\'adresse de retrait',
    'EFunding:Withdrawal address requires confirmation email and 2FA': 'Email de confirmation et 2FA requis pour l\'adresse de retrait',
    'EFunding:Withdrawal address requires approval and 2FA': 'Approbation et 2FA requis pour l\'adresse de retrait',
    'EFunding:Withdrawal address requires approval and confirmation email': 'Approbation et email de confirmation requis pour l\'adresse de retrait',
    'EFunding:Withdrawal address requires approval, confirmation email and 2FA': 'Approbation, email de confirmation et 2FA requis pour l\'adresse de retrait',
    'EFunding:Withdrawal address requires verification': 'Vérification requise pour l\'adresse de retrait',
    'EFunding:Withdrawal address requires verification and 2FA': 'Vérification et 2FA requis pour l\'adresse de retrait',
    'EFunding:Withdrawal address requires verification and confirmation email': 'Vérification et email de confirmation requis pour l\'adresse de retrait',
    'EFunding:Withdrawal address requires verification, confirmation email and 2FA': 'Vérification, email de confirmation et 2FA requis pour l\'adresse de retrait',
    'EFunding:Withdrawal address requires verification and approval': 'Vérification et approbation requises pour l\'adresse de retrait',
    'EFunding:Withdrawal address requires verification, approval and 2FA': 'Vérification, approbation et 2FA requis pour l\'adresse de retrait',
    'EFunding:Withdrawal address requires verification, approval and confirmation email': 'Vérification, approbation et email de confirmation requis pour l\'adresse de retrait',
    'EFunding:Withdrawal address requires verification, approval, confirmation email and 2FA': 'Vérification, approbation, email de confirmation et 2FA requis pour l\'adresse de retrait',
    'EFunding:Withdrawal address requires verification and approval, confirmation email and 2FA': 'Vérification, approbation, email de confirmation et 2FA requis pour l\'adresse de retrait',
    'EFunding:Withdrawal address requires verification, approval and confirmation email, and 2FA': 'Vérification, approbation, email de confirmation et 2FA requis pour l\'adresse de retrait',
    'EFunding:Withdrawal address requires verification, approval, confirmation email and 2FA': 'Vérification, approbation, email de confirmation et 2FA requis pour l\'adresse de retrait',
}

# Paramètres par défaut pour les appels API
DEFAULT_API_PARAMS = {
    'timeout': 30,  # Timeout en secondes
    'verify': True,  # Vérifier le certificat SSL
    'proxies': None,  # Pas de proxy par défaut
    'retries': 3,  # Nombre de tentatives en cas d'échec
    'backoff_factor': 0.5,  # Facteur d'attente exponentielle entre les tentatives
}

# Configuration des logs
LOGGING_CONFIG = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'standard': {
            'format': '%(asctime)s [%(levelname)s] %(name)s: %(message)s',
            'datefmt': '%Y-%m-%d %H:%M:%S'
        },
    },
    'handlers': {
        'console': {
            'level': 'INFO',
            'class': 'logging.StreamHandler',
            'formatter': 'standard',
            'stream': 'ext://sys.stdout'
        },
        'file': {
            'level': 'DEBUG',
            'class': 'logging.handlers.RotatingFileHandler',
            'filename': 'logs/pair_selector.log',
            'maxBytes': 10 * 1024 * 1024,  # 10 MB
            'backupCount': 5,
            'formatter': 'standard',
            'encoding': 'utf8'
        },
    },
    'loggers': {
        'pair_selector': {
            'handlers': ['console', 'file'],
            'level': 'DEBUG',
            'propagate': True
        },
    }
}
