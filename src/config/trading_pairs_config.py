"""
Configuration des paires de trading recommandées pour le bot Kraken.

Ce module définit les paires de trading recommandées avec leurs paramètres spécifiques.
Les paires sont organisées par catégories basées sur leur liquidité et leur volatilité.
"""
from typing import List, Optional, Tuple, Dict, Any, Literal
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
import numpy as np
from collections import defaultdict
import logging
from .types import RiskProfile

# Configuration du logger
logger = logging.getLogger(__name__)


@dataclass
class MarketAnalysis:
    """Analyse de marché pour une paire de trading."""
    symbol: str
    current_price: float = 0.0
    volume_24h: float = 0.0
    spread_pct: float = 0.0
    trend: Literal['up', 'down', 'neutral'] = 'neutral'
    trend_strength: float = 0.0  # 0-1
    volatility: float = 0.0
    last_updated: datetime = field(
        default_factory=lambda: datetime.now(
            timezone.utc))
    price_history: List[Tuple[datetime, float]] = field(default_factory=list)


def load_trading_pairs_config() -> Dict[str, Any]:
    """Charge la configuration des paires de trading recommandées."""
    return {
        'high_liquidity': [
            'BTC/USD', 'ETH/USD', 'XRP/USD', 'LTC/USD', 'BCH/USD'
        ],
        'medium_liquidity': [
            'ADA/USD', 'DOT/USD', 'LINK/USD', 'UNI/USD', 'SOL/USD'
        ],
        'low_liquidity': [
            'DOGE/USD', 'SHIB/USD', 'TRX/USD', 'MATIC/USD', 'AVAX/USD'
        ],
        'risk_profile': RiskProfile()
    }

    def update_price(self, price: float):
        """Met à jour le prix et l'historique."""
        now = datetime.now(timezone.utc)
        self.current_price = price
        self.price_history.append((now, price))
        # Ne garder que les 100 derniers prix pour l'analyse
        self.price_history = self.price_history[-100:]
        self.last_updated = now
        self._analyze_trend()

    def _analyze_trend(self):
        """Analyse la tendance à partir de l'historique des prix."""
        if len(self.price_history) < 5:
            self.trend = 'neutral'
            self.trend_strength = 0
            return

        prices = np.array([p[1] for p in self.price_history])
        times = np.array([p[0].timestamp() for p in self.price_history])

        # Régression linéaire sur les 30 dernières minutes
        recent_cutoff = datetime.now(timezone.utc) - timedelta(minutes=30)
        recent_prices = [
            (t, p) for t, p in zip(
                times, prices) if datetime.fromtimestamp(
                t, timezone.utc) > recent_cutoff]

        if len(recent_prices) < 3:
            self.trend = 'neutral'
            self.trend_strength = 0
            return

        recent_times, recent_prices = zip(*recent_prices)
        slope, _ = np.polyfit(recent_times, recent_prices, 1)

        # Calcul de la force de la tendance (0-1)
        price_range = max(prices) - min(prices)
        if price_range > 0:
            self.trend_strength = min(abs(slope) * 10000 / price_range, 1.0)
        else:
            self.trend_strength = 0

        self.trend = 'up' if slope > 0 else 'down' if slope < 0 else 'neutral'

        # Calcul de la volatilité (écart-type des rendements)
        if len(prices) > 1:
            returns = np.diff(prices) / prices[:-1]
            # Volatilité annualisée
            self.volatility = np.std(returns) * np.sqrt(365)


@dataclass
class TradingPairConfig:
    """Configuration d'une paire de trading."""
    symbol: str           # Symbole de la paire (ex: 'XBT/EUR')
    base_currency: str    # Devise de base
    quote_currency: str   # Devise de cotation
    min_size: float       # Taille minimum d'ordre
    max_size: float       # Taille maximum d'ordre
    step_size: float      # Pas de taille d'ordre
    price_precision: int  # Précision du prix
    min_volume_btc: float  # Volume minimum 24h en BTC
    max_spread: float     # Spread maximum accepté en %
    risk_level: Literal['low', 'medium', 'high']  # Niveau de risque
    risk_profile: RiskProfile  # Profil de risque spécifique
    enabled: bool = True  # Si la paire est activée
    analysis: MarketAnalysis = field(init=False)

    def __post_init__(self):
        self.analysis = MarketAnalysis(symbol=self.symbol)
        # Gérer les symboles au format Kraken (ex: XXBTZUSD)
        if '/' in self.symbol:  # Format standard
            self.base_currency, self.quote_currency = self.symbol.split('/')
            # Convertir au format Kraken si nécessaire
            if len(self.base_currency) == 3:  # Monnaie standard
                self.base_currency = f"X{self.base_currency}"
            if len(self.quote_currency) == 3:  # Monnaie fiat
                self.quote_currency = f"Z{self.quote_currency}"
        else:  # Format Kraken
            # Vérifier la validité des préfixes
            if not self.symbol.startswith(('X', 'Z')):
                raise ValueError(
                    f"Symbole Kraken invalide: {self.symbol} - doit commencer par X ou Z")
            self.base_currency = self.symbol[:4]  # Premiers 4 caractères
            self.quote_currency = self.symbol[4:]  # Derniers 4 caractères

    def update_market_data(self, price: float, volume: float, spread: float):
        """Met à jour les données de marché de la paire."""
        self.analysis.update_price(price)
        self.analysis.volume_24h = volume
        self.analysis.spread_pct = spread

# Configuration des paires de trading recommandées


# Paires majeures (haute liquidité, faible spread)
MAJOR_PAIRS = [
    TradingPairConfig(
        symbol='XXBTZUSD',
        base_currency='XXBT',
        quote_currency='ZUSD',
        min_size=0.01,
        max_size=100.0,
        step_size=0.0001,
        price_precision=4,
        min_volume_btc=40.0,
        max_spread=0.1,
        risk_level='low',
        risk_profile={
            'max_position_size': 0.01,
            'stop_loss_percentage': 2.0,
            'take_profit_percentage': 4.0,
            'max_drawdown': 10.0
        }
    ),
    TradingPairConfig(
        symbol='XETHZUSD',
        base_currency='XETH',
        quote_currency='ZUSD',
        min_size=0.01,
        max_size=100.0,
        step_size=0.0001,
        price_precision=4,
        min_volume_btc=40.0,
        max_spread=0.1,
        risk_level='low',
        risk_profile={
            'max_position_size': 0.01,
            'stop_loss_percentage': 2.0,
            'take_profit_percentage': 4.0,
            'max_drawdown': 10.0
        }
    ),
    TradingPairConfig(
        symbol='XXBTZEUR',
        base_currency='XXBT',
        quote_currency='ZEUR',
        min_size=0.01,
        max_size=100.0,
        step_size=0.0001,
        price_precision=4,
        min_volume_btc=40.0,
        max_spread=0.1,
        risk_level='low',
        risk_profile={
            'max_position_size': 0.01,
            'stop_loss_percentage': 2.0,
            'take_profit_percentage': 4.0,
            'max_drawdown': 10.0
        }
    ),
    TradingPairConfig(
        symbol='XETHZEUR',
        base_currency='XETH',
        quote_currency='ZEUR',
        min_size=0.01,
        max_size=100.0,
        step_size=0.0001,
        price_precision=4,
        min_volume_btc=40.0,
        max_spread=0.1,
        risk_level='low',
        risk_profile={
            'max_position_size': 0.01,
            'stop_loss_percentage': 2.0,
            'take_profit_percentage': 4.0,
            'max_drawdown': 10.0
        }
    )
]

# Paires moyennes (liquidité moyenne, volatilité modérée)
MID_CAP_PAIRS = [
    TradingPairConfig(
        symbol='XDOTZUSD',
        base_currency='XDOT',
        quote_currency='ZUSD',
        min_size=0.1,
        max_size=1000.0,
        step_size=0.001,
        price_precision=4,
        min_volume_btc=10.0,
        max_spread=0.5,
        risk_level='medium',
        risk_profile={
            'max_position_size': 0.05,
            'stop_loss_percentage': 3.0,
            'take_profit_percentage': 6.0,
            'max_drawdown': 15.0
        }
    ),
    TradingPairConfig(
        symbol='XADAZUSD',
        base_currency='XADA',
        quote_currency='ZUSD',
        min_size=1.0,
        max_size=10000.0,
        step_size=0.01,
        price_precision=4,
        min_volume_btc=5.0,
        max_spread=0.5,
        risk_level='medium',
        risk_profile={
            'max_position_size': 0.05,
            'stop_loss_percentage': 3.0,
            'take_profit_percentage': 6.0,
            'max_drawdown': 15.0
        }
    )
]

# Paires à haute volatilité
HIGH_VOLATILITY_PAIRS = [
    TradingPairConfig(
        symbol='XSRMZUSD',
        base_currency='XSRM',
        quote_currency='ZUSD',
        min_size=1.0,
        max_size=1000.0,
        step_size=0.01,
        price_precision=4,
        min_volume_btc=1.0,
        max_spread=1.0,
        risk_level='high',
        risk_profile={
            'max_position_size': 0.1,
            'stop_loss_percentage': 5.0,
            'take_profit_percentage': 10.0,
            'max_drawdown': 20.0
        }
    ),
    TradingPairConfig(
        symbol='XDOGEZUSD',
        base_currency='XDOGE',
        quote_currency='ZUSD',
        min_size=100.0,
        max_size=100000.0,
        step_size=0.01,
        price_precision=4,
        min_volume_btc=0.1,
        max_spread=1.0,
        risk_level='high',
        risk_profile={
            'max_position_size': 0.1,
            'stop_loss_percentage': 5.0,
            'take_profit_percentage': 10.0,
            'max_drawdown': 20.0
        }
    )
]

# Extraire les symboles de toutes les paires
ALL_PAIR_SYMBOLS = [pair.symbol for pair in MAJOR_PAIRS] + \
    [pair.symbol for pair in MID_CAP_PAIRS] + [pair.symbol for pair in HIGH_VOLATILITY_PAIRS]

# Paires majeures (haute liquidité, faible spread)
MAJOR_PAIRS = [
    TradingPairConfig(
        symbol='XXBTZUSD',
        base_currency='XXBT',
        quote_currency='ZUSD',
        min_size=0.01,
        max_size=100.0,
        step_size=0.0001,
        price_precision=4,
        min_volume_btc=40.0,
        max_spread=0.1,
        risk_level='low',
        risk_profile={
            'max_position_size': 0.01,
            'stop_loss_percentage': 2.0,
            'take_profit_percentage': 4.0,
            'max_drawdown': 10.0
        }
    ),
    TradingPairConfig(
        symbol='XETHZUSD',
        base_currency='XETH',
        quote_currency='ZUSD',
        min_size=0.01,
        max_size=100.0,
        step_size=0.0001,
        price_precision=4,
        min_volume_btc=40.0,
        max_spread=0.1,
        risk_level='low',
        risk_profile={
            'max_position_size': 0.01,
            'stop_loss_percentage': 2.0,
            'take_profit_percentage': 4.0,
            'max_drawdown': 10.0
        }
    ),
    TradingPairConfig(
        symbol='XXBTZEUR',
        base_currency='XXBT',
        quote_currency='ZEUR',
        min_size=0.01,
        max_size=100.0,
        step_size=0.0001,
        price_precision=4,
        min_volume_btc=40.0,
        max_spread=0.1,
        risk_level='low',
        risk_profile={
            'max_position_size': 0.01,
            'stop_loss_percentage': 2.0,
            'take_profit_percentage': 4.0,
            'max_drawdown': 10.0
        }
    ),
    TradingPairConfig(
        symbol='XETHZEUR',
        base_currency='XETH',
        quote_currency='ZEUR',
        min_size=0.01,
        max_size=100.0,
        step_size=0.0001,
        price_precision=4,
        min_volume_btc=40.0,
        max_spread=0.1,
        risk_level='low',
        risk_profile={
            'max_position_size': 0.01,
            'stop_loss_percentage': 2.0,
            'take_profit_percentage': 4.0,
            'max_drawdown': 10.0
        }
    )
]

# Paires moyennes (liquidité moyenne, volatilité modérée)
MID_CAP_PAIRS = [
    TradingPairConfig(
        symbol='XADAZUSD',
        base_currency='XADA',
        quote_currency='ZUSD',
        min_size=0.01,
        max_size=100.0,
        step_size=0.0001,
        price_precision=4,
        min_volume_btc=40.0,
        max_spread=0.1,
        risk_level='medium',
        risk_profile={
            'max_position_size': 0.01,
            'stop_loss_percentage': 2.0,
            'take_profit_percentage': 4.0,
            'max_drawdown': 10.0
        }
    ),
    TradingPairConfig(
        symbol='XSOZZUSD',
        base_currency='XSOZ',
        quote_currency='ZUSD',
        min_size=0.01,
        max_size=100.0,
        step_size=0.0001,
        price_precision=4,
        min_volume_btc=40.0,
        max_spread=0.1,
        risk_level='medium',
        risk_profile={
            'max_position_size': 0.01,
            'stop_loss_percentage': 2.0,
            'take_profit_percentage': 4.0,
            'max_drawdown': 10.0
        }
    ),
    TradingPairConfig(
        symbol='XDOTZUSD',
        base_currency='XDOT',
        quote_currency='ZUSD',
        min_size=0.01,
        max_size=100.0,
        step_size=0.0001,
        price_precision=4,
        min_volume_btc=40.0,
        max_spread=0.1,
        risk_level='medium',
        risk_profile={
            'max_position_size': 0.01,
            'stop_loss_percentage': 2.0,
            'take_profit_percentage': 4.0,
            'max_drawdown': 10.0
        }
    ),
    TradingPairConfig(
        symbol='XMATICZUSD',
        base_currency='XMATIC',
        quote_currency='ZUSD',
        min_size=0.01,
        max_size=100.0,
        step_size=0.0001,
        price_precision=4,
        min_volume_btc=40.0,
        max_spread=0.1,
        risk_level='medium',
        risk_profile={
            'max_position_size': 0.01,
            'stop_loss_percentage': 2.0,
            'take_profit_percentage': 4.0,
            'max_drawdown': 10.0
        }
    )
]

# Paires à forte volatilité (potentiel de profit plus élevé mais plus risquées)
HIGH_VOLATILITY_PAIRS = [
    TradingPairConfig(
        symbol='XLINKZUSD',
        base_currency='XLINK',
        quote_currency='ZUSD',
        min_size=0.01,
        max_size=100.0,
        step_size=0.0001,
        price_precision=4,
        min_volume_btc=40.0,
        max_spread=0.1,
        risk_level='high',
        risk_profile={
            'max_position_size': 0.01,
            'stop_loss_percentage': 2.0,
            'take_profit_percentage': 4.0,
            'max_drawdown': 10.0
        }
    ),
    TradingPairConfig(
        symbol='XUNIZUSD',
        base_currency='XUNI',
        quote_currency='ZUSD',
        min_size=0.01,
        max_size=100.0,
        step_size=0.0001,
        price_precision=4,
        min_volume_btc=40.0,
        max_spread=0.1,
        risk_level='high',
        risk_profile={
            'max_position_size': 0.01,
            'stop_loss_percentage': 2.0,
            'take_profit_percentage': 4.0,
            'max_drawdown': 10.0
        }
    )
]

# Mettre à jour ALL_PAIR_SYMBOLS avec les symboles des paires définies
ALL_PAIR_SYMBOLS = [
    pair.symbol for pair in MAJOR_PAIRS +
    MID_CAP_PAIRS +
    HIGH_VOLATILITY_PAIRS]

# Paires moyennes (liquidité moyenne, volatilité modérée)
MID_CAP_PAIRS = [
    TradingPairConfig(
        symbol='XADAZUSD',
        base_currency='XADA',
        quote_currency='ZUSD',
        min_size=0.01,
        max_size=100.0,
        step_size=0.0001,
        price_precision=4,
        min_volume_btc=40.0,
        max_spread=0.1,
        risk_level='medium',
        risk_profile={
            'max_position_size': 0.01,
            'stop_loss_percentage': 2.0,
            'take_profit_percentage': 4.0,
            'max_drawdown': 10.0
        }
    ),
    TradingPairConfig(
        symbol='XSOZZUSD',
        base_currency='XSOZ',
        quote_currency='ZUSD',
        min_size=0.01,
        max_size=100.0,
        step_size=0.0001,
        price_precision=4,
        min_volume_btc=40.0,
        max_spread=0.1,
        risk_level='medium',
        risk_profile={
            'max_position_size': 0.01,
            'stop_loss_percentage': 2.0,
            'take_profit_percentage': 4.0,
            'max_drawdown': 10.0
        }
    ),
    TradingPairConfig(
        symbol='XDOTZUSD',
        base_currency='XDOT',
        quote_currency='ZUSD',
        min_size=0.01,
        max_size=100.0,
        step_size=0.0001,
        price_precision=4,
        min_volume_btc=40.0,
        max_spread=0.1,
        risk_level='medium',
        risk_profile={
            'max_position_size': 0.01,
            'stop_loss_percentage': 2.0,
            'take_profit_percentage': 4.0,
            'max_drawdown': 10.0
        }
    ),
    TradingPairConfig(
        symbol='XMATICZUSD',
        base_currency='XMATIC',
        quote_currency='ZUSD',
        min_size=0.01,
        max_size=100.0,
        step_size=0.0001,
        price_precision=4,
        min_volume_btc=40.0,
        max_spread=0.1,
        risk_level='medium',
        risk_profile={
            'max_position_size': 0.01,
            'stop_loss_percentage': 2.0,
            'take_profit_percentage': 4.0,
            'max_drawdown': 10.0
        }
    )
]

# Paires à forte volatilité (potentiel de profit plus élevé mais plus risquées)
HIGH_VOLATILITY_PAIRS = [
    TradingPairConfig(
        symbol='XLINKZUSD',
        base_currency='XLINK',
        quote_currency='ZUSD',
        min_size=0.01,
        max_size=100.0,
        step_size=0.0001,
        price_precision=4,
        min_volume_btc=40.0,
        max_spread=0.1,
        risk_level='high',
        risk_profile={
            'max_position_size': 0.01,
            'stop_loss_percentage': 2.0,
            'take_profit_percentage': 4.0,
            'max_drawdown': 10.0
        }
    ),
    TradingPairConfig(
        symbol='XUNIZUSD',
        base_currency='XUNI',
        quote_currency='ZUSD',
        min_size=0.01,
        max_size=100.0,
        step_size=0.0001,
        price_precision=4,
        min_volume_btc=40.0,
        max_spread=0.1,
        risk_level='high',
        risk_profile={
            'max_position_size': 0.01,
            'stop_loss_percentage': 2.0,
            'take_profit_percentage': 4.0,
            'max_drawdown': 10.0
        }
    ),
    TradingPairConfig(
        symbol='XAVAXZUSD',
        base_currency='AVAX',
        quote_currency='USD',
        min_size=0.01,
        max_size=100.0,
        step_size=0.0001,
        price_precision=4,
        min_volume_btc=40.0,
        max_spread=0.1,
        risk_level='high',
        risk_profile={
            'max_position_size': 0.01,
            'stop_loss_percentage': 2.0,
            'take_profit_percentage': 4.0,
            'max_drawdown': 10.0
        }
    ),
    TradingPairConfig(
        symbol='XLUNAZUSD',
        base_currency='LUNA',
        quote_currency='USD',
        min_size=0.01,
        max_size=100.0,
        step_size=0.0001,
        price_precision=4,
        min_volume_btc=40.0,
        max_spread=0.1,
        risk_level='high',
        risk_profile={
            'max_position_size': 0.01,
            'stop_loss_percentage': 2.0,
            'take_profit_percentage': 4.0,
            'max_drawdown': 10.0
        }
    )
]


class CorrelationAnalyzer:
    """Analyse les corrélations entre les paires de trading."""

    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.price_data: Dict[str,
                              List[Tuple[datetime, float]]] = defaultdict(list)

    def add_price(self, symbol: str, price: float, timestamp: datetime = None):
        """Ajoute un prix pour une paire donnée."""
        if timestamp is None:
            timestamp = datetime.now(timezone.utc)

        self.price_data[symbol].append((timestamp, price))

        # Garder uniquement les données récentes
        cutoff = timestamp - timedelta(days=7)  # 7 jours de données
        self.price_data[symbol] = [
            p for p in self.price_data[symbol] if p[0] > cutoff]

    def get_correlation(self, symbol1: str, symbol2: str) -> float:
        """Calcule la corrélation entre deux paires."""
        if symbol1 == symbol2:
            return 1.0

        prices1 = self._get_normalized_prices(symbol1)
        prices2 = self._get_normalized_prices(symbol2)

        if len(prices1) < 10 or len(prices2) < 10:
            return 0.0

        # Aligner les séries temporelles
        common_dates = set(p[0] for p in prices1) & set(p[0] for p in prices2)
        if not common_dates:
            return 0.0

        p1 = np.array([p[1] for p in prices1 if p[0] in common_dates])
        p2 = np.array([p[1] for p in prices2 if p[0] in common_dates])

        if len(p1) < 2 or len(p2) < 2:
            return 0.0

        # Calculer la corrélation de Pearson
        return float(np.corrcoef(p1, p2)[0, 1])

    def _get_normalized_prices(
            self, symbol: str) -> List[Tuple[datetime, float]]:
        """Retourne les prix normalisés pour une paire."""
        if symbol not in self.price_data or not self.price_data[symbol]:
            return []

        prices = sorted(self.price_data[symbol], key=lambda x: x[0])
        if not prices:
            return []

        # Normalisation des prix (premier prix = 1.0)
        base_price = prices[0][1]
        if base_price == 0:
            return []

        return [(t, p / base_price) for t, p in prices]


def get_recommended_pairs(
        risk_level: Optional[str] = None) -> List[TradingPairConfig]:
    """
    Retourne les paires recommandées en fonction du niveau de risque.

    Args:
        risk_level: Niveau de risque ('low', 'medium', 'high' ou None pour toutes)

    Returns:
        Liste des configurations de paires filtrées
    """
    all_pairs = MAJOR_PAIRS + MID_CAP_PAIRS + HIGH_VOLATILITY_PAIRS

    if risk_level is None:
        return [p for p in all_pairs if p.enabled]

    risk_level = risk_level.lower()
    if risk_level not in ['low', 'medium', 'high']:
        raise ValueError(
            "Le niveau de risque doit être 'low', 'medium' ou 'high'")

    return [p for p in all_pairs if p.risk_level == risk_level and p.enabled]


def filter_correlated_pairs(
        pairs: List[TradingPairConfig],
        correlation_threshold: float = 0.8,
        analyzer: Optional[CorrelationAnalyzer] = None,
        min_volume_btc: float = 1.0,
        max_spread: float = 1.0) -> List[TradingPairConfig]:
    """
    Filtre les paires corrélées en gardant les plus liquides.

    Args:
        pairs: Liste des paires à filtrer
        correlation_threshold: Seuil de corrélation au-delà duquel on considère que les paires sont corrélées
        analyzer: Analyseur de corrélation (si None, un nouveau sera créé)
        min_volume_btc: Volume minimum en BTC pour considérer une paire
        max_spread: Spread maximum en % pour considérer une paire

    Returns:
        Liste des paires filtrées
    """
    if analyzer is None:
        analyzer = CorrelationAnalyzer()

    # Filtrer les paires selon les critères de base
    filtered_pairs = [
        p for p in pairs
        if (p.analysis.volume_24h >= min_volume_btc and
            p.analysis.spread_pct <= max_spread and
            p.analysis.volatility > 0)  # Éviter les paires sans volatilité
    ]

    if not filtered_pairs:
        logger.warning("Aucune paire ne correspond aux critères de filtrage")
        return []

    # Trier les paires par score de diversification (volume / volatilité)
    def diversification_score(pair: TradingPairConfig) -> float:
        if pair.analysis.volatility == 0:
            return 0
        return (pair.analysis.volume_24h * 0.6 + (1 /
                (pair.analysis.spread_pct + 0.01)) * 0.4) / pair.analysis.volatility

    sorted_pairs = sorted(
        filtered_pairs,
        key=diversification_score,
        reverse=True
    )

    selected = []
    correlation_matrix: Dict[Tuple[str, str], float] = {}

    for i, pair in enumerate(sorted_pairs):
        if len(selected) >= 10:  # Limiter à 10 paires maximum
            break

        # Vérifier la corrélation avec les paires déjà sélectionnées
        is_highly_correlated = False

        for selected_pair in selected:
            # Utiliser le cache de la matrice de corrélation
            pair_key = tuple(sorted([pair.symbol, selected_pair.symbol]))

            if pair_key not in correlation_matrix:
                correlation_matrix[pair_key] = analyzer.get_correlation(
                    pair.symbol, selected_pair.symbol
                )

            correlation = correlation_matrix[pair_key]

            if abs(correlation) > correlation_threshold:
                logger.debug(
                    f"Paire {pair.symbol} fortement corrélée "
                    f"({correlation:.2f}) avec {selected_pair.symbol}"
                )
                is_highly_correlated = True
                break

        # Si la paire n'est pas trop corrélée, l'ajouter à la sélection
        if not is_highly_correlated:
            selected.append(pair)
            logger.info(
                f"Paire sélectionnée: {pair.symbol} | "
                f"Vol: {pair.analysis.volume_24h:.2f} BTC | "
                f"Spread: {pair.analysis.spread_pct:.2f}% | "
                f"Volatilité: {pair.analysis.volatility*100:.2f}%"
            )

    # Si aucune paire n'est sélectionnée, retourner les meilleures paires non
    # corrélées
    if not selected and sorted_pairs:
        selected = [sorted_pairs[0]]
        for pair in sorted_pairs[1:]:
            if len(selected) >= 5:  # Maximum 5 paires si forte corrélation
                break
            selected.append(pair)

    return selected


def get_pair_config(symbol: str) -> Optional[TradingPairConfig]:
    """
    Récupère la configuration d'une paire spécifique avec mise en cache.

    Args:
        symbol: Symbole de la paire (ex: 'XBT/USD')

    Returns:
        La configuration de la paire ou None si non trouvée
    """
    # Mise en cache des résultats pour améliorer les performances
    if not hasattr(get_pair_config, '_cache'):
        get_pair_config._cache: Dict[str, TradingPairConfig] = {}

    # Vérifier le cache d'abord
    cache_key = symbol.upper()
    if cache_key in get_pair_config._cache:
        return get_pair_config._cache[cache_key]

    # Si pas dans le cache, chercher dans toutes les paires
    all_pairs = MAJOR_PAIRS + MID_CAP_PAIRS + HIGH_VOLATILITY_PAIRS
    for pair in all_pairs:
        if pair.symbol.upper() == cache_key:
            # Mettre en cache le résultat pour les appels futurs
            get_pair_config._cache[cache_key] = pair
            return pair

    # Si la paire n'est pas trouvée, essayer de la créer dynamiquement
    # avec des paramètres par défaut basés sur la devise de cotation
    try:
        base, quote = symbol.upper().split('/')
        is_major = quote in ['USD', 'EUR', 'GBP', 'JPY', 'USDT', 'USDC']

        if is_major:
            # Pour les paires majeures, utiliser des paramètres stricts
            pair = TradingPairConfig(
                symbol=symbol.upper(),
                min_volume_btc=50.0,
                max_spread=0.2,
                risk_level='low' if base in ['XBT', 'ETH', 'XRP'] else 'medium'
            )
        else:
            # Pour les paires mineures, être plus permissif
            pair = TradingPairConfig(
                symbol=symbol.upper(),
                min_volume_btc=5.0,
                max_spread=0.5,
                risk_level='high'
            )

        # Mettre en cache la nouvelle paire
        get_pair_config._cache[cache_key] = pair
        logger.warning(
            f"Paire {symbol} non trouvée, configuration par défaut appliquée")
        return pair

    except (ValueError, AttributeError) as e:
        logger.error(f"Format de paire invalide: {symbol} - {str(e)}")
        return None
