"""
Module helpers - Contient des fonctions utilitaires générales.
"""
import uuid
from typing import Any, Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


def generate_unique_id() -> str:
    """Génère un identifiant unique."""
    return str(uuid.uuid4())


def safe_get(d: Dict[Any, Any], key: Any,
             default: Optional[Any] = None) -> Any:
    """Récupère une valeur d'un dictionnaire de manière sécurisée."""
    try:
        return d[key]
    except KeyError:
        logger.warning(f"Clé '{key}' non trouvée dans le dictionnaire")
        return default


def chunk_list(lst: List[Any], chunk_size: int) -> List[List[Any]]:
    """Divise une liste en sous-listes de taille maximale chunk_size."""
    return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]


def calculate_position_size(
        balance: float,
        risk_per_trade: float,
        stop_loss: float) -> float:
    """
    Calcule la taille de position en fonction du risque et du stop loss.
    """
    try:
        position_size = (balance * risk_per_trade) / stop_loss
        return position_size
    except Exception as e:
        logger.error(
            f"Erreur lors du calcul de la taille de position: {str(e)}")
        return 0.0


def calculate_stop_loss(entry_price: float, stop_loss_percent: float) -> float:
    """
    Calcule le niveau du stop loss.
    """
    try:
        stop_loss = entry_price * (1 - stop_loss_percent)
        return stop_loss
    except Exception as e:
        logger.error(f"Erreur lors du calcul du stop loss: {str(e)}")
        return 0.0


def calculate_take_profit(
        entry_price: float,
        take_profit_percent: float) -> float:
    """
    Calcule le niveau du take profit.
    """
    try:
        take_profit = entry_price * (1 + take_profit_percent)
        return take_profit
    except Exception as e:
        logger.error(f"Erreur lors du calcul du take profit: {str(e)}")
        return 0.0


def calculate_pnl(
        entry_price: float,
        exit_price: float,
        size: float,
        leverage: float = 1.0) -> float:
    """
    Calcule le profit ou la perte.
    """
    try:
        pnl = (exit_price - entry_price) * size * leverage
        return pnl
    except Exception as e:
        logger.error(f"Erreur lors du calcul du P&L: {str(e)}")
        return 0.0


def calculate_drawdown(balance_history: list) -> float:
    """
    Calcule le drawdown maximum.
    """
    try:
        if not balance_history:
            return 0.0
        high_watermark = 0.0
        drawdowns = []
        for balance in balance_history:
            high_watermark = max(high_watermark, balance)
            drawdown = (high_watermark - balance) / high_watermark
            drawdowns.append(drawdown)
        return max(drawdowns)
    except Exception as e:
        logger.error(f"Erreur lors du calcul du drawdown: {str(e)}")
        return 0.0


def calculate_sharpe_ratio(
        returns: list,
        risk_free_rate: float = 0.02) -> float:
    """
    Calcule le ratio Sharpe.
    """
    import numpy as np
    try:
        if not returns:
            return 0.0
        returns_np = np.array(returns)
        excess_returns = returns_np - risk_free_rate / 252  # Annualisé
        if np.std(excess_returns) == 0:
            return 0.0
        sharpe_ratio = np.mean(excess_returns) / np.std(excess_returns)
        return float(sharpe_ratio)
    except Exception as e:
        logger.error(f"Erreur lors du calcul du ratio Sharpe: {str(e)}")
        return 0.0


def calculate_sortino_ratio(returns: list,
                            risk_free_rate: float = 0.02) -> float:
    """
    Calcule le ratio Sortino.
    """
    import numpy as np
    try:
        if not returns:
            return 0.0
        returns_np = np.array(returns)
        excess_returns = returns_np - risk_free_rate / 252  # Annualisé
        downside_returns = excess_returns[excess_returns < 0]
        downside_std = np.std(downside_returns)
        if downside_std == 0:
            return 0.0
        sortino_ratio = np.mean(excess_returns) / downside_std
        return float(sortino_ratio)
    except Exception as e:
        logger.error(f"Erreur lors du calcul du ratio Sortino: {str(e)}")
        return 0.0


def calculate_max_drawdown(returns: list) -> float:
    """
    Calcule le drawdown maximum à partir des retours.
    """
    import numpy as np
    try:
        if not returns:
            return 0.0
        returns_np = np.array(returns)
        cumulative_returns = np.cumprod(1 + returns_np)
        running_max = np.maximum.accumulate(cumulative_returns)
        drawdowns = (running_max - cumulative_returns) / running_max
        return float(np.max(drawdowns))
    except Exception as e:
        logger.error(f"Erreur lors du calcul du drawdown maximum: {str(e)}")
        return 0.0


def calculate_win_rate(trades: list) -> float:
    """
    Calcule le taux de trades gagnants.
    """
    try:
        if not trades:
            return 0.0
        winning_trades = [t for t in trades if t['pnl'] > 0]
        win_rate = len(winning_trades) / len(trades)
        return win_rate
    except Exception as e:
        logger.error(
            f"Erreur lors du calcul du taux de trades gagnants: {str(e)}")
        return 0.0


def calculate_profit_factor(trades: list) -> float:
    """
    Calcule le facteur de profit.
    """
    try:
        if not trades:
            return 0.0

        gross_profit = sum(trade['profit'] for trade in trades if trade['profit'] > 0)
        gross_loss = abs(sum(trade['profit'] for trade in trades if trade['profit'] < 0))

        if gross_loss == 0:
            return float('inf')

        return gross_profit / gross_loss
    except Exception as e:
        logger.error(f"Erreur lors du calcul du facteur de profit: {str(e)}")
        return 0.0


def calculate_support_resistance(df, window: int = 20, threshold: float = 0.05):
    """
    Calcule les niveaux de support et de résistance à partir des données OHLC.
    
    Args:
        df: DataFrame contenant les colonnes 'high', 'low', 'close'
        window: Taille de la fenêtre pour la détection des points pivots
        threshold: Seuil de pourcentage pour considérer un niveau comme significatif
        
    Returns:
        Tuple de (supports, resistances) où chaque élément est une liste de niveaux
    """
    import pandas as pd
    import numpy as np
    
    # Calcul des points pivots locaux
    df = df.copy()
    df['pivot_high'] = df['high'].rolling(window=window, center=True).apply(
        lambda x: x.iloc[window//2] if x.iloc[window//2] == x.max() else np.nan, raw=False
    )
    
    df['pivot_low'] = df['low'].rolling(window=window, center=True).apply(
        lambda x: x.iloc[window//2] if x.iloc[window//2] == x.min() else np.nan, raw=False
    )
    
    # Extraction des niveaux de support et résistance
    resistance_levels = df['pivot_high'].dropna().unique()
    support_levels = df['pivot_low'].dropna().unique()
    
    # Filtrage des niveaux trop proches
    def filter_levels(levels, threshold=threshold):
        if len(levels) == 0:
            return levels
            
        # Trier les niveaux
        levels = np.sort(levels)
        
        # Regrouper les niveaux proches
        filtered = []
        i = 0
        
        while i < len(levels):
            current = levels[i]
            group = [current]
            
            # Trouver tous les niveaux proches
            j = i + 1
            while j < len(levels) and (levels[j] - current) / current <= threshold:
                group.append(levels[j])
                j += 1
                
            # Prendre la moyenne du groupe
            filtered.append(np.mean(group))
            i = j
            
        return np.array(filtered)
    
    # Filtrer et retourner les niveaux
    return filter_levels(support_levels), filter_levels(resistance_levels)
