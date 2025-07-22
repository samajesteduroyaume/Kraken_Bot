"""
Module contenant des implémentations d'indicateurs techniques avancés.
"""
import numpy as np
import pandas as pd
from typing import Tuple, Optional, Union, List, Dict


def calculate_atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """
    Calcule l'Average True Range (ATR).
    
    Args:
        high: Série des prix hauts
        low: Série des prix bas
        close: Série des prix de clôture
        period: Période de calcul (par défaut: 14)
        
    Returns:
        Série pandas contenant les valeurs de l'ATR
    """
    # Calculer les True Ranges
    high_low = high - low
    high_close = (high - close.shift(1)).abs()
    low_close = (low - close.shift(1)).abs()
    
    # Prendre le maximum des trois valeurs
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    
    # Calculer l'ATR avec une moyenne mobile exponentielle
    atr = true_range.ewm(span=period, min_periods=period, adjust=False).mean()
    
    return atr


def calculate_supertrend(
    high: pd.Series, 
    low: pd.Series, 
    close: pd.Series, 
    period: int = 10, 
    multiplier: float = 3.0
) -> Tuple[pd.Series, pd.Series]:
    """
    Calcule l'indicateur SuperTrend.
    
    Args:
        high: Série des prix hauts
        low: Série des prix bas
        close: Série des prix de clôture
        period: Période de calcul (par défaut: 10)
        multiplier: Multiplicateur pour les bandes (par défaut: 3.0)
        
    Returns:
        Tuple contenant (SuperTrend, Tendance) où Tendance est 1 pour haussier et -1 pour baissier
    """
    # Calculer l'ATR
    atr = calculate_atr(high, low, close, period)
    
    # Calculer les bandes supérieure et inférieure
    hl2 = (high + low) / 2
    upper_band = hl2 + (multiplier * atr)
    lower_band = hl2 - (multiplier * atr)
    
    # Initialiser les tableaux pour SuperTrend et la tendance
    supertrend = pd.Series(index=close.index, dtype=float)
    trend = pd.Series(1, index=close.index, dtype=int)
    
    # Calculer le SuperTrend
    for i in range(1, len(close)):
        if close.iloc[i-1] > supertrend.iloc[i-1]:
            trend.iloc[i] = 1
        else:
            trend.iloc[i] = -1
            
        if trend.iloc[i] < 0 and trend.iloc[i-1] > 0:
            supertrend.iloc[i] = upper_band.iloc[i]
        elif trend.iloc[i] > 0 and trend.iloc[i-1] < 0:
            supertrend.iloc[i] = lower_band.iloc[i]
        else:
            if trend.iloc[i] > 0:
                supertrend.iloc[i] = min(lower_band.iloc[i], supertrend.iloc[i-1] if i > 1 else lower_band.iloc[i])
            else:
                supertrend.iloc[i] = max(upper_band.iloc[i], supertrend.iloc[i-1] if i > 1 else upper_band.iloc[i])
    
    return supertrend, trend


def calculate_ichimoku(
    high: pd.Series, 
    low: pd.Series, 
    close: pd.Series,
    tenkan_period: int = 9,
    kijun_period: int = 26,
    senkou_span_b_period: int = 52,
    chikou_shift: int = 26,
    senkou_shift: int = 26
) -> Dict[str, pd.Series]:
    """
    Calcule les composantes de l'indicateur Ichimoku Kinko Hyo.
    
    Args:
        high: Série des prix hauts
        low: Série des prix bas
        close: Série des prix de clôture
        tenkan_period: Période pour la ligne Tenkan-sen (par défaut: 9)
        kijun_period: Période pour la ligne Kijun-sen (par défaut: 26)
        senkou_span_b_period: Période pour le Senkou Span B (par défaut: 52)
        chikou_shift: Décalage pour la ligne Chikou Span (par défaut: 26)
        senkou_shift: Décalage pour les nuages Senkou (par défaut: 26)
        
    Returns:
        Dictionnaire contenant toutes les composantes d'Ichimoku
    """
    # Tenkan-sen (Conversion Line)
    tenkan_high = high.rolling(window=tenkan_period).max()
    tenkan_low = low.rolling(window=tenkan_period).min()
    tenkan_sen = (tenkan_high + tenkan_low) / 2
    
    # Kijun-sen (Base Line)
    kijun_high = high.rolling(window=kijun_period).max()
    kijun_low = low.rolling(window=kijun_period).min()
    kijun_sen = (kijun_high + kijun_low) / 2
    
    # Senkou Span A (Leading Span A)
    senkou_span_a = ((tenkan_sen + kijun_sen) / 2).shift(senkou_shift)
    
    # Senkou Span B (Leading Span B)
    senkou_high = high.rolling(window=senkou_span_b_period).max()
    senkou_low = low.rolling(window=senkou_span_b_period).min()
    senkou_span_b = ((senkou_high + senkou_low) / 2).shift(senkou_shift)
    
    # Chikou Span (Lagging Span)
    chikou_span = close.shift(-chikou_shift)
    
    return {
        'tenkan_sen': tenkan_sen,
        'kijun_sen': kijun_sen,
        'senkou_span_a': senkou_span_a,
        'senkou_span_b': senkou_span_b,
        'chikou_span': chikou_span
    }


def calculate_adx(
    high: pd.Series, 
    low: pd.Series, 
    close: pd.Series, 
    period: int = 14
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Calcule l'Average Directional Index (ADX) avec les lignes +DI et -DI.
    
    Args:
        high: Série des prix hauts
        low: Série des prix bas
        close: Série des prix de clôture
        period: Période de calcul (par défaut: 14)
        
    Returns:
        Tuple contenant (ADX, +DI, -DI)
    """
    # Calculer les mouvements directionnels
    high_diff = high.diff()
    low_diff = low.diff()
    
    plus_dm = high_diff.copy()
    minus_dm = -low_diff.copy()
    
    # Filtrer les valeurs négatives
    plus_dm[plus_dm < 0] = 0
    minus_dm[minus_dm < 0] = 0
    
    # Mettre à zéro si la condition n'est pas respectée
    cond1 = high_diff < low_diff.abs()
    cond2 = high_diff <= low_diff
    
    plus_dm[cond1 | cond2] = 0
    minus_dm[~cond1 | cond2] = 0
    
    # Calculer le True Range
    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low - close.shift(1)).abs()
    ], axis=1).max(axis=1)
    
    # Lisser les valeurs avec une moyenne mobile exponentielle
    atr = tr.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    plus_di = 100 * (plus_dm.ewm(alpha=1/period, min_periods=period, adjust=False).mean() / atr)
    minus_di = 100 * (minus_dm.ewm(alpha=1/period, min_periods=period, adjust=False).mean() / atr)
    
    # Calculer le DX et l'ADX
    dx = 100 * ((plus_di - minus_di).abs() / (plus_di + minus_di)).fillna(0)
    adx = dx.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    
    return adx, plus_di, minus_di
