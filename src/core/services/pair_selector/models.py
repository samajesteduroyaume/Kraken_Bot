"""
Modèles de données pour le module PairSelector.

Ce module définit les structures de données utilisées par le sélecteur de paires,
y compris les résultats d'analyse et les métadonnées associées.
"""
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional


@dataclass
class PairAnalysis:
    """Représente le résultat de l'analyse d'une paire de trading.
    
    Attributes:
        pair: Identifiant de la paire (ex: 'XBT/USD')
        score: Score global de la paire (0.0-1.0)
        metrics: Dictionnaire des métriques calculées (volume, volatilité, etc.)
        indicators: Dictionnaire des indicateurs techniques (RSI, ATR, etc.)
        last_updated: Horodatage ISO de la dernière mise à jour
        cached: Indique si le résultat provient du cache
        error: Message d'erreur éventuel
    """
    pair: str
    score: float = 0.0
    metrics: Dict[str, float] = field(default_factory=dict)
    indicators: Dict[str, float] = field(default_factory=dict)
    last_updated: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    cached: bool = False
    error: Optional[str] = None


@dataclass
class CacheStats:
    """Statistiques d'utilisation du cache.
    
    Attributes:
        hits: Nombre de succès de lecture dans le cache
        misses: Nombre d'échecs de lecture dans le cache
        currsize: Nombre actuel d'entrées dans le cache
        maxsize: Taille maximale du cache
        last_reset: Date de la dernière réinitialisation du cache
    """
    hits: int = 0
    misses: int = 0
    currsize: int = 0
    maxsize: int = 100
    last_reset: str = field(default_factory=lambda: datetime.utcnow().isoformat())


@dataclass
class PairAnalysisResult:
    """Résultat complet de l'analyse d'une paire.
    
    Inclut à la fois les données analysées et les métadonnées d'exécution.
    
    Attributes:
        success: Indique si l'analyse a réussi
        data: Données d'analyse de la paire
        execution_time: Temps d'exécution en secondes
        timestamp: Horodatage de l'analyse
    """
    success: bool
    data: PairAnalysis
    execution_time: float
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
