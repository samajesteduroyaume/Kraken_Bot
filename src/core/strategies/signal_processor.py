"""
Module de traitement des signaux de trading.

Ce module fournit des fonctionnalités pour combiner, filtrer et prioriser
les signaux de trading provenant de différentes stratégies.
"""
from typing import List, Dict, Optional, Tuple
import numpy as np
from dataclasses import dataclass, field
from enum import Enum, auto
import logging
from datetime import datetime, timedelta

from .types import TradingSignal, SignalAction


class SignalCombinationMethod(Enum):
    """Méthodes de combinaison des signaux."""
    WEIGHTED_AVERAGE = auto()  # Moyenne pondérée des signaux
    VOTE_MAJORITY = auto()    # Vote majoritaire
    PRIORITY_QUEUE = auto()   # File de priorité basée sur la confiance
    THRESHOLD = auto()        # Seuil de confiance


@dataclass
class SignalAggregationConfig:
    """Configuration pour l'agrégation des signaux."""
    method: SignalCombinationMethod = SignalCombinationMethod.WEIGHTED_AVERAGE
    confidence_threshold: float = 0.5
    min_votes: int = 1
    weights: Dict[str, float] = field(default_factory=dict)
    cooldown_period: int = 0  # en secondes


class SignalProcessor:
    """
    Traite et combine les signaux de trading provenant de différentes stratégies.
    """
    
    def __init__(self, config: Optional[SignalAggregationConfig] = None):
        """
        Initialise le processeur de signaux.
        
        Args:
            config: Configuration pour l'agrégation des signaux
        """
        self.config = config or SignalAggregationConfig()
        self.logger = logging.getLogger("signal_processor")
        self._last_signal_time: Dict[Tuple[str, str], datetime] = {}
    
    def process_signals(
        self, 
        signals: List[TradingSignal],
        symbol: str
    ) -> Optional[TradingSignal]:
        """
        Traite une liste de signaux et retourne un signal consolidé.
        
        Args:
            signals: Liste des signaux à traiter
            symbol: Symbole de trading concerné
            
        Returns:
            Un signal consolidé ou None si aucun signal valide
        """
        if not signals:
            return None
        
        # Filtrer les signaux par symbole
        symbol_signals = [s for s in signals if s.symbol == symbol]
        
        # Vérifier le cooldown
        current_time = datetime.utcnow()
        key = (symbol, str(signals[0].action))
        
        if key in self._last_signal_time:
            time_since_last = current_time - self._last_signal_time[key]
            if time_since_last < timedelta(seconds=self.config.cooldown_period):
                self.logger.debug(
                    f"Signal ignoré - en période de cooldown: "
                    f"{time_since_last.total_seconds():.0f}s restantes"
                )
                return None
        
        # Appliquer la méthode de combinaison sélectionnée
        combined_signal = self._combine_signals(symbol_signals)
        
        # Mettre à jour le timestamp du dernier signal
        if combined_signal is not None:
            self._last_signal_time[key] = current_time
            
        return combined_signal
    
    def _combine_signals(
        self, 
        signals: List[TradingSignal]
    ) -> Optional[TradingSignal]:
        """
        Combine plusieurs signaux en un seul en fonction de la méthode configurée.
        
        Args:
            signals: Liste des signaux à combiner
            
        Returns:
            Un signal consolidé ou None si aucun signal valide
        """
        if not signals:
            return None
        
        # Si un seul signal, le retourner directement
        if len(signals) == 1:
            return signals[0]
        
        # Grouper les signaux par action
        signals_by_action = {}
        for signal in signals:
            if signal.action not in signals_by_action:
                signals_by_action[signal.action] = []
            signals_by_action[signal.action].append(signal)
        
        # Si une seule action, utiliser la moyenne pondérée
        if len(signals_by_action) == 1:
            action = list(signals_by_action.keys())[0]
            return self._combine_weighted_average(signals_by_action[action])
        
        # Sinon, appliquer la méthode de combinaison configurée
        if self.config.method == SignalCombinationMethod.WEIGHTED_AVERAGE:
            return self._combine_weighted_average(signals)
        elif self.config.method == SignalCombinationMethod.VOTE_MAJORITY:
            return self._combine_majority_vote(signals)
        elif self.config.method == SignalCombinationMethod.PRIORITY_QUEUE:
            return self._combine_priority_queue(signals)
        elif self.config.method == SignalCombinationMethod.THRESHOLD:
            return self._combine_threshold(signals)
        else:
            self.logger.warning(
                f"Méthode de combinaison inconnue: {self.config.method}. "
                "Utilisation de la moyenne pondérée par défaut."
            )
            return self._combine_weighted_average(signals)
    
    def _combine_weighted_average(
        self, 
        signals: List[TradingSignal]
    ) -> Optional[TradingSignal]:
        """
        Combine les signaux en utilisant une moyenne pondérée.
        
        Args:
            signals: Liste des signaux à combiner
            
        Returns:
            Un signal consolidé ou None si aucun signal valide
        """
        if not signals:
            return None
            
        # Si un seul signal, le retourner directement
        if len(signals) == 1:
            return signals[0]
        
        # Utiliser les poids personnalisés si disponibles, sinon utiliser la confiance
        weights = []
        for signal in signals:
            weight = self.config.weights.get(signal.strategy, 1.0)
            weights.append(weight * signal.confidence)
        
        # Normaliser les poids
        total_weight = sum(weights)
        if total_weight <= 0:
            return None
            
        weights = [w / total_weight for w in weights]
        
        # Calculer la moyenne pondérée des prix et la confiance globale
        avg_price = sum(s.price * w for s, w in zip(signals, weights))
        avg_confidence = sum(s.confidence * w for s, w in zip(signals, weights))
        
        # Créer un nouveau signal consolidé
        combined = signals[0].copy()
        combined.price = avg_price
        combined.confidence = avg_confidence
        combined.metadata['combined_from'] = [s.strategy for s in signals]
        
        return combined
    
    def _combine_majority_vote(
        self, 
        signals: List[TradingSignal]
    ) -> Optional[TradingSignal]:
        """
        Combine les signaux en utilisant un vote majoritaire.
        
        Args:
            signals: Liste des signaux à combiner
            
        Returns:
            Un signal consolidé ou None si aucun signal valide
        """
        if not signals:
            return None
            
        # Compter les votes par action
        votes = {}
        for signal in signals:
            if signal.action not in votes:
                votes[signal.action] = {
                    'count': 0,
                    'total_confidence': 0.0,
                    'signals': []
                }
            votes[signal.action]['count'] += 1
            votes[signal.action]['total_confidence'] += signal.confidence
            votes[signal.action]['signals'].append(signal)
        
        # Trier par nombre de votes (puis par confiance totale en cas d'égalité)
        sorted_votes = sorted(
            votes.items(),
            key=lambda x: (x[1]['count'], x[1]['total_confidence']),
            reverse=True
        )
        
        # Vérifier si nous avons un gagnant clair
        if len(sorted_votes) == 1 or sorted_votes[0][1]['count'] > sorted_votes[1][1]['count']:
            best_action, best_data = sorted_votes[0]
            
            # Vérifier le seuil minimum de votes
            if best_data['count'] >= self.config.min_votes:
                # Retourner le signal avec la plus haute confiance parmi les gagnants
                best_signal = max(
                    best_data['signals'], 
                    key=lambda x: x.confidence
                )
                best_signal.metadata['vote_count'] = best_data['count']
                best_signal.metadata['total_confidence'] = best_data['total_confidence']
                return best_signal
        
        # Aucun gagnant clair ou pas assez de votes
        return None
    
    def _combine_priority_queue(
        self, 
        signals: List[TradingSignal]
    ) -> Optional[TradingSignal]:
        """
        Combine les signaux en utilisant une file de priorité basée sur la confiance.
        
        Args:
            signals: Liste des signaux à combiner
            
        Returns:
            Le signal avec la plus haute priorité ou None si aucun signal valide
        """
        if not signals:
            return None
            
        # Trier les signaux par confiance décroissante
        sorted_signals = sorted(
            signals, 
            key=lambda x: (x.confidence, self.config.weights.get(x.strategy, 1.0)),
            reverse=True
        )
        
        # Retourner le signal avec la plus haute confiance
        best_signal = sorted_signals[0]
        
        # Vérifier le seuil de confiance minimum
        if best_signal.confidence >= self.config.confidence_threshold:
            best_signal.metadata['priority_rank'] = 1
            best_signal.metadata['total_signals'] = len(signals)
            return best_signal
            
        return None
    
    def _combine_threshold(
        self, 
        signals: List[TradingSignal]
    ) -> Optional[TradingSignal]:
        """
        Combine les signaux en utilisant un seuil de confiance.
        
        Args:
            signals: Liste des signaux à combiner
            
        Returns:
            Un signal consolidé ou None si aucun signal valide
        """
        if not signals:
            return None
            
        # Filtrer les signaux en dessous du seuil de confiance
        filtered_signals = [
            s for s in signals 
            if s.confidence >= self.config.confidence_threshold
        ]
        
        if not filtered_signals:
            return None
            
        # Si un seul signal après filtrage, le retourner
        if len(filtered_signals) == 1:
            return filtered_signals[0]
            
        # Sinon, utiliser la moyenne pondérée des signaux restants
        return self._combine_weighted_average(filtered_signals)
