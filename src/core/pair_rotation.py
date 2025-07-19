from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from src.core.logging import LoggerManager
from src.core.services.pair_selector import PairSelector


class PairRotationStrategy:
    """Stratégie de rotation des paires de trading."""

    def __init__(
        self,
        pair_selector: PairSelector,
        rotation_frequency: timedelta = timedelta(hours=4),
        performance_threshold: float = 0.01,
        volatility_threshold: float = 0.05,
        logger: Optional[LoggerManager] = None
    ):
        self.pair_selector = pair_selector
        self.rotation_frequency = rotation_frequency
        self.performance_threshold = performance_threshold
        self.volatility_threshold = volatility_threshold
        self.last_rotation = datetime.now()
        self.logger = logger or LoggerManager()
        self.logger = self.logger.get_logger()

    def should_rotate(self, pair_metrics: Dict[str, Dict]) -> bool:
        """Décide si une rotation est nécessaire."""
        current_time = datetime.now()
        time_since_last = current_time - self.last_rotation

        if time_since_last < self.rotation_frequency:
            self.logger.debug(
                f"Pas encore temps de rotater (attendu: {self.rotation_frequency}, passé: {time_since_last})")
            return False

        # Analyser les performances
        underperforming = [
            pair for pair, metrics in pair_metrics.items()
            if metrics.get('total_profit', 0) < self.performance_threshold
        ]

        # Analyser la volatilité
        high_volatility = [
            pair for pair, metrics in pair_metrics.items()
            if metrics.get('volatility', 0) > self.volatility_threshold
        ]

        if underperforming:
            self.logger.info(f"Paires sous-performantes: {underperforming}")
        if high_volatility:
            self.logger.info(f"Paires à haute volatilité: {high_volatility}")

        should_rotate = len(underperforming) > 0 or len(high_volatility) > 0
        self.logger.info(f"Rotation nécessaire: {should_rotate}")
        return should_rotate

    async def rotate_pairs(self, current_pairs: List[str]) -> List[str]:
        """
        Effectue la rotation des paires en fonction des performances et de la volatilité.
        
        Args:
            current_pairs: Liste des paires actuellement en cours de trading
            
        Returns:
            Liste des nouvelles paires sélectionnées pour le trading
        """
        try:
            self.logger.info("Démarrage de la rotation des paires...")
            
            # Récupérer les paires valides (cela charge et analyse automatiquement les paires)
            new_pairs = await self.pair_selector.get_valid_pairs(min_score=0.5)
            
            if not new_pairs:
                self.logger.warning("Aucune nouvelle paire valide trouvée, conservation des paires actuelles")
                return current_pairs
                
            # Comparer avec les paires actuelles
            current_set = set(current_pairs)
            new_set = set(new_pairs)
            removed_pairs = current_set - new_set
            added_pairs = new_set - current_set

            # Journalisation des changements
            if removed_pairs:
                self.logger.info(f"Paires retirées: {', '.join(removed_pairs)}")
            if added_pairs:
                self.logger.info(f"Nouvelles paires ajoutées: {', '.join(added_pairs)}")
            if not removed_pairs and not added_pairs:
                self.logger.info("Aucun changement dans la sélection des paires")

            self.last_rotation = datetime.now()
            self.logger.info(f"Rotation terminée. {len(new_pairs)} paires sélectionnées")
            return new_pairs

        except Exception as e:
            self.logger.error(
                f"Erreur lors de la rotation des paires: {str(e)}")
            return current_pairs  # En cas d'erreur, garder les paires actuelles

    def get_rotation_metrics(
            self, pair_metrics: Dict[str, Dict]) -> Dict[str, Any]:
        """Retourne les métriques de rotation."""
        return {
            'last_rotation': self.last_rotation.isoformat(),
            'rotation_frequency': str(self.rotation_frequency),
            'performance_threshold': self.performance_threshold,
            'volatility_threshold': self.volatility_threshold,
            'underperforming_pairs': [
                pair for pair, metrics in pair_metrics.items()
                if metrics.get('total_profit', 0) < self.performance_threshold
            ],
            'high_volatility_pairs': [
                pair for pair, metrics in pair_metrics.items()
                if metrics.get('volatility', 0) > self.volatility_threshold
            ]
        }
