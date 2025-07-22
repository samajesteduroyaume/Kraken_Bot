from typing import Dict, List, Optional, Any, Set, Union
from datetime import datetime, timedelta
from src.core.logging import LoggerManager
from src.core.services.pair_selector.core import PairSelector
from src.utils.pair_utils import normalize_pair_input


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
        """
        Décide si une rotation est nécessaire en fonction des métriques des paires.
        
        Args:
            pair_metrics: Dictionnaire des métriques par paire (clé: nom de paire normalisé)
            
        Returns:
            bool: True si une rotation est nécessaire, False sinon
        """
        current_time = datetime.now()
        time_since_last = current_time - self.last_rotation

        if time_since_last < self.rotation_frequency:
            self.logger.debug(
                f"Pas encore temps de rotater (attendu: {self.rotation_frequency}, passé: {time_since_last})")
            return False

        # Normaliser les clés du dictionnaire de métriques
        normalized_metrics = {}
        for pair, metrics in pair_metrics.items():
            try:
                # Utiliser raise_on_error=False pour gérer les erreurs nous-mêmes
                normalized_pair = normalize_pair_input(pair, raise_on_error=False)
                
                if not normalized_pair:
                    # Si la paire n'est pas reconnue, essayer d'obtenir des suggestions
                    try:
                        normalize_pair_input(pair, raise_on_error=True)
                    except UnsupportedTradingPairError as e:
                        if e.alternatives:
                            self.logger.warning(
                                f"Paire non reconnue: {pair}. "
                                f"Suggestions: {', '.join(e.alternatives[:3])}"
                            )
                        else:
                            self.logger.warning(f"Paire non reconnue: {pair}. Aucune suggestion disponible.")
                    continue
                    
                normalized_metrics[normalized_pair] = metrics
                
            except Exception as e:
                self.logger.warning(f"Erreur lors de la normalisation de la paire {pair}: {e}")
                continue

        # Analyser les performances
        underperforming = [
            pair for pair, metrics in normalized_metrics.items()
            if metrics.get('total_profit', 0) < self.performance_threshold
        ]

        # Analyser la volatilité
        high_volatility = [
            pair for pair, metrics in normalized_metrics.items()
            if metrics.get('volatility', 0) > self.volatility_threshold
        ]

        if underperforming:
            self.logger.info(f"Paires sous-performantes: {underperforming}")
        if high_volatility:
            self.logger.info(f"Paires à haute volatilité: {high_volatility}")

        should_rotate = len(underperforming) > 0 or len(high_volatility) > 0
        self.logger.info(f"Rotation nécessaire: {should_rotate}")
        return should_rotate

    async def rotate_pairs(self, current_pairs: List[Union[str, Dict[str, Any]]]) -> List[str]:
        """
        Effectue la rotation des paires en fonction des performances et de la volatilité.
        
        Args:
            current_pairs: Liste des paires actuellement en cours de trading (peut être des chaînes ou des dicts)
            
        Returns:
            Liste des nouvelles paires sélectionnées pour le trading (noms normalisés)
        """
        try:
            self.logger.info("Démarrage de la rotation des paires...")
            
            # Normaliser les paires actuelles
            normalized_current_pairs = []
            for pair in current_pairs:
                try:
                    if isinstance(pair, dict) and 'pair' in pair:
                        normalized = normalize_pair_input(pair['pair'])
                    else:
                        normalized = normalize_pair_input(pair)
                    normalized_current_pairs.append(normalized)
                except ValueError as e:
                    self.logger.warning(f"Paire actuelle ignorée: {pair} - {e}")
            
            # Récupérer les paires valides (cela charge et analyse automatiquement les paires)
            valid_pairs = await self.pair_selector.get_valid_pairs(min_score=0.5)
            
            if not valid_pairs:
                self.logger.warning("Aucune nouvelle paire valide trouvée, conservation des paires actuelles")
                return normalized_current_pairs
                
            # Extraire les noms de paires normalisés
            new_pairs = []
            for pair_info in valid_pairs:
                try:
                    if isinstance(pair_info, dict) and 'pair' in pair_info:
                        normalized = normalize_pair_input(pair_info['pair'])
                    else:
                        normalized = normalize_pair_input(pair_info)
                    new_pairs.append(normalized)
                except ValueError as e:
                    self.logger.warning(f"Paire valide ignorée: {pair_info} - {e}")
            
            if not new_pairs:
                self.logger.warning("Aucune paire valide après normalisation, conservation des paires actuelles")
                return normalized_current_pairs
                
            # Comparer avec les paires actuelles
            current_set = set(normalized_current_pairs)
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
            selected_pairs = list(new_set)
            self.logger.info(f"Rotation terminée. {len(selected_pairs)} paires sélectionnées")
            return selected_pairs

        except Exception as e:
            self.logger.error(
                f"Erreur lors de la rotation des paires: {str(e)}", 
                exc_info=self.logger.isEnabledFor(LoggerManager.DEBUG)
            )
            # En cas d'erreur, essayer de retourner les paires actuelles normalisées
            try:
                return [normalize_pair_input(p) if isinstance(p, str) else normalize_pair_input(p['pair']) 
                       for p in current_pairs 
                       if (isinstance(p, str) or (isinstance(p, dict) and 'pair' in p))]
            except Exception:
                # Si la normalisation échoue, retourner une liste vide plutôt que de propager l'erreur
                return []

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
