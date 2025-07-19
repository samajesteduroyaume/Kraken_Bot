from typing import Dict, List, Tuple, Optional
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from src.core.logging.logging import LoggerManager


class PairClusterAnalyzer:
    """Analyse et clusterise les paires de trading."""

    def __init__(
        self,
        n_clusters: int = 5,
        min_pairs_per_cluster: int = 3,
        max_pairs_per_cluster: int = 10,
        logger: Optional[LoggerManager] = None
    ):
        self.n_clusters = n_clusters
        self.min_pairs_per_cluster = min_pairs_per_cluster
        self.max_pairs_per_cluster = max_pairs_per_cluster
        self.logger = logger or LoggerManager()
        self.logger = self.logger.get_logger('pair_cluster')
        self.clusters: Dict[int, List[str]] = {}
        self.cluster_metrics: Dict[int, Dict] = {}

    def _extract_features(self,
                          pair_metrics: Dict[str,
                                             Dict]) -> Tuple[np.ndarray,
                                                             List[str]]:
        """Extrait les features pour le clustering."""
        features = []
        pairs = []

        for pair, metrics in pair_metrics.items():
            features.append([
                metrics.get('volatility', 0),
                metrics.get('momentum', 0),
                metrics.get('volume', 0),
                metrics.get('score', 0),
                metrics.get('correlation', 0)
            ])
            pairs.append(pair)

        return np.array(features), pairs

    def cluster_pairs(self, pair_metrics: Dict[str, Dict]) -> Dict[str, int]:
        """Clusterise les paires en fonction de leurs caractéristiques."""
        try:
            features, pairs = self._extract_features(pair_metrics)

            if len(features) < self.n_clusters:
                self.logger.warning(
                    f"Nombre de paires ({len(features)}) inférieur au nombre de clusters ({self.n_clusters})")
                self.n_clusters = min(self.n_clusters, len(features))

            # Normaliser les données
            scaler = StandardScaler()
            scaled_features = scaler.fit_transform(features)

            # Appliquer le clustering
            kmeans = KMeans(n_clusters=self.n_clusters, random_state=42)
            cluster_labels = kmeans.fit_predict(scaled_features)

            # Créer les clusters
            clusters = {}
            for pair, cluster in zip(pairs, cluster_labels):
                if cluster not in clusters:
                    clusters[cluster] = []
                clusters[cluster].append(pair)

            # Ajuster le nombre de paires par cluster
            for cluster_id, pairs in clusters.items():
                if len(pairs) < self.min_pairs_per_cluster:
                    self.logger.warning(
                        f"Cluster {cluster_id} a moins de {self.min_pairs_per_cluster} paires")
                elif len(pairs) > self.max_pairs_per_cluster:
                    self.logger.warning(
                        f"Cluster {cluster_id} a plus de {self.max_pairs_per_cluster} paires")
                    # Sélectionner les meilleures paires du cluster
                    pairs.sort(
                        key=lambda p: pair_metrics[p]['score'],
                        reverse=True)
                    clusters[cluster_id] = pairs[:self.max_pairs_per_cluster]

            self.clusters = clusters
            self._calculate_cluster_metrics(pair_metrics)

            return {pair: cluster for cluster, pairs in clusters.items()
                    for pair in pairs}

        except Exception as e:
            self.logger.error(f"Erreur lors du clustering: {str(e)}")
            # En cas d'erreur, mettre toutes les paires dans le cluster 0
            return {pair: 0 for pair in pair_metrics.keys()}

    def _calculate_cluster_metrics(
            self, pair_metrics: Dict[str, Dict]) -> None:
        """Calcule les métriques pour chaque cluster."""
        for cluster_id, pairs in self.clusters.items():
            cluster_data = [pair_metrics[pair] for pair in pairs]

            if not cluster_data:
                continue

            # Calculer les métriques moyennes
            avg_volatility = np.mean([d['volatility'] for d in cluster_data])
            avg_momentum = np.mean([d['momentum'] for d in cluster_data])
            avg_volume = np.mean([d['volume'] for d in cluster_data])
            avg_score = np.mean([d['score'] for d in cluster_data])

            # Calculer la diversification
            correlations = []
            for i in range(len(pairs)):
                for j in range(i + 1, len(pairs)):
                    corr = pair_metrics[pairs[i]].get('correlation', 0.0)
                    correlations.append(corr)
            avg_correlation = np.mean(correlations) if correlations else 0.0

            self.cluster_metrics[cluster_id] = {
                'num_pairs': len(pairs),
                'avg_volatility': avg_volatility,
                'avg_momentum': avg_momentum,
                'avg_volume': avg_volume,
                'avg_score': avg_score,
                'avg_correlation': avg_correlation,
                'diversification_score': 1.0 - avg_correlation
            }

    def get_optimal_pairs(self, target_risk: float = 0.01) -> List[str]:
        """Sélectionne les paires optimales pour atteindre le risque cible."""
        optimal_pairs = []

        # Trier les clusters par diversification
        sorted_clusters = sorted(
            self.cluster_metrics.items(),
            key=lambda x: x[1]['diversification_score'],
            reverse=True
        )

        # Sélectionner les paires des clusters les plus diversifiés
        for cluster_id, metrics in sorted_clusters:
            pairs = self.clusters[cluster_id]
            pairs.sort(key=lambda p: pair_metrics[p]['score'], reverse=True)

            # Sélectionner le nombre optimal de paires
            num_pairs = min(
                len(pairs),
                int(target_risk * metrics['diversification_score'] * 100)
            )

            optimal_pairs.extend(pairs[:num_pairs])

            if len(optimal_pairs) >= self.max_pairs_per_cluster:
                break

        return optimal_pairs

    def get_cluster_metrics(self) -> Dict[int, Dict]:
        """Retourne les métriques de tous les clusters."""
        return self.cluster_metrics
