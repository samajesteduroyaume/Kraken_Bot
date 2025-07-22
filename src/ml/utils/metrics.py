"""
Métriques d'évaluation pour les modèles de trading.
"""
import numpy as np
from typing import Dict, List, Union, Optional
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)


class ModelMetrics:
    """Classe utilitaire pour calculer les métriques d'évaluation des modèles."""
    
    @staticmethod
    def calculate_classification_metrics(
        y_true: Union[np.ndarray, List],
        y_pred: Union[np.ndarray, List],
        labels: Optional[List] = None
    ) -> Dict[str, float]:
        """Calcule les métriques de classification standard.
        
        Args:
            y_true: Vraies étiquettes
            y_pred: Prédictions du modèle
            labels: Noms des classes pour le rapport
            
        Returns:
            Dictionnaire des métriques calculées
        """
        y_true, y_pred = np.array(y_true), np.array(y_pred)
        
        # Vérification des entrées
        if len(y_true) != len(y_pred):
            raise ValueError("Les tableaux y_true et y_pred doivent avoir la même longueur")
        
        if len(y_true) == 0:
            return {}
        
        # Calcul des métriques
        metrics = {
            'accuracy': float(accuracy_score(y_true, y_pred)),
            'precision': float(precision_score(y_true, y_pred, average='weighted', zero_division=0)),
            'recall': float(recall_score(y_true, y_pred, average='weighted', zero_division=0)),
            'f1': float(f1_score(y_true, y_pred, average='weighted', zero_division=0)),
        }
        
        # Matrice de confusion
        conf_matrix = confusion_matrix(y_true, y_pred)
        metrics['confusion_matrix'] = conf_matrix.tolist()
        
        # Rapport de classification détaillé
        if labels is not None:
            report = classification_report(
                y_true, y_pred, 
                target_names=labels,
                output_dict=True,
                zero_division=0
            )
            metrics['classification_report'] = report
        
        return metrics
    
    @staticmethod
    def calculate_trading_metrics(
        returns: List[float],
        benchmark_returns: Optional[List[float]] = None,
        risk_free_rate: float = 0.0
    ) -> Dict[str, float]:
        """Calcule les métriques spécifiques au trading.
        
        Args:
            returns: Liste des rendements du portefeuille
            benchmark_returns: Liste des rendements du benchmark (optionnel)
            risk_free_rate: Taux sans risque (pour le calcul du ratio de Sharpe)
            
        Returns:
            Dictionnaire des métriques de trading
        """
        if not returns:
            return {}
            
        returns = np.array(returns)
        metrics = {
            'total_return': float(np.prod(1 + returns) - 1),
            'annualized_return': float(np.mean(returns) * 252),  # 252 jours de trading par an
            'volatility': float(np.std(returns) * np.sqrt(252)),
            'max_drawdown': float(ModelMetrics._calculate_max_drawdown(returns)),
            'sharpe_ratio': float(ModelMetrics._calculate_sharpe_ratio(returns, risk_free_rate)),
        }
        
        if benchmark_returns is not None and len(benchmark_returns) == len(returns):
            benchmark_returns = np.array(benchmark_returns)
            metrics.update({
                'alpha': float(ModelMetrics._calculate_alpha(returns, benchmark_returns, risk_free_rate)),
                'beta': float(ModelMetrics._calculate_beta(returns, benchmark_returns)),
                'information_ratio': float(ModelMetrics._calculate_information_ratio(returns, benchmark_returns)),
            })
        
        return metrics
    
    @staticmethod
    def _calculate_max_drawdown(returns: np.ndarray) -> float:
        """Calcule le drawdown maximum."""
        cum_returns = np.cumprod(1 + returns)
        peak = np.maximum.accumulate(cum_returns)
        drawdowns = (peak - cum_returns) / peak
        return np.max(drawdowns) if len(drawdowns) > 0 else 0.0
    
    @staticmethod
    def _calculate_sharpe_ratio(returns: np.ndarray, risk_free_rate: float) -> float:
        """Calcule le ratio de Sharpe annualisé."""
        excess_returns = returns - risk_free_rate / 252  # Taux journalier
        if np.std(excess_returns) == 0:
            return 0.0
        return np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252)
    
    @staticmethod
    def _calculate_alpha(returns: np.ndarray, benchmark_returns: np.ndarray, risk_free_rate: float) -> float:
        """Calcule l'alpha de Jensen."""
        excess_returns = returns - risk_free_rate / 252
        excess_benchmark = benchmark_returns - risk_free_rate / 252
        beta = ModelMetrics._calculate_beta(returns, benchmark_returns)
        return np.mean(excess_returns) - beta * np.mean(excess_benchmark)
    
    @staticmethod
    def _calculate_beta(returns: np.ndarray, benchmark_returns: np.ndarray) -> float:
        """Calcule le bêta par rapport au benchmark."""
        covariance = np.cov(returns, benchmark_returns)[0, 1]
        variance = np.var(benchmark_returns)
        return covariance / variance if variance != 0 else 0.0
    
    @staticmethod
    def _calculate_information_ratio(returns: np.ndarray, benchmark_returns: np.ndarray) -> float:
        """Calcule le ratio d'information."""
        active_returns = returns - benchmark_returns
        if np.std(active_returns) == 0:
            return 0.0
        return np.mean(active_returns) / np.std(active_returns) * np.sqrt(252)
