import os
import logging
import json
import pandas as pd
from datetime import datetime
from typing import Dict, Any, List
from pathlib import Path
from src.utils import helpers

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class PerformanceMetrics:
    """Calcule et gère les métriques de performance."""

    def __init__(self, config: Dict[str, Any]):
        """Initialise le calculateur de métriques."""
        self.config = config
        self.metrics_dir = os.path.expanduser(
            self.config.get('METRICS_DIR', '~/kraken_bot_metrics'))

        # Créer le répertoire des métriques s'il n'existe pas
        Path(self.metrics_dir).mkdir(parents=True, exist_ok=True)

    def calculate_metrics(
            self, trades: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calcule toutes les métriques de performance."""
        try:
            df = pd.DataFrame(trades)

            metrics = {
                'basic': self._calculate_basic_metrics(df),
                'risk': self._calculate_risk_metrics(df),
                'return': self._calculate_return_metrics(df),
                'position': self._calculate_position_metrics(df)
            }

            # Sauvegarder les métriques
            self._save_metrics(metrics)

            return metrics

        except Exception as e:
            logger.error(f"Erreur lors du calcul des métriques: {str(e)}")
            return {}

    def _calculate_basic_metrics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calcule les métriques de base."""
        profits = df['profit'].tolist() if 'profit' in df.columns else []
        if not profits:
            return {}
        avg_duration = 0.0
        if 'duration' in df.columns:
            try:
                if not bool(df['duration'].isnull().all()):
                    avg_duration = float(df['duration'].mean())
            except Exception:
                avg_duration = 0.0
        return {
            'total_trades': len(df),
            'win_rate': helpers.calculate_win_rate([{'pnl': p} for p in profits]) * 100,
            'profit_factor': helpers.calculate_profit_factor([{'pnl': p} for p in profits]),
            'avg_trade_duration': avg_duration
        }

    def _calculate_risk_metrics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calcule les métriques de risque."""
        returns = (
            df['profit'] /
            df['entry_price']).tolist() if 'profit' in df.columns and 'entry_price' in df.columns else []
        if not returns:
            return {}
        return {
            'sharpe_ratio': helpers.calculate_sharpe_ratio(returns),
            'sortino_ratio': helpers.calculate_sortino_ratio(returns),
            'max_drawdown': helpers.calculate_max_drawdown(returns),
            'calmar_ratio': self._calculate_calmar_ratio(df)
        }

    def _calculate_return_metrics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calcule les métriques de rendement."""
        if 'profit' not in df.columns:
            return {}
        return {
            'total_profit': df['profit'].sum(),
            'avg_profit': df['profit'].mean(),
            'profit_std': df['profit'].std(),
            'monthly_returns': self._calculate_monthly_returns(df)
        }

    def _calculate_position_metrics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calcule les métriques de position."""
        if 'size' not in df.columns:
            return {}
        return {
            'avg_position_size': df['size'].mean(),
            'max_position_size': df['size'].max(),
            'position_turnover': self._calculate_position_turnover(df)
        }

    def _calculate_max_drawdown(self, df: pd.DataFrame) -> float:
        """Calcule le drawdown maximum."""
        returns = (
            df['profit'] /
            df['entry_price']).tolist() if 'profit' in df.columns and 'entry_price' in df.columns else []
        if not returns:
            return 0.0
        return helpers.calculate_max_drawdown(returns)

    def _calculate_calmar_ratio(self, df: pd.DataFrame) -> float:
        """Calcule le ratio de Calmar."""
        try:
            annual_return = df['profit'].sum(
            ) / len(df) if len(df) > 0 else 0.0
            max_dd = self._calculate_max_drawdown(df)
            return annual_return / abs(max_dd) if max_dd != 0 else float('inf')
        except Exception:
            return 0.0

    def _calculate_monthly_returns(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calcule les rendements mensuels."""
        if 'timestamp' not in df.columns or 'profit' not in df.columns:
            return {}
        try:
            df['month'] = pd.to_datetime(df['timestamp']).dt.to_period('M')
            monthly_returns = df.groupby('month')['profit'].sum().to_dict()
            # Convertir les clés en str pour éviter les problèmes de typage
            return {str(k): float(v) for k, v in monthly_returns.items()}
        except Exception:
            return {}

    def _calculate_position_turnover(self, df: pd.DataFrame) -> float:
        """Calcule le turnover des positions."""
        try:
            if 'size' not in df.columns:
                return 0.0
            total_size = df['size'].sum()
            n_trades = len(df)
            return float(total_size) / n_trades if n_trades > 0 else 0.0
        except Exception:
            return 0.0

    def _save_metrics(self, metrics: Dict[str, Any]):
        """Sauvegarde les métriques dans un fichier."""
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            metrics_file = os.path.join(
                self.metrics_dir, f'metrics_{timestamp}.json')

            with open(metrics_file, 'w') as f:
                json.dump(metrics, f, indent=4)

            logger.info(f"Métriques sauvegardées dans {metrics_file}")

        except Exception as e:
            logger.error(
                f"Erreur lors de la sauvegarde des métriques: {str(e)}")

    def load_metrics(self, timestamp: str = None) -> Dict[str, Any]:
        """Charge les métriques à partir d'un fichier."""
        try:
            if timestamp:
                metrics_file = os.path.join(
                    self.metrics_dir, f'metrics_{timestamp}.json')
                if not os.path.exists(metrics_file):
                    return None

                with open(metrics_file, 'r') as f:
                    return json.load(f)

            # Charger le fichier le plus récent
            metrics_files = sorted(
                Path(self.metrics_dir).glob('metrics_*.json'))
            if not metrics_files:
                return None

            with open(metrics_files[-1], 'r') as f:
                return json.load(f)

        except Exception as e:
            logger.error(f"Erreur lors du chargement des métriques: {str(e)}")
            return None


def setup_performance_metrics(config: Dict[str, Any]) -> PerformanceMetrics:
    """Configure le calculateur de métriques de performance."""
    return PerformanceMetrics(config)


def calculate_and_save_metrics(
        config: Dict[str, Any], trades: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Calcule et sauvegarde les métriques de performance."""
    metrics_calculator = setup_performance_metrics(config)
    return metrics_calculator.calculate_metrics(trades)


def load_latest_metrics(config: Dict[str, Any]) -> Dict[str, Any]:
    """Charge les métriques les plus récentes."""
    metrics_calculator = setup_performance_metrics(config)
    return metrics_calculator.load_metrics()
