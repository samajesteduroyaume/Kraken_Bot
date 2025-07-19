from typing import Dict, Any
import numpy as np
import logging
from src.utils import helpers


class MetricsManager:
    """Gestionnaire des métriques de performance."""

    def __init__(self):
        """Initialise le gestionnaire de métriques."""
        self.metrics = {
            'total_trades': 0,
            'total_profit': 0.0,
            'winning_trades': 0,
            'losing_trades': 0,
            'max_drawdown': 0.0,
            'sharpe_ratio': 0.0,
            'sortino_ratio': 0.0,
            'win_rate': 0.0,
            'avg_profit': 0.0,
            'avg_loss': 0.0,
            'max_profit': 0.0,
            'max_loss': 0.0,
            'current_balance': 0.0,
            'start_balance': 0.0,
            'end_balance': 0.0,
            'total_return': 0.0,
            'annualized_return': 0.0,
            'volatility': 0.0,
            'trades': [],
            'daily_returns': [],
            'logger': logging.getLogger('metrics_manager')
        }

    def update_metrics(self, trade_data: Dict[str, Any]):
        """
        Met à jour les métriques avec les données d'un trade.

        Args:
            trade_data: Dictionnaire avec les données du trade
        """
        try:
            # Mettre à jour les métriques de base
            self.metrics['total_trades'] += 1
            self.metrics['total_profit'] += trade_data['pnl']

            # Classer le trade comme gagnant ou perdant
            if trade_data['pnl'] > 0:
                self.metrics['winning_trades'] += 1
            else:
                self.metrics['losing_trades'] += 1

            # Mettre à jour les statistiques de trades
            self.metrics['trades'].append({
                'timestamp': trade_data['timestamp'],
                'pnl': trade_data['pnl'],
                'size': trade_data['size'],
                'leverage': trade_data['leverage']
            })

            # Calculs centralisés
            pnls = [t['pnl'] for t in self.metrics['trades']]
            self.metrics['win_rate'] = helpers.calculate_win_rate(
                [{'pnl': p} for p in pnls])
            self.metrics['avg_profit'] = float(
                np.mean([p for p in pnls if p > 0])) if any(p > 0 for p in pnls) else 0.0
            self.metrics['avg_loss'] = float(
                np.mean([p for p in pnls if p < 0])) if any(p < 0 for p in pnls) else 0.0
            self.metrics['max_profit'] = max(pnls) if pnls else 0.0
            self.metrics['max_loss'] = min(pnls) if pnls else 0.0
            self.metrics['max_drawdown'] = helpers.calculate_drawdown(pnls)
            self.metrics['profit_factor'] = helpers.calculate_profit_factor(
                [{'pnl': p} for p in pnls])
            self.metrics['sharpe_ratio'] = helpers.calculate_sharpe_ratio(pnls)
            self.metrics['sortino_ratio'] = helpers.calculate_sortino_ratio(
                pnls)
            self.metrics['total_return'] = self.metrics['total_profit'] / \
                self.metrics['start_balance'] if self.metrics['start_balance'] else 0.0
            # Annualized return et volatilité
            if self.metrics['total_trades'] > 0:
                days = len(set([t['timestamp'].date()
                           for t in self.metrics['trades']]))
                if days > 0:
                    self.metrics['annualized_return'] = (
                        1 + self.metrics['total_return']) ** (365 / days) - 1
            returns = [t['pnl'] / self.metrics['start_balance']
                       for t in self.metrics['trades']] if self.metrics['start_balance'] else []
            self.metrics['volatility'] = float(
                np.std(returns) * np.sqrt(365)) if returns else 0.0

        except Exception as e:
            self.logger.error(
                f"Erreur lors de la mise à jour des métriques: {str(e)}")

    def get_metrics(self) -> Dict[str, Any]:
        """Retourne toutes les métriques."""
        return self.metrics.copy()

    def reset_metrics(self):
        """Réinitialise les métriques."""
        self.metrics = {
            'total_trades': 0,
            'total_profit': 0.0,
            'winning_trades': 0,
            'losing_trades': 0,
            'max_drawdown': 0.0,
            'sharpe_ratio': 0.0,
            'sortino_ratio': 0.0,
            'win_rate': 0.0,
            'avg_profit': 0.0,
            'avg_loss': 0.0,
            'max_profit': 0.0,
            'max_loss': 0.0,
            'current_balance': 0.0,
            'start_balance': 0.0,
            'end_balance': 0.0,
            'total_return': 0.0,
            'annualized_return': 0.0,
            'volatility': 0.0,
            'trades': [],
            'daily_returns': [],
            'logger': logging.getLogger('metrics_manager')
        }
