"""
Gestion des risques pour le trading.

Ce module fournit des outils pour la gestion des risques, y compris
la gestion du levier, du drawdown, et du position sizing.
"""
import logging
from typing import Dict, Optional, Any
import numpy as np
from datetime import datetime

from ..leverage_manager import LeverageManager

logger = logging.getLogger(__name__)


class RiskManager:
    """Gère les risques liés aux opérations de trading."""

    def __init__(self,
                 initial_balance: float,
                 risk_profile: Optional[Dict] = None):
        """
        Initialise le gestionnaire de risques.

        Args:
            initial_balance: Solde initial du compte
            risk_profile: Profil de risque (optionnel)
        """
        self.initial_balance = initial_balance
        self.current_balance = initial_balance
        self.equity_curve = [initial_balance]
        self.risk_profile = risk_profile or {
            'max_drawdown': 0.1,  # 10% de drawdown maximum
            'risk_per_trade': 0.01,  # 1% du capital par trade
            'max_leverage': 5.0,  # Levier maximum
            'leverage_strategy': 'moderate'  # Stratégie de levier
        }

        # Initialisation du gestionnaire de levier
        config = {
            'max_leverage': self.risk_profile['max_leverage'],
            'risk_per_trade': self.risk_profile['risk_per_trade'],
            'initial_balance': initial_balance
        }
        self.leverage_manager = LeverageManager(config)

        # Suivi des performances
        self.trade_history = []
        self.max_drawdown = 0.0
        self.sharpe_ratio = 0.0
        self.sortino_ratio = 0.0

    def update_balance(self, new_balance: float) -> None:
        """
        Met à jour le solde du compte et recalcule les métriques de risque.

        Args:
            new_balance: Nouveau solde du compte
        """
        self.current_balance = new_balance
        self.equity_curve.append(new_balance)
        self._update_risk_metrics()

    def _update_risk_metrics(self) -> None:
        """Met à jour les métriques de risque."""
        # Calcul du drawdown
        peak = max(self.equity_curve)
        current = self.equity_curve[-1]
        self.max_drawdown = max(self.max_drawdown,
                                (peak - current) / peak if peak > 0 else 0.0)

        # Calcul des rendements
        returns = self._calculate_returns()

        if len(returns) > 1:
            # Ratio de Sharpe (simplifié, sans taux sans risque)
            self.sharpe_ratio = np.sqrt(252) * (returns.mean() / returns.std())

            # Ratio de Sortino (ne considère que la volatilité à la baisse)
            downside_returns = returns[returns < 0]
            if len(downside_returns) > 0:
                self.sortino_ratio = np.sqrt(
                    252) * (returns.mean() / downside_returns.std())

    def _calculate_returns(self) -> np.ndarray:
        """Calcule les rendements à partir de la courbe d'équité."""
        if len(self.equity_curve) < 2:
            return np.array([])

        equity = np.array(self.equity_curve)
        return np.diff(equity) / equity[:-1]

    def calculate_position_size(self,
                                entry_price: float,
                                stop_loss: float,
                                risk_percent: Optional[float] = None) -> float:
        """
        Calcule la taille de position en fonction du risque.

        Args:
            entry_price: Prix d'entrée
            stop_loss: Niveau de stop-loss
            risk_percent: Pourcentage du capital à risquer (optionnel)

        Returns:
            Taille de la position dans l'unité de base
        """
        if risk_percent is None:
            risk_percent = self.risk_profile.get('risk_per_trade', 0.01)

        if entry_price <= 0 or stop_loss >= entry_price:
            raise ValueError("Prix d'entrée et stop-loss invalides")

        # Montant en capital à risquer
        risk_amount = self.current_balance * risk_percent

        # Distance au stop-loss en pourcentage
        stop_distance = (entry_price - stop_loss) / entry_price

        # Taille de la position en unités de base
        position_size = risk_amount / (entry_price * stop_distance)

        return position_size

    def get_leverage(self,
                     market_volatility: float,
                     signal_confidence: float,
                     recent_performance: float) -> float:
        """
        Calcule le levier approprié en fonction du risque.

        Args:
            market_volatility: Volatilité du marché (0-1)
            signal_confidence: Confiance dans le signal (0-1)
            recent_performance: Performance récente du système (0-1)

        Returns:
            Levier à utiliser
        """
        # Mettre à jour le levier en fonction des conditions du marché
        leverage = self.leverage_manager.update_leverage_based_on_market_conditions(
            market_volatility=market_volatility,
            signal_confidence=signal_confidence,
            recent_performance=recent_performance,
            max_drawdown=self.max_drawdown,
            position_size=0.0  # À ajuster si nécessaire
        )

        return min(leverage, self.risk_profile.get('max_leverage', 5.0))

    def get_risk_metrics(self) -> Dict[str, float]:
        """
        Récupère les métriques de risque actuelles.

        Returns:
            Dictionnaire des métriques de risque
        """
        return {
            'current_balance': self.current_balance,
            'max_drawdown': self.max_drawdown,
            'sharpe_ratio': self.sharpe_ratio,
            'sortino_ratio': self.sortino_ratio,
            'equity_curve': self.equity_curve[-100:],  # Derniers 100 points
            'current_leverage': self.leverage_manager.get_current_leverage(),
            # 50 dernières entrées
            'leverage_history': self.leverage_manager.get_leverage_history()[-50:]
        }

    def log_trade(self,
                  pair: str,
                  side: str,
                  amount: float,
                  entry_price: float,
                  exit_price: Optional[float] = None,
                  stop_loss: Optional[float] = None,
                  take_profit: Optional[float] = None) -> Dict[str, Any]:
        """
        Enregistre un trade dans l'historique.

        Args:
            pair: Paire tradée
            side: 'buy' ou 'sell'
            amount: Montant tradé
            entry_price: Prix d'entrée
            exit_price: Prix de sortie (optionnel)
            stop_loss: Niveau de stop-loss (optionnel)
            take_profit: Niveau de take-profit (optionnel)

        Returns:
            Dictionnaire avec les informations du trade
        """
        trade = {
            'id': len(
                self.trade_history) + 1,
            'pair': pair,
            'side': side,
            'amount': amount,
            'entry_price': entry_price,
            'entry_time': datetime.now(),
            'exit_price': exit_price,
            'exit_time': datetime.now() if exit_price is not None else None,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'status': 'open' if exit_price is None else 'closed',
            'pnl': (
                exit_price - entry_price) * amount * (
                1 if side == 'buy' else -1) if exit_price is not None else 0.0,
            'pnl_pct': (
                (exit_price / entry_price) - 1) * 100 if exit_price is not None else 0.0}

        self.trade_history.append(trade)
        return trade

    def update_trade(self,
                     trade_id: int,
                     exit_price: Optional[float] = None,
                     status: str = 'closed') -> bool:
        """
        Met à jour un trade existant.

        Args:
            trade_id: ID du trade à mettre à jour
            exit_price: Prix de sortie (optionnel)
            status: Nouveau statut du trade

        Returns:
            True si la mise à jour a réussi, False sinon
        """
        if trade_id < 0 or trade_id >= len(self.trade_history):
            return False

        trade = self.trade_history[trade_id]

        if exit_price is not None:
            trade['exit_price'] = exit_price
            trade['exit_time'] = datetime.now()
            trade['pnl'] = (exit_price - trade['entry_price']) * \
                trade['amount'] * (1 if trade['side'] == 'buy' else -1)
            trade['pnl_pct'] = ((exit_price / trade['entry_price']) - 1) * 100

        trade['status'] = status

        return True

    def process_signals(self, signals: list) -> list:
        """Filtre ou ajuste les signaux de trading selon la gestion du risque réelle."""
        # Exemple : filtrer les signaux selon le drawdown ou le risque par trade
        filtered = []
        for signal in signals:
            # Exemple : n'accepter que les signaux si le drawdown max n'est pas dépassé
            if self.max_drawdown < self.risk_profile.get('max_drawdown', 0.1):
                filtered.append(signal)
        return filtered
