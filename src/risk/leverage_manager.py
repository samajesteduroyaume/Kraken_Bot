"""
Gestion avancée du levier pour le trading sur marge.

Ce module fournit des fonctionnalités pour gérer l'effet de levier de manière dynamique
en fonction des conditions du marché, du profil de risque et des performances récentes.
"""
import json
import logging
import os
from typing import Dict, Optional, Tuple, Any
from datetime import datetime
from enum import Enum
import numpy as np

# Configuration du logger
logger = logging.getLogger(__name__)


class LeverageStrategy(Enum):
    """Stratégies de gestion du levier."""
    CONSERVATIVE = 1
    MODERATE = 2
    AGGRESSIVE = 3


class LeverageManager:
    """Gestionnaire de levier pour le trading sur marge."""

    def __init__(self, risk_profile: Dict, initial_balance: float):
        """
        Initialise le gestionnaire de levier.

        Args:
            risk_profile: Profil de risque contenant les paramètres de levier
            initial_balance: Solde initial du compte
        """
        self.risk_profile = risk_profile
        self.current_leverage = 1.0
        self.max_leverage = min(
            risk_profile.get(
                'max_leverage',
                1.0),
            20.0)  # Limite de sécurité
        self.initial_balance = initial_balance
        self.current_balance = initial_balance
        self.leverage_history = []
        self.equity_curve = [initial_balance]

        # Configuration de la stratégie de levier
        self.leverage_strategy = self._parse_leverage_strategy(
            risk_profile.get('leverage_strategy', 'conservative')
        )

        # Initialisation des attributs pour le suivi des performances
        self.drawdown = 0.0
        self.volatility = 0.0
        self.leverage_history = []  # Sera utilisé pour stocker les entrées complètes
        self.max_leverage = min(risk_profile.get('max_leverage', 1.0), 20.0)

        # Paramètres de la stratégie
        self.leverage_params = {
            LeverageStrategy.CONSERVATIVE: {
                'base_leverage': 1.0,
                'max_drawdown': 0.05,
                'volatility_factor': 0.5,
                'confidence_threshold': 0.7
            },
            LeverageStrategy.MODERATE: {
                'base_leverage': 2.0,
                'max_drawdown': 0.1,
                'volatility_factor': 0.7,
                'confidence_threshold': 0.6
            },
            LeverageStrategy.AGGRESSIVE: {
                'base_leverage': 3.0,
                'max_drawdown': 0.15,
                'volatility_factor': 0.9,
                'confidence_threshold': 0.5
            }
        }

    def _parse_leverage_strategy(self, strategy_str: str) -> LeverageStrategy:
        """Convertit une chaîne de stratégie en énumération."""
        strategy_map = {
            'conservative': LeverageStrategy.CONSERVATIVE,
            'moderate': LeverageStrategy.MODERATE,
            'aggressive': LeverageStrategy.AGGRESSIVE
        }
        return strategy_map.get(
            strategy_str.lower(),
            LeverageStrategy.MODERATE)

    def update_balance(self, new_balance: float) -> None:
        """Met à jour le solde du compte et l'historique."""
        self.current_balance = new_balance
        self.equity_curve.append(new_balance)

    def calculate_drawdown(self) -> float:
        """Calcule le drawdown actuel en pourcentage."""
        if not self.equity_curve:
            return 0.0

        peak = max(self.equity_curve)
        current = self.equity_curve[-1]
        return (peak - current) / peak if peak > 0 else 0.0

    def calculate_volatility(self, window: int = 20) -> float:
        """
        Calcule la volatilité récente du compte.

        Args:
            window: Nombre de périodes pour le calcul de la volatilité

        Returns:
            float: Volatilité sur la période (écart-type des rendements)
        """
        if len(self.equity_curve) < 2:
            return 0.0

        returns = np.diff(self.equity_curve) / self.equity_curve[:-1]
        returns = returns[~np.isnan(returns)]

        if len(returns) < 2:
            return 0.0

        return float(np.std(returns[-window:]))

    def calculate_position_size(
        self,
        entry_price: float,
        stop_loss: float,
        account_balance: Optional[float] = None,
        confidence: float = 1.0
    ) -> Tuple[float, float]:
        """
        Calcule la taille de position optimale avec le levier.

        Args:
            entry_price: Prix d'entrée
            stop_loss: Prix du stop-loss
            account_balance: Solde du compte (optionnel, utilise self.current_balance si None)
            confidence: Niveau de confiance du trade (0-1)

        Returns:
            Tuple[float, float]: Taille de la position et levier utilisé
        """
        if account_balance is None:
            account_balance = self.current_balance

        # Calcul du risque par trade en fonction du profil de risque
        risk_per_trade = self.risk_profile.get('risk_per_trade', 0.01)
        max_position_size = self.risk_profile.get('max_position_size', 0.1)

        # Calcul du risque par unité
        risk_per_share = abs(entry_price - stop_loss)
        if risk_per_share == 0:
            return 0.0, 1.0

        # Calcul du levier en fonction de la stratégie
        leverage = self.calculate_leverage(confidence)

        # Calcul de la taille de position de base
        risk_amount = account_balance * risk_per_trade * leverage
        position_size = risk_amount / risk_per_share

        # Ajustement selon la taille maximale de position
        max_position = (account_balance * leverage *
                        max_position_size) / entry_price
        position_size = min(position_size, max_position)

        # Vérification de la marge disponible
        required_margin = (entry_price * position_size) / leverage
        if required_margin > account_balance:
            # Ajustement de la position pour respecter la marge disponible
            position_size = (account_balance * leverage) / entry_price

        return position_size, leverage

    def calculate_leverage(self, confidence: float = 1.0) -> float:
        """
        Calcule le levier optimal en fonction des conditions actuelles.

        Args:
            confidence: Niveau de confiance du signal (0-1)

        Returns:
            float: Levier calculé
        """
        # Récupération des paramètres de la stratégie
        params = self.leverage_params[self.leverage_strategy]

        # Calcul des facteurs d'ajustement
        drawdown = self.calculate_drawdown()
        volatility = self.calculate_volatility()

        # Ajustement basé sur le drawdown
        drawdown_factor = 1.0 - (drawdown / params['max_drawdown'])
        # Borné entre 0.1 et 1.0
        drawdown_factor = max(0.1, min(1.0, drawdown_factor))

        # Ajustement basé sur la volatilité
        volatility_factor = 1.0 / \
            (1.0 + volatility * params['volatility_factor'])

        # Ajustement basé sur la confiance
        confidence_factor = min(
            1.0, confidence / params['confidence_threshold'])

        # Calcul du levier final
        leverage = params['base_leverage'] * drawdown_factor * \
            volatility_factor * confidence_factor

        # Application des limites
        leverage = max(1.0, min(leverage, self.max_leverage))

        # Mise à jour du levier actuel
        self.leverage_history.append(leverage)
        if len(self.leverage_history) > 100:  # Garder un historique limité
            self.leverage_history.pop(0)

        return leverage

    def get_leverage_stats(self) -> Dict[str, Any]:
        """
        Retourne des statistiques sur l'utilisation du levier.

        Returns:
            Dictionnaire contenant les statistiques de levier
        """
        return {
            'current_leverage': self.current_leverage,
            'max_leverage': self.max_leverage,
            # Derniers 100 points
            'leverage_history': self.leverage_history[-100:],
            'drawdown': self.calculate_drawdown(),
            'volatility': self.calculate_volatility(),
            'last_updated': datetime.now().isoformat()
        }

    def update_leverage_based_on_market_conditions(
        self,
        market_volatility: float,
        signal_confidence: float,
        recent_performance: float,
        max_drawdown: float,
        position_size: float,
        max_position_size: float = 0.1  # 10% du capital par défaut
    ) -> float:
        """
        Ajuste dynamiquement le levier en fonction des conditions du marché.

        Args:
            market_volatility: Volatilité actuelle du marché (0-1)
            signal_confidence: Confiance dans le signal de trading (0-1)
            recent_performance: Performance récente du bot (0-1)
            max_drawdown: Drawdown maximum autorisé (0-1)
            position_size: Taille actuelle de la position (en % du capital)
            max_position_size: Taille maximale de position (en % du capital)

        Returns:
            float: Nouveau levier recommandé
        """
        try:
            # Facteurs d'ajustement
            # Réduit le levier en cas de forte volatilité
            volatility_factor = 1.0 / (1.0 + market_volatility * 3)
            # Augmente le levier avec la confiance
            confidence_factor = min(1.0, signal_confidence * 1.5)
            # Réduit le levier si les performances sont mauvaises
            performance_factor = 0.5 + (recent_performance * 0.5)

            # Ajustement basé sur la taille de la position
            position_usage = position_size / max_position_size  # 0-1
            # Réduit le levier si la position est importante
            position_factor = 1.0 - (position_usage * 0.5)

            # Calcul du levier de base
            base_leverage = self.max_leverage * confidence_factor * performance_factor

            # Ajustement pour la volatilité et la taille de la position
            adjusted_leverage = base_leverage * volatility_factor * position_factor

            # Ajustement pour le drawdown
            if self.calculate_drawdown() > max_drawdown * \
                    0.5:  # Si on dépasse 50% du drawdown max
                drawdown_factor = 1.0 - \
                    ((self.calculate_drawdown() - (max_drawdown * 0.5)) / (max_drawdown * 0.5))
                # Réduit progressivement le levier
                adjusted_leverage *= max(0.1, drawdown_factor)

            # Application des limites
            new_leverage = max(1.0, min(adjusted_leverage, self.max_leverage))

            # Mise à jour de l'historique
            self.current_leverage = new_leverage
            self.leverage_history.append({
                'timestamp': datetime.now().isoformat(),
                'leverage': new_leverage,
                'volatility': market_volatility,
                'confidence': signal_confidence,
                'drawdown': self.calculate_drawdown()
            })

            # Mise à jour des statistiques
            self._update_volatility()
            self._update_drawdown()

            logger.info(
                f"Mise à jour du levier: {new_leverage:.2f}x "
                f"(Vol: {market_volatility:.2f}, Conf: {signal_confidence:.2f}, "
                f"Perf: {recent_performance:.2f}, DD: {self.calculate_drawdown():.2f})")

            return new_leverage

        except Exception as e:
            logger.error(f"Erreur lors de la mise à jour du levier: {e}")
            return self.current_leverage  # Retourne le levier actuel en cas d'erreur

    def _update_volatility(self) -> None:
        """Met à jour la volatilité."""
        self.volatility = self.calculate_volatility()

    def _update_drawdown(self) -> None:
        """Met à jour le drawdown."""
        self.drawdown = self.calculate_drawdown()

    def calculate_drawdown(self, window: int = 30) -> float:
        """
        Calcule le drawdown sur une fenêtre glissante.

        Args:
            window: Taille de la fenêtre en nombre de points

        Returns:
            float: Drawdown maximum sur la fenêtre (0-1)
        """
        if len(self.leverage_history) < 2:
            return 0.0

        # Extraire les valeurs de levier sur la fenêtre
        leverages = [entry['leverage']
                     for entry in self.leverage_history[-window:]]

        # Calculer le drawdown
        peak = max(leverages)
        trough = min(leverages)

        if peak == 0:
            return 0.0

        return (peak - trough) / peak

    def calculate_volatility(self, window: int = 20) -> float:
        """
        Calcule la volatilité des changements de levier.

        Args:
            window: Taille de la fenêtre en nombre de points

        Returns:
            float: Volatilité (écart-type des rendements)
        """
        if len(self.leverage_history) < 2:
            return 0.0

        # Extraire les valeurs de levier sur la fenêtre
        leverages = [entry['leverage']
                     for entry in self.leverage_history[-window:]]

        # Calculer les rendements
        returns = []
        for i in range(1, len(leverages)):
            if leverages[i - 1] != 0:
                returns.append(
                    (leverages[i] - leverages[i - 1]) / leverages[i - 1])

        if not returns:
            return 0.0

        # Calculer l'écart-type des rendements
        return float(np.std(returns))

    def save_state(self, filepath: str) -> bool:
        """
        Sauvegarde l'état du gestionnaire de levier dans un fichier.

        Args:
            filepath: Chemin vers le fichier de sauvegarde

        Returns:
            bool: True si la sauvegarde a réussi, False sinon
        """
        try:
            state = {
                'current_leverage': self.current_leverage,
                'leverage_history': self.leverage_history,
                'max_leverage': self.max_leverage,
                'drawdown': self.drawdown,
                'volatility': self.volatility,
                'last_updated': datetime.now().isoformat()
            }

            # Créer le répertoire parent si nécessaire
            os.makedirs(
                os.path.dirname(
                    os.path.abspath(filepath)),
                exist_ok=True)

            # Sauvegarder dans un fichier JSON
            with open(filepath, 'w') as f:
                json.dump(state, f, indent=2)

            logger.info(
                f"État du gestionnaire de levier sauvegardé dans {filepath}")
            return True

        except Exception as e:
            logger.error(f"Erreur lors de la sauvegarde de l'état: {e}")
            return False

    @classmethod
    def load_state(
            cls,
            filepath: str,
            max_leverage: float = 5.0) -> 'LeverageManager':
        """
        Charge l'état du gestionnaire de levier depuis un fichier.

        Args:
            filepath: Chemin vers le fichier de sauvegarde
            max_leverage: Levier maximum autorisé (peut écraser la valeur sauvegardée)

        Returns:
            Instance de LeverageManager avec l'état chargé
        """
        try:
            with open(filepath, 'r') as f:
                state = json.load(f)

            # Créer une nouvelle instance
            manager = cls(max_leverage=max_leverage)

            # Restaurer l'état
            manager.current_leverage = state.get('current_leverage', 1.0)
            manager.leverage_history = state.get('leverage_history', [])
            manager.drawdown = state.get('drawdown', 0.0)
            manager.volatility = state.get('volatility', 0.0)

            logger.info(
                f"État du gestionnaire de levier chargé depuis {filepath}. "
                f"Dernière mise à jour: {state.get('last_updated', 'inconnue')}")

            return manager

        except FileNotFoundError:
            logger.warning(
                f"Aucun fichier d'état trouvé pour le gestionnaire de levier. Création d'une nouvelle instance.")
            return cls(max_leverage=max_leverage)

        except Exception as e:
            logger.error(f"Erreur lors du chargement de l'état: {e}")
            return cls(max_leverage=max_leverage)
