"""
Implémentation de la stratégie de trading par grille (Grid Trading).
Cette stratégie place des ordres d'achat et de vente à des niveaux de prix prédéfinis.
"""
from typing import Dict, Optional
import pandas as pd
from .base_strategy import BaseStrategy


class GridStrategy(BaseStrategy):
    """
    Stratégie de trading par grille qui place des ordres à des niveaux de prix prédéfinis.

    La grille est définie par un prix central et des niveaux espacés de manière égale
    au-dessus et en dessous de ce prix.
    """

    def __init__(self, config: Optional[Dict] = None):
        """
        Initialise la stratégie de grille.

        Args:
            config: Configuration de la stratégie avec les paramètres suivants:
                - grid_size: Nombre de niveaux de la grille (par défaut: 5)
                - grid_spacing: Espacement entre les niveaux en pourcentage (par défaut: 1.0%)
                - take_profit: Pourcentage de profit pour fermer la position (par défaut: 0.5%)
                - stop_loss: Pourcentage de perte maximale (par défaut: 2.0%)
        """
        super().__init__(config)

        # Paramètres par défaut
        self.default_config = {
            'grid_size': 5,         # Nombre de niveaux de grille
            'grid_spacing': 1.0,    # Espacement en pourcentage entre les niveaux
            'take_profit': 0.5,     # Pourcentage de profit pour fermer la position
            'stop_loss': 2.0,       # Pourcentage de perte maximale
            # Taille maximale de position (10% du capital)
            'max_position_size': 0.1
        }

        # Fusion avec la configuration fournie
        self.config = {**self.default_config, **(config or {})}

        # État de la grille
        self.grid_levels = []
        self.active_orders = []

    def calculate_signals(self, data: pd.DataFrame) -> Dict[str, any]:
        """
        Calcule les signaux de trading basés sur la stratégie de grille.

        Args:
            data: DataFrame contenant les données OHLCV et indicateurs techniques

        Returns:
            Dictionnaire contenant les signaux et les niveaux de la grille
        """
        if data.empty:
            return {'signal': 'HOLD', 'grid_levels': []}

        # Récupérer le dernier prix
        current_price = data['close'].iloc[-1]

        # Si c'est le premier appel, initialiser la grille autour du prix
        # actuel
        if not self.grid_levels:
            self._initialize_grid(current_price)

        # Vérifier si le prix a atteint un niveau de la grille
        signal = 'HOLD'
        target_price = None

        for level in self.grid_levels:
            if level['type'] == 'BUY' and current_price <= level['price'] and not level['triggered']:
                signal = 'BUY'
                target_price = level['price']
                level['triggered'] = True
                break

            elif level['type'] == 'SELL' and current_price >= level['price'] and not level['triggered']:
                signal = 'SELL'
                target_price = level['price']
                level['triggered'] = True
                break

        return {
            'signal': signal,
            'price': current_price,
            'target_price': target_price,
            'grid_levels': self.grid_levels
        }

    def _initialize_grid(self, current_price: float):
        """
        Initialise les niveaux de la grille autour du prix actuel.

        Args:
            current_price: Prix actuel du marché
        """
        grid_size = self.config['grid_size']
        grid_spacing = self.config['grid_spacing'] / \
            100.0  # Conversion en décimal

        self.grid_levels = []

        # Créer des niveaux d'achat (en dessous du prix actuel)
        for i in range(1, grid_size + 1):
            price = current_price * (1 - i * grid_spacing)
            self.grid_levels.append({
                'type': 'BUY',
                'price': price,
                'triggered': False
            })

        # Créer des niveaux de vente (au-dessus du prix actuel)
        for i in range(1, grid_size + 1):
            price = current_price * (1 + i * grid_spacing)
            self.grid_levels.append({
                'type': 'SELL',
                'price': price,
                'triggered': False
            })

        # Trier les niveaux par prix
        self.grid_levels.sort(key=lambda x: x['price'])

    def calculate_position_size(
            self,
            current_price: float,
            balance: float) -> float:
        """
        Calcule la taille de la position basée sur le capital disponible.

        Args:
            current_price: Prix actuel du marché
            balance: Solde disponible pour le trading

        Returns:
            Taille de la position dans l'unité de base
        """
        max_position_value = balance * self.config['max_position_size']
        position_size = max_position_value / current_price
        return position_size

    def calculate_stop_loss(self, entry_price: float, signal: str) -> float:
        """
        Calcule le prix du stop loss.

        Args:
            entry_price: Prix d'entrée de la position
            signal: Type de signal ('BUY' ou 'SELL')

        Returns:
            Prix du stop loss
        """
        if signal == 'BUY':
            return entry_price * (1 - self.config['stop_loss'] / 100.0)
        else:  # SELL
            return entry_price * (1 + self.config['stop_loss'] / 100.0)

    def calculate_take_profit(self, entry_price: float, signal: str) -> float:
        """
        Calcule le prix du take profit.

        Args:
            entry_price: Prix d'entrée de la position
            signal: Type de signal ('BUY' ou 'SELL')

        Returns:
            Prix du take profit
        """
        if signal == 'BUY':
            return entry_price * (1 + self.config['take_profit'] / 100.0)
        else:  # SELL
            return entry_price * (1 - self.config['take_profit'] / 100.0)
