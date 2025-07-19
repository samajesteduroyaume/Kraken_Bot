from dataclasses import dataclass
from typing import Dict, Optional
import logging
from datetime import datetime

logger = logging.getLogger('kraken_position')


@dataclass
class Position:
    """Représente une position de trading."""
    pair: str
    size: float
    entry_price: float
    entry_time: datetime
    exit_price: Optional[float] = None
    exit_time: Optional[datetime] = None
    leverage: float = 1.0
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    fee_rate: float = 0.0026  # 0.26% par défaut

    def __post_init__(self):
        """Initialisation après la création de l'objet."""
        self._validate()

    def _validate(self) -> None:
        """Valide les paramètres de la position."""
        if self.size <= 0:
            raise ValueError("La taille de la position doit être positive")

        if self.entry_price <= 0:
            raise ValueError("Le prix d'entrée doit être positif")

        if self.leverage < 1:
            raise ValueError("Le levier doit être supérieur ou égal à 1")

        if self.stop_loss is not None and self.stop_loss <= 0:
            raise ValueError("Le stop-loss doit être positif")

        if self.take_profit is not None and self.take_profit <= 0:
            raise ValueError("Le take-profit doit être positif")

    @property
    def is_long(self) -> bool:
        """Retourne True si la position est longue."""
        return self.size > 0

    @property
    def is_short(self) -> bool:
        """Retourne True si la position est courte."""
        return self.size < 0

    @property
    def is_closed(self) -> bool:
        """Retourne True si la position est fermée."""
        return self.exit_time is not None

    def calculate_pnl(self, current_price: float) -> float:
        """
        Calcule le P&L de la position.

        Args:
            current_price: Prix actuel

        Returns:
            P&L de la position
        """
        if self.is_long:
            pnl = (current_price - self.entry_price) * self.size
        else:
            pnl = (self.entry_price - current_price) * abs(self.size)

        # Application des frais
        pnl -= self.calculate_fees(current_price)
        return pnl

    def calculate_fees(self, current_price: float) -> float:
        """
        Calcule les frais totaux de la position.

        Args:
            current_price: Prix actuel

        Returns:
            Frais totaux
        """
        entry_fee = self.entry_price * abs(self.size) * self.fee_rate
        exit_fee = current_price * abs(self.size) * self.fee_rate
        return entry_fee + exit_fee

    def close(self, exit_price: float, exit_time: datetime = None) -> float:
        """
        Ferme la position.

        Args:
            exit_price: Prix de sortie
            exit_time: Heure de sortie

        Returns:
            P&L de la position
        """
        if self.is_closed:
            raise ValueError("La position est déjà fermée")

        self.exit_price = exit_price
        self.exit_time = exit_time or datetime.now()

        pnl = self.calculate_pnl(exit_price)
        logger.info(f"Position fermée - P&L: {pnl:.2f}")
        return pnl

    def check_stop_loss(self, current_price: float) -> bool:
        """
        Vérifie si le stop-loss est atteint.

        Args:
            current_price: Prix actuel

        Returns:
            True si le stop-loss est atteint
        """
        if self.stop_loss is None:
            return False

        if self.is_long and current_price <= self.stop_loss:
            return True

        if self.is_short and current_price >= self.stop_loss:
            return True

        return False

    def check_take_profit(self, current_price: float) -> bool:
        """
        Vérifie si le take-profit est atteint.

        Args:
            current_price: Prix actuel

        Returns:
            True si le take-profit est atteint
        """
        if self.take_profit is None:
            return False

        if self.is_long and current_price >= self.take_profit:
            return True

        if self.is_short and current_price <= self.take_profit:
            return True

        return False

    def to_dict(self) -> Dict:
        """Convertit la position en dictionnaire."""
        return {
            'pair': self.pair,
            'size': self.size,
            'entry_price': self.entry_price,
            'entry_time': self.entry_time.isoformat(),
            'exit_price': self.exit_price,
            'exit_time': self.exit_time.isoformat() if self.exit_time else None,
            'leverage': self.leverage,
            'stop_loss': self.stop_loss,
            'take_profit': self.take_profit,
            'fee_rate': self.fee_rate}
