"""
Classe de base pour les stratégies de trading.
"""
from abc import ABC, abstractmethod
from typing import Dict, List
import pandas as pd
import logging
from decimal import Decimal, ROUND_HALF_UP

logger = logging.getLogger(__name__)


class BaseStrategy(ABC):
    """
    Classe abstraite de base pour toutes les stratégies de trading.

    Attributes:
        name (str): Nom de la stratégie
        config (dict): Configuration de la stratégie
        indicators (dict): Dictionnaire des indicateurs calculés
        timeframes (list): Liste des timeframes utilisés
        required_columns (list): Colonnes de données requises
        portfolio_value (float): Valeur du portefeuille
        risk_per_trade (float): Risque maximal par trade
        max_position_size (float): Taille maximale de position
    """

    def __init__(self, config: dict = None):
        """
        Initialise la stratégie de trading.

        Args:
            config (dict, optional): Configuration de la stratégie. Par défaut None.
        """
        self.config = config or {}
        self.name = "BaseStrategy"
        self.portfolio_value = self.config.get('portfolio_value', 100000.0)
        self.risk_per_trade = self.config.get('risk_per_trade', 0.01)
        self.max_position_size = self.config.get('max_position_size', 0.15)
        self.timeframes = ['15m', '1h', '4h', '1d']
        self.last_update = None
        self.is_real_time = self.config.get(
            'real_time', True)  # Mode temps réel par défaut
        self.last_real_time_update = datetime.datetime.now()
        # Intervalle de mise à jour en temps réel
        self.real_time_interval = timedelta(seconds=10)

        # Validation de la configuration
        self.validate_config()

    async def update_real_time(self, market_data: Dict[str, Dict]) -> None:
        """
        Met à jour les données en temps réel de la stratégie.

        Args:
            market_data (Dict[str, Dict]): Données de marché en temps réel
        """
        if not self.is_real_time:
            return

        current_time = datetime.datetime.now()
        if current_time - self.last_real_time_update < self.real_time_interval:
            return

        try:
            # Mise à jour des indicateurs en temps réel
            for tf, data in market_data.items():
                if tf in self.timeframes:
                    df = pd.DataFrame(data['candles'])
                    df.set_index('timestamp', inplace=True)

                    # Mise à jour des indicateurs
                    self.calculate_indicators({tf: df}, data.get('order_book'))

                    # Génération des signaux si nécessaire
                    if tf == self.timeframes[0]:  # Timeframe principale
                        self.generate_signals({tf: df}, data.get('order_book'))

            self.last_real_time_update = current_time

        except Exception as e:
            logger.error(f"Erreur lors de la mise à jour en temps réel: {e}")

    def set_real_time_mode(self, enabled: bool) -> None:
        """
        Active ou désactive le mode temps réel.

        Args:
            enabled (bool): True pour activer le mode temps réel
        """
        self.is_real_time = enabled
        logger.info(
            f"Mode temps réel {'activé' if enabled else 'désactivé'} pour {self.name}")

    def validate_config(self):
        """
        Valide la configuration de la stratégie.
        """
        if not isinstance(self.risk_per_trade, (float, int)
                          ) or not 0 < self.risk_per_trade <= 1:
            raise ValueError(
                "Le risque par trade doit être un nombre entre 0 et 1")

        if not isinstance(self.max_position_size, (float, int)
                          ) or not 0 < self.max_position_size <= 1:
            raise ValueError(
                "La taille maximale de position doit être un nombre entre 0 et 1")

        if not isinstance(self.portfolio_value, (float, int)
                          ) or self.portfolio_value <= 0:
            raise ValueError(
                "La valeur du portefeuille doit être un nombre positif")

    def validate_data(self, data: Dict[str, pd.DataFrame]) -> bool:
        """
        Vérifie que les données fournies sont valides pour cette stratégie.

        Args:
            data (Dict[str, pd.DataFrame]): Données OHLCV par timeframe

        Returns:
            bool: True si les données sont valides, False sinon
        """
        if not isinstance(data, dict):
            logger.error("Les données doivent être un dictionnaire")
            return False

        for tf in self.timeframes:
            if tf not in data:
                logger.error(f"Données manquantes pour le timeframe {tf}")
                return False

            df = data[tf]
            if not isinstance(df, pd.DataFrame):
                logger.error(
                    f"Les données pour {tf} doivent être un DataFrame")
                return False

            missing_cols = [
                col for col in self.required_columns if col not in df.columns]
            if missing_cols:
                logger.error(
                    f"Colonnes manquantes dans les données pour {tf}: {missing_cols}")
                return False

            if len(df) < self.get_minimum_required_bars(tf):
                logger.error(f"Nombre insuffisant de données pour {tf}")
                return False

        return True

    def get_minimum_required_bars(self, timeframe: str) -> int:
        """
        Calcule le nombre minimum de bars requis pour une timeframe.

        Args:
            timeframe (str): Timeframe à analyser

        Returns:
            int: Nombre minimum de bars requis
        """
        # Par défaut, on prend la période la plus longue parmi les indicateurs
        periods = [
            self.config.get('rsi_period', 14),
            self.config.get('macd_slow', 26),
            self.config.get('sma_period', 50),
            self.config.get('atr_period', 14)
        ]
        return max(periods) + 10  # Ajout d'un tampon de 10 bars

    def calculate_position_size(
            self,
            current_price: float,
            stop_loss: float) -> float:
        """
        Calcule la taille optimale de la position en fonction du risque.

        Args:
            current_price (float): Prix actuel
            stop_loss (float): Niveau de stop loss

        Returns:
            float: Taille optimale de la position
        """
        if current_price <= 0 or stop_loss <= 0:
            raise ValueError("Les prix doivent être positifs")

        risk_amount = self.portfolio_value * self.risk_per_trade
        potential_risk = abs(current_price - stop_loss)

        if potential_risk == 0:
            raise ValueError("Le risque potentiel ne peut pas être nul")

        position_size = (risk_amount / potential_risk)
        position_size = min(
            position_size,
            self.portfolio_value *
            self.max_position_size)

        # Arrondi à 8 décimales pour la précision
        return float(
            Decimal(
                str(position_size)).quantize(
                Decimal('0.00000001'),
                rounding=ROUND_HALF_UP))

    def validate_data(self, data: Dict[str, pd.DataFrame]) -> bool:
        """
        Vérifie que les données fournies sont valides pour cette stratégie.

        Args:
            data (Dict[str, pd.DataFrame]): Données OHLCV par timeframe

        Returns:
            bool: True si les données sont valides, False sinon
        """
        if not data:
            logger.error("Aucune donnée fournie")
            return False

        for tf in self.timeframes:
            if tf not in data:
                logger.error(f"Données manquantes pour le timeframe {tf}")
                return False

            df = data[tf]
            missing_cols = [
                col for col in self.required_columns if col not in df.columns]
            if missing_cols:
                logger.error(
                    f"Colonnes manquantes dans les données: {missing_cols}")
                return False

        return True

    @abstractmethod
    def calculate_indicators(
            self, data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """
        Calcule les indicateurs techniques nécessaires pour la stratégie.

        Args:
            data (Dict[str, pd.DataFrame]): Données OHLCV par timeframe

        Returns:
            Dict[str, pd.DataFrame]: Données avec les indicateurs ajoutés
        """

    @abstractmethod
    def generate_signals(self, data: Dict[str, pd.DataFrame]) -> pd.Series:
        """
        Génère les signaux de trading basés sur la stratégie.

        Args:
            data (Dict[str, pd.DataFrame]): Données avec indicateurs calculés

        Returns:
            pd.Series: Série contenant les signaux (1 = Achat, -1 = Vente, 0 = Neutre)
        """

    def calculate_confidence(
            self, data: Dict[str, pd.DataFrame], signals: pd.Series) -> pd.Series:
        """
        Calcule un score de confiance pour chaque signal.

        Args:
            data (Dict[str, pd.DataFrame]): Données avec indicateurs
            signals (pd.Series): Signaux générés

        Returns:
            pd.Series: Score de confiance pour chaque signal (0.0 à 1.0)
        """
        confidence = pd.Series(0.0, index=signals.index)

        for tf in self.timeframes:
            tf_data = data[tf]

            # Score basé sur la force de la tendance
            if 'trend_strength' in tf_data.columns:
                confidence += tf_data['trend_strength']

            # Score basé sur la confirmation multi-timeframe
            if 'signal_confirmation' in tf_data.columns:
                confidence += tf_data['signal_confirmation']

            # Score basé sur la volatilité
            if 'atr' in tf_data.columns:
                atr_score = 1.0 - (tf_data['atr'] / tf_data['close'])
                confidence += atr_score.clip(0, 1)

        # Normalisation entre 0 et 1
        max_conf = len(self.timeframes) * 3  # 3 composants par timeframe
        confidence = confidence / max_conf

        return confidence.clip(0, 1)

    def get_required_timeframes(self) -> List[str]:
        """
        Retourne la liste des timeframes nécessaires pour cette stratégie.

        Returns:
            List[str]: Liste des timeframes requis
        """
        return self.timeframes

    def get_required_columns(self) -> List[str]:
        """
        Retourne la liste des colonnes de données requises.

        Returns:
            List[str]: Liste des noms de colonnes requis
        """
        return self.required_columns

    def __str__(self) -> str:
        """Représentation en chaîne de la stratégie."""
        return f"{self.name}Strategy"

    def __repr__(self) -> str:
        """Représentation officielle de la stratégie."""
        return f"<{self.__class__.__name__} config={self.config}>"
