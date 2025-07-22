"""
Classe de base pour les stratégies de trading avancées.
"""
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Optional, Union
import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta, timezone
from decimal import Decimal, ROUND_HALF_UP
from enum import Enum, auto
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from datetime import datetime

class SignalStrength(Enum):
    STRONG = 3
    MODERATE = 2
    WEAK = 1
    NEUTRAL = 0
    WEAK_SELL = -1
    MODERATE_SELL = -2
    STRONG_SELL = -3

@dataclass
class TradeSignal:
    """Classe pour représenter un signal de trading."""
    symbol: str
    direction: int  # 1 pour achat, -1 pour vente
    strength: SignalStrength
    price: Decimal
    stop_loss: Optional[Decimal] = None
    take_profit: Optional[Decimal] = None
    timestamp: datetime = datetime.now(timezone.utc)
    metadata: Dict[str, Any] = None

@dataclass
class PositionSizing:
    """Classe pour la gestion de la taille des positions."""
    size: Decimal
    risk_reward_ratio: Decimal
    stop_loss_pct: Decimal
    take_profit_pct: Decimal
    max_position_size: Decimal

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
        Initialise la stratégie de trading avancée.

        Args:
            config (dict, optional): Configuration de la stratégie. Par défaut None.
        """
        self.config = config or {}
        self.name = "BaseStrategy"
        self.portfolio_value = Decimal(str(self.config.get('portfolio_value', 100000.0)))
        self.risk_per_trade = Decimal(str(self.config.get('risk_per_trade', 0.01)))
        self.max_position_size = Decimal(str(self.config.get('max_position_size', 0.15)))
        self.timeframes = self.config.get('timeframes', ['15m', '1h', '4h', '1d'])
        self.primary_timeframe = self.timeframes[0]
        self.last_update = None
        self.is_real_time = self.config.get('real_time', True)
        self.last_real_time_update = datetime.now()
        self.real_time_interval = timedelta(seconds=10)
        
        # Configuration du risque avancé
        self.volatility_adjusted_risk = self.config.get('volatility_adjusted_risk', True)
        self.dynamic_position_sizing = self.config.get('dynamic_position_sizing', True)
        self.max_drawdown = Decimal(str(self.config.get('max_drawdown', 0.10)))  # 10% max de drawdown
        
        # Configuration des indicateurs
        self.indicators = {}
        self.ml_model = None
        self.meta_strategy_weights = {}
        
        # Suivi des performances
        self.trade_history = []
        self.performance_metrics = {}
        self.equity_curve = []
        self.max_equity = self.portfolio_value
        self.max_drawdown = Decimal('0')
        
        # Validation de la configuration (uniquement si la méthode n'est pas surchargée)
        if type(self).validate_config == BaseStrategy.validate_config:
            self.validate_config()
        
    # ===== Méthodes de gestion du risque avancé =====
    
    def calculate_position_size(self, entry_price: Decimal, stop_loss: Decimal) -> PositionSizing:
        """
        Calcule la taille de position optimale en fonction du risque et de la volatilité.
        
        Args:
            entry_price: Prix d'entrée proposé
            stop_loss: Niveau de stop-loss proposé
            
        Returns:
            PositionSizing: Détails de la taille de position calculée
        """
        if entry_price <= Decimal('0'):
            raise ValueError("Le prix d'entrée doit être supérieur à zéro")
            
        # Calcul du risque en pourcentage du prix
        risk_percent = (abs(entry_price - stop_loss) / entry_price).quantize(Decimal('0.0001'))
        
        if risk_percent <= Decimal('0'):
            # Éviter la division par zéro
            risk_percent = Decimal('0.01')  # 1% par défaut
            
        # Taille de position basée sur le risque
        risk_amount = self.portfolio_value * self.risk_per_trade
        position_value = (risk_amount / risk_percent).quantize(Decimal('0.01'))
        
        # Limiter à la taille de position maximale
        max_position_value = self.portfolio_value * self.max_position_size
        position_value = min(position_value, max_position_value)
        
        # Calcul du take-profit basé sur le ratio risque/rendement
        risk_reward_ratio = Decimal(str(self.config.get('risk_reward_ratio', 2.0)))
        take_profit_pct = risk_percent * risk_reward_ratio
        
        return PositionSizing(
            size=position_value,
            risk_reward_ratio=risk_reward_ratio,
            stop_loss_pct=risk_percent,
            take_profit_pct=take_profit_pct,
            max_position_size=max_position_value
        )
    
    def update_equity_curve(self, pnl: Decimal) -> None:
        """
        Met à jour la courbe d'équité et calcule les métriques de performance.
        
        Args:
            pnl: Profit ou perte du dernier trade
        """
        self.portfolio_value += pnl
        self.equity_curve.append({
            'timestamp': datetime.utcnow(),
            'equity': float(self.portfolio_value),
            'pnl': float(pnl)
        })
        
        # Mise à jour du drawdown
        self.max_equity = max(self.max_equity, self.portfolio_value)
        if self.max_equity > 0:
            drawdown = (self.max_equity - self.portfolio_value) / self.max_equity
            self.max_drawdown = max(self.max_drawdown, drawdown)
    
    # ===== Méthodes de génération de signaux =====
    
    @abstractmethod
    def generate_signals(self, data: Dict[str, pd.DataFrame]) -> List[TradeSignal]:
        """
        Méthode abstraite pour générer des signaux de trading.
        
        Args:
            data: Dictionnaire de DataFrames avec les données de marché par timeframe
            
        Returns:
            Liste des signaux de trading générés
        """
        pass
    
    def combine_signals(self, signals: List[TradeSignal]) -> Optional[TradeSignal]:
        """
        Combine plusieurs signaux en un seul signal consolidé.
        
        Args:
            signals: Liste des signaux à combiner
            
        Returns:
            Signal consolidé ou None si aucun signal clair
        """
        if not signals:
            return None
            
        # Compter les signaux par direction
        buy_count = sum(1 for s in signals if s.direction > 0)
        sell_count = sum(1 for s in signals if s.direction < 0)
        neutral_count = len(signals) - buy_count - sell_count
        
        # Décider de la direction globale
        if buy_count > sell_count and buy_count > neutral_count:
            direction = 1
            strength = max(s.strength.value for s in signals if s.direction > 0)
        elif sell_count > buy_count and sell_count > neutral_count:
            direction = -1
            strength = min(s.strength.value for s in signals if s.direction < 0)
        else:
            return None
            
        # Créer un nouveau signal consolidé
        avg_price = sum(float(s.price) for s in signals) / len(signals)
        return TradeSignal(
            symbol=signals[0].symbol,
            direction=direction,
            strength=SignalStrength(strength),
            price=Decimal(str(avg_price))
        )
    
    # ===== Méthodes d'intégration ML =====
    
    def set_ml_model(self, model) -> None:
        """
        Définit le modèle de machine learning à utiliser.
        
        Args:
            model: Modèle de ML compatible avec scikit-learn (doit implémenter predict_proba)
        """
        self.ml_model = model
        
    def predict_with_ml(self, features: np.ndarray) -> np.ndarray:
        """
        Effectue une prédiction avec le modèle ML.
        
        Args:
            features: Tableau de caractéristiques pour la prédiction
            
        Returns:
            Prédictions du modèle
        """
        if self.ml_model is None:
            raise ValueError("Aucun modèle ML n'a été défini")
            
        try:
            return self.ml_model.predict_proba(features)
        except Exception as e:
            logger.error(f"Erreur lors de la prédiction ML: {e}")
            raise
    
    # ===== Méthodes de métadonnées et de journalisation =====
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convertit la configuration de la stratégie en dictionnaire.
        
        Returns:
            Dictionnaire de configuration
        """
        return {
            'name': self.name,
            'portfolio_value': float(self.portfolio_value),
            'risk_per_trade': float(self.risk_per_trade),
            'max_position_size': float(self.max_position_size),
            'timeframes': self.timeframes,
            'is_real_time': self.is_real_time,
            'volatility_adjusted_risk': self.volatility_adjusted_risk,
            'dynamic_position_sizing': self.dynamic_position_sizing,
            'max_drawdown': float(self.max_drawdown)
        }

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
        Valide la configuration de la stratégie avancée.
        """
        # Validation des paramètres de risque
        if not (Decimal('0') < self.risk_per_trade <= Decimal('1')):
            raise ValueError("Le risque par trade doit être un nombre entre 0 et 1")

        if not (Decimal('0') < self.max_position_size <= Decimal('1')):
            raise ValueError("La taille maximale de position doit être un nombre entre 0 et 1")

        if not self.timeframes:
            raise ValueError("Au moins un timeframe doit être spécifié pour la stratégie")
            
        if not isinstance(self.timeframes, list) or not all(isinstance(tf, str) for tf in self.timeframes):
            raise ValueError("Les timeframes doivent être une liste de chaînes de caractères")
            
        # Validation des paramètres de gestion du risque
        if not (Decimal('0') <= self.max_drawdown < Decimal('1')):
            raise ValueError("Le drawdown maximum doit être un nombre entre 0 et 1")

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
