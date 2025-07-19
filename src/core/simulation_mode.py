from enum import Enum
from typing import Optional
from datetime import datetime, timedelta
import pandas as pd
from .ml_predictor import MLPredictor
from .technical_analyzer import TechnicalAnalyzer


class SimulationMode(Enum):
    """Modes de simulation disponibles."""
    HISTORICAL = "historical"  # Utilise uniquement les données historiques
    REAL_TIME = "real_time"     # Trading en temps réel
    BACKTEST = "backtest"      # Backtesting avec données historiques et prédictions
    HYBRID = "hybrid"          # Mélange de données historiques et temps réel


class SimulationConfig:
    """Configuration de la simulation."""

    def __init__(
        self,
        mode: SimulationMode,
        historical_days: int = 30,
        real_time_window: timedelta = timedelta(hours=24),
        prediction_window: int = 1,  # Nombre de jours à prédire
        ml_predictor: Optional[MLPredictor] = None,
        technical_analyzer: Optional[TechnicalAnalyzer] = None,
        available_pairs: Optional[list] = None
    ):
        self.mode = mode
        self.historical_days = historical_days
        self.real_time_window = real_time_window
        self.prediction_window = prediction_window
        self.ml_predictor = ml_predictor or MLPredictor()
        self.technical_analyzer = technical_analyzer or TechnicalAnalyzer()
        self.available_pairs = available_pairs or []
        self.adaptive_leverage = None
        self.backtest_manager = None
        self.sentiment_analyzer = None
        self.signal_generator = None
        self.database_manager = None
        self.metrics_manager = None
        self.utils = None

    def get_time_range(self) -> tuple[datetime, datetime]:
        """Retourne la plage de temps pour la simulation."""
        now = datetime.now()
        if self.mode == SimulationMode.HISTORICAL:
            return now - timedelta(days=self.historical_days), now
        elif self.mode == SimulationMode.REAL_TIME:
            return now - self.real_time_window, now
        elif self.mode == SimulationMode.BACKTEST:
            return now - timedelta(days=self.historical_days), now + \
                timedelta(days=self.prediction_window)
        elif self.mode == SimulationMode.HYBRID:
            return now - timedelta(days=self.historical_days), now + \
                timedelta(days=self.prediction_window)
        return now - timedelta(days=self.historical_days), now

    def get_prediction_range(self) -> tuple[datetime, datetime]:
        """Retourne la plage de temps pour les prédictions."""
        now = datetime.now()
        return now, now + timedelta(days=self.prediction_window)

    def get_data_splits(self,
                        data: pd.DataFrame) -> tuple[pd.DataFrame,
                                                     pd.DataFrame,
                                                     pd.DataFrame]:
        """Divise les données en train, validation et test."""
        if len(data) < 3:
            raise ValueError("Pas assez de données pour la simulation")

        train_size = int(len(data) * 0.7)
        val_size = int(len(data) * 0.2)

        return (
            data.iloc[:train_size],
            data.iloc[train_size:train_size + val_size],
            data.iloc[train_size + val_size:]
        )
