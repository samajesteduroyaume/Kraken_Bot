"""
Configuration centralisée du bot de trading Kraken.
Ce fichier contient toute la configuration de l'application.
"""
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field, field_validator, ConfigDict
from pydantic_settings import BaseSettings
import os
from pathlib import Path

# Chemin de base du projet
BASE_DIR = Path(__file__).resolve().parent.parent

class Settings(BaseSettings):
    # Configuration de Pydantic pour autoriser les champs supplémentaires
    model_config = ConfigDict(
        extra="allow",  # Permet les champs supplémentaires sans erreur de validation
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        validate_default=True
    )
    
    # ===== Configuration de l'environnement =====
    ENV: str = "development"
    DEBUG: bool = False
    
    # ===== Configuration API Kraken =====
    KRAKEN_API_KEY: str = Field(
        default="test_api_key",
        json_schema_extra={"env": "KRAKEN_API_KEY"}
    )
    KRAKEN_PRIVATE_KEY: str = Field(
        default="test_private_key",
        json_schema_extra={"env": "KRAKEN_PRIVATE_KEY"}
    )
    KRAKEN_API_URL: str = Field(
        default="https://api.kraken.com",
        json_schema_extra={"env": "KRAKEN_API_URL"}
    )
    IS_TEST_ENV: bool = Field(default=False, exclude=True)  # Flag pour les environnements de test
    
    # ===== Paramètres de trading =====
    DEFAULT_PAIR: str = "XBT/USD"
    TRADING_INTERVAL: str = "15m"  # 1m, 5m, 15m, 1h, 4h, 1d
    RISK_PER_TRADE: float = 1.0  # % du capital à risquer par trade
    MAX_OPEN_TRADES: int = 5
    LEVERAGE: int = 2  # Effet de levier par défaut
    
    # ===== Paramètres de stratégie =====
    ENABLED_STRATEGIES: List[str] = ["trend_following", "mean_reversion"]
    DEFAULT_STRATEGY: str = "trend_following"
    
    # Configuration des stratégies
    STRATEGY_CONFIG: Dict[str, Dict[str, Any]] = {
        'trend_following': {
            'rsi_period': 14,
            'rsi_overbought': 70,
            'rsi_oversold': 30,
            'macd_fast': 12,
            'macd_slow': 26,
            'macd_signal': 9,
            'position_size': 0.1
        },
        'mean_reversion': {
            'bb_period': 20,
            'bb_std': 2.0,
            'rsi_period': 14,
            'rsi_overbought': 70,
            'rsi_oversold': 30,
            'position_size': 0.1
        }
    }
    
    # ===== Gestion des risques =====
    STOP_LOSS_PCT: float = 1.0  # % de stop loss
    TAKE_PROFIT_PCT: float = 2.0  # % de take profit
    MAX_DRAWDOWN_PCT: float = 10.0  # Drawdown maximum autorisé
    MAX_LEVERAGE: int = 5  # Effet de levier maximum
    
    # ===== Journalisation =====
    LOG_LEVEL: str = "INFO"
    LOG_FILE: str = "logs/kraken_bot.log"
    LOG_FORMAT: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # ===== Base de données =====
    DATABASE_URL: str = Field(
        default="sqlite:///./kraken_bot.db",
        json_schema_extra={"env": "DATABASE_URL"}
    )
    
    # ===== Configuration ML =====
    ML_ENABLED: bool = True
    ML_MODEL_PATH: str = "models"
    ML_MIN_CONFIDENCE: float = 0.6
    
    # ===== Chemins des fichiers =====
    @property
    def LOG_DIR(self) -> Path:
        return BASE_DIR / "logs"
    
    @property
    def MODELS_DIR(self) -> Path:
        return BASE_DIR / "models"
    
    @property
    def DATA_DIR(self) -> Path:
        return BASE_DIR / "data"
    
    # ===== Validation =====
    @field_validator('ENV')
    @classmethod
    def validate_env(cls, v: str) -> str:
        if v not in ["development", "testing", "production"]:
            raise ValueError("ENV doit être 'development', 'testing' ou 'production'")
        return v
    
    @field_validator('TRADING_INTERVAL')
    @classmethod
    def validate_trading_interval(cls, v: str) -> str:
        valid_intervals = ["1m", "5m", "15m", "30m", "1h", "4h", "1d"]
        if v not in valid_intervals:
            raise ValueError(f"Intervalle de trading invalide. Doit être l'un de: {valid_intervals}")
        return v

# Instance des paramètres
settings = Settings()

# Création des répertoires nécessaires
os.makedirs(settings.LOG_DIR, exist_ok=True)
os.makedirs(settings.MODELS_DIR, exist_ok=True)
os.makedirs(settings.DATA_DIR, exist_ok=True)
