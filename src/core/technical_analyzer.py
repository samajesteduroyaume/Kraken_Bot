from typing import Dict, List, Optional, Union
import pandas as pd
import numpy as np
from ta import (
    momentum, volatility, trend, volume
)
from dataclasses import dataclass
from enum import Enum
import logging
from datetime import datetime
import json
from pathlib import Path

# Configuration des dossiers
INDICATORS_CONFIG_DIR = Path("config/indicators")
INDICATORS_CONFIG_DIR.mkdir(parents=True, exist_ok=True)

class IndicatorType(Enum):
    TREND = "trend"
    MOMENTUM = "momentum"
    VOLATILITY = "volatility"
    VOLUME = "volume"
    CUSTOM = "custom"

@dataclass
class IndicatorConfig:
    """Configuration d'un indicateur technique."""
    name: str
    indicator_type: IndicatorType
    params: dict
    enabled: bool = True
    display_name: Optional[str] = None
    description: Optional[str] = None
    
    def to_dict(self) -> dict:
        return {
            'name': self.name,
            'type': self.indicator_type.value,
            'params': self.params,
            'enabled': self.enabled,
            'display_name': self.display_name or self.name,
            'description': self.description or ""
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'IndicatorConfig':
        return cls(
            name=data['name'],
            indicator_type=IndicatorType(data['type']),
            params=data.get('params', {}),
            enabled=data.get('enabled', True),
            display_name=data.get('display_name'),
            description=data.get('description')
        )

class IndicatorManager:
    """Gestionnaire des indicateurs techniques."""
    
    def __init__(self, config_file: Optional[str] = None):
        self.indicators: Dict[str, IndicatorConfig] = {}
        self.config_file = config_file or INDICATORS_CONFIG_DIR / "default_indicators.json"
        self.logger = logging.getLogger('technical_analyzer.IndicatorManager')
        self._load_default_indicators()
        
    def _load_default_indicators(self):
        """Charge les indicateurs par défaut."""
        default_indicators = [
            # Indicateurs de tendance
            IndicatorConfig(
                name="sma_20",
                indicator_type=IndicatorType.TREND,
                params={"window": 20},
                display_name="SMA 20",
                description="Moyenne mobile simple sur 20 périodes"
            ),
            IndicatorConfig(
                name="ema_50",
                indicator_type=IndicatorType.TREND,
                params={"window": 50},
                display_name="EMA 50",
                description="Moyenne mobile exponentielle sur 50 périodes"
            ),
            
            # Indicateurs de momentum
            IndicatorConfig(
                name="rsi_14",
                indicator_type=IndicatorType.MOMENTUM,
                params={"window": 14},
                display_name="RSI 14",
                description="Relative Strength Index sur 14 périodes"
            ),
            
            # Indicateurs de volatilité
            IndicatorConfig(
                name="bollinger_bands",
                indicator_type=IndicatorType.VOLATILITY,
                params={"window": 20, "window_dev": 2},
                display_name="Bandes de Bollinger",
                description="Bandes de Bollinger (20, 2)"
            ),
            
            # Indicateurs de volume
            IndicatorConfig(
                name="volume_ma_20",
                indicator_type=IndicatorType.VOLUME,
                params={"window": 20},
                display_name="Volume MA 20",
                description="Moyenne mobile du volume sur 20 périodes"
            )
        ]
        
        for indicator in default_indicators:
            self.add_indicator(indicator)
    
    def add_indicator(self, indicator: IndicatorConfig):
        """Ajoute un indicateur à la configuration."""
        self.indicators[indicator.name] = indicator
    
    def remove_indicator(self, name: str) -> bool:
        """Supprime un indicateur de la configuration."""
        if name in self.indicators:
            del self.indicators[name]
            return True
        return False
    
    def get_indicator(self, name: str) -> Optional[IndicatorConfig]:
        """Récupère un indicateur par son nom."""
        return self.indicators.get(name)
    
    def enable_indicator(self, name: str, enable: bool = True) -> bool:
        """Active ou désactive un indicateur."""
        if name in self.indicators:
            self.indicators[name].enabled = enable
            return True
        return False
    
    def save_config(self, filepath: Optional[str] = None) -> bool:
        """Sauvegarde la configuration des indicateurs dans un fichier."""
        filepath = filepath or self.config_file
        try:
            config = {
                'indicators': [ind.to_dict() for ind in self.indicators.values()],
                'last_updated': datetime.now().isoformat()
            }
            with open(filepath, 'w') as f:
                json.dump(config, f, indent=2)
            return True
        except Exception as e:
            self.logger.error(f"Erreur lors de la sauvegarde de la configuration: {e}")
            return False
    
    def load_config(self, filepath: Optional[str] = None) -> bool:
        """Charge la configuration des indicateurs depuis un fichier."""
        filepath = filepath or self.config_file
        try:
            if not Path(filepath).exists():
                self.logger.warning(f"Fichier de configuration {filepath} non trouvé. Utilisation des valeurs par défaut.")
                return False
                
            with open(filepath, 'r') as f:
                config = json.load(f)
            
            self.indicators.clear()
            for ind_data in config.get('indicators', []):
                try:
                    indicator = IndicatorConfig.from_dict(ind_data)
                    self.add_indicator(indicator)
                except Exception as e:
                    self.logger.error(f"Erreur lors du chargement de l'indicateur {ind_data.get('name')}: {e}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Erreur lors du chargement de la configuration: {e}")
            return False


class TechnicalAnalyzer:
    """
    Analyse technique des données de marché avec support pour de nombreux indicateurs.
    
    Gère le calcul, la mise en cache et la configuration des indicateurs techniques.
    """

    def __init__(self, config_file: Optional[str] = None):
        """
        Initialise l'analyseur technique.
        
        Args:
            config_file: Chemin vers un fichier de configuration d'indicateurs personnalisé
        """
        self.logger = logging.getLogger('technical_analyzer')
        self.indicator_manager = IndicatorManager(config_file)
        self._cache = {}
        
        # Charger la configuration des indicateurs
        self.indicator_manager.load_config()
        
        self.logger.info(f"Analyseur technique initialisé avec {len(self.indicator_manager.indicators)} indicateurs")

    def _get_cached(self, key: str, func: callable, *args, **kwargs):
        """
        Récupère une valeur depuis le cache ou la calcule si nécessaire.
        
        Args:
            key: Clé de cache unique
            func: Fonction à exécuter si le cache est vide
            *args, **kwargs: Arguments à passer à la fonction
            
        Returns:
            Résultat du calcul ou du cache
        """
        if key in self._cache:
            return self._cache[key]
            
        result = func(*args, **kwargs)
        self._cache[key] = result
        return result
        
    def clear_cache(self):
        """Vide le cache des indicateurs calculés."""
        self._cache.clear()
    
    def _validate_price_data(self, price_data: pd.Series):
        """Valide les données de prix d'entrée."""
        if not isinstance(price_data, pd.Series):
            raise ValueError("price_data doit être une pandas Series")
            
        if len(price_data) < 30:  # Minimum pour la plupart des indicateurs
            raise ValueError("Pas assez de données (minimum 30 périodes requises)")
            
        if price_data.isnull().any():
            raise ValueError("Les données de prix contiennent des valeurs manquantes")
    
    def calculate_indicators(
        self, 
        price_data: pd.Series,
        volume_data: Optional[pd.Series] = None,
        indicators: Optional[List[str]] = None,
        **kwargs
    ) -> Dict[str, pd.Series]:
        """
        Calcule les indicateurs techniques demandés.

        Args:
            price_data: Série des prix (typiquement close)
            volume_data: Série des volumes (optionnel, requis pour certains indicateurs)
            indicators: Liste des noms d'indicateurs à calculer (tous si None)
            **kwargs: Paramètres additionnels pour les indicateurs

        Returns:
            Dictionnaire {nom_indicateur: série_valeurs}
        """
        try:
            self._validate_price_data(price_data)
            
            # Créer un DataFrame pour stocker les résultats
            results = {}
            df = pd.DataFrame(index=price_data.index)
            df['price'] = price_data
            
            # Si volume est fourni, l'ajouter au DataFrame
            if volume_data is not None:
                if not isinstance(volume_data, pd.Series) or len(volume_data) != len(price_data):
                    raise ValueError("volume_data doit être une pandas Series de même longueur que price_data")
                df['volume'] = volume_data
            
            # Déterminer quels indicateurs calculer
            indicators_to_calculate = indicators or list(self.indicator_manager.indicators.keys())
            
            # Calculer chaque indicateur demandé
            for indicator_name in indicators_to_calculate:
                indicator = self.indicator_manager.get_indicator(indicator_name)
                if not indicator or not indicator.enabled:
                    continue
                    
                try:
                    if indicator.indicator_type == IndicatorType.TREND:
                        results.update(self._calculate_trend_indicator(df, indicator, **kwargs))
                    elif indicator.indicator_type == IndicatorType.MOMENTUM:
                        results.update(self._calculate_momentum_indicator(df, indicator, **kwargs))
                    elif indicator.indicator_type == IndicatorType.VOLATILITY:
                        results.update(self._calculate_volatility_indicator(df, indicator, **kwargs))
                    elif indicator.indicator_type == IndicatorType.VOLUME and 'volume' in df.columns:
                        results.update(self._calculate_volume_indicator(df, indicator, **kwargs))
                except Exception as e:
                    self.logger.error(f"Erreur lors du calcul de l'indicateur {indicator_name}: {e}")
            
            return results
            
        except Exception as e:
            self.logger.error(f"Erreur dans calculate_indicators: {e}")
            raise
            df['macd_signal'] = trend.MACD(df['price']).macd_signal()

            # Indicateurs de volume
            df['obv'] = volume.OnBalanceVolumeIndicator(
                df['price'], df['price']
            ).on_balance_volume()

            # Nettoyer les NaN
            indicators = df.dropna().to_dict('series')

            return indicators

        except Exception as e:
            self.logger.error(
                f"Erreur lors du calcul des indicateurs: {str(e)}")
            return {}

    def analyze_trend(self, price_history: pd.Series) -> str:
        """
        Analyse la tendance du marché.

        Args:
            price_history: Historique des prix

        Returns:
            'bullish', 'bearish' ou 'neutral'
        """
        try:
            # Calculer les EMAs
            ema_short = trend.EMAIndicator(
                price_history, window=12).ema_indicator()
            ema_long = trend.EMAIndicator(
                price_history, window=26).ema_indicator()

            # Analyser la tendance
            if ema_short.iloc[-1] > ema_long.iloc[-1]:
                return 'bullish'
            elif ema_short.iloc[-1] < ema_long.iloc[-1]:
                return 'bearish'
            else:
                return 'neutral'

        except Exception as e:
            self.logger.error(
                f"Erreur lors de l'analyse de la tendance: {str(e)}")
            return 'neutral'

    def analyze_volatility(self, price_history: pd.Series) -> float:
        """
        Analyse la volatilité du marché.

        Args:
            price_history: Historique des prix

        Returns:
            Score de volatilité (0-1)
        """
        try:
            # Calculer le RSI
            rsi = momentum.RSIIndicator(price_history, window=14).rsi()

            # Calculer la volatilité
            volatility = price_history.pct_change().rolling(window=20).std()

            # Normaliser la volatilité
            normalized_volatility = volatility.iloc[-1] / volatility.mean()

            return float(normalized_volatility)

        except Exception as e:
            self.logger.error(
                f"Erreur lors de l'analyse de la volatilité: {str(e)}")
            return 0.0
            
    def _calculate_trend_indicator(self, df: pd.DataFrame, indicator: IndicatorConfig, **kwargs) -> Dict[str, pd.Series]:
        """
        Calcule les indicateurs de tendance.
        
        Args:
            df: DataFrame contenant les données de prix
            indicator: Configuration de l'indicateur
            
        Returns:
            Dictionnaire avec les séries calculées
        """
        results = {}
        
        if indicator.name.startswith('sma_'):
            window = indicator.params.get('window', 20)
            sma = trend.SMAIndicator(close=df['price'], window=window).sma_indicator()
            results[f'sma_{window}'] = sma
            
        elif indicator.name.startswith('ema_'):
            window = indicator.params.get('window', 50)
            ema = trend.EMAIndicator(close=df['price'], window=window).ema_indicator()
            results[f'ema_{window}'] = ema
            
        elif indicator.name == 'macd':
            window_slow = indicator.params.get('window_slow', 26)
            window_fast = indicator.params.get('window_fast', 12)
            window_sign = indicator.params.get('window_sign', 9)
            
            macd = trend.MACD(
                close=df['price'],
                window_slow=window_slow,
                window_fast=window_fast,
                window_sign=window_sign
            )
            
            results['macd_line'] = macd.macd()
            results['macd_signal'] = macd.macd_signal()
            results['macd_diff'] = macd.macd_diff()
            
        return results
        
    def _calculate_momentum_indicator(self, df: pd.DataFrame, indicator: IndicatorConfig, **kwargs) -> Dict[str, pd.Series]:
        """
        Calcule les indicateurs de momentum.
        
        Args:
            df: DataFrame contenant les données de prix
            indicator: Configuration de l'indicateur
            
        Returns:
            Dictionnaire avec les séries calculées
        """
        results = {}
        
        if indicator.name.startswith('rsi_'):
            window = indicator.params.get('window', 14)
            rsi = momentum.RSIIndicator(close=df['price'], window=window).rsi()
            results[f'rsi_{window}'] = rsi
            
        elif indicator.name == 'stoch_rsi':
            window = indicator.params.get('window', 14)
            smooth1 = indicator.params.get('smooth1', 3)
            smooth2 = indicator.params.get('smooth2', 3)
            
            stoch_rsi = momentum.StochRSIIndicator(
                close=df['price'],
                window=window,
                smooth1=smooth1,
                smooth2=smooth2
            )
            
            results['stoch_rsi'] = stoch_rsi.stochrsi()
            results['stoch_rsi_k'] = stoch_rsi.stochrsi_k()
            results['stoch_rsi_d'] = stoch_rsi.stochrsi_d()
            
        return results
        
    def _calculate_volatility_indicator(self, df: pd.DataFrame, indicator: IndicatorConfig, **kwargs) -> Dict[str, pd.Series]:
        """
        Calcule les indicateurs de volatilité.
        
        Args:
            df: DataFrame contenant les données de prix
            indicator: Configuration de l'indicateur
            
        Returns:
            Dictionnaire avec les séries calculées
        """
        results = {}
        
        if indicator.name == 'bollinger_bands':
            window = indicator.params.get('window', 20)
            window_dev = indicator.params.get('window_dev', 2)
            
            bb = volatility.BollingerBands(
                close=df['price'],
                window=window,
                window_dev=window_dev
            )
            
            results['bb_upper'] = bb.bollinger_hband()
            results['bb_middle'] = bb.bollinger_mavg()
            results['bb_lower'] = bb.bollinger_lband()
            results['bb_width'] = bb.bollinger_wband()
            results['bb_pct'] = bb.bollinger_pband()
            
        elif indicator.name == 'atr':
            window = indicator.params.get('window', 14)
            high = kwargs.get('high', df['price'] * 1.01)  # Valeur par défaut si non fournie
            low = kwargs.get('low', df['price'] * 0.99)    # Valeur par défaut si non fournie
            
            atr = volatility.AverageTrueRange(
                high=high,
                low=low,
                close=df['price'],
                window=window
            ).average_true_range()
            
            results['atr'] = atr
            
        return results
        
    def _calculate_volume_indicator(self, df: pd.DataFrame, indicator: IndicatorConfig, **kwargs) -> Dict[str, pd.Series]:
        """
        Calcule les indicateurs de volume.
        
        Args:
            df: DataFrame contenant les données de volume
            indicator: Configuration de l'indicateur
            
        Returns:
            Dictionnaire avec les séries calculées
        """
        results = {}
        
        if indicator.name == 'obv':
            obv = volume.OnBalanceVolumeIndicator(
                close=df['price'],
                volume=df['volume']
            ).on_balance_volume()
            results['obv'] = obv
            
        elif indicator.name.startswith('volume_ma_'):
            window = indicator.params.get('window', 20)
            volume_ma = df['volume'].rolling(window=window).mean()
            results[f'volume_ma_{window}'] = volume_ma
            
        elif indicator.name == 'vwap':
            if 'high' in df.columns and 'low' in df.columns and 'close' in df.columns:
                vwap = volume.VolumeWeightedAveragePrice(
                    high=df['high'],
                    low=df['low'],
                    close=df['close'],
                    volume=df['volume']
                ).volume_weighted_average_price()
                results['vwap'] = vwap
            else:
                self.logger.warning("Données manquantes pour calculer le VWAP (high, low, close requis)")
        
        return results
