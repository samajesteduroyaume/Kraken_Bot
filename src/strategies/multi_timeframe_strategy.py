"""
Module implémentant une stratégie de trading multi-timeframes.

Cette stratégie analyse plusieurs horizons temporels pour générer des signaux
plus robustes en combinant les analyses de différents timeframes.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta

@dataclass
class TimeframeConfig:
    """Configuration pour un timeframe spécifique."""
    interval: str
    weight: float
    indicators: List[str]
    
    @classmethod
    def from_dict(cls, config: Dict[str, Any]) -> 'TimeframeConfig':
        """Crée une configuration à partir d'un dictionnaire."""
        return cls(
            interval=config['interval'],
            weight=float(config['weight']),
            indicators=config.get('indicators', [])
        )

class MultiTimeframeStrategy:
    """
    Stratégie de trading multi-timeframes.
    
    Cette stratégie combine les signaux de plusieurs horizons temporels
    pour générer des signaux plus fiables et réduire les faux signaux.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialise la stratégie multi-timeframes.
        
        Args:
            config: Configuration de la stratégie
        """
        self.enabled = config.get('enabled', True)
        self.weight = config.get('weight', 0.3)
        self.risk_multiplier = config.get('risk_multiplier', 1.0)
        
        # Configuration des timeframes
        self.timeframes = [
            TimeframeConfig.from_dict(tf) 
            for tf in config.get('timeframes', [])
        ]
        
        # Règles de confirmation entre timeframes
        self.confirmation_rules = config.get('confirmation_rules', {
            'min_timeframes': 2,
            'priority': 'higher'  # 'higher' ou 'consensus'
        })
        
        # Fournisseur de données (sera injecté par le framework)
        self.data_provider = None
    
    def analyze_timeframe(
        self, 
        interval: str, 
        tf_config: TimeframeConfig, 
        market_data: pd.DataFrame
    ) -> Dict[str, Any]:
        """
        Analyse un timeframe spécifique.
        
        Args:
            interval: Intervalle du timeframe (ex: '15m', '1h')
            tf_config: Configuration du timeframe
            market_data: Données de marché pour ce timeframe
            
        Returns:
            Dict: Résultats de l'analyse
        """
        if market_data.empty:
            return {
                'interval': interval,
                'signal': 'HOLD',
                'strength': 0.0,
                'indicators': {},
                'error': 'No data available'
            }
        
        try:
            # Calcul des indicateurs pour ce timeframe
            indicators = self.data_provider.calculate_indicators(
                market_data, 
                tf_config.indicators
            )
            
            # Génération du signal basé sur les indicateurs
            signal, strength = self._generate_signal(indicators, interval)
            
            return {
                'interval': interval,
                'signal': signal,
                'strength': strength,
                'indicators': indicators,
                'volatility': self._calculate_volatility(market_data),
                'trend_strength': self._calculate_trend_strength(indicators, interval)
            }
            
        except Exception as e:
            return {
                'interval': interval,
                'signal': 'HOLD',
                'strength': 0.0,
                'indicators': {},
                'error': str(e)
            }
    
    def _generate_signal(
        self, 
        indicators: Dict[str, Any], 
        interval: str
    ) -> Tuple[str, float]:
        """
        Génère un signal basé sur les indicateurs d'un timeframe.
        
        Args:
            indicators: Dictionnaire des indicateurs calculés
            interval: Intervalle du timeframe
            
        Returns:
            Tuple: (signal, force_du_signal)
        """
        # Logique simplifiée de génération de signal
        # À adapter selon les indicateurs disponibles
        
        # Exemple avec RSI
        rsi = indicators.get('rsi')
        if rsi is not None:
            if rsi > 70:
                return 'SELL', (rsi - 70) / 30  # Normalisé entre 0 et 1
            elif rsi < 30:
                return 'BUY', (30 - rsi) / 30    # Normalisé entre 0 et 1
        
        # Exemple avec MACD
        macd = indicators.get('macd')
        macd_signal = indicators.get('macd_signal')
        if macd is not None and macd_signal is not None:
            if macd > macd_signal:
                return 'BUY', min(1.0, (macd - macd_signal) / (macd_signal + 1e-9))
            else:
                return 'SELL', min(1.0, (macd_signal - macd) / (abs(macd) + 1e-9))
        
        # Par défaut, on reste en position
        return 'HOLD', 0.0
    
    def _calculate_volatility(self, market_data: pd.DataFrame) -> float:
        """
        Calcule la volatilité des prix sur le timeframe.
        
        Args:
            market_data: Données de marché
            
        Returns:
            float: Volatilité (écart-type des rendements)
        """
        returns = market_data['close'].pct_change().dropna()
        return returns.std() * np.sqrt(252)  # Annualisation
    
    def _calculate_trend_strength(
        self, 
        indicators: Dict[str, Any], 
        interval: str
    ) -> float:
        """
        Calcule la force de la tendance sur le timeframe.
        
        Args:
            indicators: Indicateurs calculés
            interval: Intervalle du timeframe
            
        Returns:
            float: Force de la tendance (0-1)
        """
        # Exemple avec plusieurs indicateurs de tendance
        trend_indicators = []
        
        # ADX
        if 'adx' in indicators:
            adx = indicators['adx']
            trend_indicators.append(adx / 100)  # Normalisé entre 0 et 1
        
        # Pente des moyennes mobiles
        for ma in ['ema_20', 'ema_50', 'sma_200']:
            if ma in indicators:
                # Logique simplifiée pour détecter la pente
                trend_indicators.append(0.5)  # Valeur factice
        
        # Moyenne des indicateurs de tendance
        if trend_indicators:
            return float(np.mean(trend_indicators))
        
        return 0.5  # Valeur par défaut
    
    def combine_timeframe_signals(
        self, 
        signals: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Combine les signaux de plusieurs timeframes.
        
        Args:
            signals: Liste des signaux par timeframe
            
        Returns:
            Dict: Signal combiné
        """
        if not signals:
            return {'signal': 'HOLD', 'strength': 0.0, 'timeframe_contributions': {}}
        
        # Poids cumulés par signal
        signal_weights = {'BUY': 0.0, 'SELL': 0.0, 'HOLD': 0.0}
        timeframe_contributions = {}
        
        for tf_signal in signals:
            interval = tf_signal['interval']
            signal = tf_signal['signal']
            strength = tf_signal.get('strength', 0.0)
            
            # Récupère le poids du timeframe
            tf_config = next((tf for tf in self.timeframes if tf.interval == interval), None)
            if not tf_config:
                continue
                
            weight = tf_config.weight
            
            # Met à jour les poids cumulés
            if signal in signal_weights:
                signal_weights[signal] += strength * weight
                
            # Enregistre la contribution de ce timeframe
            timeframe_contributions[interval] = {
                'signal': signal,
                'strength': strength,
                'weight': weight
            }
        
        # Détermine le signal final
        final_signal = max(signal_weights, key=signal_weights.get)  # type: ignore
        
        # Normalise la force du signal
        total_weight = sum(tf.weight for tf in self.timeframes)
        strength = signal_weights[final_signal] / (total_weight + 1e-9)
        
        return {
            'signal': final_signal,
            'strength': min(max(strength, 0.0), 1.0),  # Borne entre 0 et 1
            'timeframe_contributions': timeframe_contributions
        }
    
    def check_confirmation(self, signals: List[Dict[str, Any]]) -> bool:
        """
        Vérifie si les signaux sont confirmés selon les règles définies.
        
        Args:
            signals: Liste des signaux par timeframe
            
        Returns:
            bool: True si la confirmation est obtenue
        """
        if not signals:
            return False
            
        min_timeframes = self.confirmation_rules.get('min_timeframes', 2)
        priority = self.confirmation_rules.get('priority', 'higher')
        
        # Compte les signaux par type
        signal_counts = {'BUY': 0, 'SELL': 0, 'HOLD': 0}
        for sig in signals:
            signal = sig.get('signal', 'HOLD')
            if signal in signal_counts:
                signal_counts[signal] += 1
        
        # Vérifie la confirmation selon la priorité
        if priority == 'consensus':
            # Nécessite un consensus entre les timeframes
            max_count = max(signal_counts.values())
            if max_count < min_timeframes:
                return False
                
            # Vérifie qu'il n'y a pas d'opposition forte
            for sig, count in signal_counts.items():
                if sig != 'HOLD' and count > 0 and sig != max(signal_counts, key=signal_counts.get):
                    if count >= min_timeframes / 2:  # Opposition significative
                        return False
            
            return True
            
        elif priority == 'higher':
            # Priorité aux timeframes plus longs
            # Trie les signaux par ordre d'importance du timeframe
            sorted_signals = sorted(
                signals,
                key=lambda x: self._get_timeframe_priority(x.get('interval', '')),
                reverse=True
            )
            
            # Prend en compte les signaux des timeframes les plus importants
            confirmed = 0
            for sig in sorted_signals[:min_timeframes]:
                if sig.get('signal') != 'HOLD':
                    confirmed += 1
            
            return confirmed >= min_timeframes
        
        return False
    
    def _get_timeframe_priority(self, interval: str) -> int:
        """
        Retourne la priorité d'un intervalle (plus le chiffre est élevé, plus c'est prioritaire).
        """
        priorities = {
            '1m': 1, '5m': 2, '15m': 3, '30m': 4,
            '1h': 5, '4h': 6, '1d': 7, '1w': 8
        }
        return priorities.get(interval.lower(), 0)
    
    def assess_risk(self, timeframe_analysis: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        Évalue le risque global basé sur l'analyse des différents timeframes.
        
        Args:
            timeframe_analysis: Résultats d'analyse par timeframe
            
        Returns:
            Dict: Métriques de risque
        """
        if not timeframe_analysis:
            return {
                'composite_volatility': 0.0,
                'composite_trend': 0.5,
                'position_size': 0.0
            }
        
        # Calcule la volatilité composite (moyenne pondérée)
        total_weight = 0.0
        vol_sum = 0.0
        trend_sum = 0.0
        
        for analysis in timeframe_analysis:
            tf_config = next((tf for tf in self.timeframes if tf.interval == analysis['interval']), None)
            if not tf_config:
                continue
                
            weight = tf_config.weight
            vol = analysis.get('volatility', 0.0)
            trend = analysis.get('trend_strength', 0.5)
            
            vol_sum += vol * weight
            trend_sum += trend * weight
            total_weight += weight
        
        if total_weight > 0:
            composite_vol = vol_sum / total_weight
            composite_trend = trend_sum / total_weight
        else:
            composite_vol = 0.0
            composite_trend = 0.5
        
        # Calcule la taille de position en fonction de la volatilité et de la tendance
        # Plus la tendance est forte et la volatilité faible, plus la position peut être importante
        if composite_vol > 0:
            position_size = (composite_trend * 0.5 + 0.5) / (composite_vol + 0.1)  # Évite la division par zéro
            position_size = min(max(position_size, 0.0), 1.0)  # Borne entre 0 et 1
        else:
            position_size = 0.0
        
        return {
            'composite_volatility': composite_vol,
            'composite_trend': composite_trend,
            'position_size': position_size * self.risk_multiplier
        }
    
    def calculate_performance_metrics(self, returns: pd.Series) -> Dict[str, float]:
        """
        Calcule les métriques de performance à partir des rendements.
        
        Args:
            returns: Série des rendements
            
        Returns:
            Dict: Dictionnaire des métriques de performance
        """
        if len(returns) < 2:
            return {
                'total_return': 0.0,
                'volatility': 0.0,
                'sharpe': 0.0,
                'max_drawdown': 0.0,
                'win_rate': 0.0
            }
            
        # Calcul du rendement total
        total_return = float((1 + returns).prod() - 1)
        
        # Volatilité annualisée
        volatility = returns.std() * np.sqrt(252)  # 252 jours de bourse dans une année
        
        # Ratio de Sharpe (sans taux sans risque pour simplifier)
        sharpe = (returns.mean() / (returns.std() + 1e-9)) * np.sqrt(252)
        
        # Maximum Drawdown
        cum_returns = (1 + returns).cumprod()
        rolling_max = cum_returns.cummax()
        drawdowns = (cum_returns - rolling_max) / rolling_max
        max_drawdown = drawdowns.min()
        
        # Taux de réussite (pourcentage de trades gagnants)
        win_rate = (returns > 0).mean()
        
        return {
            'total_return': float(total_return),
            'volatility': float(volatility),
            'sharpe': float(sharpe),
            'max_drawdown': float(max_drawdown),
            'win_rate': float(win_rate)
        }
        
    def generate_signals(self, symbol: str) -> Dict[str, Any]:
        """
        Génère des signaux de trading en combinant plusieurs timeframes.
        
        Args:
            symbol: Symbole du marché à analyser
            
        Returns:
            Dict: Signaux générés et métriques associées
        """
        if not self.enabled or not self.timeframes:
            return {
                'signal': 'HOLD',
                'strength': 0.0,
                'timeframe_analysis': [],
                'error': 'Strategy disabled or no timeframes configured'
            }
        
        try:
            # Analyse de chaque timeframe
            timeframe_analysis = []
            
            for tf_config in self.timeframes:
                # Récupère les données pour ce timeframe
                market_data = self.data_provider.get_historical_data(
                    symbol=symbol,
                    interval=tf_config.interval
                )
                
                # Analyse le timeframe
                analysis = self.analyze_timeframe(
                    interval=tf_config.interval,
                    tf_config=tf_config,
                    market_data=market_data
                )
                
                timeframe_analysis.append(analysis)
            
            # Combine les signaux des différents timeframes
            combined_signal = self.combine_timeframe_signals(timeframe_analysis)
            
            # Vérifie la confirmation du signal
            is_confirmed = self.check_confirmation(timeframe_analysis)
            
            # Évalue le risque
            risk_assessment = self.assess_risk(timeframe_analysis)
            
            return {
                'symbol': symbol,
                'signal': combined_signal['signal'] if is_confirmed else 'HOLD',
                'strength': combined_signal['strength'] if is_confirmed else 0.0,
                'is_confirmed': is_confirmed,
                'timeframe_analysis': timeframe_analysis,
                'risk_assessment': risk_assessment,
                'timestamp': datetime.utcnow()
            }
            
        except Exception as e:
            return {
                'signal': 'HOLD',
                'strength': 0.0,
                'timeframe_analysis': [],
                'error': f'Error generating signals: {str(e)}'
            }
