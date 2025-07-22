"""
Analyse technique des paires de trading.

Ce module fournit des fonctions pour analyser les paires de trading
en utilisant divers indicateurs techniques et métriques de marché.
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple

from .models import PairAnalysis


class PairAnalyzer:
    """Analyseur de paires de trading.
    
    Cette classe encapsule la logique d'analyse technique des paires de trading,
    y compris le calcul des indicateurs techniques et des métriques de marché.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """Initialise l'analyseur avec une configuration optionnelle.
        
        Args:
            config: Dictionnaire de configuration pour l'analyseur
        """
        self.config = config or {}
        self.logger = self.config.get('logger')
    
    async def analyze_pair(self, pair: str, ohlc_data: Dict) -> PairAnalysis:
        """Analyse une paire de trading à partir de ses données OHLC.
        
        Args:
            pair: Identifiant de la paire (ex: 'XBT/USD')
            ohlc_data: Données OHLC au format Kraken
            
        Returns:
            Objet PairAnalysis contenant les résultats de l'analyse
        """
        analysis = PairAnalysis(pair=pair)
        
        try:
            # Convertir les données OHLC en DataFrame pandas
            df = self._prepare_ohlc_data(ohlc_data)
            if df.empty:
                analysis.error = "Aucune donnée valide pour l'analyse"
                return analysis
            
            # Extraire les séries de prix
            close_prices = df['close'].to_numpy()
            high_prices = df['high'].to_numpy()
            low_prices = df['low'].to_numpy()
            volumes = df['volume'].to_numpy()
            
            # Calculer les indicateurs techniques
            indicators = self._calculate_technical_indicators(
                close=close_prices,
                high=high_prices,
                low=low_prices,
                volume=volumes
            )
            
            # Calculer les métriques de marché
            metrics = self._calculate_market_metrics(
                close=close_prices,
                high=high_prices,
                low=low_prices,
                volume=volumes,
                indicators=indicators
            )
            
            # Calculer le score global
            score = self._calculate_score(metrics, indicators)
            
            # Mettre à jour l'objet d'analyse
            analysis.score = score
            analysis.metrics = metrics
            analysis.indicators = indicators
            analysis.last_updated = datetime.now(timezone.utc).isoformat()
            
        except Exception as e:
            error_msg = f"Erreur lors de l'analyse de {pair}: {str(e)}"
            analysis.error = error_msg
            if self.logger and hasattr(self.logger, 'error'):
                self.logger.error(error_msg, exc_info=True)
        
        return analysis
    
    def _prepare_ohlc_data(self, ohlc_data: Dict) -> pd.DataFrame:
        """Prépare les données OHLC pour l'analyse.
        
        Args:
            ohlc_data: Données OHLC brutes
            
        Returns:
            DataFrame pandas avec les données formatées
        """
        if not ohlc_data:
            return pd.DataFrame()
            
        try:
            # Convertir les données en DataFrame
            data = []
            for timestamp, ohlc in ohlc_data.items():
                data.append([
                    pd.to_datetime(timestamp, unit='s'),
                    float(ohlc['open']),
                    float(ohlc['high']),
                    float(ohlc['low']),
                    float(ohlc['close']),
                    float(ohlc['volume'])
                ])
            
            df = pd.DataFrame(
                data,
                columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
            )
            
            # Trier par date (du plus ancien au plus récent)
            df = df.sort_values('timestamp')
            
            # Supprimer les doublons et valeurs manquantes
            df = df.drop_duplicates().dropna()
            
            return df
            
        except Exception as e:
            if self.logger and hasattr(self.logger, 'error'):
                self.logger.error(f"Erreur préparation données OHLC: {str(e)}", exc_info=True)
            return pd.DataFrame()
    
    def _calculate_technical_indicators(
        self,
        close: np.ndarray,
        high: np.ndarray,
        low: np.ndarray,
        volume: np.ndarray
    ) -> Dict[str, float]:
        """Calcule les indicateurs techniques pour une paire.
        
        Args:
            close: Tableau des prix de clôture
            high: Tableau des plus hauts prix
            low: Tableau des plus bas prix
            volume: Tableau des volumes
            
        Returns:
            Dictionnaire des indicateurs calculés
        """
        indicators = {}
        
        try:
            # Calcul des rendements logarithmiques
            returns = np.diff(np.log(close))
            
            # Volatilité (écart-type des rendements annualisé)
            if len(returns) > 1:
                indicators['volatility'] = float(np.std(returns, ddof=1) * np.sqrt(252))
            else:
                indicators['volatility'] = 0.0
            
            # Momentum (variation sur 10 périodes)
            lookback = min(10, len(close) - 1)
            if lookback > 0:
                momentum = ((close[-1] - close[-lookback-1]) / close[-lookback-1]) * 100
                indicators['momentum'] = float(momentum)
            else:
                indicators['momentum'] = 0.0
            
            # Volume moyen (sur 20 périodes)
            volume_window = min(20, len(volume))
            if volume_window > 0:
                indicators['volume_ma'] = float(np.mean(volume[-volume_window:]))
            else:
                indicators['volume_ma'] = 0.0
            
            # Spread moyen (écart relatif haut/bas en %)
            spread_pct = (high - low) / ((high + low) / 2) * 100
            indicators['avg_spread'] = float(np.mean(spread_pct))
            
            # RSI (14 périodes)
            if len(close) >= 15:  # Besoin d'au moins 15 points pour RSI(14)
                deltas = np.diff(close)
                gains = np.where(deltas > 0, deltas, 0)
                pertes = np.where(deltas < 0, -deltas, 0)
                
                # Moyenne mobile des gains et pertes
                avg_gain = np.mean(gains[:14])
                avg_loss = np.mean(pertes[:14])
                
                # Calcul du RS et RSI
                if avg_loss != 0:
                    rs = avg_gain / avg_loss
                    indicators['rsi'] = float(100 - (100 / (1 + rs)))
                else:
                    indicators['rsi'] = 100.0  # Si pas de perte, RSI = 100
            else:
                indicators['rsi'] = 50.0  # Valeur neutre si pas assez de données
            
            # ATR (Average True Range) - 14 périodes
            if len(close) >= 15:
                high = np.asarray(high, dtype=np.float64)
                low = np.asarray(low, dtype=np.float64)
                close = np.asarray(close, dtype=np.float64)
                
                # Calcul du True Range
                tr1 = high[1:] - low[1:]
                tr2 = np.abs(high[1:] - close[:-1])
                tr3 = np.abs(low[1:] - close[:-1])
                
                true_range = np.maximum(np.maximum(tr1, tr2), tr3)
                
                # Moyenne mobile simple sur 14 périodes
                atr = np.mean(true_range[-14:]) if len(true_range) >= 14 else np.mean(true_range)
                indicators['atr'] = float(atr)
            else:
                indicators['atr'] = 0.0
            
            # ADX (Average Directional Index) - 14 périodes
            if len(close) >= 28:  # Besoin de plus de données pour ADX
                # Calcul des mouvements directionnels
                up = high[1:] - high[:-1]
                down = low[:-1] - low[1:]
                
                plus_dm = np.where((up > down) & (up > 0), up, 0.0)
                minus_dm = np.where((down > up) & (down > 0), down, 0.0)
                
                # Lissage des DM avec une moyenne mobile
                def smooth(series, period):
                    return np.convolve(
                        series, 
                        np.ones(period) / period, 
                        mode='valid'
                    )
                
                # Calcul de l'ADX
                tr = true_range[-28:]
                plus_di = 100 * smooth(plus_dm[-28:], 14) / smooth(tr, 14)
                minus_di = 100 * smooth(minus_dm[-28:], 14) / smooth(tr, 14)
                
                dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
                adx = np.mean(dx[-14:]) if len(dx) >= 14 else np.mean(dx)
                
                indicators['adx'] = float(adx)
                indicators['plus_di'] = float(plus_di[-1] if len(plus_di) > 0 else 0)
                indicators['minus_di'] = float(minus_di[-1] if len(minus_di) > 0 else 0)
            else:
                indicators['adx'] = 0.0
                indicators['plus_di'] = 0.0
                indicators['minus_di'] = 0.0
            
        except Exception as e:
            if self.logger and hasattr(self.logger, 'error'):
                self.logger.error(f"Erreur calcul indicateurs: {str(e)}", exc_info=True)
            
            # En cas d'erreur, retourner des valeurs par défaut
            for key in ['volatility', 'momentum', 'volume_ma', 'avg_spread', 'rsi', 'atr', 'adx']:
                if key not in indicators:
                    indicators[key] = 0.0
        
        return indicators
    
    def _calculate_market_metrics(
        self,
        close: np.ndarray,
        high: np.ndarray,
        low: np.ndarray,
        volume: np.ndarray,
        indicators: Dict[str, float]
    ) -> Dict[str, float]:
        """Calcule les métriques de marché pour une paire.
        
        Args:
            close: Tableau des prix de clôture
            high: Tableau des plus hauts prix
            low: Tableau des plus bas prix
            volume: Tableau des volumes
            indicators: Dictionnaire des indicateurs déjà calculés
            
        Returns:
            Dictionnaire des métriques calculées
        """
        metrics = {}
        
        try:
            # Dernier prix
            metrics['last_price'] = float(close[-1]) if len(close) > 0 else 0.0
            
            # Variation sur différentes périodes
            periods = [1, 7, 30]  # 1 jour, 1 semaine, 1 mois
            for period in periods:
                if len(close) > period:
                    change = ((close[-1] - close[-period-1]) / close[-period-1]) * 100
                    metrics[f'change_{period}d'] = float(change)
                else:
                    metrics[f'change_{period}d'] = 0.0
            
            # Volume sur 24h
            metrics['volume_24h'] = float(np.sum(volume[-24:])) if len(volume) >= 24 else float(np.sum(volume))
            
            # Prix moyen pondéré par le volume (VWAP)
            if len(close) > 0 and np.sum(volume) > 0:
                vwap = np.sum(close * volume) / np.sum(volume)
                metrics['vwap'] = float(vwap)
            else:
                metrics['vwap'] = metrics.get('last_price', 0.0)
            
            # Spread moyen
            metrics['avg_spread'] = indicators.get('avg_spread', 0.0)
            
            # Volatilité historique
            metrics['volatility'] = indicators.get('volatility', 0.0)
            
            # Liquidité (volume * prix)
            metrics['liquidity'] = metrics['volume_24h'] * metrics['last_price']
            
            # Ratio de Sharpe (simplifié, sans taux sans risque)
            returns = np.diff(np.log(close))
            if len(returns) > 1 and np.std(returns) > 0:
                sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252)
                metrics['sharpe_ratio'] = float(sharpe)
            else:
                metrics['sharpe_ratio'] = 0.0
            
            # Drawdown maximum
            cum_returns = np.cumsum(returns)
            peak = np.maximum.accumulate(cum_returns)
            drawdown = (peak - cum_returns) / (peak + 1e-10)  # Éviter la division par zéro
            metrics['max_drawdown'] = float(np.max(drawdown)) if len(drawdown) > 0 else 0.0
            
        except Exception as e:
            if self.logger and hasattr(self.logger, 'error'):
                self.logger.error(f"Erreur calcul métriques: {str(e)}", exc_info=True)
            
            # En cas d'erreur, retourner des valeurs par défaut
            for key in ['last_price', 'volume_24h', 'vwap', 'liquidity', 'sharpe_ratio', 'max_drawdown']:
                if key not in metrics:
                    metrics[key] = 0.0
                    
            for period in [1, 7, 30]:
                if f'change_{period}d' not in metrics:
                    metrics[f'change_{period}d'] = 0.0
        
        return metrics
    
    def _calculate_score(
        self, 
        metrics: Dict[str, float], 
        indicators: Dict[str, float]
    ) -> float:
        """Calcule un score global pour la paire basé sur les métriques et indicateurs.
        
        Args:
            metrics: Dictionnaire des métriques de marché
            indicators: Dictionnaire des indicateurs techniques
            
        Returns:
            Score global entre 0 et 1
        """
        score = 0.5  # Score neutre par défaut
        
        try:
            # Facteurs de pondération (ajustables)
            weights = {
                # Métriques de marché
                'liquidity': 0.25,      # Plus c'est liquide, mieux c'est
                'volatility': 0.20,     # Volatilité modérée est préférée
                'volume_24h': 0.15,     # Volume élevé est préféré
                'avg_spread': -0.10,    # Spread faible est préféré
                'sharpe_ratio': 0.15,   # Meilleur ratio de Sharpe est préféré
                'max_drawdown': -0.15,  # Drawdown faible est préféré
                
                # Indicateurs techniques
                'rsi': 0.10,           # Proche de 50 est préféré (ni surachat ni survente)
                'adx': 0.05,           # Tendance forte est préférée
                'momentum': 0.05       # Momentum positif est préféré
            }
            
            # Valeurs normalisées pour chaque facteur (0-1)
            factors = {}
            
            # Liquidité (log pour réduire l'échelle)
            liquidity = metrics.get('liquidity', 0)
            factors['liquidity'] = min(np.log10(liquidity + 1) / 6.0, 1.0) if liquidity > 0 else 0.0
            
            # Volatilité (optimale autour de 0.5-1.5)
            volatility = metrics.get('volatility', 0)
            factors['volatility'] = 1.0 - min(abs(volatility - 1.0), 1.0)
            
            # Volume 24h (log pour réduire l'échelle)
            volume = metrics.get('volume_24h', 0)
            factors['volume_24h'] = min(np.log10(volume + 1) / 5.0, 1.0) if volume > 0 else 0.0
            
            # Spread moyen (moins c'est mieux, max à 2%)
            spread = metrics.get('avg_spread', 0)
            factors['avg_spread'] = max(0, 1.0 - (spread / 2.0)) if spread > 0 else 1.0
            
            # Ratio de Sharpe (normalisé entre 0 et 1)
            sharpe = metrics.get('sharpe_ratio', 0)
            factors['sharpe_ratio'] = min(max((sharpe + 2.0) / 4.0, 0.0), 1.0)
            
            # Drawdown maximum (moins c'est mieux)
            drawdown = metrics.get('max_drawdown', 0)
            factors['max_drawdown'] = 1.0 - min(drawdown, 1.0)
            
            # RSI (proche de 50 est préféré)
            rsi = indicators.get('rsi', 50)
            factors['rsi'] = 1.0 - (abs(rsi - 50) / 50.0)
            
            # ADX (plus c'est élevé, mieux c'est, max à 50)
            adx = indicators.get('adx', 0)
            factors['adx'] = min(adx / 50.0, 1.0)
            
            # Momentum (normalisé)
            momentum = indicators.get('momentum', 0)
            factors['momentum'] = min(max((momentum + 20.0) / 40.0, 0.0), 1.0)
            
            # Calcul du score pondéré
            total_weight = sum(abs(w) for w in weights.values())
            if total_weight > 0:
                score = sum(factors.get(k, 0.5) * weights[k] for k in weights) / total_weight
                score = max(0.0, min(1.0, score))  # Borne entre 0 et 1
            
        except Exception as e:
            if self.logger and hasattr(self.logger, 'error'):
                self.logger.error(f"Erreur calcul score: {str(e)}", exc_info=True)
            score = 0.5  # En cas d'erreur, retourner un score neutre
        
        return score
