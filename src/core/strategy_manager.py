"""
Gestionnaire de stratégies pour le bot de trading.
Permet d'exécuter plusieurs stratégies en parallèle et de basculer entre elles.
"""
import logging
from typing import Dict, Tuple, Any
import pandas as pd
import numpy as np
from datetime import datetime
import asyncio
from ta.trend import ADXIndicator

from src.strategies import (
    MomentumStrategy,
    MeanReversionStrategy,
    BreakoutStrategy
)

logger = logging.getLogger(__name__)


class StrategyManager:
    """
    Gère l'exécution de plusieurs stratégies de trading et la sélection de la meilleure.
    """

    def __init__(self, config: dict = None, risk_profile: dict = None):
        """
        Initialise le gestionnaire de stratégies.

        Args:
            config (dict, optional): Configuration du gestionnaire. Par défaut None.
            risk_profile (dict, optional): Profil de risque contenant les paramètres de levier. Par défaut None.
        """
        self.config = config or {}
        self.risk_profile = risk_profile or {}
        self.strategies = {}
        self.strategy_weights = {}
        self.strategy_performance = {}
        self.leverage = self.risk_profile.get('max_leverage', 1.0)
        self.max_leverage = min(
            self.risk_profile.get(
                'max_leverage',
                5.0),
            20.0)  # Limite de sécurité à 20x
        self.market_regime = "neutral"  # trending, ranging, volatile, neutral
        self.last_regime_change = datetime.now()
        self.initialize_strategies()

    def initialize_strategies(self):
        """Initialise les stratégies avec leur configuration."""
        # Configuration par défaut des stratégies
        default_config = {
            'momentum': {
                'rsi_period': 14,
                'rsi_overbought': 70,
                'rsi_oversold': 30,
                'macd_fast': 12,
                'macd_slow': 26,
                'macd_signal': 9,
                'min_confidence': 0.6
            },
            'mean_reversion': {
                'bb_window': 20,
                'bb_std': 2.0,
                'rsi_period': 14,
                'rsi_overbought': 70,
                'rsi_oversold': 30,
                'min_confidence': 0.6
            },
            'breakout': {
                'adx_period': 14,
                'adx_threshold': 25,
                'atr_period': 14,
                'min_confidence': 0.65
            }
        }

        # Fusion avec la configuration fournie
        strategy_configs = self.config.get('strategies', {})
        for name, cfg in strategy_configs.items():
            if name in default_config:
                default_config[name].update(cfg)

        # Initialisation des stratégies
        self.strategies = {
            'momentum': MomentumStrategy(
                default_config['momentum']), 'mean_reversion': MeanReversionStrategy(
                default_config['mean_reversion']), 'breakout': BreakoutStrategy(
                default_config['breakout'])}

        # Poids initiaux des stratégies (peuvent évoluer en fonction des
        # performances)
        self.strategy_weights = {
            'momentum': 0.4,
            'mean_reversion': 0.3,
            'breakout': 0.3
        }

        # Historique des performances (pour le calcul des poids)
        for name in self.strategies:
            self.strategy_performance[name] = {
                'returns': [],
                'win_rate': 0.5,
                'sharpe_ratio': 0.0,
                'last_update': datetime.now()
            }

    async def analyze_market_regime(
            self, data: Dict[str, pd.DataFrame]) -> str:
        """
        Analyse le régime de marché actuel (tendance, range, volatile).

        Args:
            data (Dict[str, pd.DataFrame]): Données de marché par timeframe

        Returns:
            str: Régime de marché ('trending', 'ranging', 'volatile', 'neutral')
        """
        # Utiliser le timeframe le plus long disponible
        tf = max(data.keys(), key=lambda x: int(
            x[:-1]) if x[-1] in ['m', 'h'] else 0)
        df = data[tf].copy()

        # Calcul des indicateurs de régime
        adx = ADXIndicator(
            high=df['high'],
            low=df['low'],
            close=df['close'],
            window=14
        )
        df['adx'] = adx.adx()

        # ATR pour la volatilité
        atr = AverageTrueRange(
            high=df['high'],
            low=df['low'],
            close=df['close'],
            window=14
        )
        df['atr'] = atr.average_true_range()

        # Moyenne mobile pour la tendance
        df['sma_50'] = df['close'].rolling(window=50).mean()
        df['sma_200'] = df['close'].rolling(window=200).mean()

        # Dernières valeurs
        last_adx = df['adx'].iloc[-1]
        last_atr = df['atr'].iloc[-1]
        atr_ma = df['atr'].rolling(window=20).mean().iloc[-1]

        # Détection du régime
        is_trending = last_adx > 25  # Tendance forte si ADX > 25
        is_volatile = last_atr > atr_ma * 1.5  # Volatilité élevée

        # Logique de décision
        if is_trending and not is_volatile:
            return "trending"
        elif not is_trending and not is_volatile:
            return "ranging"
        elif is_volatile:
            return "volatile"
        else:
            return "neutral"

    def update_strategy_weights(
            self, performance_data: Dict[str, Dict[str, float]]):
        """
        Met à jour les poids des stratégies en fonction de leurs performances récentes.

        Args:
            performance_data (Dict[str, Dict[str, float]]): Données de performance par stratégie
        """
        total_performance = sum(perf.get('sharpe_ratio', 0)
                                for perf in performance_data.values())

        if total_performance > 0:
            # Ajuster les poids en fonction des performances relatives
            for name, strategy in self.strategies.items():
                perf = performance_data.get(name, {})
                sharpe = perf.get('sharpe_ratio', 0)

                # Poids basé sur la performance relative (Sharpe ratio)
                if total_performance > 0:
                    self.strategy_weights[name] = max(
                        0.1, min(0.7, sharpe / total_performance))

                # Ajustement supplémentaire basé sur le régime de marché
                if self.market_regime == 'trending' and name == 'momentum':
                    self.strategy_weights[name] *= 1.2
                elif self.market_regime == 'ranging' and name == 'mean_reversion':
                    self.strategy_weights[name] *= 1.2
                elif self.market_regime == 'volatile' and name == 'breakout':
                    self.strategy_weights[name] *= 1.2

            # Normaliser les poids pour qu'ils totalisent 1.0
            total_weight = sum(self.strategy_weights.values())
            for name in self.strategy_weights:
                self.strategy_weights[name] /= total_weight

    async def get_combined_signals(
            self, data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """
        Obtient les signaux combinés de toutes les stratégies.

        Args:
            data (Dict[str, pd.DataFrame]): Données de marché par timeframe

        Returns:
            Dict[str, Any]: Signaux combinés et métadonnées
        """
        # Mettre à jour le régime de marché
        self.market_regime = await self.analyze_market_regime(data)

        # Exécuter toutes les stratégies en parallèle
        tasks = []
        for name, strategy in self.strategies.items():
            task = asyncio.create_task(
                self._run_strategy(
                    name, strategy, data))
            tasks.append(task)

        # Attendre que toutes les stratégies aient terminé
        results = await asyncio.gather(*tasks)
        strategy_signals = {
            name: result for name,
            result in results if result is not None}

        # Mettre à jour les poids des stratégies en fonction des performances
        self.update_strategy_weights({
            name: signals.get('performance', {})
            for name, signals in strategy_signals.items()
        })

        # Combiner les signaux en fonction des poids
        combined_signal = self._combine_signals(strategy_signals)

        return {
            'signals': combined_signal,
            'strategy_details': strategy_signals,
            'market_regime': self.market_regime,
            'strategy_weights': self.strategy_weights.copy()
        }

    async def _run_strategy(self, name: str, strategy,
                            data: Dict[str, pd.DataFrame]) -> Tuple[str, Dict]:
        """
        Exécute une stratégie et retourne ses signaux.

        Args:
            name (str): Nom de la stratégie
            strategy: Instance de la stratégie
            data (Dict[str, pd.DataFrame]): Données de marché

        Returns:
            Tuple[str, Dict]: Nom de la stratégie et résultats
        """
        try:
            # Filtrer les données pour ne garder que les timeframes nécessaires
            required_tfs = strategy.get_required_timeframes()
            strategy_data = {tf: data[tf] for tf in required_tfs if tf in data}

            if not strategy_data:
                logger.warning(
                    f"Aucune donnée valide pour la stratégie {name}")
                return None

            # Calculer les indicateurs
            strategy_data = strategy.calculate_indicators(strategy_data)

            # Générer les signaux
            signals = strategy.generate_signals(strategy_data)

            # Calculer la confiance des signaux
            confidence = strategy.calculate_confidence(strategy_data, signals)

            # Mettre à jour les performances (simplifié)
            self.strategy_performance[name]['last_update'] = datetime.now()

            return (name, {
                'signals': signals,
                'confidence': confidence,
                'indicators': strategy_data,
                'performance': {
                    # À remplacer par un calcul réel
                    'sharpe_ratio': np.random.normal(0.5, 0.2),
                    # À remplacer par un calcul réel
                    'win_rate': np.random.uniform(0.4, 0.7),
                    'last_trade': 'win' if np.random.random() > 0.5 else 'loss'  # Exemple
                }
            })

        except Exception as e:
            logger.error(
                f"Erreur lors de l'exécution de la stratégie {name}: {e}",
                exc_info=True)
            return None

    def calculate_position_size(
            self,
            entry_price: float,
            stop_loss: float,
            account_balance: float) -> float:
        """
        Calcule la taille de la position en tenant compte du levier.

        Args:
            entry_price: Prix d'entrée du trade
            stop_loss: Prix du stop-loss
            account_balance: Solde total du compte

        Returns:
            float: Taille de la position ajustée avec le levier
        """
        # Calcul du risque de base
        risk_per_trade = self.risk_profile.get('risk_per_trade', 0.01)
        risk_amount = account_balance * risk_per_trade * self.leverage

        # Calcul du risque par unité
        risk_per_share = abs(entry_price - stop_loss)
        if risk_per_share == 0:
            return 0

        # Calcul de la taille de position de base
        position_size = risk_amount / risk_per_share

        # Ajustement selon la taille maximale de position
        max_position = (account_balance * self.leverage *
                        self.risk_profile.get('max_position_size', 0.1)) / entry_price

        # Vérification de la marge disponible
        required_margin = (entry_price * position_size) / self.leverage
        if required_margin > account_balance:
            # Ajustement de la position pour respecter la marge disponible
            position_size = (account_balance * self.leverage) / entry_price

        return min(position_size, max_position)

    def adjust_leverage(
            self,
            market_volatility: float,
            confidence: float) -> None:
        """
        Ajuste dynamiquement le levier en fonction de la volatilité et de la confiance.

        Args:
            market_volatility: Mesure de la volatilité du marché (0-1)
            confidence: Niveau de confiance du signal (0-1)
        """
        # Facteurs d'ajustement
        # Réduit le levier quand la volatilité est élevée
        volatility_factor = 1.0 / (1.0 + market_volatility * 2)
        # Augmente le levier avec la confiance
        confidence_factor = min(1.0, confidence * 1.5)

        # Calcul du nouveau levier
        base_leverage = min(
            self.max_leverage,
            self.risk_profile.get(
                'max_leverage',
                1.0))
        new_leverage = base_leverage * volatility_factor * confidence_factor

        # Contraintes de sécurité
        new_leverage = max(1.0, min(new_leverage, self.max_leverage))

        # Application du nouveau levier avec vérification des limites
        if new_leverage != self.leverage:
            logger.info(
                f"Ajustement du levier de {self.leverage}x à {new_leverage:.2f}x"
                f" (volatilité: {market_volatility:.2f}, confiance: {confidence:.2f})")
            self.leverage = new_leverage

    def _combine_signals(
            self, strategy_signals: Dict[str, Dict]) -> Dict[str, Any]:
        """
        Combine les signaux de plusieurs stratégies en fonction de leurs poids.

        Args:
            strategy_signals (Dict[str, Dict]): Signaux de chaque stratégie

        Returns:
            Dict[str, Any]: Signaux combinés
        """
        if not strategy_signals:
            return {}

        # Prendre le premier DataFrame comme référence
        ref_name = next(iter(strategy_signals))
        ref_df = strategy_signals[ref_name]['signals']

        # Initialiser les signaux combinés
        combined = pd.DataFrame(index=ref_df.index)
        combined['signal'] = 0.0
        combined['confidence'] = 0.0
        total_weight = 0.0

        # Combiner les signaux pondérés
        for name, signals in strategy_signals.items():
            weight = self.strategy_weights.get(name, 0.0)
            if weight <= 0:
                continue

            # Aligner les données avec l'index de référence
            aligned_signals = signals['signals'].reindex_like(
                ref_df, method='ffill')
            aligned_conf = signals['confidence'].reindex_like(
                ref_df, method='ffill')

            # Ajouter au signal combiné (pondéré)
            combined['signal'] += aligned_signals * weight
            combined['confidence'] += aligned_conf * weight
            total_weight += weight

        # Normaliser la confiance
        if total_weight > 0:
            combined['confidence'] /= total_weight

        # Arrondir les signaux (-1, 0, 1)
        combined['signal'] = combined['signal'].apply(
            lambda x: 1 if x > 0.5 else (-1 if x < -0.5 else 0)
        )

        # Ajouter des métadonnées
        combined['market_regime'] = self.market_regime

        return combined.to_dict('index')
