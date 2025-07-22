"""
Module pour le système de switching de stratégies.

Ce module permet de basculer dynamiquement entre différentes stratégies de trading
en fonction des conditions du marché et des performances passées.
"""
import logging
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
from enum import Enum, auto

# Import des indicateurs techniques
from src.core.utils.technical_indicators import (
    calculate_atr,
    calculate_supertrend,
    calculate_ichimoku,
    calculate_adx
)


class MarketCondition(Enum):
    """Énumération des conditions de marché possibles."""
    TRENDING_UP = auto()
    TRENDING_DOWN = auto()
    RANGING = auto()
    HIGH_VOLATILITY = auto()
    LOW_VOLATILITY = auto()
    BREAKOUT = auto()
    BREAKDOWN = auto()


class StrategyPerformance:
    """Classe pour suivre les performances des stratégies."""
    
    def __init__(self, lookback_period: int = 30):
        """
        Initialise le suivi des performances.
        
        Args:
            lookback_period: Nombre de jours pour le calcul des performances
        """
        self.lookback_period = lookback_period
        self.performance_data = {}
        self.logger = logging.getLogger(__name__)
    
    def update_performance(self, strategy_name: str, pnl: float, timestamp: datetime):
        """
        Met à jour les performances d'une stratégie.
        
        Args:
            strategy_name: Nom de la stratégie
            pnl: Profit/Perte réalisé
            timestamp: Horodatage de la transaction (peut être avec ou sans fuseau horaire)
        """
        if strategy_name not in self.performance_data:
            self.performance_data[strategy_name] = []
        
        # Normaliser le fuseau horaire si nécessaire
        if timestamp.tzinfo is None:
            timestamp = timestamp.replace(tzinfo=timezone.utc)
        else:
            timestamp = timestamp.astimezone(timezone.utc)
        
        # Ajouter la performance avec l'horodatage normalisé
        self.performance_data[strategy_name].append({
            'timestamp': timestamp,
            'pnl': pnl
        })
        
        # Nettoyer les anciennes données
        self._cleanup_old_data()
    
    def get_performance_metrics(self, strategy_name: str) -> Dict:
        """
        Calcule les métriques de performance pour une stratégie.
        
        Args:
            strategy_name: Nom de la stratégie
            
        Returns:
            Dictionnaire contenant les métriques de performance
        """
        if strategy_name not in self.performance_data or not self.performance_data[strategy_name]:
            print(f"Aucune donnée de performance pour la stratégie: {strategy_name}")
            return {}
        
        # Récupérer les données de performance
        data = self.performance_data[strategy_name]
        pnls = [d['pnl'] for d in data]
        
        # Calculer les métriques
        total_return = sum(pnls)
        win_rate = len([p for p in pnls if p > 0]) / len(pnls) if pnls else 0
        avg_win = np.mean([p for p in pnls if p > 0]) if any(p > 0 for p in pnls) else 0
        avg_loss = np.mean([p for p in pnls if p < 0]) if any(p < 0 for p in pnls) else 0
        profit_factor = -avg_win / avg_loss if avg_loss < 0 else float('inf')
        
        # Afficher les métriques pour le débogage
        print(f"\nMétriques pour {strategy_name}:")
        print(f"- PnLs: {pnls}")
        print(f"- Total return: {total_return:.4f}")
        print(f"- Win rate: {win_rate:.2f}")
        print(f"- Average win: {avg_win:.4f}")
        print(f"- Average loss: {avg_loss:.4f}")
        print(f"- Profit factor: {profit_factor:.4f}")
        print(f"- Nombre de trades: {len(pnls)}")
        
        return {
            'total_return': total_return,
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'num_trades': len(pnls)
        }
    
    def _cleanup_old_data(self):
        """Supprime les données de performance plus anciennes que la période de lookback."""
        cutoff_date = datetime.now(timezone.utc) - timedelta(days=self.lookback_period)
        
        for strategy in list(self.performance_data.keys()):
            # Filtrer les données plus récentes que la date de coupure
            filtered_data = []
            for d in self.performance_data[strategy]:
                # Normaliser le fuseau horaire pour la comparaison
                timestamp = d['timestamp']
                if timestamp.tzinfo is None:
                    timestamp = timestamp.replace(tzinfo=timezone.utc)
                else:
                    timestamp = timestamp.astimezone(timezone.utc)
                
                if timestamp >= cutoff_date:
                    filtered_data.append(d)
            
            self.performance_data[strategy] = filtered_data
            
            # Supprimer la stratégie si elle n'a plus de données
            if not self.performance_data[strategy]:
                del self.performance_data[strategy]


class StrategySwitcher:
    """
    Classe pour gérer le switching entre différentes stratégies de trading
    en fonction des conditions du marché et des performances passées.
    """
    
    def __init__(self, config: Dict):
        """
        Initialise le système de switching de stratégies.
        
        Args:
            config: Configuration du système de switching
        """
        self.config = config
        self.performance_tracker = StrategyPerformance(
            lookback_period=config.get('performance_lookback_days', 30)
        )
        self.logger = logging.getLogger(__name__)
        
        # Dictionnaire des conditions de marché et des stratégies associées
        self.strategy_conditions = {
            MarketCondition.TRENDING_UP: ['trend_following', 'momentum'],
            MarketCondition.TRENDING_DOWN: ['trend_following', 'mean_reversion'],
            MarketCondition.RANGING: ['mean_reversion', 'grid_trading'],
            MarketCondition.HIGH_VOLATILITY: ['volatility_breakout', 'grid_trading'],
            MarketCondition.LOW_VOLATILITY: ['mean_reversion', 'grid_trading'],
            MarketCondition.BREAKOUT: ['momentum', 'trend_following'],
            MarketCondition.BREAKDOWN: ['mean_reversion', 'grid_trading']
        }
        
        # Poids initiaux pour chaque stratégie
        self.strategy_weights = {
            'trend_following': 1.0,
            'mean_reversion': 1.0,
            'momentum': 1.0,
            'volatility_breakout': 1.0,
            'grid_trading': 1.0,
            'ml': 1.0
        }
        
        # Dernière condition de marché détectée
        self.last_market_condition = None
        
    def analyze_market_conditions(self, market_data) -> MarketCondition:
        """
        Analyse les conditions actuelles du marché.
        
        Args:
            market_data: Données de marché (OHLCV). Peut être un DataFrame pandas,
                        un dictionnaire avec clé 'candles' contenant un DataFrame ou un objet itérable,
                        ou un objet avec une interface compatible (comme un mock de test).
            
        Returns:
            Condition de marché détectée
            
        Raises:
            ValueError: Si le format des données de marché n'est pas pris en charge
        """
        try:
            # Si c'est déjà un DataFrame, l'utiliser directement
            if isinstance(market_data, pd.DataFrame):
                df = market_data.copy()
            # Si c'est un dictionnaire avec une clé 'candles'
            elif isinstance(market_data, dict) and 'candles' in market_data:
                # Si 'candles' est un DataFrame ou un objet compatible
                if hasattr(market_data['candles'], 'to_dict'):
                    df = pd.DataFrame(market_data['candles'])
                # Si 'candles' est déjà un itérable de dictionnaires
                elif isinstance(market_data['candles'], (list, tuple)) and all(isinstance(x, dict) for x in market_data['candles']):
                    df = pd.DataFrame(market_data['candles'])
                # Si c'est un objet avec une interface de type DataFrame (comme notre mock)
                elif hasattr(market_data['candles'], '__getitem__') and hasattr(market_data['candles'], 'iloc'):
                    df = market_data['candles']
                else:
                    # Essayer de convertir en DataFrame directement
                    df = pd.DataFrame(market_data['candles'])
            # Si c'est un objet avec une interface de type DataFrame (comme notre mock)
            elif hasattr(market_data, '__getitem__') and hasattr(market_data, 'iloc'):
                df = market_data
            # Autres cas non gérés
            else:
                raise ValueError(
                    "Format de données de marché non pris en charge. "
                    "Attendu: un DataFrame, un dictionnaire avec clé 'candles', "
                    "ou un objet avec une interface compatible."
                )
            
            # Calculer les indicateurs
            df['atr'] = calculate_atr(df['high'], df['low'], df['close'])
            df['sma_50'] = df['close'].rolling(window=50).mean()
            df['sma_200'] = df['close'].rolling(window=200).mean()
            
            # Calculer l'ADX pour la force de la tendance
            df['adx'], df['plus_di'], df['minus_di'] = calculate_adx(
                df['high'], df['low'], df['close']
            )
            
            # Dernières valeurs
            last = df.iloc[-1]
            prev = df.iloc[-2]
            
            # Détecter la tendance
            adx_threshold = 25
            di_crossover = (last['plus_di'] > last['minus_di']) and (prev['plus_di'] <= prev['minus_di'])
            di_crossunder = (last['plus_di'] < last['minus_di']) and (prev['plus_di'] >= prev['minus_di'])
            
            # Détecter la volatilité
            atr_percent = last['atr'] / last['close'] * 100
            high_volatility = atr_percent > 2.0  # Plus de 2% de volatilité
            low_volatility = atr_percent < 0.5   # Moins de 0.5% de volatilité
            
            # Détecter un range
            is_ranging = last['adx'] < 20 and not high_volatility
            
            # Détecter un breakout/breakdown
            is_breakout = (last['close'] > df['high'].rolling(window=20).max().shift(1).iloc[-1])
            is_breakdown = (last['close'] < df['low'].rolling(window=20).min().shift(1).iloc[-1])
            
            # Déterminer la condition de marché
            if is_breakout:
                return MarketCondition.BREAKOUT
            elif is_breakdown:
                return MarketCondition.BREAKDOWN
            elif last['adx'] > adx_threshold and last['plus_di'] > last['minus_di']:
                return MarketCondition.TRENDING_UP
            elif last['adx'] > adx_threshold and last['plus_di'] < last['minus_di']:
                return MarketCondition.TRENDING_DOWN
            elif is_ranging:
                return MarketCondition.RANGING
            elif high_volatility:
                return MarketCondition.HIGH_VOLATILITY
            elif low_volatility:
                return MarketCondition.LOW_VOLATILITY
            else:
                return MarketCondition.RANGING
                
        except Exception as e:
            self.logger.error(f"Erreur lors de l'analyse des conditions du marché: {e}")
            return MarketCondition.RANGING  # Par défaut, considérer un marché en range
    
    def update_strategy_performance(self, strategy_name: str, trade_result: Dict):
        """
        Met à jour les performances d'une stratégie.
        
        Args:
            strategy_name: Nom de la stratégie
            trade_result: Résultats du trade (doit contenir 'pnl' et 'timestamp')
        """
        if 'pnl' in trade_result and 'timestamp' in trade_result:
            self.performance_tracker.update_performance(
                strategy_name,
                trade_result['pnl'],
                trade_result['timestamp']
            )
    
    def get_best_strategies(self, market_data: Dict, top_n: int = 2) -> List[Tuple[str, float]]:
        """
        Retourne les meilleures stratégies pour les conditions de marché actuelles.
        
        Args:
            market_data: Données de marché
            top_n: Nombre de stratégies à retourner
            
        Returns:
            Liste de tuples (nom_stratégie, score)
        """
        # Analyser les conditions du marché
        market_condition = self.analyze_market_conditions(market_data)
        self.last_market_condition = market_condition
        
        # Obtenir les stratégies recommandées pour ces conditions
        recommended_strategies = self.strategy_conditions.get(market_condition, [])
        
        # Si aucune stratégie n'est recommandée, retourner une liste vide
        if not recommended_strategies:
            return []
        
        # Calculer les scores pour chaque stratégie
        strategy_scores = []
        print("\n=== DÉBOGAGE: Calcul des scores pour les stratégies recommandées ===")
        for strategy in recommended_strategies:
            # Récupérer les performances de la stratégie
            perf = self.performance_tracker.get_performance_metrics(strategy)
            
            # Si on a des données de performance, calculer un score
            if perf and perf['num_trades'] > 0:
                # Afficher les métriques brutes pour le débogage
                print(f"\n--- {strategy} ---")
                print(f"- Profit Factor: {perf['profit_factor']:.4f}")
                print(f"- Win Rate: {perf['win_rate']:.4f}")
                print(f"- Total Return: {perf['total_return']:.4f}")
                
                # Score basé sur le facteur de profit, le taux de réussite et le rendement total
                # On donne plus de poids au win_rate (40%) et au profit_factor (40%) qu'au rendement total (20%)
                # pour équilibrer entre consistance et performance absolue
                profit_factor_score = min(perf['profit_factor'], 10.0) * 0.4  # Limiter le profit factor à 10 pour éviter les valeurs extrêmes
                win_rate_score = perf['win_rate'] * 0.4
                # Normaliser le rendement total entre 0 et 1 pour les besoins du score
                total_return_score = np.tanh(abs(perf['total_return'])) * 0.2
                
                score = profit_factor_score + win_rate_score + total_return_score
                
                # Afficher les scores intermédiaires pour le débogage
                print(f"- Profit Factor Score: {profit_factor_score:.4f}")
                print(f"- Win Rate Score: {win_rate_score:.4f}")
                print(f"- Total Return Score: {total_return_score:.4f}")
                print(f"- SCORE FINAL: {score:.4f}")
                
                # Log des métriques pour le débogage
                self.logger.debug(f"\nCalcul du score pour {strategy}:")
                self.logger.debug(f"- Profit Factor: {perf['profit_factor']:.4f} (limité à 10) * 0.4 = {profit_factor_score:.4f}")
                self.logger.debug(f"- Win Rate: {perf['win_rate']:.4f} * 0.4 = {win_rate_score:.4f}")
                self.logger.debug(f"- Total Return Score: {perf['total_return']:.4f} (tanh) * 0.2 = {total_return_score:.4f}")
                self.logger.debug(f"- Score total: {score:.4f}")
                
                strategy_scores.append((strategy, score))
            else:
                # Score par défaut si pas assez de données
                self.logger.debug(f"\nStratégie: {strategy} - Pas assez de données, score par défaut: 1.0")
                strategy_scores.append((strategy, 1.0))
        
        # Trier par score décroissant
        strategy_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Retourner les top_n stratégies
        return strategy_scores[:top_n]
    
    def adjust_strategy_weights(self, market_data: Dict):
        """
        Ajuste les poids des stratégies en fonction des performances récentes.
        
        Args:
            market_data: Données de marché
        """
        print("\n=== DÉBUT adjust_strategy_weights ===")
        print("Meilleures stratégies avec scores:")
        
        # Obtenir les meilleures stratégies actuelles avec leurs scores
        best_strategies = self.get_best_strategies(market_data, top_n=len(self.strategy_weights))
        
        if not best_strategies:
            return
        
        # Afficher les scores pour le débogage
        for strategy, score in best_strategies:
            print(f"- {strategy}: {score:.4f}")
        
        # Extraire les noms des stratégies et leurs scores
        strategies, scores = zip(*best_strategies)
        
        # Calculer les scores normalisés entre 0.5 et 2.0
        min_score = min(scores)
        max_score = max(scores)
        score_range = max_score - min_score if max_score > min_score else 1.0
        
        print(f"\nScores normalisés entre 0.5 et 2.0:")
        print(f"- Min score: {min_score:.4f}")
        print(f"- Max score: {max_score:.4f}")
        print(f"- Plage de scores: {score_range:.4f}")
        
        # Calculer les poids bruts pour chaque stratégie recommandée
        print("\nCalcul des poids bruts:")
        raw_weights = {}
        
        # 1. Calculer les poids bruts basés sur les scores normalisés entre 0.5 et 2.0
        for strategy, score in best_strategies:
            if score_range > 0:
                # Normaliser le score entre 0.5 et 2.0
                normalized_score = 0.5 + 1.5 * ((score - min_score) / score_range)
                raw_weights[strategy] = normalized_score
            else:
                # Si tous les scores sont identiques, utiliser un poids de 1.0
                raw_weights[strategy] = 1.0
            print(f"- {strategy}: score={score:.4f} -> poids_brut={raw_weights[strategy]:.4f}")
        
        # 2. Trier les stratégies par score décroissant
        sorted_strategies = sorted(raw_weights.items(), key=lambda x: x[1], reverse=True)
        
        # 3. Initialiser les poids finaux
        final_weights = {s: 0.0 for s in self.strategy_weights}
        
        # 4. Attribuer les poids en respectant l'ordre des scores
        #    et en garantissant que la stratégie avec le meilleur score ait le poids le plus élevé
        print("\nAttribution des poids finaux:")
        
        # La stratégie avec le meilleur score reçoit le poids le plus élevé (2.0)
        best_strategy = sorted_strategies[0][0]
        final_weights[best_strategy] = 2.0
        print(f"- {best_strategy}: Poids fixé à 2.0 (meilleure stratégie)")
        
        # Les autres stratégies reçoivent un poids proportionnel à leur score, mais pas plus que 1.0
        if len(sorted_strategies) > 1:
            # Calculer la somme des scores pour les stratégies restantes
            remaining_score_sum = sum(score for _, score in sorted_strategies[1:])
            
            if remaining_score_sum > 0:
                # Calculer le poids total restant à attribuer (1.0 - 2.0 = -1.0, donc on ajuste)
                total_weight_remaining = max(0.5 * (len(sorted_strategies) - 1), 0.1)
                
                # Répartir le poids restant proportionnellement aux scores
                for strategy, score in sorted_strategies[1:]:
                    weight = (score / remaining_score_sum) * total_weight_remaining
                    final_weights[strategy] = max(min(weight, 1.0), 0.1)  # Bornes entre 0.1 et 1.0
                    print(f"- {strategy}: Poids fixé à {final_weights[strategy]:.4f}")
        
        # Normaliser les poids pour que leur somme soit égale à 1.0
        total_weight = sum(final_weights.values())
        if total_weight > 0:
            final_weights = {s: w / total_weight for s, w in final_weights.items()}
        
        # Mettre à jour les poids des stratégies
        self.strategy_weights = final_weights
        
        # Afficher les poids finaux
        print("\nPoids finaux après ajustement:")
        for strategy, weight in sorted(self.strategy_weights.items(), key=lambda x: x[1], reverse=True):
            print(f"- {strategy}: {weight:.4f}")
        
        print("=== FIN adjust_strategy_weights ===\n")
    def get_strategy_weights(self) -> Dict[str, float]:
        """
        Retourne les poids actuels des stratégies.
        
        Returns:
            Dictionnaire des poids des stratégies
        """
        return self.strategy_weights.copy()
    
    def get_market_condition(self) -> Optional[MarketCondition]:
        """
        Retourne la dernière condition de marché détectée.
        
        Returns:
            Dernière condition de marché détectée ou None si aucune
        """
        return self.last_market_condition
