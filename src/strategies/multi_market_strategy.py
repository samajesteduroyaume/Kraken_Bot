"""
Module implémentant une stratégie de trading multi-marchés.

Cette stratégie permet de gérer plusieurs marchés simultanément en tenant compte
des corrélations entre les actifs et en optimisant l'allocation du capital.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
from datetime import datetime

@dataclass
class MarketGroup:
    """Classe représentant un groupe de marchés corrélés."""
    name: str
    symbols: List[str]
    max_exposure: float

class MultiMarketStrategy:
    """
    Stratégie de trading multi-marchés.
    
    Cette stratégie permet de :
    - Gérer plusieurs marchés simultanément
    - Prendre en compte les corrélations entre actifs
    - Optimiser l'allocation du capital
    - Gérer les risques de manière globale
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialise la stratégie multi-marchés.
        
        Args:
            config: Configuration de la stratégie
        """
        self.enabled = config.get('enabled', True)
        self.weight = config.get('weight', 0.5)
        self.risk_multiplier = config.get('risk_multiplier', 1.0)
        
        # Initialisation des groupes de marchés
        self.market_groups = [
            MarketGroup(
                name=group['name'],
                symbols=group['symbols'],
                max_exposure=group['max_exposure']
            )
            for group in config.get('market_groups', [])
        ]
        
        # Paramètres de diversification
        self.diversification = config.get('diversification', {
            'max_per_market': 0.2,
            'min_correlation': -0.3
        })
        
        # Fournisseur de données (sera injecté par le framework)
        self.data_provider = None
    
    def calculate_correlations(self) -> pd.DataFrame:
        """
        Calcule la matrice de corrélation entre les marchés.
        
        Returns:
            DataFrame: Matrice de corrélation entre les paires de marchés
        """
        symbols = []
        for group in self.market_groups:
            symbols.extend(group.symbols)
        
        # Création d'une matrice de corrélation vide
        corr_matrix = pd.DataFrame(
            np.eye(len(symbols)), 
            index=symbols, 
            columns=symbols
        )
        
        # Remplissage de la matrice avec les corrélations
        for i, sym1 in enumerate(symbols):
            for j, sym2 in enumerate(symbols[i+1:], i+1):
                if sym1 != sym2:
                    corr = self.data_provider.get_correlation(sym1, sym2)
                    corr_matrix.loc[sym1, sym2] = corr
                    corr_matrix.loc[sym2, sym1] = corr
        
        return corr_matrix
    
    def allocate_capital(self, portfolio_value: float) -> Dict[str, float]:
        """
        Alloue le capital entre les différents marchés.
        
        Args:
            portfolio_value: Valeur totale du portefeuille
            
        Returns:
            Dict: Dictionnaire des allocations par marché
        """
        allocations = {}
        
        # Allocation de base proportionnelle aux poids des groupes
        for group in self.market_groups:
            group_allocation = portfolio_value * group.max_exposure
            per_symbol = group_allocation / len(group.symbols)
            
            for symbol in group.symbols:
                # Limite par marché
                max_per_market = self.diversification['max_per_market'] * portfolio_value
                allocations[symbol] = min(per_symbol, max_per_market)
        
        return allocations
    
    def generate_signals(self) -> Dict[str, Dict[str, Any]]:
        """
        Génère des signaux de trading pour chaque marché.
        
        Returns:
            Dict: Dictionnaire des signaux par marché
        """
        signals = {}
        
        for group in self.market_groups:
            for symbol in group.symbols:
                # Récupération des données de marché
                market_data = self.data_provider.get_historical_data(symbol)
                
                # Calcul des indicateurs
                indicators = self.data_provider.calculate_indicators(market_data)
                
                # Génération du signal (stratégie simplifiée)
                signal_strength = self._calculate_signal_strength(indicators)
                signal = self._generate_signal(indicators, signal_strength)
                
                signals[symbol] = {
                    'signal': signal,
                    'strength': signal_strength,
                    'timestamp': datetime.utcnow()
                }
        
        return signals
    
    def _calculate_signal_strength(self, indicators: Dict[str, Any]) -> float:
        """
        Calcule la force du signal basée sur les indicateurs.
        
        Args:
            indicators: Dictionnaire des indicateurs techniques
            
        Returns:
            float: Force du signal entre -1 (fort vente) et 1 (fort achat)
        """
        # Implémentation simplifiée
        rsi = indicators.get('rsi', 50)
        macd = indicators.get('macd', 0)
        
        # Normalisation des indicateurs
        rsi_strength = (rsi - 50) / 50  # Entre -1 et 1
        macd_strength = np.tanh(macd / 100)  # Normalisation approximative
        
        # Moyenne pondérée des indicateurs
        return (rsi_strength * 0.4 + macd_strength * 0.6) * self.risk_multiplier
    
    def _generate_signal(self, indicators: Dict[str, Any], strength: float) -> str:
        """
        Génère un signal de trading basé sur la force du signal.
        
        Args:
            indicators: Dictionnaire des indicateurs techniques
            strength: Force du signal
            
        Returns:
            str: Signal ('BUY', 'SELL' ou 'HOLD')
        """
        if strength > 0.2:
            return 'BUY'
        elif strength < -0.2:
            return 'SELL'
        else:
            return 'HOLD'
    
    def assess_risk(self, portfolio: Dict[str, Dict[str, float]]) -> Dict[str, Any]:
        """
        Évalue les risques du portefeuille actuel.
        
        Args:
            portfolio: État actuel du portefeuille
            
        Returns:
            Dict: Évaluation des risques
        """
        total_risk = 0.0
        per_market_risk = {}
        
        for symbol, position in portfolio.items():
            # Calcul simplifié du risque par marché
            market_risk = position.get('position', 0) * position.get('volatility', 0.1)
            per_market_risk[symbol] = market_risk
            total_risk += market_risk
        
        # Normalisation du risque total
        total_risk = min(total_risk, 1.0)
        
        return {
            'total_risk': total_risk,
            'per_market_risk': per_market_risk
        }
    
    def calculate_rebalancing(self, current_positions: Dict[str, float]) -> Dict[str, float]:
        """
        Calcule les ajustements nécessaires pour rééquilibrer le portefeuille.
        
        Args:
            current_positions: Positions actuelles par marché
            
        Returns:
            Dict: Ajustements à apporter à chaque position
        """
        rebalancing = {}
        
        # Calcul des allocations cibles
        total_value = sum(current_positions.values())
        target_allocations = self.allocate_capital(total_value)
        
        # Calcul des ajustements
        for symbol, current in current_positions.items():
            target = target_allocations.get(symbol, 0)
            rebalancing[symbol] = target - current
        
        return rebalancing
    
    def analyze_correlations(self) -> Dict[str, Any]:
        """
        Analyse les corrélations entre les marchés.
        
        Returns:
            Dict: Résultats de l'analyse des corrélations
        """
        corr_matrix = self.calculate_correlations()
        
        # Détection des paires fortement corrélées
        high_correlation_pairs = []
        symbols = corr_matrix.columns
        
        for i, sym1 in enumerate(symbols):
            for sym2 in symbols[i+1:]:
                corr = corr_matrix.loc[sym1, sym2]
                if abs(corr) > 0.7:  # Seuil de corrélation élevée
                    high_correlation_pairs.append((sym1, sym2, corr))
        
        return {
            'correlation_matrix': corr_matrix,
            'high_correlation_pairs': high_correlation_pairs
        }
    
    def calculate_performance_metrics(self, returns: pd.Series) -> Dict[str, float]:
        """
        Calcule les métriques de performance.
        
        Args:
            returns: Série des rendements
            
        Returns:
            Dict: Métriques de performance
        """
        if len(returns) < 2:
            return {}
        
        # Calcul du rendement total
        total_return = (1 + returns).prod() - 1
        
        # Calcul de la volatilité
        volatility = returns.std() * np.sqrt(252)  # Annualisation
        
        # Calcul du ratio de Sharpe (simplifié)
        sharpe = (returns.mean() / (returns.std() + 1e-9)) * np.sqrt(252)
        
        # Calcul du drawdown maximum
        cum_returns = (1 + returns).cumprod()
        peak = cum_returns.expanding(min_periods=1).max()
        drawdowns = (cum_returns - peak) / peak
        max_drawdown = drawdowns.min()
        
        # Taux de réussite
        win_rate = (returns > 0).mean()
        
        return {
            'total_return': total_return,
            'volatility': volatility,
            'sharpe': sharpe,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate
        }
