from typing import Dict, Any, Optional
import pandas as pd
from .technical_analyzer import TechnicalAnalyzer
from .sentiment_analyzer import SentimentAnalyzer
from .ml_predictor import MLPredictor
from .market.order_book_signal_adapter import OrderBookSignalAdapter
from .market.order_book import OrderBookManager
import logging
from datetime import datetime


class SignalGenerator:
    """Génération de signaux de trading basée sur différents indicateurs."""

    def __init__(self, order_book_manager: Optional[OrderBookManager] = None):
        """
        Initialise le générateur de signaux.
        
        Args:
            order_book_manager: Gestionnaire de carnet d'ordres optionnel
        """
        self.technical_analyzer = TechnicalAnalyzer()
        self.sentiment_analyzer = SentimentAnalyzer()
        self.ml_predictor = MLPredictor()
        self.logger = logging.getLogger('signal_generator')
        
        # Initialisation de l'adaptateur pour les signaux du carnet d'ordres
        self.order_book_adapter = None
        if order_book_manager:
            self.order_book_adapter = OrderBookSignalAdapter(order_book_manager)

    def generate_signals(self, price_history: pd.Series) -> Dict[str, Any]:
        """
        Génère les signaux de trading.

        Args:
            price_history: Historique des prix (série pandas avec un index datetime)

        Returns:
            Dictionnaire de signaux avec les clés :
            - technical: indicateurs techniques classiques
            - order_book: métriques avancées du carnet d'ordres
            - sentiment: analyse de sentiment
            - ml: prédictions par machine learning
            - combined: signaux combinés et recommandations
        """
        try:
            signals = {}
            signals['timestamp'] = datetime.utcnow().isoformat()
            
            # 1. Analyse technique classique
            indicators = self.technical_analyzer.calculate_indicators(price_history)
            signals['technical'] = {
                'trend': self.technical_analyzer.analyze_trend(price_history),
                'volatility': self.technical_analyzer.analyze_volatility(price_history),
                'rsi': indicators.get('rsi', pd.Series([0])).iloc[-1] if 'rsi' in indicators else 0,
                'macd': indicators.get('macd', pd.Series([0])).iloc[-1] if 'macd' in indicators else 0,
                'indicators': {k: v.iloc[-1] if hasattr(v, 'iloc') else v 
                             for k, v in indicators.items()}
            }
            
            # 2. Analyse du carnet d'ordres (si disponible)
            signals['order_book'] = self._generate_order_book_signals()
            
            # 3. Analyse du sentiment
            signals['sentiment'] = {
                'price': self.sentiment_analyzer.get_price_sentiment(price_history),
                'market': 'neutral'  # À implémenter avec des données de news
            }
            
            # 4. Prédiction ML
            self.ml_predictor.train(price_history)
            signals['ml'] = {
                'prediction': self.ml_predictor.predict(price_history),
                'confidence': 0.8  # Exemple de confiance
            }
            
            # 5. Signaux combinés et recommandations
            signals['combined'] = self._combine_signals(signals)

            return signals

        except Exception as e:
            self.logger.error(f"Erreur lors de la génération des signaux: {e}", exc_info=True)
            return {'error': str(e)}

    def _generate_order_book_signals(self) -> Dict[str, Any]:
        """
        Génère des signaux basés sur l'analyse du carnet d'ordres.
        
        Returns:
            Dictionnaire des signaux du carnet d'ordres
        """
        if not self.order_book_adapter:
            return {'available': False, 'message': 'Order book manager non disponible'}
            
        try:
            # Mettre à jour les indicateurs avec les dernières données
            if not self.order_book_adapter.update():
                return {'available': False, 'message': 'Impossible de mettre à jour les données du carnet'}
            
            # Récupérer les signaux de base
            signals = self.order_book_adapter.get_signals()
            
            # Ajouter les niveaux de support/résistance
            support_resistance = self.order_book_adapter.get_support_resistance_levels(num_levels=3)
            signals['support_resistance'] = support_resistance
            
            # Ajouter l'analyse du flux d'ordres
            order_flow = self.order_book_adapter.get_order_flow_signals()
            signals.update(order_flow)
            
            # Ajouter les zones de liquidité
            liquidity_zones = self.order_book_adapter.get_liquidity_zones(num_zones=3)
            signals['liquidity_zones'] = liquidity_zones
            
            # Ajouter un indicateur de qualité de marché
            signals['market_quality'] = self._assess_market_quality(signals)
            
            signals['available'] = True
            return signals
            
        except Exception as e:
            self.logger.error(f"Erreur lors de la génération des signaux du carnet d'ordres: {e}", exc_info=True)
            return {
                'available': False,
                'error': str(e),
                'message': 'Erreur lors de l\'analyse du carnet d\'ordres'
            }
    
    def _assess_market_quality(self, signals: Dict[str, Any]) -> Dict[str, Any]:
        """
        Évalue la qualité du marché basée sur les signaux du carnet d'ordres.
        
        Args:
            signals: Dictionnaire des signaux du carnet d'ordres
            
        Returns:
            Dictionnaire d'évaluation de la qualité du marché
        """
        quality = {
            'liquidity': 'medium',
            'volatility': 'medium',
            'spread': 'medium',
            'order_imbalance': 'neutral',
            'overall_quality': 'medium',
            'suggested_strategy': 'mean_reversion'  # Par défaut
        }
        
        try:
            # Évaluer la liquidité
            bid_liq = signals.get('liquidity_bid_0.5%', 0)
            ask_liq = signals.get('liquidity_ask_0.5%', 0)
            
            if bid_liq > 100 and ask_liq > 100:  # Valeurs arbitraires à ajuster
                quality['liquidity'] = 'high'
            elif bid_liq < 10 or ask_liq < 10:
                quality['liquidity'] = 'low'
            
            # Évaluer la volatilité (basée sur le spread et les pentes)
            spread = signals.get('spread', 0)
            mid_price = signals.get('mid_price', 1)
            spread_pct = (spread / mid_price) * 100 if mid_price > 0 else 0
            
            if spread_pct > 0.5:  # 0.5% de spread
                quality['volatility'] = 'high'
                quality['spread'] = 'wide'
            elif spread_pct < 0.1:
                quality['volatility'] = 'low'
                quality['spread'] = 'tight'
            
            # Évaluer le déséquilibre d'ordres
            imbalance = signals.get('order_flow_imbalance', 0)
            if imbalance > 0.3:
                quality['order_imbalance'] = 'strong_buy'
            elif imbalance > 0.1:
                quality['order_imbalance'] = 'buy'
            elif imbalance < -0.3:
                quality['order_imbalance'] = 'strong_sell'
            elif imbalance < -0.1:
                quality['order_imbalance'] = 'sell'
            
            # Déterminer la qualité globale
            if quality['liquidity'] == 'high' and quality['volatility'] == 'medium' and quality['spread'] == 'tight':
                quality['overall_quality'] = 'high'
                quality['suggested_strategy'] = 'market_making'
            elif quality['volatility'] == 'high':
                quality['suggested_strategy'] = 'breakout'
            elif quality['order_imbalance'] != 'neutral':
                quality['suggested_strategy'] = 'momentum'
            
            return quality
            
        except Exception as e:
            self.logger.error(f"Erreur lors de l'évaluation de la qualité du marché: {e}")
            return quality
    
    def _combine_signals(self, signals: Dict[str, Any]) -> Dict[str, Any]:
        """
        Combine les différents signaux en un signal final et des recommandations.
        
        Args:
            signals: Dictionnaire des signaux à combiner
            
        Returns:
            Dictionnaire du signal final avec recommandations
        """
        combined = {
            'final_signal': 'neutral',
            'strength': 0.0,
            'confidence': 0.5,
            'recommended_actions': [],
            'risk_level': 'medium',
            'market_conditions': {}
        }
        
        try:
            # 1. Évaluer les conditions du marché
            market_quality = signals.get('order_book', {}).get('market_quality', {})
            combined['market_conditions'] = market_quality
            
            # 2. Combiner les signaux techniques et du carnet d'ordres
            tech_signal = signals.get('technical', {})
            ob_signals = signals.get('order_book', {})
            
            # 3. Détection de tendance
            trend = tech_signal.get('trend', 'neutral')
            trend_strength = 0.0
            
            if trend == 'up':
                trend_strength = 1.0
            elif trend == 'down':
                trend_strength = -1.0
            
            # 4. Déséquilibre d'ordres
            order_imbalance = ob_signals.get('order_flow_imbalance', 0)
            
            # 5. Signaux de support/résistance
            support_resistance = ob_signals.get('support_resistance', {})
            
            # 6. Combinaison des signaux (exemple simplifié)
            combined_signal = (trend_strength * 0.4) + (order_imbalance * 0.6)
            
            # 7. Déterminer le signal final
            if combined_signal > 0.3:
                combined['final_signal'] = 'strong_buy'
                combined['strength'] = min(1.0, (combined_signal - 0.3) * 1.4)
            elif combined_signal > 0.1:
                combined['final_signal'] = 'buy'
                combined['strength'] = (combined_signal - 0.1) * 5
            elif combined_signal < -0.3:
                combined['final_signal'] = 'strong_sell'
                combined['strength'] = min(1.0, (-combined_signal - 0.3) * 1.4)
            elif combined_signal < -0.1:
                combined['final_signal'] = 'sell'
                combined['strength'] = (-combined_signal - 0.1) * 5
            
            # 8. Ajuster la confiance
            confidence = 0.7  # Valeur de base
            
            # Augmenter la confiance si les signaux sont cohérents
            if (trend_strength > 0 and order_imbalance > 0) or \
               (trend_strength < 0 and order_imbalance < 0):
                confidence = min(0.9, confidence + 0.15)
            
            # Réduire la confiance en cas de divergence
            if (trend_strength > 0 and order_imbalance < -0.2) or \
               (trend_strength < 0 and order_imbalance > 0.2):
                confidence = max(0.3, confidence - 0.2)
            
            combined['confidence'] = confidence
            
            # 9. Recommandations
            if market_quality.get('suggested_strategy') == 'market_making' and abs(combined_signal) < 0.2:
                combined['recommended_actions'].append('market_making')
            elif combined['final_signal'] in ['strong_buy', 'strong_sell']:
                combined['recommended_actions'].append('trend_following')
            elif market_quality.get('volatility') == 'high':
                combined['recommended_actions'].append('breakout_trading')
            
            # 10. Niveau de risque
            if market_quality.get('liquidity') == 'low' or market_quality.get('volatility') == 'high':
                combined['risk_level'] = 'high'
            elif market_quality.get('liquidity') == 'high' and market_quality.get('volatility') == 'low':
                combined['risk_level'] = 'low'
            
            return combined
            
        except Exception as e:
            self.logger.error(f"Erreur lors de la combinaison des signaux: {e}", exc_info=True)
            combined['error'] = str(e)
            return combined
