import logging
from typing import Dict, List, Optional
from decimal import Decimal
from datetime import datetime
from src.core.types.market_data import MarketData
from src.core.types import TradeSignal
from src.core.analysis.technical import TechnicalAnalyzer


class SignalGenerator:
    """
    Générateur de signaux de trading basé sur l'analyse technique et ML.

    Cette classe génère des signaux de trading en utilisant :
    - Analyse technique (indicateurs, patterns, divergences)
    - Modèles de machine learning
    - Gestion du risque
    """

    def __init__(self,
                 predictor=None,  # Instance du prédicteur ML
                 config: Optional[Dict] = None):
        """
        Initialise le générateur de signaux.

        Args:
            predictor: Instance du prédicteur ML
            config: Configuration du générateur
        """
        self.predictor = predictor
        self.config = config or {}
        self.technical_analyzer = TechnicalAnalyzer(
            window_size=self.config.get('window_size', 20)
        )
        self.logger = logging.getLogger(__name__)

    def generate_signals(self, market_data: MarketData) -> List[TradeSignal]:
        """
        Génère les signaux de trading pour une paire.

        Args:
            market_data: Données de marché pour la paire

        Returns:
            Liste des signaux de trading générés
        """
        if not market_data.get('candles'):
            self.logger.warning(
                f"Pas de données de bougies pour {market_data['symbol']}")
            return []

        # Analyse technique
        technical_indicators = self.technical_analyzer.analyze_candles(
            market_data['candles'])
        patterns = self.technical_analyzer.detect_patterns(
            market_data['candles'])
        divergences = self.technical_analyzer.detect_divergences(
            market_data['candles'],
            [candle['close'] for candle in market_data['candles']]
        )

        # Prédictions ML
        ml_signal = None
        if self.predictor:
            try:
                ml_signal = self._generate_ml_signal(market_data)
            except Exception as e:
                self.logger.error(
                    f"Erreur lors de la génération du signal ML: {e}")

        # Génération des signaux basés sur les règles
        signals = []

        # Signal basé sur RSI
        if technical_indicators.get('rsi'):
            signals.extend(self._generate_rsi_signals(
                market_data['symbol'],
                technical_indicators['rsi'],
                market_data['candles'][-1]['close']
            ))

        # Signal basé sur MACD
        if all([technical_indicators.get(k)
               for k in ['macd', 'macd_signal', 'macd_hist']]):
            signals.extend(self._generate_macd_signals(
                market_data['symbol'],
                technical_indicators['macd'],
                technical_indicators['macd_signal'],
                technical_indicators['macd_hist'],
                market_data['candles'][-1]['close']
            ))

        # Signal basé sur les patterns
        if any(patterns.values()):
            signals.extend(self._generate_pattern_signals(
                market_data['symbol'],
                patterns,
                market_data['candles'][-1]['close']
            ))

        # Signal basé sur les divergences
        if any(divergences.values()):
            signals.extend(self._generate_divergence_signals(
                market_data['symbol'],
                divergences,
                market_data['candles'][-1]['close']
            ))

        # Signal ML
        if ml_signal:
            signals.append(ml_signal)

        # Filtrage et consolidation des signaux
        return self._filter_and_combine_signals(signals)

    def _generate_rsi_signals(self,
                              symbol: str,
                              rsi: Decimal,
                              current_price: Decimal) -> List[TradeSignal]:
        """Génère des signaux basés sur le RSI."""
        signals = []

        if rsi < Decimal('30'):  # Survente
            signals.append({
                'symbol': symbol,
                'action': 'buy',
                'price': current_price,
                'confidence': float(Decimal('1.0') - (rsi / Decimal('30'))),
                'reason': 'RSI survente',
                'timestamp': datetime.now(),
                'rsi': float(rsi)
            })

        if rsi > Decimal('70'):  # Surachat
            signals.append({
                'symbol': symbol,
                'action': 'sell',
                'price': current_price,
                'confidence': float((rsi - Decimal('70')) / Decimal('30')),
                'reason': 'RSI surachat',
                'timestamp': datetime.now(),
                'rsi': float(rsi)
            })

        return signals

    def _generate_macd_signals(self,
                               symbol: str,
                               macd: Decimal,
                               macd_signal: Decimal,
                               macd_hist: Decimal,
                               current_price: Decimal) -> List[TradeSignal]:
        """Génère des signaux basés sur le MACD."""
        signals = []

        if macd > macd_signal and macd_hist > Decimal('0'):
            signals.append({
                'symbol': symbol,
                'action': 'buy',
                'price': current_price,
                'confidence': float(macd_hist / Decimal('100')),
                'reason': 'MACD croisement haussier',
                'timestamp': datetime.now(),
                'macd': float(macd),
                'macd_signal': float(macd_signal),
                'macd_hist': float(macd_hist)
            })

        if macd < macd_signal and macd_hist < Decimal('0'):
            signals.append({
                'symbol': symbol,
                'action': 'sell',
                'price': current_price,
                'confidence': float(-macd_hist / Decimal('100')),
                'reason': 'MACD croisement baissier',
                'timestamp': datetime.now(),
                'macd': float(macd),
                'macd_signal': float(macd_signal),
                'macd_hist': float(macd_hist)
            })

        return signals

    def _generate_pattern_signals(self,
                                  symbol: str,
                                  patterns: Dict[str, bool],
                                  current_price: Decimal) -> List[TradeSignal]:
        """Génère des signaux basés sur les patterns de bougies."""
        signals = []

        if patterns.get('bullish_engulfing'):
            signals.append({
                'symbol': symbol,
                'action': 'buy',
                'price': current_price,
                'confidence': float(Decimal('0.8')),
                'reason': 'Pattern Bullish Engulfing',
                'timestamp': datetime.now(),
                'pattern': 'bullish_engulfing'
            })

        if patterns.get('bearish_engulfing'):
            signals.append({
                'symbol': symbol,
                'action': 'sell',
                'price': current_price,
                'confidence': float(Decimal('0.8')),
                'reason': 'Pattern Bearish Engulfing',
                'timestamp': datetime.now(),
                'pattern': 'bearish_engulfing'
            })

        return signals

    def _generate_divergence_signals(self,
                                     symbol: str,
                                     divergences: Dict[str, bool],
                                     current_price: Decimal) -> List[TradeSignal]:
        """Génère des signaux basés sur les divergences."""
        signals = []

        if divergences.get('bullish_divergence'):
            signals.append({
                'symbol': symbol,
                'action': 'buy',
                'price': current_price,
                'confidence': float(Decimal('0.9')),
                'reason': 'Divergence haussière',
                'timestamp': datetime.now(),
                'divergence': 'bullish'
            })

        if divergences.get('bearish_divergence'):
            signals.append({
                'symbol': symbol,
                'action': 'sell',
                'price': current_price,
                'confidence': float(Decimal('0.9')),
                'reason': 'Divergence baissière',
                'timestamp': datetime.now(),
                'divergence': 'bearish'
            })

        return signals

    def _generate_ml_signal(
            self,
            market_data: MarketData) -> Optional[TradeSignal]:
        """Génère un signal basé sur le modèle ML."""
        if not self.predictor:
            return None

        try:
            prediction = self.predictor.predict(market_data)
            if prediction.get('action') and prediction.get('confidence'):
                return {
                    'symbol': market_data['symbol'],
                    'action': prediction['action'],
                    'price': market_data['candles'][-1]['close'],
                    'confidence': prediction['confidence'],
                    'reason': 'Prédiction ML',
                    'timestamp': datetime.now(),
                    'ml_prediction': prediction
                }
        except Exception as e:
            self.logger.error(f"Erreur lors de la prédiction ML: {e}")

        return None

    def _filter_and_combine_signals(
            self, signals: List[TradeSignal]) -> List[TradeSignal]:
        """
        Filtre et combine les signaux pour éviter les conflits.

        Cette méthode :
        - Supprime les signaux contradictoires
        - Combine les signaux similaires
        - Calcule une confiance consolidée

        Args:
            signals: Liste des signaux à filtrer

        Returns:
            Liste des signaux filtrés et consolidés
        """
        if not signals:
            return []

        # Regrouper par symbole et action
        grouped_signals = {}
        for signal in signals:
            key = (signal['symbol'], signal['action'])
            if key not in grouped_signals:
                grouped_signals[key] = []
            grouped_signals[key].append(signal)

        # Consolider les signaux
        consolidated_signals = []
        for (symbol, action), sig_list in grouped_signals.items():
            # Calculer la confiance moyenne
            avg_confidence = sum(s['confidence']
                                 for s in sig_list) / len(sig_list)

            # Combiner les raisons
            reasons = set(s['reason'] for s in sig_list)

            # Créer le signal consolidé
            consolidated_signals.append({
                'symbol': symbol,
                'action': action,
                # Utiliser le prix du premier signal
                'price': sig_list[0]['price'],
                'confidence': avg_confidence,
                'reason': ', '.join(reasons),
                'timestamp': datetime.now(),
                'indicators': {k: v for s in sig_list for k, v in s.items()
                               if k not in ['symbol', 'action', 'price', 'confidence', 'reason', 'timestamp']}
            })

        return consolidated_signals
