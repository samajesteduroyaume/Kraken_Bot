import logging
import pandas as pd


class AdaptiveLeverage:
    """Gestionnaire de levier adaptatif avancé."""

    def __init__(self,
                 min_leverage: float = 1.0,
                 max_leverage: float = 5.0,
                 volatility_threshold: float = 0.02,
                 risk_factor: float = 0.05,
                 volatility_window: int = 20,
                 trend_threshold: float = 0.01,
                 market_sentiment: str = 'neutral'):
        """
        Initialise le gestionnaire de levier adaptatif.

        Args:
            min_leverage: Levier minimum
            max_leverage: Levier maximum
            volatility_threshold: Seuil de volatilité
            risk_factor: Facteur de risque
            volatility_window: Fenêtre pour le calcul de la volatilité
            trend_threshold: Seuil pour détecter les tendances
            market_sentiment: Sentiment du marché ('bullish', 'bearish', 'neutral')
        """
        self.min_leverage = min_leverage
        self.max_leverage = max_leverage
        self.volatility_threshold = volatility_threshold
        self.risk_factor = risk_factor
        self.volatility_window = volatility_window
        self.trend_threshold = trend_threshold
        self.market_sentiment = market_sentiment
        self.logger = logging.getLogger('adaptive_leverage')

    def calculate_leverage(self,
                           volatility: float,
                           trend: float,
                           market_sentiment: str) -> float:
        """
        Calcule le levier en fonction de multiples facteurs.

        Args:
            volatility: Volatilité actuelle
            trend: Tendance du marché (-1 à 1)
            market_sentiment: Sentiment du marché

        Returns:
            float: Levier ajusté
        """
        # Calcul du levier de base
        base_leverage = self.max_leverage

        # Ajustement en fonction de la volatilité
        if volatility >= self.volatility_threshold:
            base_leverage = self.min_leverage

        # Ajustement en fonction de la tendance
        # Multiplie par 0.5 pour limiter l'effet
        trend_factor = 1 + (trend * 0.5)
        base_leverage *= trend_factor

        # Ajustement en fonction du sentiment du marché
        sentiment_factor = {
            'bullish': 1.2,
            'bearish': 0.8,
            'neutral': 1.0
        }[market_sentiment]
        base_leverage *= sentiment_factor

        # Limitation entre min et max
        return max(self.min_leverage, min(self.max_leverage, base_leverage))

    def calculate_position_size(self,
                                balance: float,
                                volatility: float,
                                entry_price: float,
                                stop_loss_percent: float,
                                trend: float,
                                market_sentiment: str) -> float:
        """
        Calcule la taille de position en fonction de multiples facteurs.

        Args:
            balance: Solde disponible
            volatility: Volatilité actuelle
            entry_price: Prix d'entrée
            stop_loss_percent: Pourcentage de stop-loss
            trend: Tendance du marché
            market_sentiment: Sentiment du marché

        Returns:
            float: Taille de position ajustée
        """
        # Calcul du levier adaptatif
        leverage = self.calculate_leverage(volatility, trend, market_sentiment)

        # Calcul du risque maximum ajusté
        risk_amount = balance * self.risk_factor

        # Ajustement du risque en fonction de la volatilité
        if volatility >= self.volatility_threshold:
            risk_amount *= 0.8  # Réduit le risque en cas de forte volatilité

        # Calcul de la taille de position
        stop_loss_price = entry_price * (1 - stop_loss_percent)
        position_size = (risk_amount / (entry_price -
                         stop_loss_price)) * leverage

        # Ajustement supplémentaire en fonction de la tendance
        position_size *= (1 + (trend * 0.3))

        return position_size

    def calculate_volatility(self, price_history: pd.Series) -> float:
        """
        Calcule la volatilité sur une série de prix avec une méthode avancée.

        Args:
            price_history: Série de prix historiques

        Returns:
            float: Volatilité calculée
        """
        if len(price_history) < self.volatility_window:
            return 0.0

        # Calcul des rendements
        returns = price_history.pct_change().dropna()

        # Calcul de la volatilité avec une moyenne mobile
        volatility = returns.rolling(
            window=self.volatility_window).std().iloc[-1] * (252 ** 0.5)

        # Ajustement pour les périodes de forte volatilité
        if volatility > self.volatility_threshold:
            volatility *= 1.2  # Amplifie la volatilité détectée

        return float(volatility)

    def calculate_trend(self, price_history: pd.Series) -> float:
        """
        Calcule la tendance du marché.

        Args:
            price_history: Série de prix historiques

        Returns:
            float: Tendance (-1 à 1)
        """
        if len(price_history) < 2:
            return 0.0

        # Calcul des rendements
        returns = price_history.pct_change().dropna()

        # Moyenne mobile rapide et lente
        fast_ma = returns.rolling(window=10).mean()
        slow_ma = returns.rolling(window=30).mean()

        # Tendance basée sur la différence entre les moyennes mobiles
        trend = (fast_ma.iloc[-1] - slow_ma.iloc[-1]) / \
            (abs(fast_ma.iloc[-1]) + abs(slow_ma.iloc[-1]) + 1e-8)

        # Limitation entre -1 et 1
        return max(-1, min(1, trend))

    def detect_market_sentiment(self, price_history: pd.Series) -> str:
        """
        Détermine le sentiment du marché.

        Args:
            price_history: Série de prix historiques

        Returns:
            str: Sentiment ('bullish', 'bearish', 'neutral')
        """
        if len(price_history) < 20:
            return 'neutral'

        # Calcul des rendements
        returns = price_history.pct_change().dropna()

        # Moyenne mobile rapide et lente
        fast_ma = returns.rolling(window=10).mean()
        slow_ma = returns.rolling(window=30).mean()

        # Tendance générale
        trend = fast_ma.iloc[-1] > slow_ma.iloc[-1]

        # Volatilité
        volatility = self.calculate_volatility(price_history)

        # Volume
        volume = price_history.diff().abs().rolling(window=20).mean()

        # Règles pour le sentiment
        if trend and volatility < self.volatility_threshold and volume.iloc[-1] > volume.mean(
        ):
            return 'bullish'
        elif not trend and volatility > self.volatility_threshold and volume.iloc[-1] < volume.mean():
            return 'bearish'
        else:
            return 'neutral'
