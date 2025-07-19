# 🛠️ Création de Stratégies Personnalisées

Ce guide explique comment créer et intégrer de nouvelles stratégies de trading dans Kraken_Bot.

## Structure d'une Stratégie

Chaque stratégie doit hériter de la classe `BaseStrategy` et implémenter les méthodes requises :

```python
from abc import abstractmethod
from typing import Dict, Any, Optional
from dataclasses import dataclass
from ..core.strategy import BaseStrategy
from ..core.types import Symbol, Timeframe, Signal, Position

@dataclass
class MyCustomStrategy(BaseStrategy):
    """
    Exemple de stratégie personnalisée.
    """
    # Paramètres de la stratégie avec valeurs par défaut
    param1: float = 1.0
    param2: int = 14
    
    def __post_init__(self):
        super().__init__()
        # Initialisation des indicateurs
        self._setup_indicators()
    
    def _setup_indicators(self):
        """Initialise les indicateurs techniques."""
        pass
    
    async def analyze(self, symbol: Symbol, timeframe: Timeframe) -> Optional[Signal]:
        """
        Analyse le marché et retourne un signal de trading.
        
        Args:
            symbol: Paire de trading à analyser
            timeframe: Période de temps des données
            
        Returns:
            Signal: Un signal de trading ou None si aucun signal
        """
        # 1. Récupérer les données OHLCV
        df = await self.data_provider.get_ohlcv(symbol, timeframe)
        
        # 2. Calculer les indicateurs
        # ...
        
        # 3. Générer un signal basé sur la logique de trading
        signal = self._generate_signal(df)
        
        return signal
    
    def _generate_signal(self, df) -> Optional[Signal]:
        """Génère un signal basé sur les données techniques."""
        # Logique de génération de signal
        pass
    
    async def on_position_opened(self, position: Position):
        """Callback appelé quand une position est ouverte."""
        self.logger.info(f"Position ouverte: {position}")
    
    async def on_position_closed(self, position: Position):
        """Callback appelé quand une position est fermée."""
        self.logger.info(f"Position fermée: {position}")
        
    def get_parameters(self) -> Dict[str, Any]:
        """Retourne les paramètres actuels de la stratégie."""
        return {
            'param1': self.param1,
            'param2': self.param2
        }
```

## Étapes pour Ajouter une Nouvelle Stratégie

### 1. Créer un Nouveau Fichier

Créez un nouveau fichier dans `src/strategies/` avec un nom descriptif (ex: `my_strategy.py`).

### 2. Implémenter la Stratégie

Suivez la structure ci-dessus pour implémenter votre logique de trading.

### 3. Enregistrer la Stratégie

Ajoutez votre stratégie au dictionnaire `STRATEGIES` dans `src/strategies/__init__.py` :

```python
from .my_strategy import MyCustomStrategy

STRATEGIES = {
    'momentum': MomentumStrategy,
    'mean_reversion': MeanReversionStrategy,
    'my_custom_strategy': MyCustomStrategy,  # Nouvelle stratégie
}
```

### 4. Créer un Fichier de Configuration

Créez un fichier de configuration dans `config/strategies/my_custom_strategy.yaml` :

```yaml
# Paramètres par défaut de la stratégie
param1: 1.0
param2: 14

# Paramètres de risque
risk:
  stop_loss: -0.05
  take_profit: 0.10
  max_position_size: 1000

# Paires et timeframes
pairs:
  - BTC/EUR
  - ETH/EUR
  
timeframes:
  - 1h
  - 4h

# Activer/désactiver
enabled: false
```

## Bonnes Pratiques

1. **Documentation** : Documentez clairement votre stratégie
2. **Tests** : Ajoutez des tests unitaires
3. **Backtesting** : Testez sur des données historiques
4. **Gestion des Erreurs** : Gérez toutes les erreurs potentielles
5. **Logging** : Utilisez le système de logging intégré

## Exemple Complet : Stratégie de Croisement de Moyennes Mobiles

```python
from ..core.strategy import BaseStrategy
from ..core.types import Symbol, Timeframe, Signal, SignalType, Position

class MovingAverageCrossover(BaseStrategy):
    """
    Stratégie de croisement de moyennes mobiles.
    
    Achete quand la moyenne mobile rapide croise au-dessus de la lente.
    Vends quand la moyenne mobile rapide croise en-dessous de la lente.
    """
    
    def __init__(self, fast_ma: int = 9, slow_ma: int = 21):
        super().__init__()
        self.fast_ma = fast_ma
        self.slow_ma = slow_ma
        self.name = f"MA_Crossover_{fast_ma}_{slow_ma}"
    
    async def analyze(self, symbol: Symbol, timeframe: Timeframe) -> Optional[Signal]:
        # Récupérer les données OHLCV
        df = await self.data_provider.get_ohlcv(symbol, timeframe)
        
        if len(df) < self.slow_ma + 1:
            return None
        
        # Calculer les moyennes mobiles
        df['fast_ma'] = df['close'].rolling(window=self.fast_ma).mean()
        df['slow_ma'] = df['close'].rolling(window=self.slow_ma).mean()
        
        # Détecter les croisements
        df['position'] = 0
        df.loc[df['fast_ma'] > df['slow_ma'], 'position'] = 1
        df.loc[df['fast_ma'] <= df['slow_ma'], 'position'] = -1
        
        # Détecter les changements de position
        df['signal'] = df['position'].diff()
        
        # Générer un signal si croisement détecté
        if df['signal'].iloc[-1] > 0:
            return Signal(
                symbol=symbol,
                signal_type=SignalType.BUY,
                strength=1.0,
                timestamp=df.index[-1],
                metadata={
                    'fast_ma': df['fast_ma'].iloc[-1],
                    'slow_ma': df['slow_ma'].iloc[-1]
                }
            )
        elif df['signal'].iloc[-1] < 0:
            return Signal(
                symbol=symbol,
                signal_type=SignalType.SELL,
                strength=1.0,
                timestamp=df.index[-1],
                metadata={
                    'fast_ma': df['fast_ma'].iloc[-1],
                    'slow_ma': df['slow_ma'].iloc[-1]
                }
            )
        
        return None
```

## Tests et Validation

1. **Tests Unitaires** :
   ```python
   # tests/strategies/test_my_strategy.py
   def test_ma_crossover():
       strategy = MovingAverageCrossover(fast_ma=9, slow_ma=21)
       # Ajoutez des tests unitaires ici
   ```

2. **Backtesting** :
   ```bash
   python -m src.core.backtesting.backtester --strategy my_custom_strategy --pairs "BTC/EUR,ETH/EUR" --timerange 20230101-20231231
   ```

## Prochaines Étapes

- [Architecture du Projet](architecture.md)
- [Guide d'API](../api_reference/overview.md)
- [Tests et Intégration Continue](testing.md)
