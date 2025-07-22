# Intégration des Modèles ML avec les Stratégies

Ce document explique comment intégrer les modèles d'apprentissage automatique avec les stratégies de trading du bot Kraken pour améliorer la prise de décision.

## Vue d'ensemble

L'intégration ML permet aux stratégies de trading de bénéficier des prédictions des modèles d'apprentissage automatique. Les prédictions peuvent être utilisées pour :

- Générer des signaux de trading
- Filtrer les signaux existants
- Ajuster la taille des positions
- Gérer le risque

## Configuration de l'Intégration ML

### Configuration de la Stratégie

Pour activer l'intégration ML dans une stratégie, ajoutez la section `ml_integration` à la configuration de la stratégie :

```python
strategy_config = {
    'symbol': 'BTC/EUR',
    'timeframes': ['1h', '4h'],
    
    # Configuration de l'intégration ML
    'ml_integration': {
        'enabled': True,                # Activer/désactiver l'intégration ML
        'model_name': 'trend_predictor',# Nom du modèle à utiliser
        'min_confidence': 0.65,         # Confiance minimale requise
        'weight': 0.7,                  # Poids du signal ML (0.0 à 1.0)
        'features': [                   # Caractéristiques à extraire
            'rsi_14',
            'macd',
            'atr_14',
            'volume_ma_20'
        ]
    },
    
    'risk_management': {
        'stop_loss': 0.02,    # Stop-loss à 2%
        'take_profit': 0.04,  # Take-profit à 4%
        'position_size': 0.1  # Taille de position à 10% du capital
    }
}
```

### Format des Données d'Entrée

Les modèles ML s'attendent à recevoir des données dans un format spécifique. Utilisez la classe `FeatureExtractor` pour préparer les données :

```python
from src.ml.utils.feature_extractor import FeatureExtractor

# Extraire les caractéristiques des données OHLCV
features = FeatureExtractor.extract(
    ohlcv_data,
    indicators=['rsi_14', 'macd', 'atr_14', 'volume_ma_20'],
    timeframes=['1h', '4h']
)
```

## Utilisation dans une Stratégie

### Exemple de Stratégie avec ML

Voici comment intégrer les prédictions ML dans une stratégie personnalisée :

```python
from src.strategies.base_strategy import BaseStrategy
from src.ml import MLPredictor

class MLEnhancedStrategy(BaseStrategy):
    def __init__(self, config):
        super().__init__(config)
        self.ml_config = config.get('ml_integration', {})
        self.ml_enabled = self.ml_config.get('enabled', False)
        
        if self.ml_enabled:
            self.ml_predictor = MLPredictor()
            self.ml_model = self.ml_predictor.load_model(
                self.ml_config['model_name']
            )
    
    def analyze(self, data):
        # Analyse technique standard
        signal = self._technical_analysis(data)
        
        # Intégration ML si activée
        if self.ml_enabled:
            ml_signal = self._get_ml_signal(data)
            signal = self._combine_signals(signal, ml_signal)
        
        return signal
    
    def _get_ml_signal(self, data):
        # Extraire les caractéristiques
        features = FeatureExtractor.extract(
            data,
            indicators=self.ml_config.get('features', []),
            timeframes=self.timeframes
        )
        
        # Obtenir la prédiction
        prediction = self.ml_model.predict(features)
        confidence = self.ml_model.predict_proba(features).max()
        
        return {
            'direction': prediction,
            'confidence': confidence,
            'features': features
        }
    
    def _combine_signals(self, tech_signal, ml_signal):
        """Combine les signaux techniques et ML."""
        if not self.ml_enabled or ml_signal['confidence'] < self.ml_config.get('min_confidence', 0.6):
            return tech_signal
        
        # Combinaison pondérée des signaux
        ml_weight = self.ml_config.get('weight', 0.5)
        combined_strength = (
            (1 - ml_weight) * tech_signal['strength'] +
            ml_weight * ml_signal['confidence']
        )
        
        # Mise à jour du signal
        tech_signal['strength'] = combined_strength
        tech_signal['ml_confidence'] = ml_signal['confidence']
        tech_signal['ml_prediction'] = ml_signal['direction']
        
        return tech_signal
```

## Bonnes Pratiques

### Sélection des Caractéristiques

Choisissez des caractéristiques qui sont :
1. **Pertinentes** : Liées au mouvement des prix
2. **Non corrélées** : Évitez la redondance
3. **Stables** : Pas trop sensibles au bruit

### Gestion du Risque

- Utilisez la confiance du modèle pour ajuster la taille de la position
- Mettez en place des garde-fous pour les prédictions peu fiables
- Testez toujours les stratégies ML en backtest avant le déploiement

### Surveillance des Performances

Surveillez régulièrement :
- La précision du modèle
- Le retour sur investissement (ROI)
- Le ratio de Sharpe
- Le drawdown maximum

## Dépannage

### Problèmes Courants

1. **Données manquantes** : Vérifiez que toutes les caractéristiques requises sont disponibles
2. **Décalage temporel** : Assurez-vous qu'il n'y a pas de fuite d'information future
3. **Sur-apprentissage** : Surveillez les performances sur des données non vues

### Journalisation

Activez la journalisation détaillée pour le débogage :

```python
import logging

# Configuration des logs
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Journalisation spécifique au ML
ml_logger = logging.getLogger('ml')
ml_logger.setLevel(logging.DEBUG)
```

## Exemple Complet

Consultez le fichier `examples/ml_enhanced_strategy.py` pour un exemple complet d'implémentation d'une stratégie avec intégration ML.
