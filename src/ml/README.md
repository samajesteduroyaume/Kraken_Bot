# Module d'Apprentissage Automatique

Ce module fournit des fonctionnalités de prédiction de tendance de marché pour le bot de trading Kraken, en utilisant diverses techniques d'apprentissage automatique.

## Structure du Projet

```
src/ml/
├── __init__.py           # Interface principale du module
├── core/                 # Classes de base et interfaces
│   ├── __init__.py
│   ├── base_model.py     # Classe de base pour tous les modèles
│   ├── model_factory.py  # Fabrique pour créer des instances de modèles
│   └── predictor.py      # Classe principale pour les prédictions
├── models/               # Implémentations des modèles
│   ├── __init__.py
│   ├── random_forest.py  # Modèle Random Forest
│   ├── xgboost.py        # Modèle XGBoost
│   ├── neural_network.py # Réseau de neurones
│   └── lstm.py           # Modèle LSTM
├── utils/                # Utilitaires
│   ├── __init__.py
│   ├── data_loader.py    # Chargement des données
│   ├── metrics.py        # Métriques d'évaluation
│   └── preprocessor.py   # Prétraitement des données
└── config/               # Configuration
    ├── __init__.py
    └── model_config.py   # Paramètres par défaut des modèles
```

## Utilisation

### Initialisation

```python
from src.ml import MLPredictor

# Configuration du prédicteur
config = {
    'model_dir': 'models',
    'default_model_type': 'random_forest',
    'test_size': 0.2,
    'model_params': {
        'random_forest': {
            'n_estimators': 100,
            'max_depth': 10
        }
    }
}

# Création du prédicteur
predictor = MLPredictor(config)
```

### Entraînement et Prédiction

```python
import numpy as np

# Données d'exemple
X_train = np.random.rand(100, 10)
y_train = np.random.randint(0, 2, 100)

# Entraînement
metrics = predictor.fit(X_train, y_train, model_name='mon_modele')
print(f"Précision: {metrics['accuracy']:.2f}")

# Prédiction
X_test = np.random.rand(10, 10)
predictions = predictor.predict(X_test)
print(f"Prédictions: {predictions}")
```

### Sauvegarde et Chargement

```python
# Sauvegarder un modèle
predictor.save_model('mon_modele')

# Charger un modèle
predictor.load_model('mon_modele')

# Lister les modèles disponibles
models = predictor.get_available_models()
print(f"Modèles disponibles: {models}")
```

## Fonctionnalités

- **Multi-modèles**: Support de plusieurs algorithmes d'apprentissage automatique
- **Persistance**: Sauvegarde et chargement des modèles entraînés
- **Évaluation**: Métriques complètes pour évaluer les performances
- **Prétraitement**: Gestion automatique du prétraitement des données
- **Extensible**: Architecture modulaire pour ajouter facilement de nouveaux modèles

## Ajout d'un Nouveau Modèle

1. Créer une nouvelle classe dans le dossier `models/` qui hérite de `BaseModel`
2. Implémenter les méthodes requises (`fit`, `predict`, `save`, `load`)
3. Enregistrer le modèle dans `ModelFactory`

## Tests

Pour exécuter les tests unitaires :

```bash
pytest tests/test_predictor.py -v
```
