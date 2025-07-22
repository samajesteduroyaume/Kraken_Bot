# Module d'Apprentissage Automatique

## Vue d'ensemble

Le module d'apprentissage automatique (ML) du bot de trading Kraken est conçu pour fournir des prédictions de marché avancées en utilisant diverses techniques d'apprentissage automatique. Ce document explique comment utiliser ce module pour entraîner, évaluer et déployer des modèles de prédiction.

## Fonctionnalités clés

- **Multi-modèles** : Support de plusieurs algorithmes (Random Forest, XGBoost, Réseaux de Neurones, LSTM)
- **Prétraitement intégré** : Gestion automatique de la normalisation et du prétraitement des données
- **Persistance** : Sauvegarde et chargement faciles des modèles entraînés
- **Évaluation complète** : Métriques détaillées pour évaluer les performances des modèles
- **Intégration avec les stratégies** : Utilisation transparente des prédictions dans les stratégies de trading

## Architecture du module

```
src/ml/
├── __init__.py           # Interface principale
├── core/                # Classes de base et interfaces
│   ├── base_model.py    # Classe de base pour tous les modèles
│   └── model_factory.py # Fabrique pour créer des instances de modèles
├── models/              # Implémentations des modèles
│   ├── random_forest.py
│   ├── xgboost.py
│   ├── neural_network.py
│   └── lstm.py
├── utils/               # Utilitaires
│   ├── data_loader.py   # Chargement des données
│   ├── metrics.py       # Métriques d'évaluation
│   └── preprocessor.py  # Prétraitement des données
└── config/              # Configuration
    └── model_config.py  # Paramètres par défaut
```

## Utilisation de base

### Initialisation du prédicteur

```python
from src.ml import MLPredictor

# Configuration du prédicteur
config = {
    'model_dir': 'models',          # Répertoire pour sauvegarder les modèles
    'default_model_type': 'random_forest',  # Type de modèle par défaut
    'test_size': 0.2,               # Taille du jeu de test
    'model_params': {               # Paramètres spécifiques aux modèles
        'random_forest': {
            'n_estimators': 100,
            'max_depth': 10
        }
    }
}

# Création d'une instance du prédicteur
predictor = MLPredictor(config)
```

### Entraînement d'un modèle

```python
import numpy as np

# Générer des données d'exemple
X_train = np.random.rand(1000, 10)  # 1000 échantillons, 10 caractéristiques
y_train = np.random.randint(0, 3, 1000)  # 3 classes

# Entraîner un nouveau modèle
metrics = predictor.fit(X_train, y_train, model_name='mon_modele')
print(f"Précision: {metrics['accuracy']:.2f}")
```

### Faire des prédictions

```python
# Données de test
X_test = np.random.rand(10, 10)

# Faire des prédictions
predictions = predictor.predict(X_test)
print(f"Prédictions: {predictions}")

# Obtenir les probabilités de classe
probabilities = predictor.predict_proba(X_test)
print(f"Probabilités: {probabilities}")
```

## Modèles disponibles

### Random Forest
- **Avantages** : Robuste, peu sensible au bruit, bon pour les données non linéaires
- **Cas d'utilisation** : Classification de tendance, détection de motifs

### XGBoost
- **Avantages** : Haute performance, gestion des valeurs manquantes
- **Cas d'utilisation** : Prédiction de mouvement de prix, détection d'anomalies

### Réseau de Neurones
- **Avantages** : Capacité à apprendre des motifs complexes
- **Cas d'utilisation** : Reconnaissance de motifs temporels, analyse de sentiment

### LSTM (Long Short-Term Memory)
- **Avantages** : Excellente mémoire à long terme pour les séries temporelles
- **Cas d'utilisation** : Prédiction de séries temporelles, analyse de tendances à long terme

## Bonnes pratiques

1. **Prétraitement des données** : Toujours normaliser les données avant l'entraînement
2. **Validation croisée** : Utiliser la validation croisée pour une évaluation robuste
3. **Surveillance des performances** : Surveiller les métriques sur un ensemble de validation
4. **Mise à jour des modèles** : Réentraîner périodiquement les modèles avec de nouvelles données
5. **Gestion des versions** : Versionner les modèles pour faciliter le suivi des performances

## Dépannage

### Problèmes courants

1. **Erreur de dimension** : Vérifiez que les dimensions des données d'entrée correspondent à celles attendues par le modèle
2. **Données manquantes** : Assurez-vous que toutes les valeurs manquantes sont traitées avant l'entraînement
3. **Fuites de données** : Évitez toute fuite d'information entre les ensembles d'entraînement et de test

### Journalisation

Le module utilise le module `logging` de Python. Activez les logs de débogage pour plus d'informations :

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Prochaines étapes

- [ ] Intégration avec les stratégies de trading
- [ ] Optimisation des hyperparamètres
- [ ] Déploiement en production
