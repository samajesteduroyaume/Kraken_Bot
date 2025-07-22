# Entraînement et Évaluation des Modèles ML

Ce guide explique comment entraîner, évaluer et optimiser les modèles d'apprentissage automatique pour le trading.

## Préparation des Données

### Format des Données

Les modèles attendent des données dans le format suivant :

- **X** : Tableau NumPy 2D de forme (n_échantillons, n_caractéristiques)
- **y** : Vecteur NumPy 1D de forme (n_échantillons,) contenant les étiquettes

### Extraction des Caractéristiques

Utilisez la classe `FeatureExtractor` pour préparer les données :

```python
from src.ml.utils.feature_extractor import FeatureExtractor

# Données OHLCV au format pandas DataFrame
# Index: DatetimeIndex
# Colonnes: 'open', 'high', 'low', 'close', 'volume'
ohlcv_data = ...  # Charger les données

# Extraire les caractéristiques
features = FeatureExtractor.extract(
    ohlcv_data,
    indicators=['rsi_14', 'macd', 'atr_14', 'sma_50', 'sma_200'],
    timeframes=['1h', '4h']
)

# Préparer les étiquettes (rendements futurs)
targets = FeatureExtractor.create_targets(
    ohlcv_data['close'],
    horizon=5,  # Nombre de bougies dans le futur
    method='classification',  # ou 'regression'
    bins=[-np.inf, -0.01, 0.01, np.inf],  # Pour la classification
    labels=[-1, 0, 1]  # Vendre, Neutre, Acheter
)
```

## Entraînement d'un Modèle

### Configuration de l'Entraînement

```python
from src.ml import MLPredictor

# Configuration du prédicteur
config = {
    'model_dir': 'models',
    'default_model_type': 'random_forest',
    'test_size': 0.2,
    'random_state': 42,
    'model_params': {
        'random_forest': {
            'n_estimators': 100,
            'max_depth': 10,
            'random_state': 42,
            'class_weight': 'balanced'
        }
    }
}

predictor = MLPredictor(config)
```

### Entraînement et Validation Croisée

```python
# Entraînement simple
metrics = predictor.fit(X_train, y_train, model_name='trend_predictor')

# Validation croisée
cv_metrics = predictor.cross_validate(
    X, y, 
    cv=5,  # 5 folds
    model_type='random_forest',
    scoring=['accuracy', 'f1_weighted', 'roc_auc_ovr']
)

print(f"Moyenne des scores de validation croisée: {cv_metrics}")
```

## Évaluation des Performances

### Métriques d'Évaluation

Le module fournit plusieurs métriques d'évaluation :

```python
from src.ml.utils.metrics import ModelMetrics

# Calcul des métriques
metrics = ModelMetrics.calculate(
    y_true=y_test,
    y_pred=predictions,
    y_proba=probabilities,
    prefix='test_'  # Préfixe optionnel pour les clés du dictionnaire
)

# Afficher les métriques
for name, value in metrics.items():
    print(f"{name}: {value:.4f}")
```

### Visualisation des Résultats

```python
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay

# Matrice de confusion
ConfusionMatrixDisplay.from_predictions(
    y_test, predictions,
    display_labels=['Sell', 'Neutral', 'Buy'],
    normalize='true'
)
plt.title('Matrice de Confusion (Normalisée)')
plt.show()

# Courbe ROC (pour la classification binaire)
RocCurveDisplay.from_predictions(
    y_test_binary,
    probabilities[:, 1],  # Probabilités de la classe positive
    name='Modèle de Trading'
)
plt.plot([0, 1], [0, 1], 'k--')
plt.title('Courbe ROC')
plt.show()
```

## Optimisation des Hyperparamètres

### Recherche par Grille

```python
from sklearn.model_selection import GridSearchCV

# Définir la grille des hyperparamètres
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [5, 10, 20, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Créer et exécuter la recherche par grille
grid_search = GridSearchCV(
    estimator=predictor.model,
    param_grid=param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
    verbose=1
)

grid_search.fit(X_train, y_train)

# Meilleurs paramètres
print(f"Meilleurs paramètres: {grid_search.best_params_}")
print(f"Meilleur score: {grid_search.best_score_:.4f}")

# Mettre à jour le modèle avec les meilleurs paramètres
predictor.model = grid_search.best_estimator_
```

### Optimisation Bayésienne

Pour une recherche plus efficace, utilisez l'optimisation bayésienne avec `optuna` :

```python
import optuna
from sklearn.model_selection import cross_val_score

def objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 300),
        'max_depth': trial.suggest_int('max_depth', 3, 15),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
        'class_weight': 'balanced'
    }
    
    model = RandomForestClassifier(**params, random_state=42)
    score = cross_val_score(
        model, X_train, y_train,
        cv=5, scoring='accuracy', n_jobs=-1
    ).mean()
    
    return score

# Créer et exécuter l'étude
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=50)

# Meilleurs paramètres
print(f"Meilleurs paramètres: {study.best_params}")
print(f"Meilleur score: {study.best_value:.4f}")
```

## Sauvegarde et Chargement des Modèles

### Sauvegarder un Modèle

```python
# Sauvegarder le modèle
model_path = predictor.save_model('trend_predictor')
print(f"Modèle sauvegardé dans: {model_path}")

# Sauvegarder les métriques d'évaluation
metrics = predictor.evaluate(X_test, y_test)
with open(f"{model_path}/metrics.json", 'w') as f:
    json.dump(metrics, f, indent=2)
```

### Charger un Modèle

```python
# Charger un modèle existant
predictor.load_model('trend_predictor')

# Faire des prédictions
predictions = predictor.predict(X_new)
```

## Bonnes Pratiques

1. **Séparation des Données** :
   - Utilisez des ensembles distincts pour l'entraînement, la validation et les tests
   - Évitez toute fuite d'information entre les ensembles

2. **Validation Temporelle** :
   - Pour les séries temporelles, utilisez `TimeSeriesSplit`
   - Ne pas mélanger les données futures avec les données passées

3. **Équilibrage des Classes** :
   - Utilisez `class_weight='balanced'` ou suréchantillonnez les classes minoritaires
   - Évaluez les performances sur chaque classe séparément

4. **Surveillance du Modèle** :
   - Surveillez la dérive des données (data drift)
   - Réentraînez régulièrement le modèle avec de nouvelles données

## Dépannage

### Problèmes Courants

1. **Sur-apprentissage** :
   - Symptômes : Bonnes performances sur l'ensemble d'entraînement, mauvaises sur l'ensemble de test
   - Solutions :
     - Réduisez la complexité du modèle
     - Augmentez la régularisation
     - Obtenez plus de données d'entraînement

2. **Sous-apprentissage** :
   - Symptômes : Mauvaises performances à la fois sur l'entraînement et le test
   - Solutions :
     - Augmentez la complexité du modèle
     - Ajoutez des caractéristiques pertinentes
     - Réduisez la régularisation

3. **Données déséquilibrées** :
   - Symptômes : Bonne précision globale mais mauvaise détection des classes minoritaires
   - Solutions :
     - Utilisez `class_weight='balanced'`
     - Suréchantillonnez les classes minoritaires (SMOTE)
     - Sous-échantillonnez les classes majoritaires
