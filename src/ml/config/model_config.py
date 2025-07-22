"""
Configuration des paramètres par défaut pour les modèles ML.
"""
from typing import Dict, Any

# Paramètres par défaut pour Random Forest
RANDOM_FOREST_DEFAULT_PARAMS: Dict[str, Any] = {
    'n_estimators': 100,
    'max_depth': None,
    'min_samples_split': 2,
    'min_samples_leaf': 1,
    'random_state': 42,
    'n_jobs': -1
}

# Paramètres par défaut pour XGBoost
XGBOOST_DEFAULT_PARAMS: Dict[str, Any] = {
    'n_estimators': 100,
    'max_depth': 6,
    'learning_rate': 0.1,
    'objective': 'binary:logistic',
    'random_state': 42,
    'n_jobs': -1
}

# Paramètres par défaut pour le réseau de neurones
NEURAL_NETWORK_DEFAULT_PARAMS: Dict[str, Any] = {
    'hidden_layer_sizes': (100, 50),
    'activation': 'relu',
    'solver': 'adam',
    'alpha': 0.0001,
    'learning_rate': 'constant',
    'max_iter': 200,
    'random_state': 42
}

# Paramètres par défaut pour LSTM
LSTM_DEFAULT_PARAMS: Dict[str, Any] = {
    'input_dim': 10,  # Doit être défini en fonction des données
    'hidden_dim': 50,
    'output_dim': 1,
    'num_layers': 1,
    'dropout': 0.2,
    'learning_rate': 0.001,
    'num_epochs': 50,
    'batch_size': 32,
    'sequence_length': 10
}
