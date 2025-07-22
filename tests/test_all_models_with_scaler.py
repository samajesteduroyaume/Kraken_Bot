"""
Test de compatibilité du PersistentStandardScaler avec tous les types de modèles.
"""
import os
import numpy as np
import pandas as pd
import tempfile
import shutil
from pathlib import Path
from typing import Dict, Any, Type, Tuple, Optional, Union

from src.ml.core.model_factory import ModelFactory
from src.ml.core.base_model import BaseModel


def generate_test_data(n_samples: int = 100, n_features: int = 5, seq_len: int = 10) -> Tuple[np.ndarray, np.ndarray]:
    """Génère des données de test pour l'entraînement.
    
    Args:
        n_samples: Nombre d'échantillons
        n_features: Nombre de caractéristiques
        seq_len: Longueur de la séquence temporelle (pour LSTM)
        
    Returns:
        Tuple de (X, y) pour l'entraînement
    """
    np.random.seed(42)
    
    # Générer des séquences temporelles pour LSTM
    X = np.random.rand(n_samples, seq_len, n_features)
    
    # Pour les modèles non-séquentiels, aplatir les deux premières dimensions
    if seq_len > 1:
        X_flat = X.reshape(-1, seq_len * n_features)
    else:
        X_flat = X.reshape(-1, n_features)
    
    # Générer des étiquettes (une par séquence)
    y = np.random.randint(0, 2, n_samples)  # Classification binaire
    
    return X, y, X_flat


def test_model_with_scaler(
    model_type: str, 
    temp_dir: Path, 
    X_train: np.ndarray, 
    y_train: np.ndarray,
    X_test: np.ndarray,
    X_train_flat: Optional[np.ndarray] = None,
    X_test_flat: Optional[np.ndarray] = None
) -> bool:
    """Teste un modèle spécifique avec le PersistentStandardScaler.
    
    Args:
        model_type: Type de modèle à tester
        temp_dir: Répertoire temporaire pour la sauvegarde
        X_train: Données d'entraînement
        y_train: Cibles d'entraînement
        X_test: Données de test pour la validation
        X_train_flat: Données d'entraînement aplaties (pour les modèles non-séquentiels)
        X_test_flat: Données de test aplaties (pour les modèles non-séquentiels)
        
    Returns:
        True si le test réussit, False sinon
    """
    print(f"\n=== Test du modèle {model_type} ===")
    
    try:
        # 1. Créer et entraîner le modèle
        print("1. Création et entraînement...")
        
        # Paramètres spécifiques pour LSTM
        model_params = {'name': f"{model_type}_test_model"}
        
        if model_type == 'lstm':
            # Configuration spécifique pour LSTM
            model_params.update({
                'input_dim': X_train.shape[2],  # n_features
                'hidden_dim': 32,
                'num_layers': 2,
                'num_classes': 2,  # Classification binaire
                'seq_len': X_train.shape[1],
                'batch_size': 16,
                'epochs': 3  # Moins d'époques pour les tests
            })
        else:
            # Paramètres pour les autres modèles
            model_params['random_state'] = 42
        
        model = ModelFactory.create_model(
            model_type=model_type,
            params=model_params
        )
        
        # 2. Entraîner le modèle avec les données appropriées
        if model_type == 'lstm':
            # Pour LSTM, utiliser X_train 3D
            model.train(X_train, y_train)
            original_predictions = model.predict(X_test)
        else:
            # Pour les autres modèles, utiliser X_train_flat 2D
            if X_train_flat is None or X_test_flat is None:
                raise ValueError("X_train_flat et X_test_flat sont requis pour les modèles non-LSTM")
            model.train(X_train_flat, y_train)
            original_predictions = model.predict(X_test_flat)
        
        # 3. Sauvegarder le modèle
        print("2. Sauvegarde du modèle...")
        model_save_dir = temp_dir / model_type
        model_save_dir.mkdir(exist_ok=True)
        model.save(str(model_save_dir))
        
        # 4. Recharger le modèle
        print("3. Rechargement du modèle...")
        loaded_model = ModelFactory.create_model(
            model_type=model_type,
            params={'name': f"{model_type}_loaded"},
            is_loading=True  # Indiquer que c'est un chargement, pour ne pas écraser les paramètres sauvegardés
        )
        loaded_model.load(str(model_save_dir))
        
        # 5. Vérifier les prédictions
        print("4. Vérification des prédictions...")
        
        # Utiliser les données appropriées selon le type de modèle
        if model_type == 'lstm':
            loaded_predictions = loaded_model.predict(X_test)
        else:
            if X_test_flat is None:
                raise ValueError("X_test_flat est requis pour les modèles non-LSTM")
            loaded_predictions = loaded_model.predict(X_test_flat)
        
        # Vérifier que les prédictions sont identiques
        if not np.array_equal(original_predictions, loaded_predictions):
            print(f"❌ Les prédictions avant/après chargement diffèrent pour {model_type}!")
            return False
        
        # Vérifier l'état du scaler
        if not hasattr(loaded_model, 'scaler') or loaded_model.scaler is None:
            print(f"❌ Le modèle {model_type} n'a pas de scaler après chargement!")
            return False
            
        if not hasattr(loaded_model.scaler, 'mean_') or loaded_model.scaler.mean_ is None:
            print(f"❌ Le scaler du modèle {model_type} n'a pas d'attribut mean_ valide!")
            return False
            
        if not hasattr(loaded_model.scaler, 'scale_') or loaded_model.scaler.scale_ is None:
            print(f"❌ Le scaler du modèle {model_type} n'a pas d'attribut scale_ valide!")
            return False
        
        print(f"✅ Test réussi pour le modèle {model_type}")
        return True
        
    except Exception as e:
        print(f"❌ Erreur lors du test du modèle {model_type}: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_all_models():
    """Teste tous les modèles avec le PersistentStandardScaler."""
    # Créer un répertoire temporaire pour les tests
    temp_dir = Path(tempfile.mkdtemp(prefix="test_models_"))
    
    try:
        # Générer des données de test
        print("Génération des données de test...")
        seq_len = 10  # Longueur des séquences pour LSTM
        n_features = 5
        
        # Générer des données 3D pour LSTM et 2D pour les autres modèles
        X_train, y_train, X_train_flat = generate_test_data(
            n_samples=100, 
            n_features=n_features,
            seq_len=seq_len
        )
        
        # Générer des données de test
        X_test, _, X_test_flat = generate_test_data(
            n_samples=10,
            n_features=n_features,
            seq_len=seq_len
        )
        
        # Liste des modèles à tester
        model_types = [
            'random_forest',
            'xgboost',
            'neural_network',
            'lstm'
        ]
        
        # Exécuter les tests
        results = {}
        for model_type in model_types:
            # Pour tous les modèles, passer à la fois les données 3D et 2D
            results[model_type] = test_model_with_scaler(
                model_type, 
                temp_dir, 
                X_train,  # Données 3D pour LSTM
                y_train, 
                X_test,   # Données 3D pour LSTM
                X_train_flat,  # Données 2D pour les autres modèles
                X_test_flat    # Données 2D pour les autres modèles
            )
        
        # Afficher le résumé
        print("\n=== Résumé des tests ===")
        all_passed = True
        for model_type, success in results.items():
            status = "✅ RÉUSSI" if success else "❌ ÉCHEC"
            print(f"{status} - {model_type}")
            if not success:
                all_passed = False
        
        if all_passed:
            print("\n✅ Tous les tests ont réussi!")
        else:
            print("\n❌ Certains tests ont échoué. Voir les messages d'erreur ci-dessus.")
        
        return all_passed
        
    finally:
        # Nettoyer le répertoire temporaire
        shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    test_all_models()
