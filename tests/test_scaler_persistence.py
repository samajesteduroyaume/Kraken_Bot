"""
Test de persistance du PersistentStandardScaler avec un modèle de test.
"""
import os
import numpy as np
import pandas as pd
import tempfile
import shutil
from pathlib import Path

from src.ml.models.random_forest import RandomForestModel
from src.ml.core.base_model import BaseModel


def test_scaler_persistence():
    """Teste la persistance du scaler avec un modèle RandomForest."""
    # Créer un répertoire temporaire pour le test
    temp_dir = Path(tempfile.mkdtemp(prefix="test_scaler_"))
    try:
        # 1. Créer et entraîner un modèle
        print("1. Création et entraînement d'un modèle RandomForest...")
        X_train = np.random.rand(100, 5)  # 100 échantillons, 5 caractéristiques
        y_train = np.random.randint(0, 2, 100)  # Classification binaire
        
        model = RandomForestModel("test_model")
        model.train(X_train, y_train)
        
        # Faire une prédiction de référence
        X_test = np.random.rand(10, 5)
        original_predictions = model.predict(X_test)
        
        # 2. Sauvegarder le modèle
        print("2. Sauvegarde du modèle...")
        model_save_dir = temp_dir / "saved_model"
        model_save_dir.mkdir(exist_ok=True)
        model.save(str(model_save_dir))
        
        # 3. Recharger le modèle
        print("3. Rechargement du modèle...")
        loaded_model = RandomForestModel("test_model_loaded")
        loaded_model.load(str(model_save_dir))
        
        # 4. Vérifier que les prédictions sont identiques
        print("4. Vérification des prédictions...")
        loaded_predictions = loaded_model.predict(X_test)
        
        # Vérifier que les prédictions sont identiques
        assert np.array_equal(original_predictions, loaded_predictions), \
            "Les prédictions avant/après chargement diffèrent!"
        
        # Vérifier que le scaler est bien chargé
        assert hasattr(loaded_model, 'scaler'), "Le modèle rechargé n'a pas de scaler"
        assert loaded_model.scaler is not None, "Le scaler est None après chargement"
        assert hasattr(loaded_model.scaler, 'mean_'), "Le scaler n'a pas d'attribut mean_"
        assert loaded_model.scaler.mean_ is not None, "mean_ est None après chargement"
        assert hasattr(loaded_model.scaler, 'scale_'), "Le scaler n'a pas d'attribut scale_"
        assert loaded_model.scaler.scale_ is not None, "scale_ est None après chargement"
        
        print("✅ Test réussi: le scaler persiste correctement après chargement")
        
    finally:
        # Nettoyer le répertoire temporaire
        shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    test_scaler_persistence()
