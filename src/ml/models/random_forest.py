"""
Implémentation du modèle Random Forest pour le trading.
"""
from sklearn.ensemble import RandomForestClassifier
from ..core.base_model import BaseModel


class RandomForestModel(BaseModel):
    """Modèle Random Forest pour la prédiction de marché."""
    
    def __init__(self, name: str = 'random_forest', model_params: dict = None):
        """Initialise le modèle Random Forest.
        
        Args:
            name: Nom du modèle. Defaults to 'random_forest'.
            model_params: Paramètres du modèle. Defaults to None.
        """
        default_params = {
            'n_estimators': 100,
            'max_depth': 5,
            'random_state': 42,
            'n_jobs': -1,
            'class_weight': 'balanced'
        }
        
        if model_params:
            default_params.update(model_params)
            
        super().__init__(name, default_params)
    
    def _initialize_model(self) -> None:
        """Initialise le modèle Random Forest."""
        self.model = RandomForestClassifier(**self.model_params)
    
    def get_feature_importances(self) -> dict:
        """Retourne l'importance des caractéristiques du modèle.
        
        Returns:
            Dictionnaire des caractéristiques et leur importance
        """
        if not hasattr(self.model, 'feature_importances_'):
            return {}
            
        return dict(zip(
            self.feature_names if hasattr(self, 'feature_names') else 
            [f'feature_{i}' for i in range(len(self.model.feature_importances_))],
            self.model.feature_importances_
        ))
    
    def get_params(self) -> dict:
        """Retourne les paramètres du modèle."""
        return self.model.get_params()
