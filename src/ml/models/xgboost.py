"""
Implémentation du modèle XGBoost pour le trading.
"""
from xgboost import XGBClassifier
from ..core.base_model import BaseModel


class XGBoostModel(BaseModel):
    """Modèle XGBoost pour la prédiction de marché."""
    
    def __init__(self, name: str = 'xgboost', model_params: dict = None):
        """Initialise le modèle XGBoost.
        
        Args:
            name: Nom du modèle. Defaults to 'xgboost'.
            model_params: Paramètres du modèle. Defaults to None.
        """
        default_params = {
            'n_estimators': 100,
            'max_depth': 3,
            'learning_rate': 0.1,
            'random_state': 42,
            'n_jobs': -1,
            'scale_pos_weight': 1,
            'use_label_encoder': False,
            'eval_metric': 'logloss'
        }
        
        if model_params:
            default_params.update(model_params)
            
        super().__init__(name, default_params)
    
    def _initialize_model(self) -> None:
        """Initialise le modèle XGBoost."""
        self.model = XGBClassifier(**self.model_params)
    
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
