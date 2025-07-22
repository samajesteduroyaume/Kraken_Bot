"""
Implémentation du modèle de réseau de neurones pour le trading.
"""
from sklearn.neural_network import MLPClassifier
from ..core.base_model import BaseModel


class NeuralNetworkModel(BaseModel):
    """Modèle de réseau de neurones pour la prédiction de marché."""
    
    def __init__(self, name: str = 'neural_network', model_params: dict = None):
        """Initialise le modèle de réseau de neurones.
        
        Args:
            name: Nom du modèle. Defaults to 'neural_network'.
            model_params: Paramètres du modèle. Defaults to None.
        """
        default_params = {
            'hidden_layer_sizes': (100, 50),
            'activation': 'relu',
            'solver': 'adam',
            'alpha': 0.0001,
            'batch_size': 'auto',
            'learning_rate': 'adaptive',
            'max_iter': 200,
            'random_state': 42,
            'early_stopping': True,
            'validation_fraction': 0.1,
            'n_iter_no_change': 10
        }
        
        if model_params:
            default_params.update(model_params)
            
        super().__init__(name, default_params)
    
    def _initialize_model(self) -> None:
        """Initialise le modèle de réseau de neurones."""
        self.model = MLPClassifier(**self.model_params)
    
    def get_params(self) -> dict:
        """Retourne les paramètres du modèle."""
        return self.model.get_params()
