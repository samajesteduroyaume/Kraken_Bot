"""
Module de fabrique pour créer des instances de modèles de machine learning.

Ce module fournit une classe `ModelFactory` qui permet de créer des instances de différents
types de modèles de manière dynamique, en fonction des besoins de l'application.
"""
from typing import Dict, Any, Type, Optional
from ..models.random_forest import RandomForestModel
from ..models.xgboost import XGBoostModel
from ..models.neural_network import NeuralNetworkModel
from ..models.lstm import LSTMPredictor


class ModelFactory:
    """Fabrique pour créer des instances de modèles de machine learning.
    
    Cette classe implémente le modèle de conception Factory pour créer des instances
    de différents types de modèles de manière dynamique. Elle maintient un registre
    des modèles disponibles et peut créer des instances avec des paramètres personnalisés.
    
    Attributes:
        _model_registry (dict): Dictionnaire mappant les noms de modèles aux classes de modèles.
                               Les clés sont des chaînes identifiant les modèles, et les valeurs
                               sont les classes de modèles correspondantes.
    """
    
    _model_registry = {
        'random_forest': RandomForestModel,    # Modèle de forêt aléatoire
        'xgboost': XGBoostModel,              # Modèle XGBoost
        'neural_network': NeuralNetworkModel,  # Réseau de neurones
        'lstm': LSTMPredictor                 # Modèle LSTM pour séries temporelles
    }
    
    @classmethod
    def create_model(
        cls, 
        model_type: str, 
        params: Optional[Dict[str, Any]] = None, 
        is_loading: bool = False
    ) -> Any:
        """Crée une instance du modèle spécifié.
        
        Cette méthode crée une nouvelle instance du type de modèle demandé, en utilisant
        les paramètres fournis. Elle gère également le cas particulier du chargement
        d'un modèle existant, où seuls les paramètres essentiels doivent être transmis.
        
        Args:
            model_type: Type de modèle à créer. Doit être une clé présente dans _model_registry.
                       Valeurs possibles : 'random_forest', 'xgboost', 'neural_network', 'lstm'.
            params: Dictionnaire des paramètres à passer au constructeur du modèle.
                   Si None, un dictionnaire vide est utilisé.
                   Doit contenir une clé 'name' ou un nom par défaut sera généré.
            is_loading: Si True, indique que le modèle est en cours de chargement depuis le disque.
                      Dans ce cas, seuls les paramètres essentiels (comme le nom) sont transmis
                      au modèle pour éviter d'écraser les paramètres sauvegardés.
            
        Returns:
            Une instance du modèle demandé, initialisée avec les paramètres fournis.
            
        Raises:
            ValueError: Si le type de modèle spécifié n'est pas pris en charge.
            
        Example:
            >>> params = {'name': 'mon_modele', 'n_estimators': 100}
            >>> model = ModelFactory.create_model('random_forest', params)
            >>> isinstance(model, RandomForestModel)
            True
        """
        model_class = cls._model_registry.get(model_type.lower())
        if not model_class:
            raise ValueError(f"Type de modèle non reconnu: {model_type}")
            
        # S'assurer que params est un dictionnaire
        params = params or {}
        
        # Extraire le nom du modèle des paramètres ou utiliser le type comme nom par défaut
        name = params.pop('name', f"{model_type}_model")
        
        if is_loading:
            # Lors du chargement, transmettre les paramètres pour l'initialisation
            # Le load() du modèle se chargera de les remplacer par les valeurs sauvegardées
            return model_class(name=name, model_params=params)
        else:
            # Créer une nouvelle instance avec tous les paramètres fournis
            return model_class(name=name, model_params=params)
    
    @classmethod
    def register_model(cls, name: str, model_class: Type):
        """Enregistre un nouveau type de modèle.
        
        Args:
            name: Nom du modèle
            model_class: Classe du modèle (doit hériter de BaseModel)
        """
        if not hasattr(model_class, 'predict') or not hasattr(model_class, 'train'):
            raise TypeError("La classe du modèle doit implémenter les méthodes train() et predict()")
            
        cls._model_registry[name.lower()] = model_class
    
    @classmethod
    def list_available_models(cls) -> list:
        """Retourne la liste des modèles disponibles."""
        return list(cls._model_registry.keys())
