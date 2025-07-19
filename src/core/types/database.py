from typing import Dict, Any, Optional, Union, List, TypedDict

# Types pour la configuration de la base de données
DatabaseConfig = Dict[str, Union[str, int, bool]]

# Types pour les modèles de données


class ModelConfig(TypedDict):
    """Configuration d'un modèle de données."""
    table_name: str
    fields: Dict[str, Any]
    indexes: Optional[List[str]]


# Types pour les requêtes
QueryParams = Dict[str, Union[str, int, float, bool]]

# Types pour les résultats
QueryResult = Union[List[Dict[str, Any]], Dict[str, Any]]
