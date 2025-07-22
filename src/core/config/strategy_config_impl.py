"""
Implémentation de la configuration des stratégies de trading.

Ce module contient l'implémentation complète de la configuration des stratégies,
y compris la classe StrategyConfig qui sert de conteneur pour toutes les configurations.
"""
from typing import Dict, Any, Union
from pydantic import BaseModel, Field
from .strategy_config import (
    StrategyType,
    BaseStrategyConfig,
    TrendFollowingConfig,
    MeanReversionConfig,
    MomentumConfig
)


class StrategyConfig(BaseModel):
    """Configuration complète pour toutes les stratégies."""
    trend_following: TrendFollowingConfig = Field(
        default_factory=TrendFollowingConfig,
        description="Configuration pour la stratégie de suivi de tendance"
    )
    mean_reversion: MeanReversionConfig = Field(
        default_factory=MeanReversionConfig,
        description="Configuration pour la stratégie de réversion à la moyenne"
    )
    momentum: MomentumConfig = Field(
        default_factory=MomentumConfig,
        description="Configuration pour la stratégie de momentum"
    )
    
    def get_strategy_config(self, strategy_type: Union[str, StrategyType]) -> BaseStrategyConfig:
        """
        Récupère la configuration pour un type de stratégie spécifique.
        
        Args:
            strategy_type: Type de stratégie ou nom de la stratégie
            
        Returns:
            La configuration de la stratégie demandée
            
        Raises:
            ValueError: Si le type de stratégie n'est pas reconnu
        """
        if isinstance(strategy_type, str):
            strategy_type = strategy_type.lower()
            if strategy_type in [t.value for t in StrategyType]:
                strategy_type = StrategyType(strategy_type)
            else:
                # Essayer de faire correspondre avec le nom de la classe de configuration
                strategy_type = strategy_type.lower().replace('config', '').strip()
        
        if strategy_type == StrategyType.TREND_FOLLOWING:
            return self.trend_following
        elif strategy_type == StrategyType.MEAN_REVERSION:
            return self.mean_reversion
        elif strategy_type == StrategyType.MOMENTUM:
            return self.momentum
        else:
            raise ValueError(f"Type de stratégie non reconnu: {strategy_type}")
    
    def update_from_dict(self, config_dict: Dict[str, Any]) -> None:
        """
        Met à jour la configuration à partir d'un dictionnaire.
        
        Args:
            config_dict: Dictionnaire contenant les nouvelles valeurs de configuration
        """
        for strategy_name, strategy_config in config_dict.items():
            if hasattr(self, strategy_name):
                strategy = getattr(self, strategy_name)
                if isinstance(strategy, BaseStrategyConfig):
                    # Mettre à jour uniquement les champs valides
                    valid_fields = strategy.__fields__.keys()
                    update_data = {
                        k: v for k, v in strategy_config.items() 
                        if k in valid_fields
                    }
                    if update_data:
                        setattr(self, strategy_name, strategy.__class__(**{
                            **strategy.dict(),
                            **update_data
                        }))
