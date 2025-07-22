"""Conteneur d'injection de dépendances simplifié."""
from dependency_injector import containers, providers
from .api.kraken import KrakenAPI
from .strategy import StrategyManager
from config.settings import settings

class Container(containers.DeclarativeContainer):
    """Conteneur d'injection de dépendances."""
    
    # Configuration
    config = providers.Configuration()
    
    # API Kraken
    kraken_api = providers.Singleton(
        KrakenAPI,
        api_key=settings.KRAKEN_API_KEY,
        private_key=settings.KRAKEN_PRIVATE_KEY
    )
    
    # Gestionnaire de stratégies
    strategy_manager = providers.Singleton(
        StrategyManager,
        config=settings.STRATEGY_CONFIG
    )
