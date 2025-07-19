"""
Module contenant les exceptions personnalisées pour l'API Kraken.
"""


class KrakenAPIError(Exception):
    """Erreur spécifique à l'API Kraken."""


class APIError(KrakenAPIError):
    """Erreur de l'API Kraken."""


class RateLimitError(KrakenAPIError):
    """Erreur de limite de taux."""


class ValidationError(KrakenAPIError):
    """Erreur de validation des données."""


class APIConnectionError(KrakenAPIError):
    """Erreur de connexion à l'API."""


class ConfigurationError(KrakenAPIError):
    """Erreur de configuration de l'API Kraken."""


class SessionError(KrakenAPIError):
    """Erreur de session de l'API Kraken."""


class AuthenticationError(KrakenAPIError):
    """Erreur d'authentification pour l'API Kraken."""
