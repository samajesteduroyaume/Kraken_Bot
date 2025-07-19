"""
Module d'API pour interagir avec l'API Kraken.
"""

from .client import KrakenAPI
from .exceptions import (
    KrakenAPIError,
    APIError,
    RateLimitError,
    ValidationError,
    APIConnectionError
)

__all__ = [
    'KrakenAPI',
    'KrakenAPIError',
    'APIError',
    'RateLimitError',
    'ValidationError',
    'APIConnectionError'
]
