"""
Gestionnaire avancé de rate limiting avec détection de blacklist IP et backoff exponentiel.
"""

from typing import Dict, Any, Optional, List, Tuple
from collections import defaultdict, deque
import time
import asyncio
import logging
import random
import socket
import aiohttp
from enum import Enum, auto

logger = logging.getLogger(__name__)

class RateLimitStrategy(Enum):
    EXPONENTIAL = auto()
    LINEAR = auto()
    CONSTANT = auto()

class BlacklistStatus(Enum):
    CLEAN = auto()
    SUSPECTED = auto()
    BLACKLISTED = auto()

class RateLimiter:
    """
    Gestionnaire avancé de rate limiting avec :
    - Détection de blacklist IP
    - Stratégies de backoff personnalisables
    - Métriques détaillées
    - Gestion des pics de charge
    """

    def __init__(self,
                 max_requests: int = 15,
                 window_seconds: int = 60,
                 burst_factor: float = 1.5,
                 retry_strategy: str = 'exponential',
                 health_check_interval: int = 300,
                 max_retries: int = 5,
                 initial_backoff: float = 1.0,
                 max_backoff: float = 300.0,
                 jitter: float = 0.1):
        """
        Initialise le gestionnaire de rate limiting avancé.

        Args:
            max_requests: Nombre maximum de requêtes par fenêtre
            window_seconds: Durée de la fenêtre en secondes
            burst_factor: Facteur de tolérance pour les pics de charge (1.0 = pas de tolérance)
            retry_strategy: Stratégie de retry ('exponential', 'linear', 'constant')
            health_check_interval: Intervalle entre les vérifications de santé (secondes)
            max_retries: Nombre maximum de tentatives en cas d'échec
            initial_backoff: Délai initial avant nouvelle tentative (secondes)
            max_backoff: Délai maximum entre les tentatives (secondes)
            jitter: Variation aléatoire à ajouter aux délais (0.0-1.0)
        """
        # Configuration de base
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.burst_factor = burst_factor
        self.retry_strategy = getattr(RateLimitStrategy, retry_strategy.upper(), RateLimitStrategy.EXPONENTIAL)
        self.health_check_interval = health_check_interval
        self.max_retries = max_retries
        self.initial_backoff = initial_backoff
        self.max_backoff = max_backoff
        self.jitter = jitter

        # État interne
        self.requests = deque(maxlen=max_requests * 2)  # Garder un peu d'historique
        self._last_reset = time.time()
        self._current_window = 0
        self._last_health_check = time.time()
        self._blacklist_status = BlacklistStatus.CLEAN
        self._blacklist_until = 0
        self._consecutive_errors = 0
        self._last_error = None

        # Métriques
        self.metrics = {
            'total_requests': 0,
            'rate_limited_requests': 0,
            'blacklist_events': 0,
            'average_response_time': 0.0,
            'total_retries': 0
        }

        # Vérification des paramètres
        self._validate_parameters()

    def _validate_parameters(self):
        """Valide les paramètres de configuration."""
        if not isinstance(self.max_requests, int) or self.max_requests <= 0:
            raise ValueError("max_requests doit être un entier positif")
        if not isinstance(self.window_seconds, (int, float)) or self.window_seconds <= 0:
            raise ValueError("window_seconds doit être un nombre positif")
        if not isinstance(self.burst_factor, (int, float)) or self.burst_factor < 1.0:
            raise ValueError("burst_factor doit être un nombre >= 1.0")
        if not isinstance(self.health_check_interval, (int, float)) or self.health_check_interval <= 0:
            raise ValueError("health_check_interval doit être un nombre positif")
        if not isinstance(self.max_retries, int) or self.max_retries < 0:
            raise ValueError("max_retries doit être un entier >= 0")
        if not isinstance(self.initial_backoff, (int, float)) or self.initial_backoff <= 0:
            raise ValueError("initial_backoff doit être un nombre positif")
        if not isinstance(self.max_backoff, (int, float)) or self.max_backoff < self.initial_backoff:
            raise ValueError("max_backoff doit être >= initial_backoff")
        if not isinstance(self.jitter, (int, float)) or not (0 <= self.jitter <= 1):
            raise ValueError("jitter doit être entre 0.0 et 1.0")

    async def __aenter__(self):
        """Gestion du contexte avec gestion des erreurs et backoff."""
        retry_count = 0
        last_error = None

        while retry_count <= self.max_retries:
            try:
                # Vérifier si nous sommes blacklistés
                await self._check_blacklist()
                
                # Gérer le rate limiting
                await self._enforce_rate_limit()
                
                # Enregistrer la requête
                self.requests.append(time.time())
                self.metrics['total_requests'] += 1
                
                # Réinitialiser le compteur d'erreurs si la requête réussit
                if retry_count > 0:
                    self._consecutive_errors = 0
                    self.metrics['total_retries'] += retry_count
                
                return self
                
            except (RateLimitExceeded, BlacklistDetected) as e:
                last_error = e
                retry_count += 1
                
                # Calculer le délai avant nouvelle tentative
                delay = self._calculate_backoff(retry_count)
                logger.warning(f"Tentative {retry_count}/{self.max_retries} - Nouvel essai dans {delay:.2f}s - {str(e)}")
                
                # Attendre avant de réessayer
                await asyncio.sleep(delay)
        
        # Si on arrive ici, toutes les tentatives ont échoué
        logger.error(f"Échec après {retry_count} tentatives: {str(last_error)}")
        raise last_error

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Nettoyage et mise à jour des métriques."""
        current_time = time.time()
        
        # Mettre à jour les métriques de réponse
        if self.requests:
            response_time = current_time - self.requests[-1]
            self.metrics['average_response_time'] = (
                self.metrics['average_response_time'] * 0.9 + 
                response_time * 0.1
            )
        
        # Vérification périodique de la santé
        if current_time - self._last_health_check >= self.health_check_interval:
            await self._check_health()
            self._last_health_check = current_time

    async def _enforce_rate_limit(self):
        """Applique les limites de taux."""
        current_time = time.time()
        
        # Nettoyer les requêtes hors de la fenêtre
        while self.requests and current_time - self.requests[0] > self.window_seconds:
            self.requests.popleft()
        
        # Vérifier les limites
        if len(self.requests) >= self.max_requests * self.burst_factor:
            wait_time = self.requests[0] + self.window_seconds - current_time
            if wait_time > 0:
                self.metrics['rate_limited_requests'] += 1
                raise RateLimitExceeded(
                    f"Limite de taux dépassée. Attente de {wait_time:.2f}s",
                    retry_after=wait_time
                )

    async def _check_blacklist(self):
        """Vérifie si l'IP est potentiellement blacklistée."""
        current_time = time.time()
        
        if self._blacklist_status == BlacklistStatus.BLACKLISTED:
            if current_time < self._blacklist_until:
                remaining = self._blacklist_until - current_time
                raise BlacklistDetected(
                    f"IP potentiellement blacklistée. Réessayez dans {remaining:.0f}s",
                    retry_after=remaining
                )
            else:
                # Réinitialiser l'état après la période de blacklist
                self._blacklist_status = BlacklistStatus.CLEAN
                self._consecutive_errors = 0
        
        # Si trop d'erreurs consécutives, on suspecte un blacklist
        elif self._consecutive_errors >= 3:  # Seuil configurable
            self._blacklist_status = BlacklistStatus.SUSPECTED
            self.metrics['blacklist_events'] += 1
            logger.warning("Suspicion de blacklist IP détectée")

    def _calculate_backoff(self, attempt: int) -> float:
        """Calcule le délai avant nouvelle tentative avec jitter."""
        if self.retry_strategy == RateLimitStrategy.EXPONENTIAL:
            delay = min(self.initial_backoff * (2 ** (attempt - 1)), self.max_backoff)
        elif self.retry_strategy == RateLimitStrategy.LINEAR:
            delay = min(self.initial_backoff * attempt, self.max_backoff)
        else:  # CONSTANT
            delay = self.initial_backoff
        
        # Ajouter du jitter pour éviter les effets de synchronisation
        jitter = 1 + (random.random() * 2 - 1) * self.jitter
        return delay * jitter

    async def _check_health(self) -> Dict[str, Any]:
        """
        Vérifie la santé de la connexion et retourne des métriques.
        
        Returns:
            dict: État de santé et métriques
        """
        current_time = time.time()
        
        # Calculer le taux de requêtes par seconde
        recent_requests = [r for r in self.requests 
                          if current_time - r < self.window_seconds]
        request_rate = len(recent_requests) / self.window_seconds
        
        # Déterminer l'état de santé
        health_status = {
            'status': 'healthy',
            'request_rate': f"{request_rate:.2f}/s",
            'window_usage': f"{len(recent_requests)}/{self.max_requests}",
            'blacklist_status': self._blacklist_status.name,
            'consecutive_errors': self._consecutive_errors,
            'average_response_time': f"{self.metrics['average_response_time']*1000:.2f}ms",
            'total_requests': self.metrics['total_requests'],
            'rate_limited_requests': self.metrics['rate_limited_requests']
        }
        
        # Mettre à jour le statut en fonction des métriques
        if self._blacklist_status == BlacklistStatus.BLACKLISTED:
            health_status['status'] = 'critical'
            health_status['message'] = 'IP potentiellement blacklistée'
        elif request_rate > self.max_requests / self.window_seconds * 0.8:
            health_status['status'] = 'warning'
            health_status['message'] = 'Taux de requêtes élevé'
        
        return health_status

    def report_error(self, error: Exception):
        """Signale une erreur pour le suivi du blacklist."""
        self._consecutive_errors += 1
        self._last_error = str(error)
        
        # Si l'erreur indique un blacklist
        if any(msg in str(error).lower() for msg in ['blacklist', 'banned', 'forbidden', '403', '429']):
            self._blacklist_status = BlacklistStatus.BLACKLISTED
            # Blacklist pour 5-15 minutes (avec backoff exponentiel)
            self._blacklist_until = time.time() + min(
                900,  # 15 minutes max
                max(300, 60 * (2 ** min(self.metrics['blacklist_events'], 4)))  # 5-15 min
            )
            logger.error(f"Blacklist IP détectée. Réessai après {self._blacklist_until - time.time():.0f}s")


class RateLimitExceeded(Exception):
    """Exception levée lorsque la limite de taux est dépassée."""
    def __init__(self, message: str, retry_after: float = None):
        super().__init__(message)
        self.retry_after = retry_after


class BlacklistDetected(Exception):
    """Exception levée lorsqu'un blacklist IP est détecté."""
    def __init__(self, message: str, retry_after: float = None):
        super().__init__(message)
        self.retry_after = retry_after
