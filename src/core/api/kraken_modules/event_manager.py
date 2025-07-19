from typing import Dict, List, Optional, Callable, Any
from ..kraken_api.validators import KrakenValidator
from ..kraken_api.endpoints import KrakenEndpoints
from ..kraken_api.metrics import KrakenMetrics
from ..kraken_api.websocket import KrakenWebSocket

class KrakenEventManager:
    def __init__(self, validator: KrakenValidator, endpoints: KrakenEndpoints, metrics: KrakenMetrics, websocket: KrakenWebSocket):
        self._validator = validator
        self._endpoints = endpoints
        self._metrics = metrics
        self._websocket = websocket
        self._event_listeners = {}

    async def add_event_listener(self, event_type: str, callback: Callable) -> None:
        """Ajoute un callback pour un type d'événement spécifique."""
        if event_type not in self._event_listeners:
            self._event_listeners[event_type] = []
        self._event_listeners[event_type].append(callback)

    async def remove_event_listener(self, event_type: str, callback: Callable) -> None:
        """Supprime un callback pour un type d'événement."""
        if event_type in self._event_listeners:
            self._event_listeners[event_type].remove(callback)
            if not self._event_listeners[event_type]:
                del self._event_listeners[event_type]

    async def emit_event(self, event_type: str, data: Any) -> None:
        """Émet un événement avec les données associées."""
        if event_type in self._event_listeners:
            for callback in self._event_listeners[event_type]:
                try:
                    await callback(data)
                except Exception as e:
                    self._metrics.metrics['errors']['event_handlers'] += 1
                    self._metrics.metrics['errors']['event_handlers_by_type'][str(e)] += 1

    async def get_event_listeners(self) -> Dict:
        """Récupère la liste des écouteurs d'événements."""
        return {k: len(v) for k, v in self._event_listeners.items()}

    async def clear_event_listeners(self) -> None:
        """Supprime tous les écouteurs d'événements."""
        self._event_listeners.clear()

    async def set_ws_message_handler(self, handler: Callable) -> None:
        """Configure le handler de messages WebSocket."""
        self._websocket.message_handler = handler

    async def get_ws_message_handler(self) -> Optional[Callable]:
        """Récupère le handler de messages WebSocket actuel."""
        return self._websocket.message_handler

    async def set_ws_error_handler(self, handler: Callable) -> None:
        """Configure le handler d'erreurs WebSocket."""
        self._websocket.error_handler = handler

    async def get_ws_error_handler(self) -> Optional[Callable]:
        """Récupère le handler d'erreurs WebSocket actuel."""
        return self._websocket.error_handler

    async def set_ws_close_handler(self, handler: Callable) -> None:
        """Configure le handler de fermeture WebSocket."""
        self._websocket.close_handler = handler

    async def get_ws_close_handler(self) -> Optional[Callable]:
        """Récupère le handler de fermeture WebSocket actuel."""
        return self._websocket.close_handler
