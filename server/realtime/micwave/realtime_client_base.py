"""
Abstract base class for realtime audio-text clients.
Provides a common interface for different providers (OpenAI, x.ai, etc.)
"""
from abc import ABC, abstractmethod
from typing import Callable, Dict, List, Optional
import asyncio
import logging

logger = logging.getLogger(__name__)


class RealtimeClientBase(ABC):
    """Abstract base class for realtime audio-text API clients"""
    
    def __init__(self, api_key: str):
        """
        Initialize the realtime client.

        Args:
            api_key: API key for the provider
        """
        self.api_key = api_key
        self.ws = None
        self.session_id = None
        self.receive_task = None
        self.handlers: Dict[str, Callable[[dict], asyncio.Future]] = {}
        self.queue = asyncio.Queue()
        self._on_disconnect_callback: Optional[Callable[[], asyncio.Future]] = None

    def set_on_disconnect(self, callback: Callable[[], asyncio.Future]):
        """Register async callback invoked when the provider WebSocket drops."""
        self._on_disconnect_callback = callback

    async def _fire_on_disconnect(self):
        """Invoke the on_disconnect callback if registered."""
        if self._on_disconnect_callback:
            try:
                await self._on_disconnect_callback()
            except Exception as e:
                logger.error(f"Error in on_disconnect callback: {e}", exc_info=True)
    
    @abstractmethod
    async def connect(
        self,
        modalities: List[str] = None,
        instructions: Optional[str] = None,
    ):
        """
        Connect to the realtime API and configure the session.
        
        Args:
            modalities: List of modalities to use (e.g., ["text", "audio"])
            instructions: Optional system prompt for the session
        """
        pass
    
    @abstractmethod
    async def send_audio(self, audio_data: bytes):
        """
        Send audio data to the API.
        
        Args:
            audio_data: Raw PCM16 audio bytes
        """
        pass
    
    @abstractmethod
    async def commit_audio(self):
        """Commit the audio buffer."""
        pass
    
    @abstractmethod
    async def clear_audio_buffer(self):
        """Clear the audio buffer."""
        pass

    @abstractmethod
    async def refresh_session(
        self,
        modalities: List[str] = None,
        instructions: Optional[str] = None,
    ):
        """Refresh session configuration (for reused provider sessions)."""
        pass
    
    @abstractmethod
    async def close(self):
        """Close the WebSocket connection and cleanup."""
        pass
    
    def register_handler(self, message_type: str, handler: Callable[[dict], asyncio.Future]):
        """
        Register a handler for a specific message type.
        
        Args:
            message_type: Type of message to handle
            handler: Async function that processes the message
        """
        self.handlers[message_type] = handler
    
    def _is_ws_open(self) -> bool:
        """
        Check if WebSocket connection is open.
        Compatibility check for different websockets versions.
        
        Returns:
            True if connection is open, False otherwise
        """
        if not self.ws:
            return False
        # Prefer 'closed' if available (newer versions)
        if hasattr(self.ws, "closed"):
            try:
                return not bool(self.ws.closed)
            except Exception:
                pass
        # Fallback to 'open' (older versions)
        if hasattr(self.ws, "open"):
            try:
                return bool(self.ws.open)
            except Exception:
                pass
        # If neither attribute is reliable, assume open if object exists
        return True
    
    async def default_handler(self, data: dict):
        """
        Default handler for unhandled message types.
        Override in subclasses if needed.
        
        Args:
            data: Message data
        """
        message_type = data.get("type", "unknown")
        logger.warning(f"Unhandled message type received: {message_type}")
