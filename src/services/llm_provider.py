"""
LLM Provider Service with API Key Rotation and Fallback.

This service manages multiple Gemini API keys and automatically
rotates to fallback keys when rate limits are hit.
"""

import os
import time
import logging
from typing import List, Optional, Dict, Any
from dataclasses import dataclass, field
from threading import Lock
from datetime import datetime, timedelta

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.language_models.chat_models import BaseChatModel

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class APIKeyState:
    """State tracking for an API key."""
    key: str
    is_exhausted: bool = False
    exhausted_at: Optional[datetime] = None
    requests_count: int = 0
    last_used: Optional[datetime] = None
    cooldown_until: Optional[datetime] = None
    
    def mark_exhausted(self, cooldown_seconds: int = 60):
        """Mark this key as exhausted with a cooldown period."""
        self.is_exhausted = True
        self.exhausted_at = datetime.now()
        self.cooldown_until = datetime.now() + timedelta(seconds=cooldown_seconds)
        logger.warning(f"API key ...{self.key[-8:]} marked exhausted. Cooldown until {self.cooldown_until}")
    
    def is_available(self) -> bool:
        """Check if this key is available for use."""
        if not self.is_exhausted:
            return True
        
        # Check if cooldown period has passed
        if self.cooldown_until and datetime.now() > self.cooldown_until:
            self.is_exhausted = False
            self.cooldown_until = None
            logger.info(f"API key ...{self.key[-8:]} cooldown ended, now available")
            return True
        
        return False
    
    def record_usage(self):
        """Record that this key was used."""
        self.requests_count += 1
        self.last_used = datetime.now()


class GeminiKeyManager:
    """
    Manages multiple Gemini API keys with automatic rotation and fallback.
    
    Features:
    - Round-robin key rotation
    - Automatic fallback on rate limit errors
    - Cooldown tracking for exhausted keys
    - Thread-safe operations
    """
    
    _instance = None
    _lock = Lock()
    
    def __new__(cls):
        """Singleton pattern to ensure one key manager across the app."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        
        self._keys: List[APIKeyState] = []
        self._current_index = 0
        self._model_name = "gemini-flash-lite-latest"
        self._fallback_model = "gemini-2.5-flash-lite"
        self._lock = Lock()
        self._load_keys()
        self._initialized = True
    
    def _load_keys(self):
        """Load API keys from environment variables."""
        # Primary key
        primary_key = os.getenv("GOOGLE_API_KEY")
        if primary_key:
            self._keys.append(APIKeyState(key=primary_key))
            logger.info(f"Loaded primary API key: ...{primary_key[-8:]}")
        
        # Load additional keys (GOOGLE_API_KEY_1, GOOGLE_API_KEY_2, etc.)
        for i in range(1, 20):  # Support up to 20 fallback keys
            key = os.getenv(f"GOOGLE_API_KEY_{i}")
            if key:
                self._keys.append(APIKeyState(key=key))
                logger.info(f"Loaded fallback API key {i}: ...{key[-8:]}")
        
        # Also check for comma-separated keys in GOOGLE_API_KEYS
        multi_keys = os.getenv("GOOGLE_API_KEYS", "")
        if multi_keys:
            for key in multi_keys.split(","):
                key = key.strip()
                if key and not any(k.key == key for k in self._keys):
                    self._keys.append(APIKeyState(key=key))
                    logger.info(f"Loaded API key from GOOGLE_API_KEYS: ...{key[-8:]}")
        
        if not self._keys:
            logger.error("No Gemini API keys found! Set GOOGLE_API_KEY or GOOGLE_API_KEY_1, etc.")
        else:
            logger.info(f"Total API keys loaded: {len(self._keys)}")
    
    def add_key(self, api_key: str):
        """Dynamically add a new API key."""
        with self._lock:
            if not any(k.key == api_key for k in self._keys):
                self._keys.append(APIKeyState(key=api_key))
                logger.info(f"Added new API key: ...{api_key[-8:]}")
    
    def get_available_key(self) -> Optional[str]:
        """Get the next available API key using round-robin."""
        with self._lock:
            if not self._keys:
                return None
            
            # Try to find an available key starting from current index
            attempts = 0
            while attempts < len(self._keys):
                key_state = self._keys[self._current_index]
                
                if key_state.is_available():
                    key_state.record_usage()
                    current_key = key_state.key
                    # Move to next key for next request (round-robin)
                    self._current_index = (self._current_index + 1) % len(self._keys)
                    return current_key
                
                # Try next key
                self._current_index = (self._current_index + 1) % len(self._keys)
                attempts += 1
            
            # All keys exhausted, return the one with earliest cooldown end
            earliest_available = min(
                self._keys,
                key=lambda k: k.cooldown_until or datetime.max
            )
            logger.warning(f"All keys exhausted. Returning key with earliest cooldown: ...{earliest_available.key[-8:]}")
            return earliest_available.key
    
    def mark_key_exhausted(self, api_key: str, cooldown_seconds: int = 60):
        """Mark a specific key as exhausted."""
        with self._lock:
            for key_state in self._keys:
                if key_state.key == api_key:
                    key_state.mark_exhausted(cooldown_seconds)
                    break
    
    def get_status(self) -> Dict[str, Any]:
        """Get status of all API keys."""
        with self._lock:
            return {
                "total_keys": len(self._keys),
                "available_keys": sum(1 for k in self._keys if k.is_available()),
                "current_index": self._current_index,
                "keys": [
                    {
                        "key_suffix": f"...{k.key[-8:]}",
                        "is_available": k.is_available(),
                        "requests_count": k.requests_count,
                        "is_exhausted": k.is_exhausted,
                        "cooldown_until": str(k.cooldown_until) if k.cooldown_until else None,
                    }
                    for k in self._keys
                ]
            }
    
    def get_model(self, model_name: str = None) -> ChatGoogleGenerativeAI:
        """
        Get a ChatGoogleGenerativeAI instance with an available API key.
        
        Args:
            model_name: Optional model name override
        
        Returns:
            Configured ChatGoogleGenerativeAI instance
        """
        api_key = self.get_available_key()
        if not api_key:
            raise ValueError("No Gemini API keys available")
        
        return ChatGoogleGenerativeAI(
            model=model_name or self._model_name,
            google_api_key=api_key,
            temperature=0,
            max_retries=2,
        )


# Global singleton key manager
_global_key_manager = None

def _get_key_manager() -> GeminiKeyManager:
    """Get the global key manager instance."""
    global _global_key_manager
    if _global_key_manager is None:
        _global_key_manager = GeminiKeyManager()
    return _global_key_manager


class RobustGeminiChat(ChatGoogleGenerativeAI):
    """
    A LangChain-compatible wrapper around ChatGoogleGenerativeAI that handles rate limits
    by automatically rotating to fallback API keys.
    
    This class properly inherits from ChatGoogleGenerativeAI to maintain full LangChain
    compatibility while adding automatic key rotation on 429 errors.
    """
    
    # Use model_config to allow arbitrary types and extra fields
    model_config = {"arbitrary_types_allowed": True, "extra": "allow"}
    
    def __init__(self, model_name: str = "gemini-2.5-flash-lite", **kwargs):
        km = _get_key_manager()
        current_key = km.get_available_key()
        
        if not current_key:
            raise ValueError("No Gemini API keys available")
        
        super().__init__(
            model=model_name,
            google_api_key=current_key,
            temperature=kwargs.get("temperature", 0),
            max_retries=0,  # Disable internal retries, we handle them ourselves
            **{k: v for k, v in kwargs.items() if k not in ["temperature", "max_retries"]}
        )
        
        # Store the current key in object's __dict__ to bypass Pydantic
        object.__setattr__(self, '_robust_current_key', current_key)
    
    def _get_current_key(self) -> str:
        """Get the current API key."""
        return object.__getattribute__(self, '_robust_current_key')
    
    def _set_current_key(self, key: str):
        """Set the current API key."""
        object.__setattr__(self, '_robust_current_key', key)
    
    def _rotate_key(self) -> bool:
        """Rotate to the next available API key."""
        km = _get_key_manager()
        current_key = self._get_current_key()
        
        if current_key:
            km.mark_key_exhausted(current_key, cooldown_seconds=60)
        
        new_key = km.get_available_key()
        if new_key and new_key != current_key:
            self._set_current_key(new_key)
            # Update the google_api_key attribute
            object.__setattr__(self, 'google_api_key', new_key)
            # Force client refresh by clearing cached client
            if hasattr(self, '_client'):
                object.__setattr__(self, '_client', None)
            logger.info(f"Rotated to new API key: ...{new_key[-8:]}")
            return True
        return False
    
    def _generate(self, *args, **kwargs):
        """Override _generate to add key rotation on rate limit errors."""
        km = _get_key_manager()
        max_attempts = len(km._keys) + 1 if km._keys else 1
        last_error = None
        
        for attempt in range(max_attempts):
            try:
                return super()._generate(*args, **kwargs)
            except Exception as e:
                error_str = str(e).lower()
                last_error = e
                
                # Check if it's a rate limit error
                if "429" in str(e) or "quota" in error_str or "rate" in error_str or "resource" in error_str:
                    current_key = self._get_current_key()
                    logger.warning(f"Rate limit hit (attempt {attempt + 1}/{max_attempts}) for key ...{current_key[-8:] if current_key else 'None'}: {str(e)[:100]}")
                    
                    if self._rotate_key() and attempt < max_attempts - 1:
                        continue
                
                raise
        
        raise last_error or Exception("All API keys exhausted")
    
    async def _agenerate(self, *args, **kwargs):
        """Override _agenerate for async calls with key rotation."""
        km = _get_key_manager()
        max_attempts = len(km._keys) + 1 if km._keys else 1
        last_error = None
        
        for attempt in range(max_attempts):
            try:
                return await super()._agenerate(*args, **kwargs)
            except Exception as e:
                error_str = str(e).lower()
                last_error = e
                
                # Check if it's a rate limit error
                if "429" in str(e) or "quota" in error_str or "rate" in error_str or "resource" in error_str:
                    current_key = self._get_current_key()
                    logger.warning(f"Rate limit hit (async, attempt {attempt + 1}/{max_attempts}) for key ...{current_key[-8:] if current_key else 'None'}: {str(e)[:100]}")
                    
                    if self._rotate_key() and attempt < max_attempts - 1:
                        continue
                
                raise
        
        raise last_error or Exception("All API keys exhausted")


# Singleton instance
key_manager = GeminiKeyManager()


def get_model(model_name: str = "gemini-2.5-flash-lite") -> RobustGeminiChat:
    """
    Get a Gemini model with automatic key rotation.
    
    This is the main function to use throughout the application.
    Returns a RobustGeminiChat which is fully LangChain compatible
    but automatically rotates API keys on rate limit errors.
    """
    return RobustGeminiChat(model_name=model_name)


def get_robust_model(model_name: str = "gemini-2.5-flash-lite") -> RobustGeminiChat:
    """
    Alias for get_model - returns a robust Gemini model wrapper with automatic failover.
    """
    return RobustGeminiChat(model_name=model_name)


def get_key_status() -> Dict[str, Any]:
    """Get status of all API keys."""
    return key_manager.get_status()


def add_api_key(api_key: str):
    """Add a new API key dynamically."""
    key_manager.add_key(api_key)
