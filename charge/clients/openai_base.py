################################################################################
## Copyright 2025 Lawrence Livermore National Security, LLC. and Binghamton University.
## See the top-level LICENSE file for details.
##
## SPDX-License-Identifier: Apache-2.0
################################################################################
"""
Common OpenAI API functionality shared between AutoGen and Agent Framework clients.

This module provides an abstraction layer for OpenAI API configuration, client creation,
and common utilities that can be used by both AutoGen and Agent Framework implementations.
"""

import os
import httpx
from typing import Tuple, Optional, Dict, Any
from loguru import logger


class LoggingTransport(httpx.AsyncHTTPTransport):
    """
    Custom HTTP transport that logs all requests and responses.

    Masks API keys in logs for security.
    """

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        # Log request details (mask API key)
        headers_copy = dict(request.headers)
        if "authorization" in headers_copy:
            auth_header = headers_copy["authorization"]
            if auth_header.startswith("Bearer "):
                masked_key = auth_header[:14] + "..." + auth_header[-4:]
                headers_copy["authorization"] = masked_key

        try:
            response = await super().handle_async_request(request)
            return response
        except Exception as e:
            logger.error(f"HTTP Request Failed: {type(e).__name__}: {e}")
            raise


def get_api_key_for_backend(backend: str, provided_key: Optional[str] = None) -> Optional[str]:
    """
    Get the API key for a given backend.

    Args:
        backend: The backend name (openai, gemini, livai, etc.)
        provided_key: Optional pre-provided API key

    Returns:
        API key string or None
    """
    if provided_key:
        return provided_key

    # Map backends to environment variable names
    env_var_map = {
        "openai": "OPENAI_API_KEY",
        "gemini": "GOOGLE_API_KEY",
        "livai": "LIVAI_API_KEY",
        "livchat": "LIVAI_API_KEY",
        "llamame": "LLAMAME_API_KEY",
        "alcf": "ALCF_API_KEY",
    }

    env_var = env_var_map.get(backend)
    if env_var:
        return os.getenv(env_var)

    return None


def get_base_url_for_backend(backend: str, provided_url: Optional[str] = None) -> Optional[str]:
    """
    Get the base URL for a given backend.

    Args:
        backend: The backend name
        provided_url: Optional pre-provided base URL

    Returns:
        Base URL string or None
    """
    if provided_url:
        return provided_url

    # Map backends to environment variable names for base URLs
    env_var_map = {
        "livai": "LIVAI_BASE_URL",
        "livchat": "LIVAI_BASE_URL",
        "llamame": "LLAMAME_BASE_URL",
        "alcf": "ALCF_BASE_URL",
    }

    env_var = env_var_map.get(backend)
    if env_var:
        return os.getenv(env_var)

    return None


def get_default_model_for_backend(backend: str) -> str:
    """
    Get the default model for a given backend.

    Args:
        backend: The backend name

    Returns:
        Default model name for the backend
    """
    default_models = {
        "openai": "gpt-5",
        "gemini": "gemini-flash-latest",
        "livai": "gpt-4.1",
        "livchat": "gpt-4.1",
        "llamame": "openai/gpt-oss-120b",
        "alcf": "openai/gpt-oss-120b",
        "ollama": "gpt-oss:latest",
        "huggingface": "gpt-oss",
        "vllm": "gpt-oss",
    }

    return default_models.get(backend, "gpt-4o")


def configure_openai_backend(
    backend: str,
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
) -> Tuple[str, str, Optional[str], Dict[str, Any]]:
    """
    Configure OpenAI-compatible backend settings.

    This function handles configuration for OpenAI and OpenAI-compatible backends
    including API key retrieval, base URL configuration, and model selection.

    Args:
        backend: The backend to use (openai, gemini, livai, etc.)
        model: Optional model name (uses default if not provided)
        api_key: Optional API key (retrieves from env if not provided)
        base_url: Optional base URL for custom endpoints

    Returns:
        Tuple of (model, backend, api_key, kwargs)

    Raises:
        ValueError: If API key is required but not found
    """
    kwargs: Dict[str, Any] = {}

    # Get API key
    api_key = get_api_key_for_backend(backend, api_key)

    # Get base URL for custom endpoints
    if backend in ["livai", "livchat", "llamame", "alcf"]:
        base_url = get_base_url_for_backend(backend, base_url)
        if not base_url:
            raise ValueError(
                f"{backend.upper()} Base URL must be set via base_url parameter "
                f"or {backend.upper()}_BASE_URL environment variable"
            )
        kwargs["base_url"] = base_url
    elif base_url:
        # Custom base URL provided for standard backends
        kwargs["base_url"] = base_url

    # Validate API key for backends that require it
    if backend in ["openai", "gemini", "livai", "livchat", "llamame", "alcf"]:
        if not api_key:
            raise ValueError(
                f"API key must be set for {backend} backend. "
                f"Provide via api_key parameter or environment variable."
            )

    # Get model name
    if not model:
        model = get_default_model_for_backend(backend)

    return model, backend, api_key, kwargs


def configure_special_backends(
    backend: str,
    model: Optional[str] = None,
) -> Tuple[str, str, Optional[str], Dict[str, Any]]:
    """
    Configure special backends (ollama, huggingface, vllm).

    These backends have different configuration requirements and may not
    need API keys.

    Args:
        backend: The backend to use (ollama, huggingface, vllm)
        model: Optional model name (uses default if not provided)

    Returns:
        Tuple of (model, backend, api_key, kwargs)
    """
    kwargs: Dict[str, Any] = {}
    api_key = None

    if backend == "ollama":
        model = model or "gpt-oss:latest"
        logger.info("Ollama backend configured")

    elif backend == "huggingface":
        model = model or "gpt-oss"
        logger.info("HuggingFace backend configured")

    elif backend == "vllm":
        model = model or "gpt-oss"
        # vLLM-specific configuration
        reasoning_effort = os.getenv("OSS_REASONING", "medium")
        kwargs["reasoning_effort"] = reasoning_effort
        logger.info(f"vLLM backend configured with reasoning_effort={reasoning_effort}")

    return model, backend, api_key, kwargs


def model_configure(
    backend: str,
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
) -> Tuple[str, str, Optional[str], Dict[str, Any]]:
    """
    Universal model configuration function for all backends.

    This function provides a unified interface for configuring model clients
    across different backends. It handles API key retrieval, base URL configuration,
    and backend-specific settings.

    Args:
        backend: The backend to use (openai, gemini, ollama, etc.)
        model: Optional model name (uses backend-specific default if not provided)
        api_key: Optional API key (retrieves from environment if not provided)
        base_url: Optional base URL for custom endpoints

    Returns:
        Tuple of (model, backend, api_key, kwargs) where:
        - model: The model name to use
        - backend: The backend name (unchanged)
        - api_key: The API key (may be None for some backends)
        - kwargs: Additional configuration kwargs

    Raises:
        ValueError: If required configuration is missing

    Example:
        >>> model, backend, key, kwargs = model_configure("openai")
        >>> # model="gpt-5", backend="openai", key=<from env>, kwargs={}

        >>> model, backend, key, kwargs = model_configure(
        ...     "livai",
        ...     base_url="https://custom.com"
        ... )
    """
    # Determine if this is an OpenAI-compatible or special backend
    openai_compatible = backend in [
        "openai", "gemini", "livai", "livchat", "llamame", "alcf"
    ]

    if openai_compatible:
        return configure_openai_backend(backend, model, api_key, base_url)
    else:
        return configure_special_backends(backend, model)


def create_http_client(disable_ssl_verify: bool = False) -> httpx.AsyncClient:
    """
    Create an HTTP client with logging transport.

    Args:
        disable_ssl_verify: Whether to disable SSL verification (default: False)

    Returns:
        Configured AsyncClient with logging
    """
    return httpx.AsyncClient(
        verify=not disable_ssl_verify,
        transport=LoggingTransport()
    )


# Backend capability matrix
BACKEND_CAPABILITIES = {
    "openai": {
        "api_key_required": True,
        "supports_function_calling": True,
        "supports_streaming": True,
        "supports_vision": True,
        "supports_json_mode": True,
    },
    "gemini": {
        "api_key_required": True,
        "supports_function_calling": True,
        "supports_streaming": True,
        "supports_vision": True,
        "supports_json_mode": True,
    },
    "ollama": {
        "api_key_required": False,
        "supports_function_calling": True,
        "supports_streaming": True,
        "supports_vision": False,
        "supports_json_mode": True,
    },
    "vllm": {
        "api_key_required": False,
        "supports_function_calling": True,
        "supports_streaming": True,
        "supports_vision": False,
        "supports_json_mode": True,
    },
}


def get_backend_capabilities(backend: str) -> Dict[str, bool]:
    """
    Get capability information for a backend.

    Args:
        backend: The backend name

    Returns:
        Dictionary of capability flags
    """
    return BACKEND_CAPABILITIES.get(backend, {
        "api_key_required": True,
        "supports_function_calling": True,
        "supports_streaming": True,
        "supports_vision": False,
        "supports_json_mode": True,
    })
