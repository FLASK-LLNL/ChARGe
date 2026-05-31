"""
Unit tests for OpenAI endpoint discovery functionality.
"""

import pytest
import responses
from charge.clients.openai_base import discover_available_models, get_model_ids


@pytest.fixture
def mock_openai_models_response():
    """Mock OpenAI models API response."""
    return {
        "object": "list",
        "data": [
            {
                "id": "gpt-4",
                "object": "model",
                "created": 1687882411,
                "owned_by": "openai",
            },
            {
                "id": "gpt-3.5-turbo",
                "object": "model",
                "created": 1677610602,
                "owned_by": "openai",
            },
            {
                "id": "text-embedding-ada-002",
                "object": "model",
                "created": 1671217299,
                "owned_by": "openai-internal",
            },
        ],
    }


@pytest.fixture
def mock_vllm_models_response():
    """Mock vLLM models API response (alternative format)."""
    return [
        {"id": "meta-llama/Llama-3.1-70B-Instruct"},
        {"id": "google/Gemma-4"},
    ]


@responses.activate
def test_discover_available_models_openai_format(mock_openai_models_response):
    """Test discovering models from OpenAI-formatted response."""
    responses.add(
        responses.GET,
        "https://api.openai.com/v1/models",
        json=mock_openai_models_response,
        status=200,
    )

    models = discover_available_models("https://api.openai.com/v1", api_key="test-key")

    assert len(models) == 3
    assert models[0]["id"] == "gpt-4"
    assert models[1]["id"] == "gpt-3.5-turbo"
    assert models[2]["id"] == "text-embedding-ada-002"


@responses.activate
def test_discover_available_models_list_format(mock_vllm_models_response):
    """Test discovering models from list-formatted response."""
    responses.add(
        responses.GET,
        "http://localhost:8000/v1/models",
        json=mock_vllm_models_response,
        status=200,
    )

    models = discover_available_models("http://localhost:8000/v1")

    assert len(models) == 2
    assert models[0]["id"] == "meta-llama/Llama-3.1-70B-Instruct"
    assert models[1]["id"] == "google/Gemma-4"


@responses.activate
def test_discover_available_models_with_api_key():
    """Test that API key is correctly passed in Authorization header."""
    responses.add(
        responses.GET,
        "https://api.openai.com/v1/models",
        json={"data": []},
        status=200,
    )

    discover_available_models("https://api.openai.com/v1", api_key="sk-test123")

    # Verify the Authorization header was sent
    assert len(responses.calls) == 1
    assert responses.calls[0].request.headers["Authorization"] == "Bearer sk-test123"


@responses.activate
def test_discover_available_models_without_api_key():
    """Test discovery without API key (for local endpoints)."""
    responses.add(
        responses.GET,
        "http://localhost:11434/v1/models",
        json={"data": [{"id": "llama2"}]},
        status=200,
    )

    models = discover_available_models("http://localhost:11434/v1")

    assert len(models) == 1
    assert "Authorization" not in responses.calls[0].request.headers


@responses.activate
def test_discover_available_models_normalizes_url():
    """Test that trailing slashes are handled correctly."""
    responses.add(
        responses.GET,
        "https://api.openai.com/v1/models",
        json={"data": []},
        status=200,
    )

    # Test with trailing slash
    discover_available_models("https://api.openai.com/v1/")

    assert len(responses.calls) == 1
    assert responses.calls[0].request.url == "https://api.openai.com/v1/models"


@responses.activate
def test_discover_available_models_error_handling():
    """Test error handling for failed requests."""
    responses.add(
        responses.GET,
        "https://api.openai.com/v1/models",
        json={"error": "Unauthorized"},
        status=401,
    )

    with pytest.raises(Exception):
        discover_available_models("https://api.openai.com/v1", api_key="invalid-key")


@responses.activate
def test_get_model_ids(mock_openai_models_response):
    """Test extracting model IDs from response."""
    responses.add(
        responses.GET,
        "https://api.openai.com/v1/models",
        json=mock_openai_models_response,
        status=200,
    )

    model_ids = get_model_ids("https://api.openai.com/v1", api_key="test-key")

    assert len(model_ids) == 3
    assert "gpt-4" in model_ids
    assert "gpt-3.5-turbo" in model_ids
    assert "text-embedding-ada-002" in model_ids


@responses.activate
def test_get_model_ids_filters_invalid_entries():
    """Test that model IDs without 'id' field are filtered out."""
    responses.add(
        responses.GET,
        "https://api.openai.com/v1/models",
        json={
            "data": [
                {"id": "valid-model"},
                {"name": "invalid-entry-no-id"},
                {"id": "another-valid-model"},
            ]
        },
        status=200,
    )

    model_ids = get_model_ids("https://api.openai.com/v1")

    assert len(model_ids) == 2
    assert "valid-model" in model_ids
    assert "another-valid-model" in model_ids


@responses.activate
def test_discover_available_models_timeout():
    """Test timeout parameter is respected."""
    import requests

    responses.add(
        responses.GET,
        "https://slow-endpoint.com/v1/models",
        body=requests.exceptions.Timeout(),
    )

    with pytest.raises(requests.exceptions.Timeout):
        discover_available_models("https://slow-endpoint.com/v1", timeout=1)


@responses.activate
def test_discover_available_models_empty_response():
    """Test handling of empty model list."""
    responses.add(
        responses.GET,
        "https://api.openai.com/v1/models",
        json={"data": []},
        status=200,
    )

    models = discover_available_models("https://api.openai.com/v1")

    assert models == []


@responses.activate
def test_discover_available_models_unexpected_format():
    """Test handling of unexpected response format."""
    responses.add(
        responses.GET,
        "https://api.openai.com/v1/models",
        json={"unexpected": "format"},
        status=200,
    )

    models = discover_available_models("https://api.openai.com/v1")

    assert models == []
