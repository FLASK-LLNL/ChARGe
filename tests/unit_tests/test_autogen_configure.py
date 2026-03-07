import pytest


@pytest.fixture
def autogen_module():
    import charge.clients.autogen

    return charge.clients.autogen


def test_default_configure_ollama(autogen_module):
    model_configure = autogen_module.model_configure
    backend = "ollama"
    model, backend_out, api_key, model_kwargs = model_configure(backend=backend)
    assert backend_out == "ollama"
    assert model == "gpt-oss:latest"
    assert api_key is None
    assert model_kwargs == {}


def test_missing_api_gemini(autogen_module):
    model_configure = autogen_module.model_configure
    backend = "gemini"
    pytest.MonkeyPatch().delenv("GOOGLE_API_KEY", raising=False)
    with pytest.raises(ValueError) as excinfo:
        model_configure(backend=backend)
    assert "API key must be set for backend gemini" in str(excinfo.value)


def test_default_configure_gemini(autogen_module):
    import httpx

    model_configure = autogen_module.model_configure
    backend = "gemini"
    pytest.MonkeyPatch().setenv("GOOGLE_API_KEY", "test_key")
    model, backend_out, api_key, model_kwargs = model_configure(backend=backend)
    assert backend_out == "gemini"
    assert model == "gemini-flash-latest"
    assert api_key == "test_key"
    assert model_kwargs["parallel_tool_calls"] == False
    assert model_kwargs["reasoning_effort"] == "high"
    assert isinstance(model_kwargs["http_client"], httpx.AsyncClient)


def test_missing_api_key_openai(autogen_module):
    model_configure = autogen_module.model_configure
    backend = "openai"
    pytest.MonkeyPatch().delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(ValueError) as excinfo:
        model_configure(backend=backend)
    assert "API key must be set for backend openai" in str(excinfo.value)


def test_default_configure_openai(autogen_module):
    import httpx

    model_configure = autogen_module.model_configure
    backend = "openai"
    pytest.MonkeyPatch().setenv("OPENAI_API_KEY", "openai_test_key")
    model, backend_out, api_key, model_kwargs = model_configure(backend=backend)
    assert backend_out == "openai"
    assert model == "gpt-5"
    assert api_key == "openai_test_key"
    assert model_kwargs["reasoning_effort"] == "high"
    assert isinstance(model_kwargs["http_client"], httpx.AsyncClient)


def test_default_configure_livchat(autogen_module):
    import httpx

    model_configure = autogen_module.model_configure
    base_url = "https://test.url"
    for backend in ["livchat", "livai"]:

        pytest.MonkeyPatch().setenv("LIVAI_API_KEY", "livchat_test_key")
        pytest.MonkeyPatch().setenv("LIVAI_BASE_URL", base_url)
        model, backend_out, api_key, model_kwargs = model_configure(backend=backend)
        assert backend_out == backend
        assert model == "gpt-4.1"
        assert api_key == "livchat_test_key"

        for key, value in model_kwargs.items():
            if key == "http_client":
                assert isinstance(value, httpx.AsyncClient)
            else:
                assert (
                    value
                    == {
                        "base_url": base_url,
                    }[key]
                )
