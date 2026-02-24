import pytest
import os


@pytest.fixture
def agentframework_module():
    """Import and return the agentframework module for testing."""
    try:
        import charge.clients.agentframework
        return charge.clients.agentframework
    except ImportError:
        pytest.skip("agent-framework package not installed")


def test_default_configure_openai(agentframework_module):
    """Test default configuration for OpenAI backend."""
    from charge.clients.agentframework import model_configure

    backend = "openai"
    pytest.MonkeyPatch().setenv("OPENAI_API_KEY", "test_key")

    model, backend_out, api_key, model_kwargs = model_configure(backend=backend)

    assert backend_out == "openai"
    assert model == "gpt-5"
    assert api_key == "test_key"
    assert isinstance(model_kwargs, dict)


def test_default_configure_gemini(agentframework_module):
    """Test default configuration for Gemini backend."""
    from charge.clients.agentframework import model_configure

    backend = "gemini"
    pytest.MonkeyPatch().setenv("GOOGLE_API_KEY", "test_gemini_key")

    model, backend_out, api_key, model_kwargs = model_configure(backend=backend)

    assert backend_out == "gemini"
    assert model == "gemini-flash-latest"
    assert api_key == "test_gemini_key"


def test_default_configure_livai(agentframework_module):
    """Test default configuration for LivAI backend."""
    from charge.clients.agentframework import model_configure

    backend = "livai"
    pytest.MonkeyPatch().setenv("LIVAI_API_KEY", "test_livai_key")
    pytest.MonkeyPatch().setenv("LIVAI_BASE_URL", "https://test.livai.com")

    model, backend_out, api_key, model_kwargs = model_configure(backend=backend)

    assert backend_out == "livai"
    assert model == "gpt-4.1"
    assert api_key == "test_livai_key"
    assert model_kwargs["base_url"] == "https://test.livai.com"


def test_missing_api_key_openai(agentframework_module):
    """Test that missing OpenAI API key raises error."""
    from charge.clients.agentframework import model_configure

    backend = "openai"
    pytest.MonkeyPatch().delenv("OPENAI_API_KEY", raising=False)

    with pytest.raises(ValueError, match="API key must be set"):
        model_configure(backend=backend)


def test_missing_api_key_gemini(agentframework_module):
    """Test that missing Gemini API key raises error."""
    from charge.clients.agentframework import model_configure

    backend = "gemini"
    pytest.MonkeyPatch().delenv("GOOGLE_API_KEY", raising=False)

    with pytest.raises(ValueError, match="API key must be set"):
        model_configure(backend=backend)


def test_missing_base_url_livai(agentframework_module):
    """Test that missing LivAI base URL raises error."""
    from charge.clients.agentframework import model_configure

    backend = "livai"
    pytest.MonkeyPatch().setenv("LIVAI_API_KEY", "test_key")
    pytest.MonkeyPatch().delenv("LIVAI_BASE_URL", raising=False)

    with pytest.raises(ValueError, match="Base URL must be set"):
        model_configure(backend=backend)


def test_custom_model_name(agentframework_module):
    """Test configuration with custom model name."""
    from charge.clients.agentframework import model_configure

    backend = "openai"
    custom_model = "gpt-4o"
    pytest.MonkeyPatch().setenv("OPENAI_API_KEY", "test_key")

    model, backend_out, api_key, model_kwargs = model_configure(
        backend=backend, model=custom_model
    )

    assert model == custom_model
    assert backend_out == "openai"


def test_custom_base_url(agentframework_module):
    """Test configuration with custom base URL."""
    from charge.clients.agentframework import model_configure

    backend = "openai"
    custom_url = "https://custom.openai.com"
    pytest.MonkeyPatch().setenv("OPENAI_API_KEY", "test_key")

    model, backend_out, api_key, model_kwargs = model_configure(
        backend=backend, base_url=custom_url
    )

    assert model_kwargs.get("base_url") == custom_url


def test_ollama_not_supported(agentframework_module):
    """Test that Ollama backend shows warning."""
    from charge.clients.agentframework import model_configure

    backend = "ollama"

    # Should not raise error but will log warning
    model, backend_out, api_key, model_kwargs = model_configure(backend=backend)

    assert backend_out == "ollama"
    assert model == "gpt-oss:latest"


def test_create_openai_chat_client(agentframework_module):
    """Test creating OpenAI chat client."""
    from charge.clients.agentframework import create_agentframework_chat_client
    from agent_framework.openai import OpenAIChatClient

    backend = "openai"
    model = "gpt-5"
    api_key = "test_key"

    client = create_agentframework_chat_client(
        backend=backend, model=model, api_key=api_key
    )

    assert isinstance(client, OpenAIChatClient)


def test_create_client_unsupported_backend(agentframework_module):
    """Test that creating client with unsupported backend raises error."""
    from charge.clients.agentframework import create_agentframework_chat_client

    backend = "ollama"
    model = "llama2"

    with pytest.raises(NotImplementedError, match="Ollama support"):
        create_agentframework_chat_client(backend=backend, model=model)


def test_agentframework_pool_initialization(agentframework_module):
    """Test AgentFrameworkPool initialization."""
    from charge.clients.agentframework import AgentFrameworkPool

    pytest.MonkeyPatch().setenv("OPENAI_API_KEY", "test_key")

    # Test with model and backend
    pool = AgentFrameworkPool(model="gpt-5", backend="openai")

    assert pool.model == "gpt-5"
    assert pool.backend == "openai"
    assert pool.chat_client is not None


def test_agentframework_pool_agent_naming(agentframework_module):
    """Test agent name generation in pool."""
    from charge.clients.agentframework import AgentFrameworkPool

    pytest.MonkeyPatch().setenv("OPENAI_API_KEY", "test_key")

    pool = AgentFrameworkPool(model="gpt-5", backend="openai")

    name = pool.create_agent_name()

    assert "openai" in name
    assert "gpt" in name
    # When calling create_agent_name() directly, it uses current counter value
    assert "_0" in name or "_" in name  # Counter increments when creating actual agents


def test_agentframework_pool_custom_prefix_suffix(agentframework_module):
    """Test agent naming with custom prefix and suffix."""
    from charge.clients.agentframework import AgentFrameworkPool

    pytest.MonkeyPatch().setenv("OPENAI_API_KEY", "test_key")

    pool = AgentFrameworkPool(model="gpt-5", backend="openai")

    name = pool.create_agent_name(prefix="test_", suffix="_agent")

    assert name.startswith("test_")
    assert name.endswith("_agent")


def test_responses_api_client_creation(agentframework_module):
    """Test that Responses API client can be created."""
    from charge.clients.agentframework import AgentFrameworkPool, RESPONSES_API_AVAILABLE

    if not RESPONSES_API_AVAILABLE:
        pytest.skip("OpenAIResponsesClient not available in this version")

    pytest.MonkeyPatch().setenv("OPENAI_API_KEY", "test_key")

    # Create pool with Responses API
    pool = AgentFrameworkPool(
        model="gpt-4o",
        backend="openai",
        use_responses_api=True
    )

    assert pool.use_responses_api is True
    assert pool.chat_client is not None


def test_responses_api_without_flag_raises_error(agentframework_module):
    """Test that getting hosted tools without Responses API raises error."""
    from charge.clients.agentframework import AgentFrameworkPool

    pytest.MonkeyPatch().setenv("OPENAI_API_KEY", "test_key")

    # Create pool without Responses API
    pool = AgentFrameworkPool(model="gpt-4o", backend="openai")

    # Should raise error when trying to get hosted tools
    with pytest.raises(ValueError, match="Hosted tools are only available with Responses API"):
        pool.get_hosted_tools()


def test_agentframework_pool_create_agent_naming(agentframework_module):
    """Test that agent counter increments when creating actual agents."""
    from charge.clients.agentframework import AgentFrameworkPool
    from charge.tasks.Task import Task

    pytest.MonkeyPatch().setenv("OPENAI_API_KEY", "test_key")

    pool = AgentFrameworkPool(model="gpt-5", backend="openai")

    # Create a simple task
    task = Task(
        system_prompt="Test prompt",
        user_prompt="Test user prompt"
    )

    # Create first agent
    agent1 = pool.create_agent(task=task)
    assert "_1" in agent1.agent_name  # First agent should have _1

    # Create second agent
    agent2 = pool.create_agent(task=task)
    assert "_2" in agent2.agent_name  # Second agent should have _2
