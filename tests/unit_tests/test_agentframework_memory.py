import pytest
import json


@pytest.fixture
def agentframework_utils_module():
    """Import and return the agentframework_utils module for testing."""
    try:
        import charge.clients.agentframework_utils
        return charge.clients.agentframework_utils
    except ImportError:
        pytest.skip("agent-framework package not installed")


@pytest.mark.asyncio
async def test_agentframework_memory_initialization(agentframework_utils_module):
    """Test AgentFrameworkMemory initialization."""
    from charge.clients.agentframework_utils import AgentFrameworkMemory

    memory = AgentFrameworkMemory(name="test_memory")

    assert memory.name == "test_memory"
    assert memory.get_session() is None
    assert len(memory.get_messages()) == 0
    assert len(memory.get_source_agents()) == 0


@pytest.mark.asyncio
async def test_agentframework_memory_add_content(agentframework_utils_module):
    """Test adding content to memory."""
    from charge.clients.agentframework_utils import AgentFrameworkMemory

    memory = AgentFrameworkMemory()

    await memory.add("First message", "agent_1")
    await memory.add("Second message", "agent_2")

    messages = memory.get_messages()
    sources = memory.get_source_agents()

    assert len(messages) == 2
    assert len(sources) == 2
    assert messages[0]["content"] == "First message"
    assert messages[1]["content"] == "Second message"
    assert sources[0] == "agent_1"
    assert sources[1] == "agent_2"


@pytest.mark.asyncio
async def test_agentframework_memory_serialization(agentframework_utils_module):
    """Test memory serialization to JSON."""
    from charge.clients.agentframework_utils import AgentFrameworkMemory

    memory = AgentFrameworkMemory(name="test_memory")

    await memory.add("Test message 1", "agent_1")
    await memory.add("Test message 2", "agent_2")

    serialized = memory.serialize_memory_content()

    # Should be valid JSON
    parsed = json.loads(serialized)

    assert isinstance(parsed, list)
    assert len(parsed) == 2
    assert parsed[0]["content"] == "Test message 1"
    assert parsed[0]["source_agent"] == "agent_1"
    assert parsed[1]["content"] == "Test message 2"
    assert parsed[1]["source_agent"] == "agent_2"


@pytest.mark.asyncio
async def test_agentframework_memory_deserialization(agentframework_utils_module):
    """Test memory deserialization from JSON."""
    from charge.clients.agentframework_utils import AgentFrameworkMemory

    # Create test data
    test_data = [
        {
            "content": "Message 1",
            "role": "assistant",
            "source_agent": "agent_1"
        },
        {
            "content": "Message 2",
            "role": "assistant",
            "source_agent": "agent_2"
        }
    ]
    json_str = json.dumps(test_data)

    # Load into memory
    memory = AgentFrameworkMemory()
    memory.load_memory_content(json_str)

    messages = memory.get_messages()
    sources = memory.get_source_agents()

    assert len(messages) == 2
    assert len(sources) == 2
    assert messages[0]["content"] == "Message 1"
    assert messages[1]["content"] == "Message 2"
    assert sources[0] == "agent_1"
    assert sources[1] == "agent_2"


@pytest.mark.asyncio
async def test_agentframework_memory_roundtrip(agentframework_utils_module):
    """Test serialization and deserialization roundtrip."""
    from charge.clients.agentframework_utils import AgentFrameworkMemory

    # Create memory with content
    memory1 = AgentFrameworkMemory(name="original")
    await memory1.add("First message", "agent_1")
    await memory1.add("Second message", "agent_2")
    await memory1.add("Third message", "agent_3")

    # Serialize
    serialized = memory1.serialize_memory_content()

    # Deserialize into new memory
    memory2 = AgentFrameworkMemory(name="copy")
    memory2.load_memory_content(serialized)

    # Verify contents match
    messages1 = memory1.get_messages()
    messages2 = memory2.get_messages()
    sources1 = memory1.get_source_agents()
    sources2 = memory2.get_source_agents()

    assert len(messages1) == len(messages2)
    assert len(sources1) == len(sources2)

    for i in range(len(messages1)):
        assert messages1[i]["content"] == messages2[i]["content"]
        assert messages1[i]["role"] == messages2[i]["role"]
        assert sources1[i] == sources2[i]


@pytest.mark.asyncio
async def test_agentframework_memory_default_source_agent(agentframework_utils_module):
    """Test that default source agent is used when not specified."""
    from charge.clients.agentframework_utils import AgentFrameworkMemory

    memory = AgentFrameworkMemory()
    await memory.add("Test message")  # No source agent specified

    sources = memory.get_source_agents()
    assert len(sources) == 1
    assert sources[0] == "Agent"  # Default value


@pytest.mark.asyncio
async def test_agentframework_memory_empty_serialization(agentframework_utils_module):
    """Test serialization of empty memory."""
    from charge.clients.agentframework_utils import AgentFrameworkMemory

    memory = AgentFrameworkMemory()
    serialized = memory.serialize_memory_content()

    parsed = json.loads(serialized)
    assert isinstance(parsed, list)
    assert len(parsed) == 0


@pytest.mark.asyncio
async def test_mcp_workbench_adapter_initialization(agentframework_utils_module):
    """Test MCPWorkbenchAdapter initialization."""
    from charge.clients.agentframework_utils import MCPWorkbenchAdapter

    adapter = MCPWorkbenchAdapter(
        stdio_servers=["server1.py", "server2.py"],
        sse_servers=["http://localhost:8000/sse"]
    )

    assert len(adapter.stdio_servers) == 2
    assert len(adapter.sse_servers) == 1
    assert len(adapter.get_tools()) == 0  # No tools created yet


@pytest.mark.asyncio
async def test_mcp_workbench_adapter_empty(agentframework_utils_module):
    """Test MCPWorkbenchAdapter with no servers."""
    from charge.clients.agentframework_utils import MCPWorkbenchAdapter

    adapter = MCPWorkbenchAdapter()

    assert len(adapter.stdio_servers) == 0
    assert len(adapter.sse_servers) == 0

    tools = await adapter.create_tools()
    assert len(tools) == 0


@pytest.mark.asyncio
async def test_setup_mcp_tools_function(agentframework_utils_module):
    """Test setup_mcp_tools helper function."""
    from charge.clients.agentframework_utils import setup_mcp_tools

    # Test with empty servers
    tools = await setup_mcp_tools(stdio_servers=[], sse_servers=[])
    assert isinstance(tools, list)
    assert len(tools) == 0


def test_connection_error_exception(agentframework_utils_module):
    """Test chargeConnectionError exception."""
    from charge.clients.agentframework_utils import chargeConnectionError

    error = chargeConnectionError("Test error message")
    assert str(error) == "Test error message"
    assert isinstance(error, Exception)


def test_possible_connection_errors_tuple(agentframework_utils_module):
    """Test POSSIBLE_CONNECTION_ERRORS is a tuple."""
    from charge.clients.agentframework_utils import POSSIBLE_CONNECTION_ERRORS

    assert isinstance(POSSIBLE_CONNECTION_ERRORS, tuple)
    assert len(POSSIBLE_CONNECTION_ERRORS) > 0
    assert ConnectionError in POSSIBLE_CONNECTION_ERRORS
