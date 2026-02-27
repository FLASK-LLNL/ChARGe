import pytest


@pytest.fixture
def autogen_chARGeListMemory_module():
    from charge.clients.autogen_utils import ChARGeListMemory

    return ChARGeListMemory


@pytest.mark.asyncio
async def test_chARGeListMemory_serialization(autogen_chARGeListMemory_module):
    ChARGeListMemory = autogen_chARGeListMemory_module

    # Create an instance of ChARGeListMemory and add some content
    memory = ChARGeListMemory()
    from autogen_core.memory import MemoryContent, MemoryMimeType
    from autogen_core.model_context import UnboundedChatCompletionContext

    content1 = MemoryContent(content="Test content 1", mime_type=MemoryMimeType.TEXT)
    content2 = MemoryContent(content="Test content 2", mime_type=MemoryMimeType.TEXT)

    await memory.add(content1, "agent_1")
    await memory.add(content2, "agent_2")

    old_model_context = UnboundedChatCompletionContext()
    await memory.update_context(old_model_context)

    messages = await old_model_context.get_messages()
    assert len(messages) == 2
    assert (
        messages[0].content
        == "\nRelevant memory content (in chronological order):\n"
        + "[From Agent: agent_1] Test content 1"
    )
    assert messages[1].content == "\n[From Agent: agent_2] Test content 2"

    # Serialize the memory content to JSON
    serialized_content = memory.serialize_memory_content()
    print("Serialized memory content:")
    print(serialized_content)

    # Create a new instance and load the serialized content
    new_memory = ChARGeListMemory()
    new_memory.load_memory_content(serialized_content)

    # Verify that the loaded content matches the original
    assert len(new_memory.content) == 2
    assert new_memory.content[0].content == "Test content 1"
    assert new_memory.content[1].content == "Test content 2"
    assert new_memory.source_agent[0] == "agent_1"
    assert new_memory.source_agent[1] == "agent_2"

    new_model_context = UnboundedChatCompletionContext()
    await new_memory.update_context(new_model_context)

    messages = await new_model_context.get_messages()
    assert len(messages) == 2
    assert (
        messages[0].content
        == "\nRelevant memory content (in chronological order):\n"
        + "[From Agent: agent_1] Test content 1"
    )
    assert messages[1].content == "\n[From Agent: agent_2] Test content 2"


@pytest.mark.asyncio
async def test_AutoGenAgent_serialization(autogen_chARGeListMemory_module):
    from charge.clients.autogen import AutoGenBackend, AutoGenAgent
