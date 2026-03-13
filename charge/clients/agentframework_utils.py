################################################################################
## Copyright 2025 Lawrence Livermore National Security, LLC. and Binghamton University.
## See the top-level LICENSE file for details.
##
## SPDX-License-Identifier: Apache-2.0
################################################################################
try:
    from agent_framework import Agent, AgentSession, tool, MCPStdioTool
    from agent_framework.openai import OpenAIChatClient
except ImportError:
    raise ImportError(
        "Please install the agent-framework package to use this module."
    )

from typing import List, Optional, Union, Callable, Any
from loguru import logger
from pydantic import BaseModel
import json


# Error handling
_POSSIBLE_CONNECTION_ERRORS: List[type[Exception]] = [ConnectionError]

try:
    from openai._exceptions import (
        APIConnectionError,
        AuthenticationError,
        NotFoundError,
    )

    _POSSIBLE_CONNECTION_ERRORS += [
        APIConnectionError,
        AuthenticationError,
        NotFoundError,
    ]
except ImportError:
    pass

POSSIBLE_CONNECTION_ERRORS = tuple(_POSSIBLE_CONNECTION_ERRORS)


class chargeConnectionError(Exception):
    """Custom exception for connection errors in ChARGe with Agent Framework."""

    pass


class AgentFrameworkMemory:
    """
    Memory class for Agent Framework that provides serialization for multi-task experiments.

    Agent Framework uses AgentSession for conversation state. This class wraps session
    management and provides serialization capabilities compatible with ChARGe experiments.
    """

    def __init__(
        self,
        name: Optional[str] = None,
        session: Optional[AgentSession] = None,
    ):
        self.name = name or "AgentFrameworkMemory"
        self._session = session
        self._stored_messages: List[dict] = []
        self._source_agents: List[str] = []

    def set_session(self, session: AgentSession) -> None:
        """Set the agent session for this memory instance."""
        self._session = session

    def get_session(self) -> Optional[AgentSession]:
        """Get the current agent session."""
        return self._session

    async def add(
        self,
        content: str,
        source_agent: Optional[str] = None,
    ) -> None:
        """
        Add a memory content to the list.

        Args:
            content: The content to add.
            source_agent: The source agent of the content.
        """
        self._stored_messages.append({"content": content, "role": "assistant"})
        self._source_agents.append(source_agent if source_agent is not None else "Agent")

    def serialize_memory_content(self) -> str:
        """
        Serialize the memory contents to a JSON string.

        Returns:
            str: JSON string representation of the memory contents.
        """
        memory_dicts = []
        for msg, source in zip(self._stored_messages, self._source_agents):
            memory_dict = {
                "content": msg.get("content", ""),
                "role": msg.get("role", "assistant"),
                "source_agent": source,
            }
            memory_dicts.append(memory_dict)

        return json.dumps(memory_dicts, indent=2)

    def load_memory_content(self, json_str: str) -> None:
        """
        Load memory contents from a JSON string.

        Args:
            json_str: JSON string representation of the memory contents.
        """
        memory_dicts = json.loads(json_str)

        self._stored_messages = []
        self._source_agents = []

        for memory_dict in memory_dicts:
            source_agent = memory_dict.pop("source_agent", "Agent")
            self._stored_messages.append({
                "content": memory_dict.get("content", ""),
                "role": memory_dict.get("role", "assistant"),
            })
            self._source_agents.append(source_agent)

    def get_messages(self) -> List[dict]:
        """Get all stored messages."""
        return self._stored_messages

    def get_source_agents(self) -> List[str]:
        """Get all source agent names."""
        return self._source_agents


def generate_agent(
    chat_client: OpenAIChatClient,
    agent_name: str,
    instructions: str,
    tools: Optional[List[Any]] = None,
    max_tool_calls: Optional[int] = None,
    **kwargs,
) -> Agent:
    """
    Generate an Agent Framework agent with the given parameters.

    Args:
        chat_client: The chat client to use.
        agent_name: Name of the agent.
        instructions: System instructions for the agent.
        tools: List of tools to attach to the agent.
        max_tool_calls: Maximum number of tool calls (Note: Agent Framework doesn't have this setting).
        **kwargs: Additional keyword arguments.

    Returns:
        Agent: The created Agent Framework agent.
    """
    if max_tool_calls is not None:
        logger.warning(
            "Agent Framework doesn't support max_tool_calls setting. "
            "Agents continue until completion by default with built-in safety mechanisms."
        )

    agent = Agent(
        name=agent_name,
        client=chat_client,  # Fixed: parameter is 'client' not 'chat_client'
        instructions=instructions,
        tools=tools or [],
        **kwargs,
    )

    return agent


# Chat utilities
async def CustomConsole(stream, message_callback: Callable):
    """
    Process stream with callback.

    Args:
        stream: The stream to process.
        message_callback: Callback function for messages.

    Returns:
        The last processed message.
    """
    last_processed = None
    async for message in stream:
        last_processed = await message_callback(message)
    return last_processed


async def cli_chat_callback(message):
    """
    Default CLI chat callback for Agent Framework messages.

    Args:
        message: The message to handle.

    Returns:
        The message content or None.
    """
    # Agent Framework uses different message types than AutoGen
    # This is a placeholder and will need to be adapted based on actual message types
    if hasattr(message, 'text'):
        print(message.text, end="", flush=True)
        return message.text
    elif hasattr(message, 'content'):
        print(message.content, end="", flush=True)
        return message.content
    else:
        print(str(message), end="", flush=True)
        return str(message)


# MCP Integration
class MCPWorkbenchAdapter:
    """
    Adapter to convert AutoGen MCP workbenches to Agent Framework MCP tools.

    Agent Framework uses MCPStdioTool, MCPStreamableHTTPTool, and MCPWebsocketTool
    for MCP integration, which differs from AutoGen's McpWorkbench approach.
    """

    def __init__(self, stdio_servers: Optional[List[str]] = None, sse_servers: Optional[List[str]] = None):
        """
        Initialize the MCP adapter.

        Args:
            stdio_servers: List of STDIO server command paths.
            sse_servers: List of SSE server URLs.
        """
        self.stdio_servers = stdio_servers or []
        self.sse_servers = sse_servers or []
        self._tools: List[Any] = []

    async def create_tools(self) -> List[Any]:
        """
        Create Agent Framework MCP tools from server configurations.

        Returns:
            List of Agent Framework MCP tools.
        """
        tools = []

        # Create STDIO tools
        for server_path in self.stdio_servers:
            # Parse command and args from server path
            # Format: "command arg1 arg2..."
            parts = server_path.split()
            command = parts[0]
            args = parts[1:] if len(parts) > 1 else []

            try:
                mcp_tool = MCPStdioTool(
                    name=f"mcp_{command.split('/')[-1]}",
                    command=command,
                    args=args,
                )
                tools.append(mcp_tool)
                logger.info(f"Created STDIO MCP tool: {command}")
            except Exception as e:
                logger.error(f"Failed to create STDIO MCP tool for {command}: {e}")

        # Create SSE tools
        # Note: Agent Framework uses MCPStreamableHTTPTool for SSE
        # Import here to avoid issues if not available
        try:
            from agent_framework import MCPStreamableHTTPTool

            for url in self.sse_servers:
                try:
                    mcp_tool = MCPStreamableHTTPTool(
                        name=f"mcp_http_{url.split('/')[-1]}",
                        url=url,
                    )
                    tools.append(mcp_tool)
                    logger.info(f"Created SSE MCP tool: {url}")
                except Exception as e:
                    logger.error(f"Failed to create SSE MCP tool for {url}: {e}")
        except ImportError:
            logger.warning("MCPStreamableHTTPTool not available in this Agent Framework version")

        self._tools = tools
        return tools

    def get_tools(self) -> List[Any]:
        """Get the list of created MCP tools."""
        return self._tools


async def setup_mcp_tools(
    stdio_servers: Optional[List[str]] = None,
    sse_servers: Optional[List[str]] = None,
) -> List[Any]:
    """
    Setup MCP tools for Agent Framework from server configurations.

    Args:
        stdio_servers: List of STDIO server paths.
        sse_servers: List of SSE server URLs.

    Returns:
        List of Agent Framework MCP tools.
    """
    adapter = MCPWorkbenchAdapter(stdio_servers=stdio_servers, sse_servers=sse_servers)
    tools = await adapter.create_tools()
    return tools
