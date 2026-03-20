################################################################################
## Copyright 2025 Lawrence Livermore National Security, LLC. and Binghamton University.
## See the top-level LICENSE file for details.
##
## SPDX-License-Identifier: Apache-2.0
################################################################################
try:
    from agent_framework import MCPStdioTool, MCPStreamableHTTPTool
except ImportError:
    raise ImportError("Please install the agent-framework package to use this module.")

from typing import List, Optional, Any
from loguru import logger


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


# MCP Integration
class MCPWorkbenchAdapter:
    """
    Adapter to convert AutoGen MCP workbenches to Agent Framework MCP tools.

    Agent Framework uses MCPStdioTool, MCPStreamableHTTPTool, and MCPWebsocketTool
    for MCP integration, which differs from AutoGen's McpWorkbench approach.
    """

    def __init__(
        self,
        stdio_servers: Optional[List[str]] = None,
        mcp_servers: Optional[List[str]] = None,
    ):
        """
        Initialize the MCP adapter.

        Args:
            stdio_servers: List of STDIO server command paths.
            mcp_servers: List of MCP server URLs.
        """
        self.stdio_servers = stdio_servers or []
        self.mcp_servers = mcp_servers or []
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

        # Create MCP tools
        # Note: Agent Framework uses MCPStreamableHTTPTool for MCP
        # Import here to avoid issues if not available
        try:
            for url in self.mcp_servers:
                try:
                    mcp_tool = MCPStreamableHTTPTool(
                        name=f"mcp_http_{url.split('/')[-1]}",
                        url=url,
                    )
                    tools.append(mcp_tool)
                    logger.info(f"Created MCP tool: {url}")
                except Exception as e:
                    logger.error(f"Failed to create MCP tool for {url}: {e}")
        except ImportError:
            logger.warning(
                "MCPStreamableHTTPTool not available in this Agent Framework version"
            )

        self._tools = tools
        return tools

    def get_tools(self) -> List[Any]:
        """Get the list of created MCP tools."""
        return self._tools


async def setup_mcp_tools(
    stdio_servers: Optional[List[str]] = None,
    mcp_servers: Optional[List[str]] = None,
) -> List[Any]:
    """
    Setup MCP tools for Agent Framework from server configurations.

    Args:
        stdio_servers: List of STDIO server paths.
        mcp_servers: List of MCP server URLs.

    Returns:
        List of Agent Framework MCP tools.
    """
    adapter = MCPWorkbenchAdapter(stdio_servers=stdio_servers, mcp_servers=mcp_servers)
    tools = await adapter.create_tools()
    return tools
