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

from typing import List, Optional, Any, Dict
from loguru import logger
import httpx


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
def _normalize_server_url(url: str) -> str:
    trimmed = url.rstrip("/")
    if trimmed.endswith("/mcp"):
        return trimmed
    return f"{trimmed}/mcp"


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
        mcp_server_allowed_tools: Optional[Dict[str, List[str]]] = None,
    ):
        """
        Initialize the MCP adapter.

        Args:
            stdio_servers: List of STDIO server command paths.
            mcp_servers: List of MCP server URLs.
            mcp_server_allowed_tools: Optional mapping from MCP server URL to
                the subset of tool names that should be exposed for that server.
        """
        self.stdio_servers = stdio_servers or []
        self.mcp_servers = mcp_servers or []
        self.mcp_server_allowed_tools = {
            _normalize_server_url(server_url): list(
                dict.fromkeys(tool_name for tool_name in tool_names if tool_name)
            )
            for server_url, tool_names in (mcp_server_allowed_tools or {}).items()
            if tool_names is not None
        }
        self._tools: List[Any] = []

    async def create_tools(self, bearer_token: str) -> List[Any]:
        #    async def create_tools(self, bearer_token: Optional[str] = None) -> List[Any]:
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
        try:
            for url in self.mcp_servers:
                try:
                    allowed_tools = self.mcp_server_allowed_tools.get(
                        _normalize_server_url(url)
                    )
                    if allowed_tools == []:
                        logger.info(
                            f"Skipping MCP tool for {url} because its allowlist is empty"
                        )
                        continue

                    # Set up headers for MCP streamable HTTP - include content negotiation headers
                    headers = {
                        "Content-Type": "application/json",
                        "Accept": "text/event-stream, application/json",
                        "Cache-Control": "no-cache",
                    }

                    # Add bearer token if provided
                    if bearer_token:
                        headers["X-Token"] = bearer_token

                    # MCPStreamableHTTPTool requires an http_client with custom headers, not a headers parameter
                    http_client = httpx.AsyncClient(headers=headers, timeout=30.0)

                    mcp_tool = MCPStreamableHTTPTool(
                        name=f"mcp_http_{url.split('/')[-1]}",
                        url=url,
                        allowed_tools=allowed_tools or None,
                        http_client=http_client,
                    )
                    tools.append(mcp_tool)
                    if allowed_tools:
                        logger.info(
                            f"Created MCP tool: {url} with allowlist {allowed_tools}"
                        )
                    else:
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
    bearer_token: str,
    stdio_servers: Optional[List[str]] = None,
    mcp_servers: Optional[List[str]] = None,
    mcp_server_allowed_tools: Optional[Dict[str, List[str]]] = None,
    #    bearer_token: Optional[str] = None
) -> List[Any]:
    """
    Setup MCP tools for Agent Framework from server configurations.

    Args:
        stdio_servers: List of STDIO server paths.
        mcp_servers: List of MCP server URLs.
        mcp_server_allowed_tools: Optional mapping from MCP server URL to
            the subset of tool names that should be exposed for that server.
    Returns:
        List of Agent Framework MCP tools.
    """
    adapter = MCPWorkbenchAdapter(
        stdio_servers=stdio_servers,
        mcp_servers=mcp_servers,
        mcp_server_allowed_tools=mcp_server_allowed_tools,
    )
    tools = await adapter.create_tools(bearer_token=bearer_token)
    return tools
