################################################################################
## Copyright 2025 Lawrence Livermore National Security, LLC. and Binghamton University.
## See the top-level LICENSE file for details.
##
## SPDX-License-Identifier: Apache-2.0
################################################################################
try:

    from autogen_ext.tools.mcp import (
        StdioServerParams,
        McpWorkbench,
        StreamableHttpServerParams,
    )
except ImportError:
    raise ImportError(
        "Please install the autogen-agentchat package to use this module."
    )
import asyncio
import os
from loguru import logger
from typing import Any, Type, Optional, List
import json
import httpx
from contextlib import asynccontextmanager

try:
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client
    from mcp.client.streamable_http import streamable_http_client
except ImportError:
    ClientSession = None
    StdioServerParameters = None
    stdio_client = None
    streamable_http_client = None


class ToolResult:
    """
    Wrapper for MCP tool results to maintain backward compatibility.

    The AutoGen McpWorkbench returned objects with a .content attribute.
    This wrapper provides the same interface for the native MCP client.
    """

    def __init__(self, content):
        self.content = content

    def __repr__(self):
        return f"ToolResult(content={self.content!r})"


def _extract_content_text(content_item):
    """
    Extract text from MCP content items and wrap in ToolResult.

    MCP returns content as typed objects (TextContent, ImageContent, etc.).
    This helper extracts the actual text/data and wraps it for backward compatibility.

    Args:
        content_item: MCP content object (TextContent, ImageContent, etc.)

    Returns:
        ToolResult with extracted text as the .content attribute
    """
    # Handle TextContent objects (have .text attribute)
    if hasattr(content_item, "text"):
        return ToolResult(content_item.text)

    # Handle objects with .content attribute
    if hasattr(content_item, "content"):
        return ToolResult(content_item.content)

    # Handle dict-like objects
    if isinstance(content_item, dict):
        if "text" in content_item:
            return ToolResult(content_item["text"])
        if "content" in content_item:
            return ToolResult(content_item["content"])

    # Return wrapped as-is if we can't extract anything
    logger.warning(
        f"Could not extract text from content item type: {type(content_item)}, returning as-is"
    )
    return ToolResult(content_item)


def _create_streaming_bearer_token_header(
    bearer_token: Optional[str] = None,
) -> dict[str, str]:
    """
    Helper to create a header with a bearer token for an MCP session for HTTP/SSE servers.

    Args:
        bearer_token: Optional bearer token for authentication

    Returns:
        Dictionary with header fields
    """
    # Prepare headers with content negotiation and authentication
    headers = {
        # Streamable/streaming HTTP MCP endpoints commonly require this for content negotiation
        "Content-Type": "application/json",
        "Accept": "text/event-stream, application/json",
        "Cache-Control": "no-cache",
    }
    # Add bearer token if provided
    if bearer_token:
        headers["X-Token"] = bearer_token

    return headers


@asynccontextmanager
async def _get_http_mcp_session(url: str, bearer_token: Optional[str] = None):
    """
    Helper to create and manage an MCP session for HTTP/SSE servers.

    Args:
        url: MCP server URL (will be normalized to end with /mcp)
        bearer_token: Optional bearer token for authentication

    Yields:
        ClientSession: Initialized MCP client session

    Raises:
        ImportError: If MCP client library is not available
    """
    if ClientSession is None or streamable_http_client is None:
        raise ImportError(
            "MCP client library not available. Install with: pip install mcp"
        )

    # Normalize URL - ensure it ends with /mcp
    url_normalized = url.rstrip("/")
    if not url_normalized.endswith("/mcp"):
        url_normalized = f"{url_normalized}/mcp"

    # Prepare headers with content negotiation and authentication
    headers = _create_streaming_bearer_token_header(bearer_token)

    # Create httpx client with custom headers
    http_client = httpx.AsyncClient(headers=headers, timeout=1800.0)

    try:
        async with streamable_http_client(url_normalized, http_client=http_client) as (
            read,
            write,
            get_session_id,
        ):
            async with ClientSession(read, write) as session:
                await session.initialize()
                yield session
    finally:
        await http_client.aclose()


@asynccontextmanager
async def _get_stdio_mcp_session(path: str):
    """
    Helper to create and manage an MCP session for stdio servers.

    Args:
        path: Path to stdio server (can be command with args or just a python script path)

    Yields:
        ClientSession: Initialized MCP client session

    Raises:
        ImportError: If MCP client library is not available
    """
    if ClientSession is None or stdio_client is None:
        raise ImportError(
            "MCP client library not available. Install with: pip install mcp"
        )

    # Parse command and args from path
    parts = path.split()
    command = parts[0] if parts else "python3"
    args = parts[1:] if len(parts) > 1 else [path]

    # If path looks like a single file path, use python3 to run it
    if len(parts) == 1:
        command = "python3"
        args = [path]

    server_params = StdioServerParameters(command=command, args=args, env=None)

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            yield session


def create_servers(
    paths: List[str],
    urls: List[str],
    bearer_token: Optional[str] = None,
    timeout: Optional[int] = 60,
) -> List[Any]:
    """
    DEPRECATED: Creates AutoGen MCP server parameters from the task's server paths.

    This function creates server params for AutoGen's McpWorkbench which is deprecated.
    Only kept for backwards compatibility with autogen.py client.

    Returns:
        List[Any]: List of AutoGen MCP server parameters (StdioServerParams, StreamableHttpServerParams).
    """
    mcp_servers = []
    for path in paths:
        mcp_servers.append(
            StdioServerParams(
                command="python3",
                args=[path],
                read_timeout_seconds=timeout,
            )
        )
    headers = _create_streaming_bearer_token_header(bearer_token)

    for url in urls:
        mcp_servers.append(
            StreamableHttpServerParams(
                url=url,
                headers=headers,
                timeout=timeout,
                sse_read_timeout=timeout,
            )
        )
    return mcp_servers


# BVE FIXME so that it is not autogen specific
async def _setup_mcp_workbenches(
    paths: List[str], urls: List[str], bearer_token: Optional[str] = None
) -> List[McpWorkbench]:
    """
    DEPRECATED: Sets up AutoGen MCP workbenches from the task's server paths.

    This function uses AutoGen's McpWorkbench which is deprecated.
    Use the direct MCP client functions instead (call_mcp_tool_directly, list_mcp_tools_direct).

    Only kept for backwards compatibility with autogen.py client.

    Returns:
        List of McpWorkbench instances
    """
    mcps = create_servers(paths, urls, bearer_token)

    if len(mcps) == 0:
        return []
    workbenches = [McpWorkbench(server) for server in mcps]

    await asyncio.gather(*[workbench.start() for workbench in workbenches])
    return workbenches


async def _close_mcp_workbenches(workbenches: List[McpWorkbench]) -> None:
    """
    DEPRECATED: Closes AutoGen MCP workbenches.

    This function is deprecated along with _setup_mcp_workbenches.
    Only kept for backwards compatibility with autogen.py client.

    Returns:
        None
    """
    if not workbenches:
        return
    await asyncio.gather(*[workbench.stop() for workbench in workbenches])


async def _list_tools_on_session(session: ClientSession, server_id: str) -> list[dict]:
    """
    Helper to list all tools from an established MCP session.

    This encapsulates the common logic for both HTTP and stdio servers:
    - List available tools
    - Extract tool information (name, description, inputSchema)
    - Log the results

    Args:
        session: Initialized MCP ClientSession
        server_id: Server identifier for logging

    Returns:
        List of tool dictionaries with name, description, and inputSchema
    """
    logger.trace(f"Initialized MCP session for {server_id}")

    # List tools
    tools_result = await session.list_tools()

    if hasattr(tools_result, "tools"):
        tools = [
            {
                "name": tool.name,
                "description": (
                    tool.description if hasattr(tool, "description") else ""
                ),
                "inputSchema": (
                    tool.inputSchema if hasattr(tool, "inputSchema") else {}
                ),
            }
            for tool in tools_result.tools
        ]
    else:
        tools = []

    logger.trace(f"Found {len(tools)} tools on {server_id}:")
    for tool in tools:
        logger.trace(f"  - {tool['name']}: {tool.get('description', 'no description')}")

    return tools


async def _call_tool_on_session(
    session: ClientSession, tool_name: str, arguments: dict, server_id: str
):
    """
    Helper to call a tool on an established MCP session.

    This encapsulates the common logic for both HTTP and stdio servers:
    - List available tools
    - Check if the requested tool exists
    - Call the tool if found
    - Extract and return the result content

    Args:
        session: Initialized MCP ClientSession
        tool_name: Name of the tool to call
        arguments: Dictionary of arguments to pass to the tool
        server_id: Server identifier for logging

    Returns:
        ToolResult object (single result), list of ToolResult objects (multiple),
        or None if tool not found or no results

    Raises:
        Returns None if tool not found (doesn't raise, allowing iteration to continue)
    """
    # List tools to check if our tool exists
    tools_result = await session.list_tools()
    available_tools = (
        {tool.name for tool in tools_result.tools}
        if hasattr(tools_result, "tools")
        else set()
    )

    if tool_name not in available_tools:
        # Tool not found on this server, return None to continue searching
        return None

    logger.trace(f"Found tool {tool_name} on server {server_id}, calling it")

    # Call the tool
    result = await session.call_tool(name=tool_name, arguments=arguments)

    # Process results - extract text from content objects
    if hasattr(result, "content") and result.content:
        num_results = len(result.content)
        if num_results == 1:
            # Return extracted text from single content item
            return _extract_content_text(result.content[0])
        elif num_results > 1:
            # Return list of extracted texts
            return [_extract_content_text(item) for item in result.content]
        else:
            logger.warning(f"Tool {tool_name} returned empty content: {result}")
            return None
    else:
        logger.warning(f"Tool {tool_name} did not return valid content: {result}")
        return None


async def call_mcp_tool_directly(
    tool_name: str,
    arguments: dict,
    urls: list[str] = [],
    paths: list[str] = [],
    bearer_token: Optional[str] = None,
):
    """
    Call a tool directly from available MCP server URLs and paths using the native MCP client.

    Args:
        tool_name: Name of the tool to call
        arguments: Dictionary of arguments to pass to the tool
        urls: List of MCP server URLs (HTTP/SSE endpoints)
        paths: List of MCP server stdio paths (python scripts)
        bearer_token: Optional bearer token for authentication

    Returns:
        ToolResult object with .content attribute containing the result text (single result),
        list of ToolResult objects (multiple results), or None (no results)

        The ToolResult wrapper maintains backward compatibility with AutoGen's McpWorkbench interface.

    Raises:
        ValueError: If tool is not found in any server
        ImportError: If MCP client library is not available
    """
    # Try HTTP/SSE servers first
    for url in urls:
        try:
            logger.trace(f"Trying to call tool {tool_name} from MCP server: {url}")

            async with _get_http_mcp_session(url, bearer_token) as session:
                result = await _call_tool_on_session(session, tool_name, arguments, url)
                if result is not None:
                    return result

        except Exception as e:
            logger.trace(f"Could not call tool {tool_name} from {url}: {e}")
            continue

    # Try stdio servers
    for path in paths:
        try:
            logger.trace(f"Trying to call tool {tool_name} from stdio server: {path}")

            async with _get_stdio_mcp_session(path) as session:
                result = await _call_tool_on_session(
                    session, tool_name, arguments, path
                )
                if result is not None:
                    return result

        except Exception as e:
            logger.trace(f"Could not call tool {tool_name} from stdio {path}: {e}")
            continue

    # Tool not found in any server
    raise ValueError(
        f"Tool '{tool_name}' not found in any of the provided servers. "
        f"URLs: {urls}, Paths: {paths}"
    )


async def list_mcp_tools_direct(
    urls: list[str] = [], paths: list[str] = [], bearer_token: Optional[str] = None
) -> dict:
    """
    List all tools available from MCP servers using the native MCP client library.

    This function uses the official MCP Python client library to connect to servers
    and retrieve available tools.

    Args:
        urls: List of MCP server URLs (HTTP/SSE endpoints)
        paths: List of MCP server stdio paths (python scripts)
        bearer_token: Optional bearer token for authentication

    Returns:
        Dictionary mapping server identifier to list of tool information:
        {
            "server_url_or_path": [
                {"name": "tool_name", "description": "...", "inputSchema": {...}},
                ...
            ]
        }
    """
    all_tools = {}

    # Handle streamable HTTP MCP servers
    for url in urls:
        server_id = url
        try:
            logger.trace(f"Connecting to MCP server via streamable HTTP: {server_id}")

            async with _get_http_mcp_session(url, bearer_token) as session:
                all_tools[server_id] = await _list_tools_on_session(session, server_id)

        except Exception as e:
            logger.warning(f"Error connecting to MCP server {server_id}: {e}")
            import traceback

            logger.warning(traceback.format_exc())
            all_tools[server_id] = {"error": str(e)}

    # Handle stdio MCP servers
    for path in paths:
        server_id = path
        try:
            logger.trace(f"Connecting to MCP server via stdio: {server_id}")

            async with _get_stdio_mcp_session(path) as session:
                all_tools[server_id] = await _list_tools_on_session(session, server_id)

        except Exception as e:
            logger.trace(f"Error connecting to stdio MCP server {server_id}: {e}")
            import traceback

            logger.trace(traceback.format_exc())
            all_tools[server_id] = {"error": str(e)}

    return all_tools
