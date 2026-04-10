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

try:
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client
    from mcp.client.streamable_http import streamable_http_client
except ImportError:
    ClientSession = None
    StdioServerParameters = None
    stdio_client = None
    streamable_http_client = None


def create_servers(
    paths: List[str], urls: List[str], timeout: Optional[int] = 60
) -> List[Any]:
    """
    Creates MCP servers from the task's server paths.

    Returns:
        List[Any]: List of MCP server parameters.
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
    headers = {
        # Streamable/streaming HTTP MCP endpoints commonly require this for content negotiation
        "Content-Type": "application/json",
        "Accept": "text/event-stream, application/json",
        "Cache-Control": "no-cache",
    }
    wh_token = os.getenv("FLASK_WORMHOLE_TOKEN", None)
    if wh_token:
        headers["X-Token"] = wh_token
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


async def _setup_mcp_workbenches(
    paths: List[str], urls: List[str]
) -> List[McpWorkbench]:
    """
    Sets up MCP workbenches from the task's server paths.

    Returns:
        None
    """
    mcps = create_servers(paths, urls)

    if len(mcps) == 0:
        return []
    workbenches = [McpWorkbench(server) for server in mcps]

    await asyncio.gather(*[workbench.start() for workbench in workbenches])
    return workbenches


async def _close_mcp_workbenches(workbenches: List[McpWorkbench]) -> None:
    """
    Closes MCP workbenches.

    Returns:
        None
    """
    if not workbenches:
        return
    await asyncio.gather(*[workbench.stop() for workbench in workbenches])


async def call_mcp_tool_directly(
    tool_name: str, arguments: dict, urls: list[str] = [], paths: list[str] = []
):
    """Call a tool directly from an available server URLs and paths."""
    workbenches = await _setup_mcp_workbenches(paths, urls)

    try:
        # Find the right workbench (if you have multiple)
        unused_tools = []
        for workbench in workbenches:
            tools = await workbench.list_tools()
            tool_names = [t["name"] for t in tools]

            if tool_name in tool_names:
                results = await workbench.call_tool(name=tool_name, arguments=arguments)
                num_results = len(results.result)
                if num_results == 1:
                    return results.result[0]
                elif num_results > 1:
                    return results.result
                else:
                    logger.warning(
                        f"{tool_name} did not return a valid results message {results}"
                    )
                    return None
            else:
                unused_tools.append(tool_names)

        raise ValueError(
            f"Tool '{tool_name}' not found in any workbench: {unused_tools}"
        )

    finally:
        await _close_mcp_workbenches(workbenches)


async def list_mcp_tools_direct(urls: list[str] = [], paths: list[str] = []) -> dict:
    """
    List all tools available from MCP servers using the native MCP client library.

    This function uses the official MCP Python client library to connect to servers
    and retrieve available tools.

    Args:
        urls: List of MCP server URLs (HTTP/SSE endpoints)
        paths: List of MCP server stdio paths (python scripts)

    Returns:
        Dictionary mapping server identifier to list of tool information:
        {
            "server_url_or_path": [
                {"name": "tool_name", "description": "...", "inputSchema": {...}},
                ...
            ]
        }
    """
    if ClientSession is None:
        raise ImportError(
            "MCP client library not available. " "Install with: pip install mcp"
        )

    all_tools = {}

    # Handle streamable HTTP MCP servers
    for url in urls:
        server_id = url
        try:
            # Normalize URL - ensure it ends with /mcp
            url_normalized = url.rstrip("/")
            if not url_normalized.endswith("/mcp"):
                url_normalized = f"{url_normalized}/mcp"

            logger.trace(f"Connecting to MCP server via streamable HTTP: {server_id}")

            # Prepare headers
            headers = {
                "Content-Type": "application/json",
                "Accept": "text/event-stream, application/json",
                "Cache-Control": "no-cache",
            }

            # Add wormhole token if available
            wh_token = os.getenv("FLASK_WORMHOLE_TOKEN", None)
            if wh_token:
                headers["X-Token"] = wh_token

            # Create httpx client with custom headers
            http_client = httpx.AsyncClient(headers=headers, timeout=1800.0)

            try:
                # Connect using streamable HTTP client
                # streamable_http_client returns (read, write, get_session_id)
                async with streamable_http_client(
                    url_normalized, http_client=http_client
                ) as (read, write, get_session_id):
                    async with ClientSession(read, write) as session:
                        # Initialize the session
                        await session.initialize()
                        logger.trace(f"Initialized MCP session for {server_id}")

                        # List tools
                        tools_result = await session.list_tools()

                        if hasattr(tools_result, "tools"):
                            tools = [
                                {
                                    "name": tool.name,
                                    "description": (
                                        tool.description
                                        if hasattr(tool, "description")
                                        else ""
                                    ),
                                    "inputSchema": (
                                        tool.inputSchema
                                        if hasattr(tool, "inputSchema")
                                        else {}
                                    ),
                                }
                                for tool in tools_result.tools
                            ]
                        else:
                            tools = []

                        all_tools[server_id] = tools
                        logger.trace(f"Found {len(tools)} tools on {server_id}:")
                        for tool in tools:
                            logger.trace(
                                f"  - {tool['name']}: {tool.get('description', 'no description')}"
                            )
            finally:
                await http_client.aclose()

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

            # Parse command and args
            parts = path.split()
            command = parts[0] if parts else "python3"
            args = parts[1:] if len(parts) > 1 else [path]

            # If path looks like a single file path, use python3 to run it
            if len(parts) == 1:
                command = "python3"
                args = [path]

            server_params = StdioServerParameters(command=command, args=args, env=None)

            # Connect using stdio client
            async with stdio_client(server_params) as (read, write):
                async with ClientSession(read, write) as session:
                    # Initialize the session
                    await session.initialize()
                    logger.trace(f"Initialized MCP session for {server_id}")

                    # List tools
                    tools_result = await session.list_tools()

                    if hasattr(tools_result, "tools"):
                        tools = [
                            {
                                "name": tool.name,
                                "description": (
                                    tool.description
                                    if hasattr(tool, "description")
                                    else ""
                                ),
                                "inputSchema": (
                                    tool.inputSchema
                                    if hasattr(tool, "inputSchema")
                                    else {}
                                ),
                            }
                            for tool in tools_result.tools
                        ]
                    else:
                        tools = []

                    all_tools[server_id] = tools
                    logger.trace(f"Found {len(tools)} tools on {server_id}:")
                    for tool in tools:
                        logger.trace(
                            f"  - {tool['name']}: {tool.get('description', 'no description')}"
                        )

        except Exception as e:
            logger.trace(f"Error connecting to stdio MCP server {server_id}: {e}")
            import traceback

            logger.trace(traceback.format_exc())
            all_tools[server_id] = {"error": str(e)}

    return all_tools
