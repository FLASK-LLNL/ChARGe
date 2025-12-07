################################################################################
## Copyright 2025 Lawrence Livermore National Security, LLC. and Binghamton University.
## See the top-level LICENSE file for details.
##
## SPDX-License-Identifier: Apache-2.0
################################################################################
try:

    from autogen_ext.tools.mcp import StdioServerParams, McpWorkbench, SseServerParams
except ImportError:
    raise ImportError(
        "Please install the autogen-agentchat package to use this module."
    )
import asyncio
from typing import Any, Type, Optional, List

def create_servers(paths: List[str], urls: List[str], timeout: Optional[int] = 60) -> List[Any]:
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
    for url in urls:
        mcp_servers.append(
            SseServerParams(
                url=url,
                timeout=timeout,
                sse_read_timeout=timeout,
            )
        )
    return mcp_servers

async def _setup_mcp_workbenches(paths: List[str], urls: List[str]) -> List[McpWorkbench]:
    """
    Sets up MCP workbenches from the task's server paths.

    Returns:
        None
    """
    mcps = create_servers(paths,urls)

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

async def call_mcp_tool_directly(tool_name: str, arguments: dict, urls: list[str] = [], paths: list[str] = []):
    """Call a tool directly from an available server URLs and paths."""
    workbenches = await _setup_mcp_workbenches(paths, urls)

    try:
        # Find the right workbench (if you have multiple)
        unused_tools = []
        for workbench in workbenches:
            tools = await workbench.list_tools()
            tool_names = [t["name"] for t in tools]

            if tool_name in tool_names:
                result = await workbench.call_tool(
                    name=tool_name,
                    arguments=arguments
                )
                return result
            else:
                unused_tools.append(tool_names)

        raise ValueError(f"Tool '{tool_name}' not found in any workbench: {unused_tools}")

    finally:
        await _close_mcp_workbenches(workbenches)
