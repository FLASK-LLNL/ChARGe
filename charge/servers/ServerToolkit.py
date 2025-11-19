################################################################################
## Copyright 2025 Lawrence Livermore National Security, LLC. and Binghamton University.
## See the top-level LICENSE file for details.
##
## SPDX-License-Identifier: Apache-2.0
################################################################################

from mcp.server.fastmcp import FastMCP
from functools import wraps
from typing import Callable, Literal


class ServerToolkit:
    """
    A class that provides a toolkit for registering methods as MCP tools.
    """

    def __init__(self, mcp: FastMCP):
        self._mcp = mcp

    def _register_methods(self):
        """
        Register all methods marked with the @mcp_tool decorator.
        """
        for name in dir(self):
            # Skip private/magic methods
            if name.startswith("_"):
                continue

            attr = getattr(self, name)

            # Check if this is a method marked for registration
            if callable(attr) and hasattr(attr, "_is_mcp_tool"):
                self._register_single_method(attr)

    def _register_single_method(self, method: Callable):
        """
        Register a single method as an MCP tool.

        Args:
            method (Callable): The method to register as an MCP tool.

        Returns:
            None
        """

        # Create a closure that captures the bound method
        @wraps(method)
        def tool_wrapper(*args, **kwargs):
            return method(*args, **kwargs)

        self._mcp.tool()(tool_wrapper)

    @staticmethod
    def mcp_tool(func: Callable) -> Callable:
        """
        Decorator to mark methods for MCP registration.

        Args:
            func (Callable): The function to mark for MCP registration.

        Returns:
            Callable: The marked function.
        """
        func._is_mcp_tool = True
        return func

    @staticmethod
    def register_function_as_tool(mcp: FastMCP, func: Callable) -> None:
        """
        Register an external function as an MCP tool.

        Args:
            mcp (FastMCP): The MCP instance to register the function with.
            func (Callable): The function to register as an MCP tool.

        Returns:
            None
        """
        mcp.tool()(func)

    def register_function_to_server(func: Callable) -> None:
        """
        Register an external function to the server.

        Args:
            func (Callable): The function to register to the server.

        Returns:
            None
        """
        self._mcp.tool()(func)

    def return_mcp(self) -> FastMCP:
        """
        Return the MCP instance.

        Returns:
            FastMCP: The MCP instance.
        """
        self._register_methods()
        return self._mcp

    def run(self, transport: Literal["sse", "stdio"] = "sse") -> None:
        """
        Run the MCP server.

        Args:
            transport (Literal["sse", "stdio"], optional): The transport to use. Defaults to "sse".
        """
        self._register_methods()
        self._mcp.run(transport=transport)

    def update_mcp(self, mcp: FastMCP) -> None:
        """
        Update the MCP instance.

        Args:
            mcp (FastMCP): The new MCP instance.
        """
        self._mcp = mcp
        self._register_methods()


class MultiServerToolkit(ServerToolkit):
    """
    A class to combine multiple servers into a single object that
    can be run as a single MCP server.
    """

    def __init__(
        self,
        servers: list[ServerToolkit],
        description: str,
        host: str,
        port: int,
    ):
        """
        Initialize the MultiServerToolkit.

        Args:
            servers (list[ServerToolkit]): The list of servers to register.
            description (str): The description of the server.
            host (str): The host of the server.
            port (int): The port of the server.
        """
        mcp = FastMCP(description, host=host, port=port)
        super().__init__(mcp)

        self.servers = servers
        # Register all tools from all servers
        for server in self.servers:
            for name in dir(server):
                if name.startswith("_"):
                    continue

                attr = getattr(server, name)
                if callable(attr) and hasattr(attr, "_is_mcp_tool"):
                    self.mcp.tool()(attr)
