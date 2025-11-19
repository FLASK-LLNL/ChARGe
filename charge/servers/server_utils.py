################################################################################
## Copyright 2025 Lawrence Livermore National Security, LLC. and Binghamton University.
## See the top-level LICENSE file for details.
##
## SPDX-License-Identifier: Apache-2.0
################################################################################

import argparse
from mcp.server.fastmcp import FastMCP


def add_server_arguments(parser: argparse.ArgumentParser) -> None:
    """
    Add standard server arguments to an argparse parser.

    Args:
        parser (argparse.ArgumentParser): The parser to add arguments to.
    
    Returns:
        None
    """
    parser.add_argument(
        "--port", type=int, default=8000, help="Port to run the server on"
    )
    # It seems that for FastMCP if the host is not set or set to None, it defaults to
    # localhost
    parser.add_argument(
        "--host", type=str, default=None, help="Host to run the server on"
    )
    parser.add_argument(
        '--transport', type=str,
        help='MCP transport type',
        choices=['stdio', 'streamable-http', 'sse'],
        default='sse'
    )


def update_mcp_network(mcp: FastMCP, host: str, port: str) -> None:
    """
    Update the MCP network settings.

    Args:
        mcp (FastMCP): The MCP server to update.
        host (str): The host to run the server on.
        port (str): The port to run the server on.
    
    Returns:
        None
    """
    mcp.settings.host = host
    mcp.settings.port = port

def get_hostname() -> Tuple[str, str]:
    """
    Get the hostname and IP address of the host.

    Returns:
        Tuple[str, str]: The hostname and IP address of the host.
    """
    import socket
    hostname = socket.gethostname()
    try:
        host = socket.gethostbyname(hostname)
    except socket.gaierror as e:
        host = "127.0.0.1"
    return hostname, host

def try_get_public_hostname() -> Tuple[str, str]:
    """
    Try to get the public hostname and IP address of the host.

    Returns:
        Tuple[str, str]: The public hostname and IP address of the host.
    """
    import socket
    hostname = socket.gethostname()
    try:
        public_hostname = hostname + "-pub"
        host = socket.gethostbyname(public_hostname)
        hostname = public_hostname
    except socket.gaierror as e:
        try:
            host = socket.gethostbyname(hostname)
        except socket.gaierror as e:
            host = "127.0.0.1"

    return hostname, host
