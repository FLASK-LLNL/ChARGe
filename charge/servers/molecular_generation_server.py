################################################################################
## Copyright 2025 Lawrence Livermore National Security, LLC. and Binghamton University.
## See the top-level LICENSE file for details.
##
## SPDX-License-Identifier: Apache-2.0
################################################################################

from concurrent.futures import ThreadPoolExecutor
from loguru import logger

try:
    from rdkit import Chem

    HAS_RDKIT = True
except (ImportError, ModuleNotFoundError) as e:
    HAS_RDKIT = False
    logger.warning(
        "Please install the rdkit support packages to use this module."
        "Install it with: pip install charge[rdkit]",
    )

import json
import os
from charge.tasks.Task import Task
from charge.servers.server_utils import add_server_arguments, update_mcp_network
from mcp.server.fastmcp import FastMCP
from charge.clients.autogen import AutoGenPool
from charge.clients.Client import Client
import asyncio
from charge.servers import SMILES_utils
import charge.utils.helper_funcs as hf
import argparse
from typing import Optional
import asyncio

mcp = FastMCP(
    "SMILES Diagnosis and retrieval MCP Server",
)


JSON_FILE_PATH = f"{os.getcwd()}/known_molecules.json"
AGENT_POOL: AutoGenPool | None = None


class DiagnoseSMILESTask(Task):
    def __init__(self, smiles: str, *args, **kwargs):
        system_prompt = (
            "You are a world-class chemist. Your task is to diagnose and evaluate "
            "the quality of the provided SMILES strings. You will be given invalid"
            " SMILES strings, and your task is to identify the issues with them, "
            "correct them if possible. Return concise explanations for each SMILE string."
        )

        user_prompt = (
            f"Diagnose the followig SMILES string {smiles}. Give it a short and concise "
            "explanation of what is wrong with it, and if possible, provide a corrected "
            "version of the SMILES string. If the SMILES string is valid, simply state "
            "'The SMILES string is valid.'"
        )
        super().__init__(
            system_prompt=system_prompt, user_prompt=user_prompt, *args, **kwargs
        )


@mcp.tool()
def diagnose_smiles(smiles: str) -> str:
    """
    Diagnose a SMILES string. Returns a diagnosis of the SMILES string.

    Args:
        smiles (str): The input SMILES string.
    Returns:
        str: The diagnosis of the SMILES string.
    """
    if not HAS_RDKIT:
        raise ImportError(
            "Please install the rdkit support packages to use this module."
        )
    logger.info(f"Diagnosing SMILES string: {smiles}")
    task = DiagnoseSMILESTask(smiles=smiles)

    global AGENT_POOL
    assert (
        AGENT_POOL is not None
    ), "Agent pool is not initialized. Diagnoise Tool not available."

    diagnose_agent = AGENT_POOL.create_agent(task=task)

    async def _run_async_agent():
        return await diagnose_agent.run()

    try:
        # Check if asyncio event loop is already running
        if asyncio.get_event_loop().is_running():

            # What should we do here?
            with ThreadPoolExecutor() as pool:
                future = pool.submit(asyncio.run, _run_async_agent())
                response = future.result()

        else:
            response = asyncio.run(_run_async_agent())
        assert response is not None
        assert len(response.messages) > 0  # type: ignore
        assert response.messages[-1] is not None  # type: ignore

        diagnoses = response.messages[-1].content  # type: ignore
        logger.info(f"Diagnosis: {diagnoses}")
        return f"SMILES diagnoses: {diagnoses}"  # type: ignore

    except Exception as e:
        logger.error(f"An error occurred: {e}")
        return "Error: Unable to process the SMILES string at this time."


@mcp.tool()
def is_already_known(smiles: str) -> bool:
    """
    Check if a SMILES string provided is already known. Only provide
    valid SMILES strings. Returns True if the SMILES string is valid, and
    already in the database, False otherwise.
    Args:
        smiles (str): The input SMILES string.
    Returns:
        bool: True if the SMILES string is valid and known, False otherwise.

    Raises:
        ValueError: If the SMILES string is invalid.
    """
    if not HAS_RDKIT:
        raise ImportError(
            "Please install the rdkit support packages to use this module."
        )
    if not Chem.MolFromSmiles(smiles):
        raise ValueError("Invalid SMILES string.")

    try:
        canonical_smiles = SMILES_utils.canonicalize_smiles(smiles)

        try:
            with open(JSON_FILE_PATH) as f:
                known_mols = json.load(f)
                known_smiles = [mol["smiles"] for mol in known_mols]

        except FileNotFoundError:
            logger.warning(f"{JSON_FILE_PATH} not found. Creating a new one.")
            known_mols = []

    except Exception as e:
        raise ValueError("Error in canonicalizing SMILES string.") from e

    # Check if the SMILES string is already known (in the database)
    # This is a placeholder for the actual database check
    return canonical_smiles in known_smiles


@mcp.tool()
def get_density(smiles: str) -> float:
    """
    Calculate the density of a molecule given its SMILES string.

    Args:
        smiles (str): The input SMILES string.
    Returns:
        float: The density of the molecule.
    """
    if not HAS_RDKIT:
        raise ImportError(
            "Please install the rdkit support packages to use this module."
        )
    density = hf.get_density(smiles)
    logger.info(f"Density for SMILES {smiles}: {density}")
    return density


# Add the SMILES utility functions as MCP tools
mcp.tool()(SMILES_utils.canonicalize_smiles)
mcp.tool()(SMILES_utils.verify_smiles)
mcp.tool()(SMILES_utils.get_synthesizability)


def setup_autogen_pool(
    model: str, backend: str, api_key: Optional[str], base_url: Optional[str]
):
    global AGENT_POOL
    AGENT_POOL = AutoGenPool(
        model=model, backend=backend, api_key=api_key, base_url=base_url
    )


if __name__ == "__main__":
    if not HAS_RDKIT:
        raise ImportError(
            "Please install the rdkit support packages to use this module."
        )
    parser = argparse.ArgumentParser(description="Molecule Tools Server")

    add_server_arguments(parser)

    Client.add_std_parser_arguments(parser)
    parser.add_argument(
        "--json_file",
        type=str,
        default="known_molecules.json",
        help="Path to the JSON file containing known molecules.",
    )
    parser.add_argument(
        "--api_key",
        type=str,
        default=None,
        help="API key for the backend model, if required.",
    )

    parser.add_argument(
        "--base-url",
        type=str,
        default=None,
        help="Base URL for the AutoGen server, if applicable.",
    )

    args = parser.parse_args()

    model = args.model
    backend = args.backend
    base_url = args.server_urls
    api_key = args.api_key

    setup_autogen_pool(model, backend, api_key, base_url)

    logger.info(f"Using model: {model} on backend: {backend}")
    JSON_FILE_PATH = args.json_file if args.json_file else JSON_FILE_PATH
    logger.info(f"Using known molecules database at: {JSON_FILE_PATH}")

    update_mcp_network(mcp, args.host, args.port)
    mcp.run(transport="sse")
