################################################################################
## Copyright 2025 Lawrence Livermore National Security, LLC. and Binghamton University.
## See the top-level LICENSE file for details.
##
## SPDX-License-Identifier: Apache-2.0
################################################################################

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
except ImportError:
    raise ImportError("Please install the rdkit package to use this module.")
import json
import os
from charge.tasks.Task import Task
from charge.servers.server_utils import add_server_arguments
from charge.clients.Client import Client
import asyncio
from charge.servers import SMILES_utils
import charge.utils.helper_funcs as hf
import argparse
from charge.servers import ServerToolkit
from charge.servers.SMILES import SMILESServer
from charge.clients.autogen import AutoGenPool


class DiagnoseSMILESTask(Task):
    def __init__(self, smiles: str):
        self.smiles = smiles
        system_prompt = (
            "You are a world-class chemist. Your task is to diagnose and evaluate "
            "the quality of the provided SMILES strings. You will be given invalid"
            " SMILES strings, and your task is to identify the issues with them, "
            "correct them if possible. Return concise explanations for each SMILE string."
        )

        user_prompt = (
            f"Diagnose the following SMILES string {self.smiles}. Give it a short and concise "
            "explanation of what is wrong with it, and if possible, provide a corrected "
            "version of the SMILES string. If the SMILES string is valid, simply state "
            "'The SMILES string is valid.'"
        )
        super().__init__(system_prompt=system_prompt, user_prompt=user_prompt)


class MoleculeGenerationServer(SMILESServer):
    """
    A ChARGe server that provides molecular generation tools.
    """

    def __init__(self, mcp: FastMCP, model: str, backend: str, json_file_path: str):
        """
        Initialize the MoleculeGenerationServer. Inherits from SMILESServer.

        Args:
            mcp (FastMCP): The MCP instance to register the function with.
            model (str): The model to use for generation.
            backend (str): The backend to use for generation.
            json_file_path (str): The path to the JSON file containing known molecules.
        """
        super().__init__(mcp=mcp)
        self.model = model
        self.backend = backend
        self.json_file_path = json_file_path

        self.agent_pool = AutoGenPool(
            model=self.model,
            backend=self.backend,
        )

    @ServerToolkit.mcp_tool
    def diagnose_smiles(self, smiles: str) -> str:
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

        try:
            diagnose_agent = self.agent_pool.create_agent(task=task)
            response = asyncio.run(diagnose_agent.run())
            assert response is not None
            assert len(response.messages) > 0  # type: ignore
            assert response.messages[-1] is not None  # type: ignore

            diagnoses = response.messages[-1].content  # type: ignore
            logger.info(f"Diagnosis: {diagnoses}")
            return f"SMILES diagnoses: {diagnoses}"  # type: ignore

        except Exception as e:
            logger.error(f"An error occurred: {e}")
            return "Error: Unable to process the SMILES string at this time."

    @ServerToolkit.mcp_tool
    def is_already_known(self, smiles: str) -> bool:
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
                with open(self.json_file_path) as f:
                    known_mols = json.load(f)
                    known_smiles = [mol["smiles"] for mol in known_mols]

            except FileNotFoundError:
                logger.warning(f"{self.json_file_path} not found. Creating a new one.")
                known_mols = []

        except Exception as e:
            raise ValueError("Error in canonicalizing SMILES string.") from e

        # Check if the SMILES string is already known (in the database)
        return canonical_smiles in known_smiles

    @ServerToolkit.mcp_tool
    def get_density(self, smiles: str) -> float:
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Molecule Tools Server")
    add_server_arguments(parser)
    Client.add_std_parser_arguments(parser)
    parser.add_argument(
        "--json_file",
        type=str,
        default="known_molecules.json",
        help="Path to the JSON file containing known molecules.",
    )

    args = parser.parse_args()

    mcp = FastMCP(
        "Molecule Generation MCP Server",
        port=args.port,
        website_url=f"{args.host}",
    )

    server = MoleculeGenerationServer(
        mcp=mcp, model=args.model, backend=args.backend, json_file_path=args.json_file
    )
    server.run(transport="sse")
