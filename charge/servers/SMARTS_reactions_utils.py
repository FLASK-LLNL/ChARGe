################################################################################
## Copyright 2025 Lawrence Livermore National Security, LLC. and Binghamton University.
## See the top-level LICENSE file for details.
##
## SPDX-License-Identifier: Apache-2.0
################################################################################

from mcp.server.fastmcp import FastMCP
try:
    from rdkit import Chem
    from rdkit.Chem import AllChem, rdChemReactions
except ImportError:
    raise ImportError("Please install the rdkit package to use this module.")

from loguru import logger
from typing import Tuple

def verify_reaction_SMARTS(smarts: str) -> Tuple[bool, str]:
    """
    Verify if a SMARTS string is valid.
    Returns a tuple of (bool, str).
    The bool indicates if the SMARTS is valid, and the str is an error message if it is not.

    Args:
        smiles (str): The input SMILES string.
    Returns:
        A tuple containing:
            bool: True if the SMARTS is valid, False if it is invalid.
            str: Error message if the SMARTS reaction is valid.
    """
    try:
        logger.info(f"Verifying SMARTS: {smarts}")
        rxn = AllChem.ReactionFromSmarts(smarts)

        if not rxn:
            logger.error("Invalid reaction SMARTS.")
            return False, "Invalid reaction SMARTS."
        rxn.Initialize()
        if not rxn.IsInitialized():
            logger.error("Reaction SMARTS could not be initialized.")
            return False, "Reaction SMARTS could not be initialized."
        rdChemReactions.SanitizeRxn(rxn)
        logger.info("Reaction SMARTS is valid and sanitized.")
        sanitized_smarts = rdChemReactions.ReactionToSmarts(rxn)
        logger.info(f"Sanitized SMARTS: {sanitized_smarts}")
        if not sanitized_smarts:
            logger.error("Sanitized SMARTS is empty.")
            return False, "Sanitized SMARTS is empty."
        logger.info("SMARTS is valid.")
        return True, "SMARTS is valid."
    except Exception as e:
        logger.error(f"Invalid SMARTS string: {e}")
        return False, f"Invalid Syntax for SMARTS string. The error is: {e}"

def verify_reaction(
    smarts: str, reactants: list[str], products: list[str]
) -> Tuple[bool, str]:
    """
    Verify if a reaction can be performed given the SMARTS and reactants.
    Returns a tuple of (bool, str).
    The bool indicates if the reaction can be performed, and
    the str is an error message if it cannot be performed.

    Args:
        smarts (str): The input SMARTS string.
        reactants (list[str]): The input list of reactants in SMILES strings
        products (list[str]): The input list of products created by the reaction in SMILES strings
    Returns:
        A tuple containing:
            bool: True if a reaction can be performed given the SMARTS and reactants.
                  False if it cannot..
            str: Error message if the SMARTS reaction is valid.
    """
    try:
        logger.info(
            f"Verifying reaction with SMARTS: {smarts}, Reactants: {reactants}, Products: {products}"
        )
        reaction = AllChem.ReactionFromSmarts(smarts)
        if not reaction:
            logger.error("Invalid reaction SMARTS.")
            return False, "Invalid reaction SMARTS."

        reactant_mols = [Chem.MolFromSmiles(r) for r in reactants]
        product_mols = [Chem.MolFromSmiles(p) for p in products]

        if not all(reactant_mols) or not all(product_mols):
            for i, r in enumerate(reactant_mols):
                if r is None:
                    logger.error(f"Invalid reactant SMILES: {reactants[i]}")
                    return False, f"Invalid reactant SMILES: {reactants[i]}"
            for i, p in enumerate(product_mols):
                if p is None:
                    logger.error(f"Invalid product SMILES: {products[i]}")
                    return False, f"Invalid product SMILES: {products[i]}"

        results = reaction.RunReactants(reactant_mols)
        if not results:
            logger.error("Reaction cannot be applied to the given reactants.")
            return False, "Reaction cannot be applied to the given reactants."

        for result in results:
            result_smiles = [
                Chem.MolToSmiles(Chem.MolFromSmiles(Chem.MolToSmiles(mol)))
                for mol in result
            ]

            check_products = True
            for prod in set(product_mols):
                prod_smiles = Chem.MolToSmiles(prod)
                if prod_smiles not in result_smiles:
                    check_products = False
                    return (
                        False,
                        f"Product {prod_smiles} not found in predicted products: {result_smiles}",
                    )
            if check_products:
                logger.info("Reaction verified successfully.")
                return True, "Reaction verified successfully."

        logger.error("No matching products found from the reaction.")
        return False, "No matching products found from the reaction."
    except Exception as e:
        logger.error(f"Error verifying reaction: {e}")
        return False, f"Error verifying reaction: {e}"
    
