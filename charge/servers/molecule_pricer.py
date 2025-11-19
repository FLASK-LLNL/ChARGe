################################################################################
## Copyright 2025 Lawrence Livermore National Security, LLC. and Binghamton University.
## See the top-level LICENSE file for details.
##
## SPDX-License-Identifier: Apache-2.0
################################################################################

from loguru import logger
try:
    import chemprice
    from chemprice import PriceCollector
    HAS_CHEMPRICE = True
except (ImportError, ModuleNotFoundError) as e:
    HAS_CHEMPRICE = False
    logger.warning(
        "Please install the chemprice support packages to use this module."
        "Install it with: pip install charge[chemprice]",
    )
import os, sys

def get_chemspace_prices(SMILES_list,best_only=True):
    
    """
    Retrieve vendor pricing from ChemSpace for one or more molecules specified by SMILES.

    Valid properties
    ----------------
    ChARGe can request or consume the following properties from the returned records if best_only=False:
      - Input Smiles : str        # SMILES string for the specified molecules
      - Source : str              # Data source tag, typically "chemspace" or "molport"
      - Supplier Name : str       # Name of the chemical supplier
      - Purity : str              # Supplier catalog identifier
      - Amount : float            # Quantity of the chemical. Units are specified in the 'Measure' field below.
      - Measure : str             # Units of the size of the chemical package. E.g. "g" or "micromol".
      - Price_USD : float         # Price of the chemical in U.S. dollars.
      - USD/g : float             # Price of the chemical in U.S. dollars per gram of chemical.
      - USD/mol : float           # Price of the chemical in U.S. dollars per mol of chemical.

    Parameters
    ----------
    SMILES_list : list[str]
        A list of SMILES strings to query prices for.
    best_only : bool, default True
        If True, return only the cheapest molecule price in USD/g; if False, return
        all collected vendor offers and all properties.

    Returns
    -------
    list[float] or pandas.DataFrame
        If `best_only=True`, returns a list of floats representing the lowest
        price (in USD/g) among all vendors for the specified molecules in SMILES_list.
        Otherwise, returns a pandas DataFrame containing detailed vendor
        information for the molecule, including columns as listed above.


    Examples
    --------
    >>> get_chemspace_prices(["CCO","CO","CCC"], best_only=True)
    [0.1056, 9.57, nan]

    """    

    if not HAS_CHEMPRICE:
        raise ImportError("Please install the chemprice support packages to use this module.")

    pc = PriceCollector()
    chemspace_api_key = os.getenv("CHEMSPACE_API_KEY")
    if(chemspace_api_key):
        pc.setChemSpaceApiKey(chemspace_api_key)
    else:
        print('CHEMPROP_API_KEY environment variable not set!')
        sys.exit(2)
    print(pc.check())
    all_prices = pc.collect(SMILES_list)
    if(best_only):
        best_price=pc.selectBest(all_prices)
        return(best_price["USD/g"].astype(float).tolist())
    else:
        return(all_prices)

def main(smiles_list,price_source='Chemspace'):
    """
    Main function that retrieves prices for a list of SMILES strings. Default to Chemspace because it doesn't have API limits. Keep this function to add future price_sources (Molport).

    Args:
        smiles_list (list[str]): A list of SMILES strings.
    """

    if not HAS_CHEMPRICE:
        raise ImportError("Please install the chemprice support packages to use this module.")
    if(price_source=='Chemspace'):
        prices=get_chemspace_prices(smiles_list)
    print("Retrieved Prices from "+price_source+":")
    print(prices)

if __name__ == "__main__":
    # Example usage
    example_smiles = ["CCO"]
    main(example_smiles)
