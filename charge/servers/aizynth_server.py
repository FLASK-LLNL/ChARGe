import argparse
from aizynthfinder.aizynthfinder import AiZynthFinder
from rdkit import Chem
from mcp.server.fastmcp import FastMCP


# Initialize MCP server
mcp = FastMCP("aizynthfinder")


# Helper functions
def is_valid_smiles(smiles: str) -> bool:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return False
    return True


# Tools
@mcp.tool()
def find_synthesis_routes(product_smi: str) -> list[dict]:
    # if not is_valid_smiles(product_smi):
    #     return {'isError': True, 'content': 'Invalid SMILES string'}
    finder.target_smiles = product_smi
    finder.tree_search()
    finder.build_routes()
    return finder.routes.make_dicts()


def main():
    # CLI arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, help='Config yaml file for initializing AiZynthFinder')
    parser.add_argument('--transport', type=str, help='MCP transport type', choices=['stdio', 'streamable-http', 'sse'], default='stdio')
    args = parser.parse_args()

    # Initializer AiZynthFinder
    global finder
    finder = AiZynthFinder(configfile=args.config)
    finder.stock.select('zinc')
    finder.expansion_policy.select('uspto')
    finder.filter_policy.select("uspto")

    mcp.run(transport=args.transport)


if __name__ == "__main__":
    main()