import argparse
import asyncio
from LMOExperiment import LMOExperiment as LeadMoleculeOptimization
import os
from charge.clients.Client import Client

parser = argparse.ArgumentParser()
parser.add_argument("--lead-molecule", type=str, default="CC(=O)O[C@H](C)CCN")
parser.add_argument(
    "--client", type=str, default="autogen", choices=["autogen", "gemini"]
)

parser.add_argument(
    "--server-path",
    type=str,
    default="mol_server.py",
    help="Path to an existing MCP server script",
)

# Add standard CLI arguments
Client.add_std_parser_arguments(parser)

args = parser.parse_args()

if __name__ == "__main__":

    myexperiment = LeadMoleculeOptimization(lead_molecule=args.lead_molecule)
    server_path = args.server_path
    assert server_path is not None, "Server path must be provided"

    if args.client == "gemini":
        from charge.clients.gemini import GeminiClient

        client_key = os.getenv("GOOGLE_API_KEY")
        assert client_key is not None, "GOOGLE_API_KEY must be set in environment"
        runner = GeminiClient(experiment_type=myexperiment, api_key=client_key)
    elif args.client == "autogen":
        from charge.clients.autogen import AutoGenClient

        (model, backend, API_KEY, kwargs) = AutoGenClient.configure(args.model, args.backend)

        runner = AutoGenClient(
            experiment_type=myexperiment,
            model=model,
            backend=backend,
            api_key=API_KEY,
            model_kwargs=kwargs,
            server_path=server_path,
        )

    results = asyncio.run(runner.run())

    print(f"Experiment completed. Results: {results}")
