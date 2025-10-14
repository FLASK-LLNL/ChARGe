import argparse
import asyncio
from RetrosynthesisExperiment import RetrosynthesisExperiment as Retrosynthesis
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
    default="reaction_server.py",
    help="Path to an existing MCP server script",
)

parser.add_argument(
    "--user-prompt",
    type=str,
    default="Generate a new reaction SMARTS and reactants for the product c1cc(ccc1N)O \n\n",
    help="The product to perform retrosynthesis on, "
    + "including any further constraints",
)

# Add standard CLI arguments
Client.add_std_parser_arguments(parser)

args = parser.parse_args()

if __name__ == "__main__":

    server_path = args.server_path
    assert server_path is not None, "Server path must be provided"
    user_prompt = args.user_prompt
    assert user_prompt is not None, "User prompt must be provided"

    myexperiment = Retrosynthesis(user_prompt=user_prompt)

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
