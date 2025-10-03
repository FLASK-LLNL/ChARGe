import argparse
import asyncio
from RetrosynthesisExperiment import RetrosynthesisExperiment as Retrosynthesis
import os

parser = argparse.ArgumentParser()
parser.add_argument("--lead-molecule", type=str, default="CC(=O)O[C@H](C)CCN")
parser.add_argument(
    "--client", type=str, default="autogen", choices=["autogen", "gemini"]
)

parser.add_argument(
    "--backend",
    type=str,
    default="openai",
    choices=["openai", "gemini", "ollama", "livai", "livchat"],
    help="Backend to use for the autogen client",
)
parser.add_argument(
    "--model", type=str, default="gpt-4", help="Model to use for the autogen backend"
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
