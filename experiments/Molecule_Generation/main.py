import argparse
import asyncio
from LMOExperiment import LMOExperiment as LeadMoleculeOptimization
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
    default="mol_server.py",
    help="Path to an existing MCP server script",
)

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
        import httpx
        from charge.clients.autogen import AutoGenClient

        backend = args.backend
        model = args.model
        kwargs = {}
        API_KEY = None
        if backend in ["openai", "gemini", "livai", "livchat"]:
            if backend == "openai":
                API_KEY = os.getenv("OPENAI_API_KEY")
                model = "gpt-4"
                kwargs["parallel_tool_calls"] = False
                kwargs["reasoning_effort"] = "high"
            elif backend == "livai" or backend == "livchat":
                API_KEY = os.getenv("OPENAI_API_KEY")
                BASE_URL = os.getenv("LIVAI_BASE_URL")
                assert (
                    BASE_URL is not None
                ), "LivAI Base URL must be set in environment variable"
                model = "gpt-4.1"
                kwargs["base_url"] = BASE_URL
                kwargs["http_client"] = httpx.AsyncClient(verify=False)
            else:
                API_KEY = os.getenv("GOOGLE_API_KEY")
                model = "gemini-flash-latest"
                kwargs["parallel_tool_calls"] = False
                kwargs["reasoning_effort"] = "high"

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
