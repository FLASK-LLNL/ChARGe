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
    choices=["openai", "gemini", "ollama", "livchat"],
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
    default=None,
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
