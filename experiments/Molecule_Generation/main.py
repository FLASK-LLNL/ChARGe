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
    choices=["openai", "gemini", "ollama", "livchat"],
    help="Backend to use for the autogen client",
)
parser.add_argument(
    "--model", type=str, default="gpt-4", help="Model to use for the autogen backend"
)

args = parser.parse_args()

if __name__ == "__main__":

    myexperiment = LeadMoleculeOptimization(lead_molecule=args.lead_molecule)

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
        if backend in ["openai", "gemini", "livchat"]:
            if backend == "openai":
                API_KEY = os.getenv("OPENAI_API_KEY")
                model = "gpt-4"
                kwargs["parallel_tool_calls"] = False
                kwargs["reasoning_effort"] = "high"
            elif backend == "livchat":
                API_KEY = os.getenv("OPENAI_API_KEY")
                model = "gpt-4.1"
                kwargs["base_url"] = "https://livai-api.llnl.gov/v1"
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
        )

    # results = asyncio.run(runner.run(myexperiment))
    # print(results)

    # while True:
    #     cont = input("Additional refinement? ...")
    #     results = asyncio.run(runner.refine(cont))
    #     print(results)
