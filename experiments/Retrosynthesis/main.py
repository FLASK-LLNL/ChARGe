import argparse
import asyncio
from RetrosynthesisExperiment import (
    RetrosynthesisExperiment,
    TemplateFreeRetrosynthesisExperiment,
)
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
    default=None,
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

parser.add_argument(
    "--exp_type",
    default="template",
    choices=["template", "template-free"],
    help="Type of retrosynthesis experiment to run",
)

args = parser.parse_args()

if __name__ == "__main__":

    server_path = args.server_path

    server_urls = args.server_urls
    if server_urls is not None:
        for server_url in server_urls:
            assert server_url.endswith("/sse"), "Server URL must end with a '/'"
    user_prompt = args.user_prompt
    assert user_prompt is not None, "User prompt must be provided"

    if args.exp_type == "template":

        myexperiment = RetrosynthesisExperiment(user_prompt=user_prompt)
    elif args.exp_type == "template-free":
        myexperiment = TemplateFreeRetrosynthesisExperiment(user_prompt=user_prompt)
    else:
        raise ValueError(f"Unknown experiment type: {args.exp_type}")

    if args.client == "gemini":
        from charge.clients.gemini import GeminiClient

        client_key = os.getenv("GOOGLE_API_KEY")
        assert client_key is not None, "GOOGLE_API_KEY must be set in environment"
        runner = GeminiClient(experiment_type=myexperiment, api_key=client_key)
    elif args.client == "autogen":
        from charge.clients.autogen import AutoGenClient

        (model, backend, API_KEY, kwargs) = AutoGenClient.configure(
            args.model, args.backend
        )

        runner = AutoGenClient(
            experiment_type=myexperiment,
            model=model,
            backend=backend,
            api_key=API_KEY,
            model_kwargs=kwargs,
            server_path=server_path,
            server_url=server_urls,
        )

        results = asyncio.run(runner.run())

        print(f"Experiment completed. Results: {results}")
