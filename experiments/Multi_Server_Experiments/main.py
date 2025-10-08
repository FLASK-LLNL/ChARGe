import argparse
import asyncio
from typing import Optional
from charge.Experiment import Experiment
from charge.clients.Client import Client
from charge.clients.autogen import AutoGenClient

parser = argparse.ArgumentParser()

# Add prompt arguments
parser.add_argument(
    "--system-prompt",
    type=str,
    default=None,
    help="Custom system prompt (optional, uses default chemistry prompt if not provided)",
)

parser.add_argument(
    "--user-prompt",
    type=str,
    default=None,
    help="Custom user prompt (optional, uses default molecule generation prompt if not provided)",
)

# Add standard CLI arguments
Client.add_std_parser_arguments(parser)

# Default prompts
DEFAULT_SYSTEM_PROMPT = (
    "You are a world-class chemist. Your task is to generate unique molecules "
    "based on the lead molecule provided by the user. The generated molecules "
    "should be chemically valid and diverse, exploring different chemical spaces "
    "while maintaining some structural similarity to the lead molecule. "
    "Provide the final answer in a clear and concise manner."
)

DEFAULT_USER_PROMPT = (
    "Generate a unique molecule based on the lead molecule provided. "
    " The lead molecule is CCO. Use SMILES format for the molecules. "
    "Ensure the generated molecule is chemically valid and unique,"
    " using the tools provided. Check the price of the generated molecule "
    "using the molecule pricing tool, and get a cheap molecule. "
)

class ChargeMultiServerExperiment(Experiment):
    def __init__(
        self,
        system_prompt: Optional[str] = None,
        user_prompt: Optional[str] = None,
    ):
        # Use provided prompts or fall back to defaults
        if system_prompt is None:
            system_prompt = DEFAULT_SYSTEM_PROMPT
        if user_prompt is None:
            user_prompt = DEFAULT_USER_PROMPT
        
        super().__init__(system_prompt=system_prompt, user_prompt=user_prompt)
        print("ChargeMultiServerExperiment initialized with the provided prompts.")

        self.system_prompt = system_prompt
        self.user_prompt = user_prompt


if __name__ == "__main__":

    args = parser.parse_args()
    server_urls = args.server_urls
    assert server_urls is not None, "Server URLs must be provided"
    for url in server_urls:
        assert url.endswith("/sse"), "Server URL must end with /sse"
    
    myexperiment = ChargeMultiServerExperiment(
        system_prompt=args.system_prompt,
        user_prompt=args.user_prompt,
    )

    (model, backend, API_KEY, kwargs) = AutoGenClient.configure(
        args.model, args.backend
    )

    server_path_1 = "stdio_server_1.py"
    server_path_2 = "stdio_server_2.py"

    runner = AutoGenClient(
        experiment_type=myexperiment,
        backend=backend,
        model=model,
        api_key=API_KEY,
        model_kwargs=kwargs,
        server_path=[server_path_1, server_path_2],
        server_url=server_urls,
    )

    results = asyncio.run(runner.run())

    print(f"Experiment completed. Results: {results}")
