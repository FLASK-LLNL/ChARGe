import argparse
import asyncio
from typing import Optional, Union
import charge
from charge.tasks.task import Task
from charge.clients.client import Client
from charge.clients.autogen import AutoGenBackend

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


class ChargeMultiServerTask(Task):
    def __init__(
        self,
        system_prompt: Optional[str] = None,
        user_prompt: Optional[str] = None,
        server_urls: Optional[Union[str, list]] = None,
        server_paths: Optional[Union[str, list]] = None,
    ):
        # Use provided prompts or fall back to defaults
        if system_prompt is None:
            system_prompt = DEFAULT_SYSTEM_PROMPT
        if user_prompt is None:
            user_prompt = DEFAULT_USER_PROMPT

        super().__init__(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            server_urls=server_urls,
            server_paths=server_paths,
        )
        print("ChargeMultiServerTask initialized with the provided prompts.")


if __name__ == "__main__":

    args = parser.parse_args()
    if args.history:
        charge.enable_cmd_history_and_shell_integration(args.history)

    server_urls = args.server_urls
    server_path_1 = "stdio_server_1.py"
    server_path_2 = "stdio_server_2.py"

    mytask = ChargeMultiServerTask(
        system_prompt=args.system_prompt,
        user_prompt=args.user_prompt,
        server_urls=server_urls,
        server_paths=[server_path_1, server_path_2],
    )

    agent_backend = AutoGenBackend(model=args.model, backend=args.backend)

    runner = agent_backend.create_agent(task=mytask)

    results = asyncio.run(runner.run())

    print(f"Task completed. Results: {results}")
