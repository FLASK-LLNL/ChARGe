import argparse
import asyncio
from charge.tasks.Task import Task
from typing import Optional
from charge.clients.Client import Client
from charge.clients.autogen import AutoGenClient

parser = argparse.ArgumentParser()

# Add system prompt argument
parser.add_argument(
    "--system-prompt",
    type=str,
    default=None,
    help="Custom system prompt (optional, uses default chemistry prompt if not provided)",
)

# Add standard CLI arguments
Client.add_std_parser_arguments(parser)

# Default system prompt
DEFAULT_SYSTEM_PROMPT = (
    "You are a world-class chemist. Your task is to generate unique molecules "
    "based on the lead molecule provided by the user. The generated molecules "
    "should be chemically valid and diverse, exploring different chemical spaces "
    "while maintaining some structural similarity to the lead molecule. "
    "Provide the final answer in a clear and concise manner."
)

class ChargeChatTask(Task):
    def __init__(
        self,
        system_prompt: Optional[str] = None,
    ):
        # Use provided system prompt or fall back to default
        if system_prompt is None:
            system_prompt = DEFAULT_SYSTEM_PROMPT

        super().__init__(system_prompt=system_prompt, user_prompt=None)
        print("ChargeChatTask initialized with the provided prompts.")

        self.system_prompt = system_prompt
        self.user_prompt = None


if __name__ == "__main__":

    args = parser.parse_args()
    server_url = args.server_urls[0]
    assert server_url is not None, "Server URL must be provided"
    assert server_url.endswith("/sse"), "Server URL must end with /sse"

    mytask = ChargeChatTask(
        system_prompt=args.system_prompt,
    )

    (model, backend, API_KEY, kwargs) = AutoGenClient.configure(
        args.model, args.backend
    )

    runner = AutoGenClient(
        task=mytask,
        backend=backend,
        model=model,
        api_key=API_KEY,
        model_kwargs=kwargs,
        # server_path=server_path,
        server_url=server_url,
    )

    results = asyncio.run(runner.chat())

    print(f"Task completed. Results: {results}")
