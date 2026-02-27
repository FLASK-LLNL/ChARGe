import argparse
import asyncio
from charge.tasks.task import Task
from typing import Optional, Union
from charge.clients.client import Client
from charge.clients.autogen import AutoGenBackend

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


if __name__ == "__main__":

    args = parser.parse_args()
    server_url = args.server_urls

    mytask = Task(
        system_prompt=(
            DEFAULT_SYSTEM_PROMPT if args.system_prompt is None else args.system_prompt
        ),
        server_urls=server_url,
    )

    agent_backend = AutoGenBackend(model=args.model, backend=args.backend)

    agent = agent_backend.create_agent(task=mytask)

    agent_state = asyncio.run(agent.chat())

    print(f"Task completed. Results: {agent_state}")
