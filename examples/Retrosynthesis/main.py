import argparse
import asyncio
from charge.tasks.RetrosynthesisTask import (
    RetrosynthesisTask,
    TemplateFreeRetrosynthesisTask,
)

from charge.clients.Client import Client
from charge.clients.autogen import AutoGenPool

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

parser.add_argument(
    "--system-prompt",
    type=str,
    default=None,
    help="Custom system prompt (optional, uses default retrosynthesis prompt if not provided)",
)

# Add standard CLI arguments
Client.add_std_parser_arguments(parser)

parser.add_argument(
    "--exp_type",
    default="template",
    choices=["template", "template-free"],
    help="Type of retrosynthesis task to run",
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

        mytask = RetrosynthesisTask(
            user_prompt=user_prompt,
            system_prompt=args.system_prompt,
            server_urls=server_urls,
        )
    elif args.exp_type == "template-free":
        mytask = TemplateFreeRetrosynthesisTask(
            user_prompt=user_prompt,
            system_prompt=args.system_prompt,
            server_urls=server_urls,
        )
    else:
        raise ValueError(f"Unknown task type: {args.exp_type}")

    agent_pool = AutoGenPool(model=args.model, backend=args.backend)
    runner = agent_pool.create_agent(task=mytask)
    results = asyncio.run(runner.run())

    print(f"Task completed. Results: {results}")
