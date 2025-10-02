################################################################################
## Copyright 2025 Lawrence Livermore National Security, LLC. and Binghamton University.
## See the top-level LICENSE file for details.
##
## SPDX-License-Identifier: Apache-2.0
################################################################################

from autogen_agentchat.agents import AssistantAgent
from autogen_core.models import ModelFamily
from autogen_ext.tools.mcp import StdioServerParams, McpWorkbench
from autogen_agentchat.messages import TextMessage
import asyncio
import os
import argparse
import httpx

parser = argparse.ArgumentParser(description="Molecule Generation")
backends = ["openai", "gemini", "ollama", "livchat"]
parser.add_argument(
    "--backend",
    type=str,
    default="openai",
    choices=backends,
    help="Backend to use: openai or gemini",
)

args = parser.parse_args()
backend = args.backend

if not backend:
    raise ValueError(f"Backend must be one of {backends}")


async def main() -> None:
    model_info = {
        "vision": False,
        "function_calling": True,
        "json_output": True,
        "family": ModelFamily.UNKNOWN,
        "structured_output": True,
    }

    if backend == "openai" or backend == "gemini" or backend == "livchat":
        from autogen_ext.models.openai import OpenAIChatCompletionClient

        kwargs = {}
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
        assert API_KEY is not None, "API key must be set in environment variable"
        model_client = OpenAIChatCompletionClient(
            model=model,
            api_key=API_KEY,
            model_info=model_info,
            **kwargs,
        )
    else:
        from autogen_ext.models.ollama import OllamaChatCompletionClient

        model_client = OllamaChatCompletionClient(
            model="gpt-oss:latest",
            model_info=model_info,
        )

    mol_server = StdioServerParams(command="python3", args=["mol_server.py"], env=None)

    system_prompt = (
        "You are a helpful SMILES generator "
        + "that can generate new molecules in a SMILES format and optimize"
        + " for density and synthetic accessibility."
    )
    print("System prompt:", system_prompt)

    print("Prompt: \n")
    chat_prompt = (
        "Generate a new SMILES string"
        + " for molecules similar to CC(=O)O[C@H](C)CCN. For each molecules you suggest "
        + " verify the SMILES,"
        + " check if it already known, and calculate its density and"
        + " synthetic accessibility. Only return molecules with higher density and"
        + " the same or lower synthetic accessibility."
        + " If a molecule is known or doesn't fit the criteria, move on and"
        + " generate a different one and try again."
        + " Output a list of the unique molecules \n\n"
    )
    print(chat_prompt)

    async with McpWorkbench(mol_server) as workbench:  # type: ignore

        agent_loop = AssistantAgent(
            name="Molecule_Generator",
            model_client=model_client,
            system_message=system_prompt,
            workbench=workbench,
            reflect_on_tool_use=True,
            max_tool_iterations=15,
        )
        result = await agent_loop.run(
            task=chat_prompt,
        )
        assert isinstance(result.messages[-1], TextMessage)

        for message in result.messages:
            if isinstance(message, TextMessage):
                print(message.content)

        # Close the connection to the model client.
        await model_client.close()


if __name__ == "__main__":
    asyncio.run(main())
