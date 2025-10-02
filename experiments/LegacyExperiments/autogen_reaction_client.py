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


async def main():
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

    reaction_server = StdioServerParams(
        command="python3", args=["reaction_server.py"], env=None
    )

    system_prompt = (
        "You are an expert retrosynthesis and chemical reaction generator "
        + "that can propose new reactions by generating Reaction SMARTS"
        + "and SMILES molecules. Make sure the SMILES strings are valid first."
        + "If stuck, try different reaction SMARTS or reactants."
        + " Generate a reaction SMARTS and reactants for the given molecule."
        + " For each reaction SMARTS verify it."
        + " If the reaction SMARTS is valid, check if the reactants are valid SMILES."
        + " If they are valid, check if the reaction can be performed"
        + " Use the diagnoise tools to fix any issues that arise."
        + " and return the reaction SMARTS, reactants, and products."
        + " Prefer reactions that are more likely to be performed in a lab "
        + " setting. \n\n"
    )
    print("System prompt:", system_prompt)

    print("Prompt: \n")
    prompt = (
        "Generate a new reaction SMARTS and reactants"
        + " for the product c1cc(ccc1N)O \n\n"
    )
    print(prompt)

    async with McpWorkbench(reaction_server) as workbench:  # type: ignore

        agent_loop = AssistantAgent(
            name="Reaction_Generator",
            model_client=model_client,
            system_message=system_prompt,
            workbench=workbench,
            reflect_on_tool_use=True,
            max_tool_iterations=15,
        )
        result = await agent_loop.run(
            task=prompt,
        )
        assert isinstance(result.messages[-1], TextMessage)

        for message in result.messages:
            if isinstance(message, TextMessage):
                print(message.content)

        # Close the connection to the model client.
        await model_client.close()


if __name__ == "__main__":
    asyncio.run(main())
