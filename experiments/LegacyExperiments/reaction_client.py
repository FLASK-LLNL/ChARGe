################################################################################
## Copyright 2025 Lawrence Livermore National Security, LLC. and Binghamton University.
## See the top-level LICENSE file for details.
##
## SPDX-License-Identifier: Apache-2.0
################################################################################

from google import genai
from google.genai import types
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
import asyncio
import os
from contextlib import AsyncExitStack

client_key = os.getenv("GOOGLE_API_KEY")
client = genai.Client(api_key=client_key)

server_params = StdioServerParameters(
    command="python3",
    args=["reaction_server.py"],
    env=None,
)


async def main():

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
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
            print("System prompt: \n", system_prompt)

            print("Prompt: \n")
            prompt = (
                "Generate a new reaction SMARTS and reactants"
                + " for the product c1cc(ccc1N)O \n\n"
            )
            print(prompt)

            await session.initialize()

            response = await client.aio.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0,
                    tools=[session],
                    system_instruction=system_prompt,
                    thinking_config=types.ThinkingConfig(
                        include_thoughts=True, thinking_budget=8192
                    ),
                ),
            )
            assert response is not None
            assert response.usage_metadata is not None
            print("Thoughts tokens:", response.usage_metadata.thoughts_token_count)
            print("Output tokens:", response.usage_metadata.candidates_token_count)
            assert response.candidates is not None
            assert len(response.candidates) > 0
            assert response.candidates[0] is not None
            assert response.candidates[0].content is not None
            assert response.candidates[0].content.parts is not None

            print(response.candidates[0].content.parts)
            for part in response.candidates[0].content.parts:
                if not part.text:

                    continue
                if part.thought:
                    print("Thought summary:")
                    print(part.text)
                    print()
                else:
                    print("Answer:")
                    print(part.text)
                    print()
            print("Response text:", response.text)
            print("Response usage metadata:", response.usage_metadata)


if __name__ == "__main__":
    asyncio.run(main())
