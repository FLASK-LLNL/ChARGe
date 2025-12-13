################################################################################
## Copyright 2025 Lawrence Livermore National Security, LLC. and Binghamton University.
## See the top-level LICENSE file for details.
##
## SPDX-License-Identifier: Apache-2.0
################################################################################

from autogen_ext.models.openai import OpenAIChatCompletionClient

from openai.types.chat import ChatCompletion
from loguru import logger


## Create a custom wrapper to log responses
class LoggingModelClient(OpenAIChatCompletionClient):
    async def create(self, messages, **kwargs):
        logger.info("!!! LoggingModelClient.create() WAS CALLED !!!\n")

        # Call the parent to get the response
        response = await super().create(messages, **kwargs)

        # CRITICAL: Access the raw API response before AutoGen processes it
        # The reasoning_content is in the raw response but might not be mapped to CreateResult

        # Try to extract reasoning_content from the internal state
        # Option 1: Check if we can access the raw completion object
        reasoning_content = None

        # The response object might have internal attributes we can access
        if hasattr(response, "__dict__"):
            # Check for thought attribute (which AutoGen might populate)
            if hasattr(response, "thought") and response.thought:
                reasoning_content = response.thought

        # If reasoning_content is not in the CreateResult, we need to intercept earlier
        # We'll need to override at a lower level to catch the raw API response

        # Log what we have
        logger.info("\n" + "=" * 80 + "\n")
        logger.info("GPT-OSS MODEL RESPONSE\n")
        logger.info("=" * 80 + "\n")

        if reasoning_content:
            logger.info("\n[ANALYSIS CHANNEL]\n")
            logger.info("-" * 80 + "\n")
            logger.info(str(reasoning_content) + "\n")
            logger.info("-" * 80 + "\n")

        if hasattr(response, "content") and response.content:
            logger.info("\n[FINAL CHANNEL]\n")
            logger.info("-" * 80 + "\n")
            if isinstance(response.content, list):
                for item in response.content:
                    logger.info(str(item) + "\n")
            else:
                logger.info(str(response.content) + "\n")
            logger.info("-" * 80 + "\n")

        if hasattr(response, "usage") and response.usage:
            logger.info("\n[USAGE]\n")
            logger.info("-" * 80 + "\n")
            logger.info(f"Prompt tokens: {response.usage.prompt_tokens}\n")
            logger.info(f"Completion tokens: {response.usage.completion_tokens}\n")
            logger.info("-" * 80 + "\n")

        logger.info("\n" + "=" * 80 + "\n\n")

        return response


class InspectingModelClient(OpenAIChatCompletionClient):
    async def create(self, messages, **kwargs):
        # Make the request and inspect
        response = await super().create(messages, **kwargs)

        # The response object should contain all channels
        # gpt-oss typically puts reasoning in response.choices[0].message.reasoning_content
        # or similar field
        for choice in response.choices:
            if hasattr(choice.message, "reasoning_content"):
                logger.info("Analysis channel:", choice.message.reasoning_content)
            if hasattr(choice.message, "content"):
                logger.info("Final channel:", choice.message.content)
            if hasattr(choice.message, "tool_calls"):
                logger.info("Tool calls:", choice.message.tool_calls)

        return response
