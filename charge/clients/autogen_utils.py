################################################################################
## Copyright 2025 Lawrence Livermore National Security, LLC. and Binghamton University.
## See the top-level LICENSE file for details.
##
## SPDX-License-Identifier: Apache-2.0
################################################################################
try:
    from autogen_agentchat.agents import AssistantAgent
    from autogen_core.model_context import UnboundedChatCompletionContext
    from autogen_core.models import (
        ChatCompletionClient,
        LLMMessage,
        AssistantMessage,
    )
    from openai import AsyncOpenAI
    from autogen_ext.agents.openai import OpenAIAgent
    from autogen_ext.tools.mcp import  McpWorkbench
except ImportError:
    raise ImportError(
        "Please install the autogen-agentchat package to use this module."
    )
from typing import Type, Optional, Dict, Union, List, Callable

class ReasoningModelContext(UnboundedChatCompletionContext):
    """A model context for reasoning models."""

    def __init__(self, callback: Optional[Callable] = None):
        super().__init__()
        self.callback = callback

    async def get_messages(self) -> List[LLMMessage]:
        messages = await super().get_messages()
        # Filter out thought field from AssistantMessage.
        messages_out: List[LLMMessage] = []
        for message in messages:
            messages_out.append(message)
        return messages_out

    async def add_message(
        self,
        assistant_message: AssistantMessage,
    ) -> None:

        await super().add_message(assistant_message)

        if self.callback:
            self.callback(assistant_message)
        else:
            if (
                hasattr(assistant_message, "thought")
                and assistant_message.thought is not None
            ):
                print(f"Model thought: {assistant_message.thought}")
            else:
                print("Model: ", assistant_message.content)


def thoughts_callback(assistant_message):
    # print("In callback:", assistant_message)
    if assistant_message.type == "UserMessage":
        print(f"User: {assistant_message.content}")
    elif assistant_message.type == "AssistantMessage":

        if assistant_message.thought is not None:
            print(f"Model thought: {assistant_message.thought}")
        if isinstance(assistant_message.content, list):
            for item in assistant_message.content:
                if hasattr(item, "name") and hasattr(item, "arguments"):
                    print(
                        f"Function call: {item.name} with args {item.arguments}"
                    )
                else:
                    print(f"Model: {item}")
    elif assistant_message.type == "FunctionExecutionResultMessage":

        for result in assistant_message.content:
            if result.is_error:
                print(
                    f"Function {result.name} errored with output: {result.content}"
                )
            else:
                print(f"Function {result.name} returned: {result.content}")
    else:
        print("Model: ", assistant_message.message.content)

def generate_agent(
        model_client: Union[AsyncOpenAI,ChatCompletionClient],
        model: str,
        system_prompt: str,
        workbenches: List[McpWorkbench],
        max_tool_calls: int):
    if isinstance(model_client, AsyncOpenAI):
        agent = OpenAIAgent(
            name="Assistant",
            description='ChARGe OpenAIAgent',
            client=model_client,
            model=model,
            instructions=system_prompt,
        )
    elif isinstance(model_client, ChatCompletionClient):
        # TODO: Convert this to use custom agent in the future
        agent = AssistantAgent(
            name="Assistant",
            model_client=model_client,
            system_message=system_prompt,
            workbench=workbenches if len(workbenches) > 0 else None,
            max_tool_iterations=max_tool_calls,
            reflect_on_tool_use=True,
            model_context=ReasoningModelContext(thoughts_callback),
            # output_content_type=structured_output_schema,
        )
    else:
        raise ValueError("ERROR: Unknown model client type.")

    return agent
