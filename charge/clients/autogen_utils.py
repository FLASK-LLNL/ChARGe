################################################################################
## Copyright 2025 Lawrence Livermore National Security, LLC. and Binghamton University.
## See the top-level LICENSE file for details.
##
## SPDX-License-Identifier: Apache-2.0
################################################################################
try:
    from autogen_agentchat.agents import AssistantAgent
    from autogen_core.model_context import (
        UnboundedChatCompletionContext,
        ChatCompletionContext,
    )
    from autogen_core.memory import ListMemory
    from autogen_core.models import (
        ChatCompletionClient,
        LLMMessage,
        AssistantMessage,
        SystemMessage,
    )
    from autogen_core._cancellation_token import CancellationToken
    from autogen_core.memory._base_memory import (
        MemoryContent,
        MemoryQueryResult,
        UpdateContextResult,
    )
    from autogen_agentchat.ui._console import aprint
    from autogen_agentchat.base import Response, TaskResult
    from openai import AsyncOpenAI
    from autogen_ext.agents.openai import OpenAIAgent
    from autogen_ext.tools.mcp import McpWorkbench, StdioServerParams, SseServerParams
except ImportError:
    raise ImportError(
        "Please install the autogen-agentchat package to use this module."
    )
from charge.clients.client import Client
from typing import Type, Optional, Union, List, Callable
from loguru import logger
from pydantic import BaseModel
import json

_POSSIBLE_CONNECTION_ERRORS: List[Type[Exception]] = [ConnectionError]

try:
    from openai._exceptions import (
        APIConnectionError,
        AuthenticationError,
        NotFoundError,
    )

    _POSSIBLE_CONNECTION_ERRORS += [
        APIConnectionError,
        AuthenticationError,
        NotFoundError,
    ]
except ImportError:
    pass


try:
    from ollama import RequestError

    _POSSIBLE_CONNECTION_ERRORS.append(RequestError)
except ImportError:
    pass

POSSIBLE_CONNECTION_ERRORS = tuple(_POSSIBLE_CONNECTION_ERRORS)


class ReasoningModelContext(UnboundedChatCompletionContext):
    """A model context for reasoning models."""

    def __init__(self, name, callback: Optional[Callable] = None):
        super().__init__()
        self.name = name
        self.custom_callback = callback
        if callback:
            callback.name = name

    async def get_messages(self) -> List[LLMMessage]:
        messages = await super().get_messages()
        # Filter out thought field from AssistantMessage.
        messages_out: List[LLMMessage] = []
        for message in messages:
            messages_out.append(message)
        return messages_out

    async def add_message(
        self,
        message: LLMMessage,
    ) -> None:
        await super().add_message(message)

        if self.custom_callback:
            self.custom_callback(message)

        else:
            self.default_callback(message)

    def default_callback(self, assistant_message: LLMMessage):
        # print("In callback:", assistant_message)

        source = self.name
        if assistant_message.type == "UserMessage":
            print(f"[{source}] User: {assistant_message.content}")
        elif assistant_message.type == "AssistantMessage":

            if assistant_message.thought is not None:
                print(f"{source} Model thought: {assistant_message.thought}")
            if isinstance(assistant_message.content, list):
                for item in assistant_message.content:
                    if hasattr(item, "name") and hasattr(item, "arguments"):
                        print(
                            f"[{source}] Function call: {item.name} with args {item.arguments}"
                        )
                    else:
                        print(f"[{source}] Model: {item}")
            else:
                print(f"[{source}] Model: {assistant_message.content}")
        elif assistant_message.type == "FunctionExecutionResultMessage":

            for result in assistant_message.content:
                if result.is_error:
                    print(
                        f"Function {result.name} errored with output: {result.content}"
                    )
                else:
                    print(f"Function {result.name} returned: {result.content}")
        else:
            print(
                "[{source}] Model: ",
                getattr(assistant_message, "message", assistant_message.content),
            )


def generate_agent(
    model_client: Union[AsyncOpenAI, ChatCompletionClient],
    model: str,
    system_prompt: str,
    workbenches: List[McpWorkbench],
    max_tool_calls: int,
    name: str = "Assistant",
    callback: Optional[Callable] = None,
    output_content_type: Optional[Type[BaseModel]] = None,
    **kwargs,
):
    if isinstance(model_client, AsyncOpenAI):
        raise ValueError("ERROR: Incomplete Response API lacks tools.")
        agent = OpenAIAgent(
            name="Assistant",
            description="ChARGe OpenAIAgent",
            client=model_client,
            model=model,
            instructions=system_prompt,
        )
    elif isinstance(model_client, ChatCompletionClient):
        # TODO: Convert this to use custom agent in the future
        agent = AssistantAgent(
            name=name,
            model_client=model_client,
            system_message=system_prompt,
            workbench=workbenches if len(workbenches) > 0 else None,
            max_tool_iterations=max_tool_calls,
            reflect_on_tool_use=True,
            model_context=ReasoningModelContext(name, callback),
            **kwargs,
            output_content_type=output_content_type,
        )
    else:
        raise ValueError("ERROR: Unknown model client type.")

    return agent


async def _list_wb_tools(workbenches: List[McpWorkbench]):
    tool_list = []
    for wb in workbenches:
        tools = await wb.list_tools()
        server_params = wb._server_params
        if isinstance(server_params, SseServerParams):
            msg = server_params.url
        elif isinstance(server_params, StdioServerParams):
            msg = " ".join(server_params.args)
        else:
            msg = "Unknown server params"
        logger.info(f"Workbench: {msg}")
        for tool in tools:
            name = tool["name"]
            logger.info(f"\tTool: {name}")
            tool_list.append((name, msg))

    return tool_list


# Report on which tools are available
async def list_client_tools(
    client: Client,
):
    workbenches: List[McpWorkbench] = [
        McpWorkbench(server) for server in client.servers
    ]

    if not workbenches:
        raise ValueError(f"ERROR: client has no tools.")

    return await _list_wb_tools(workbenches)


async def CustomConsole(stream, message_callback):
    last_processed = None
    async for message in stream:
        last_processed = await message_callback(message)
    return last_processed


async def cli_chat_callback(message):
    if isinstance(message, TaskResult):
        return message
    elif isinstance(message, Response):
        content = message.chat_message.to_text()
        await aprint(content, end="", flush=True)
        return content
    else:
        if message.source != "user":
            await aprint(
                f"{'-' * 10} {message.__class__.__name__} ({message.source}) {'-' * 10}",
                end="\n",
                flush=True,
            )
            await aprint(message.to_text(), end="", flush=True)
        return None


class ChARGeListMemory(ListMemory):
    """A ListMemory that can store arbitrary objects as memory content."""

    def __init__(
        self,
        name: Optional[str] = None,
        memory_contents: Optional[List["MemoryContent"]] = None,
    ):
        super().__init__(name=name, memory_contents=memory_contents)
        self.source_agent = []

    async def update_context(
        self,
        model_context: ChatCompletionContext,
    ) -> UpdateContextResult:
        """Update the model context by appending memory content.

        This method mutates the provided model_context by adding all memories as a
        SystemMessage.

        Args:
            model_context: The context to update. Will be mutated if memories exist.

        Returns:
            UpdateContextResult containing the memories that were added to the context
        """

        if not self._contents:
            return UpdateContextResult(memories=MemoryQueryResult(results=[]))

        for i, (memory, source) in enumerate(zip(self._contents, self.source_agent)):
            if i == 0:
                content = "\nRelevant memory content (in chronological order):\n"
            else:
                content = "\n"
            content += f"[From Agent: {source}] {memory.content}"
            await model_context.add_message(
                AssistantMessage(content=content, source=source)
            )

        return UpdateContextResult(memories=MemoryQueryResult(results=self._contents))

    async def add(
        self,
        content: MemoryContent,
        cancellation_token: Optional[CancellationToken] = None,
        source_agent: Optional[str] = None,
    ) -> None:
        """Add a new memory content to the list.

        Args:
            memory_content: The memory content to add.
            source_agent: The source agent of the memory content.
        """

        self._contents.append(content)
        self.source_agent.append(source_agent or "Agent")

    def add_sync(
        self,
        content: MemoryContent,
        cancellation_token: Optional[CancellationToken] = None,
        source_agent: Optional[str] = None,
    ) -> None:
        """Add a new memory content to the list.

        Args:
            memory_content: The memory content to add.
            source_agent: The source agent of the memory content.
        """

        self._contents.append(content)
        self.source_agent.append(source_agent or "Agent")

    def serialize_memory_content(self) -> str:
        """Serialize the memory contents to a JSON string.

        Returns:
            str: JSON string representation of the memory contents.
        """

        memory_dicts = []
        for memory, source in zip(self._contents, self.source_agent):
            serialized_memory_content = memory.model_dump()
            source_agent = {"source_agent": source}
            memory_dict = {**serialized_memory_content, **source_agent}
            memory_dicts.append(memory_dict)

        _str = json.dumps(memory_dicts, indent=2)

        return _str

    def load_memory_content(self, json_str: str) -> None:
        """Load memory contents from a JSON string.

        Args:
            json_str: JSON string representation of the memory contents.
        """

        memory_dicts = json.loads(json_str)

        self._contents = []
        self.source_agent = []
        for memory_dict in memory_dicts:
            source_agent = memory_dict.pop("source_agent", "Agent")
            memory_content = MemoryContent.model_validate(memory_dict)
            self._contents.append(memory_content)
            self.source_agent.append(source_agent)
