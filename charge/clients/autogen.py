################################################################################
## Copyright 2025 Lawrence Livermore National Security, LLC. and Binghamton University.
## See the top-level LICENSE file for details.
##
## SPDX-License-Identifier: Apache-2.0
################################################################################
try:
    from autogen_agentchat.agents import AssistantAgent, UserProxyAgent
    from autogen_core.model_context import UnboundedChatCompletionContext
    from autogen_core.models import (
        ModelFamily,
        ChatCompletionClient,
        LLMMessage,
        AssistantMessage,
    )
    from openai import AsyncOpenAI

    # from autogen_ext.agents.openai import OpenAIAgent
    from autogen_ext.tools.mcp import StdioServerParams, McpWorkbench, SseServerParams
    from autogen_agentchat.messages import TextMessage, ThoughtEvent
    from autogen_agentchat.conditions import HandoffTermination, TextMentionTermination
    from autogen_agentchat.teams import RoundRobinGroupChat
    from autogen_agentchat.ui import Console
    from autogen_agentchat.conditions import TextMentionTermination
except ImportError:
    raise ImportError(
        "Please install the autogen-agentchat package to use this module."
    )

import asyncio
from functools import partial
import os
from charge.clients.Client import Client
from charge.clients.autogen_utils import generate_agent, list_client_tools
from typing import Type, Optional, Dict, Union, List, Callable
from charge.tasks.Task import Task
from loguru import logger


class AutoGenClient(Client):
    def __init__(
        self,
        task: Task,
        path: str = ".",
        max_retries: int = 3,
        backend: str = "openai",
        model: str = "gpt-4",
        model_client: Optional[Union[AsyncOpenAI, ChatCompletionClient]] = None,
        api_key: Optional[str] = None,
        model_info: Optional[dict] = None,
        model_kwargs: Optional[dict] = None,
        server_path: Optional[Union[str, list[str]]] = None,
        server_url: Optional[Union[str, list[str]]] = None,
        server_kwargs: Optional[dict] = None,
        max_tool_calls: int = 30,
        check_response: bool = False,
        max_multi_turns: int = 100,
        mcp_timeout: int = 60,
        thoughts_callback: Optional[Callable] = None,
    ):
        """Initializes the AutoGenClient.

        Args:
            task (Type[Task]): The task class to use.
            path (str, optional): Path to save generated MCP server files. Defaults to ".".
            max_retries (int, optional): Maximum number of retries for failed tasks. Defaults to 3.
            backend (str, optional): Backend to use: "openai", "gemini", "ollama", "liveai" or "livchat". Defaults to "openai".
            model (str, optional): Model name to use. Defaults to "gpt-4".
            model_client (Optional[ChatCompletionClient], optional): Pre-initialized model client. If provided, `backend`, `model`, and `api_key` are ignored. Defaults to None.
            api_key (Optional[str], optional): API key for the model. Defaults to None.
            model_info (Optional[dict], optional): Additional model info. Defaults to None.
            model_kwargs (Optional[dict], optional): Additional keyword arguments for the model client.
                                                     Defaults to None.
            server_path (Optional[Union[str, list[str]]], optional): Path or list of paths to existing MCP server script. If provided, this
                                                   server will be used instead of generating
                                                   new ones. Defaults to None.
            server_url (Optional[Union[str, list[str]]], optional): URL or list URLs of existing MCP server over the SSE transport.
                                                  If provided, this server will be used instead of generating
                                                  new ones. Defaults to None.
            server_kwargs (Optional[dict], optional): Additional keyword arguments for the server client. Defaults to None.
            max_tool_calls (int, optional): Maximum number of tool calls per task. Defaults to 15.
            check_response (bool, optional): Whether to check the response using verifier methods.
                                             Defaults to False (Will be set to True in the future).
            max_multi_turns (int, optional): Maximum number of multi-turn interactions. Defaults to 100.
            mcp_timeout (int, optional): Timeout in seconds for MCP server responses. Defaults to 60 s.
            thoughts_callback (Optional[Callable], optional): Optional callback function to handle model thoughts.
                                                            Defaults to None.
        Raises:
            ValueError: If neither `server_path` nor `server_url` is provided and MCP servers cannot be generated.
        """
        super().__init__(task, path, max_retries)
        self.backend = backend
        self.model = model
        self.api_key = api_key
        self.model_info = model_info
        self.model_kwargs = model_kwargs if model_kwargs is not None else {}
        self.max_tool_calls = max_tool_calls
        self.check_response = check_response
        self.max_multi_turns = max_multi_turns
        self.mcp_timeout = mcp_timeout
        self.thoughts_callback = thoughts_callback

        if model_client is not None:
            self.model_client = model_client
        else:
            model_info = {
                "vision": False,
                "function_calling": True,
                "json_output": True,
                "family": ModelFamily.UNKNOWN,
                "structured_output": True,
            }
            if backend == "ollama":
                from autogen_ext.models.ollama import OllamaChatCompletionClient

                self.model_client = OllamaChatCompletionClient(
                    model=model,
                    model_info=model_info,
                )
            else:
                if api_key is None:
                    if backend == "gemini":
                        api_key = os.getenv("GOOGLE_API_KEY")
                    else:
                        api_key = os.getenv("OPENAI_API_KEY")
                assert (
                    api_key is not None
                ), "API key must be provided for OpenAI or Gemini backend"

                # Disabled due to https://github.com/microsoft/autogen/issues/6937
                # if backend in ["openai", "livai", "livchat"]:
                #     self.model_client = AsyncOpenAI(
                #         **self.model_kwargs,
                #     )
                # else:
                from autogen_ext.models.openai import OpenAIChatCompletionClient

                self.model_client = OpenAIChatCompletionClient(
                    model=model,
                    api_key=api_key,
                    model_info=model_info,
                    **self.model_kwargs,
                )

        if server_path is None and server_url is None:
            self.setup_mcp_servers()
        else:
            if server_path is not None:
                if isinstance(server_path, str):
                    server_path = [server_path]
                for sp in server_path:
                    self.servers.append(
                        StdioServerParams(
                            command="python3",
                            args=[sp],
                            read_timeout_seconds=self.mcp_timeout,
                        )
                    )
            if server_url is not None:
                if isinstance(server_url, str):
                    server_url = [server_url]
                for su in server_url:
                    self.servers.append(
                        SseServerParams(
                            url=su,
                            timeout=self.mcp_timeout,
                            sse_read_timeout=self.mcp_timeout,
                            **(server_kwargs or {}),
                        )
                    )

    @staticmethod
    def configure(
        model: Optional[str], backend: str
    ) -> (str, str, str, Dict[str, str]):
        import httpx

        kwargs = {}
        API_KEY = None
        default_model = None
        if backend in ["openai", "gemini", "livai", "livchat"]:
            if backend == "openai":
                API_KEY = os.getenv("OPENAI_API_KEY")
                default_model = "gpt-5"
                # kwargs["parallel_tool_calls"] = False
                kwargs["reasoning_effort"] = "high"
            elif backend == "livai" or backend == "livchat":
                API_KEY = os.getenv("OPENAI_API_KEY")
                BASE_URL = os.getenv("LIVAI_BASE_URL")
                assert (
                    BASE_URL is not None
                ), "LivAI Base URL must be set in environment variable"
                default_model = "gpt-4.1"
                kwargs["base_url"] = BASE_URL
                kwargs["http_client"] = httpx.AsyncClient(verify=False)
            else:
                API_KEY = os.getenv("GOOGLE_API_KEY")
                default_model = "gemini-flash-latest"
                kwargs["parallel_tool_calls"] = False
                kwargs["reasoning_effort"] = "high"
        elif backend in ["ollama"]:
            default_model = "gpt-oss:latest"

        if not model:
            model = default_model
        return (model, backend, API_KEY, kwargs)

    def check_invalid_response(self, result) -> bool:
        answer_invalid = False
        for method in self.verifier_methods:
            try:
                is_valid = method(result.messages[-1].content)
                if not is_valid:
                    answer_invalid = True
                    break
            except Exception as e:
                print(f"Error during verification with {method.__name__}: {e}")
                answer_invalid = True
                break
        return answer_invalid

    async def step(self, agent, task: str):
        result = await agent.run(task=task)

        for msg in result.messages:
            if isinstance(msg, TextMessage):
                self.messages.append(msg.content)

        if not self.check_response:
            assert isinstance(result.messages[-1], TextMessage)
            return False, result

        answer_invalid = False
        if isinstance(result.messages[-1], TextMessage):
            answer_invalid = self.check_invalid_response(result.messages[-1].content)
        else:
            answer_invalid = True
        retries = 0
        while answer_invalid and retries < self.max_retries:
            new_user_prompt = (
                "The previous response was invalid. Please try again.\n\n" + task
            )
            # print("Retrying with new prompt...")
            result = await agent.run(task=new_user_prompt)
            if isinstance(result.messages[-1], TextMessage):
                answer_invalid = self.check_invalid_response(
                    result.messages[-1].content
                )
            else:
                answer_invalid = True
            retries += 1
        return answer_invalid, result

    async def run(self):
        system_prompt = self.task.get_system_prompt()
        user_prompt = self.task.get_user_prompt()
        structured_output_schema = None
        if self.task.has_structured_output_schema():
            structured_output_schema = (
                self.task.get_structured_output_schema()
            )

        assert (
            user_prompt is not None
        ), "User prompt must be provided for single-turn run."

        workbenches = [McpWorkbench(server) for server in self.servers]

        # Report on which tools are available
        await list_client_tools(self)

        # Start the servers

        await asyncio.gather(*[workbench.start() for workbench in workbenches])

        try:
            agent = generate_agent(
                self.model_client,
                self.model,
                system_prompt,
                workbenches,
                self.max_tool_calls,
                self.thoughts_callback,
            )
            answer_invalid, result = await self.step(agent, user_prompt)

        finally:

            await asyncio.gather(*[workbench.stop() for workbench in workbenches])

        if answer_invalid:
            # Maybe convert this to a warning and let the user handle it
            # warnings.warn("Failed to get a valid response after maximum retries.")
            # return None
            raise ValueError("Failed to get a valid response after maximum retries.")
        else:
            if structured_output_schema is not None:
                # Parse the output using the structured output schema
                assert isinstance(result.messages[-1], TextMessage)
                content = result.messages[-1].content
                try:
                    parsed_output = structured_output_schema.model_validate_json(
                        content
                    )
                    return parsed_output
                except Exception as e:
                    # warnings.warn(f"Failed to parse output: {e}")
                    # return result.messages[-1].content

                    # We could also potentially reprompt the model to fix the output
                    # but for now, we just raise an error
                    raise ValueError(f"Failed to parse output: {e}")
            # gracefully close the
            await agent.close()
            return result.messages[-1].content

    async def chat(self):
        system_prompt = self.task.get_system_prompt()

        handoff_termination = HandoffTermination(target="user")
        # Define a termination condition that checks for a specific text mention.
        text_termination = TextMentionTermination("TERMINATE")

        assert (
            len(self.servers) > 0
        ), "No MCP servers available. Please provide server_path or server_url."

        wokbenches = [McpWorkbench(server) for server in self.servers]

        # Start the servers
        for workbench in wokbenches:
            await workbench.start()

        agent = generate_agent(
            self.model_client, self.model, system_prompt, [], self.max_tool_calls
        )

        user = UserProxyAgent("USER", input_func=input)
        team = RoundRobinGroupChat(
            [agent, user],
            max_turns=self.max_multi_turns,
            # termination_condition=text_termination,
        )

        result = team.run_stream()
        await Console(result)
        for workbench in wokbenches:
            await workbench.stop()

        await self.model_client.close()

    async def refine(self, feedback: str):
        raise NotImplementedError(
            "TODO: Multi-turn refine currently not supported. - S.Z."
        )
