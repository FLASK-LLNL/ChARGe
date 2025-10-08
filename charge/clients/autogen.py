################################################################################
## Copyright 2025 Lawrence Livermore National Security, LLC. and Binghamton University.
## See the top-level LICENSE file for details.
##
## SPDX-License-Identifier: Apache-2.0
################################################################################
try:
    from autogen_agentchat.agents import UserProxyAgent

    from autogen_core.models import (
        ModelFamily,
        ChatCompletionClient,
        LLMMessage,
        AssistantMessage,
        ModelInfo,
    )
    from openai import AsyncOpenAI

    # from autogen_ext.agents.openai import OpenAIAgent
    from autogen_ext.tools.mcp import StdioServerParams, McpWorkbench, SseServerParams
    from autogen_agentchat.messages import TextMessage, StructuredMessage
    from autogen_agentchat.conditions import HandoffTermination, TextMentionTermination
    from autogen_agentchat.teams import RoundRobinGroupChat
    from autogen_agentchat.conditions import TextMentionTermination

except ImportError:
    raise ImportError(
        "Please install the autogen-agentchat package to use this module."
    )

import asyncio
from functools import partial
import os
import warnings
from charge.clients.AgentPool import AgentPool, Agent
from charge.clients.Client import Client
from charge.clients.huggingface_client import HuggingFaceLocalClient
from charge.clients.vllm_client import VLLMClient
from charge.clients.autogen_utils import (
    _list_wb_tools,
    generate_agent,
    list_client_tools,
    CustomConsole,
    cli_chat_callback,
)
from typing import Any, Tuple, Type, Optional, Dict, Union, List, Callable, overload
from charge.tasks.Task import Task
from loguru import logger


def model_configure(
    backend: str,
    model: Optional[str] = None,
) -> Tuple[str, str, Optional[str], Dict[str, str]]:
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
        assert API_KEY is not None, f"API key must be set for backend {backend}"
    elif backend in ["ollama"]:
        default_model = "gpt-oss:latest"
    elif backend in ["huggingface"]:
        default_model = None  # Must be provided via model path

    if not model:
        model = default_model
    assert model is not None, "Model name must be provided."
    return (model, backend, API_KEY, kwargs)


def create_autogen_model_client(
    backend: str,
    model: str,
    api_key: Optional[str] = None,
    model_kwargs: Optional[dict[str, Any]] = None,
) -> Union[AsyncOpenAI, ChatCompletionClient]:
    """
    Creates an AutoGen model client based on the specified backend and model.

    Args:
        backend (str): The backend to use: "openai", "gemini", "ollama", "huggingface", "liveai" or "livchat".
        model (str): The model name to use.
        api_key (Optional[str], optional): API key for the model. Defaults to None.
        model_kwargs (Optional[dict], optional): Additional keyword arguments for the model client. Defaults to None.
    Returns:
        Union[AsyncOpenAI, ChatCompletionClient]: The created model client.
    """
    if model_kwargs is None:
        model_kwargs = {}
    
    model_info = ModelInfo(
        vision=False,
        function_calling=True,
        json_output=True,
        family=ModelFamily.UNKNOWN,
        structured_output=True,
    )
    if backend == "ollama":
        from autogen_ext.models.ollama import OllamaChatCompletionClient

        model_client = OllamaChatCompletionClient(
            model=model,
            model_info=model_info,
        )
    elif backend == "huggingface":
        # Extract HuggingFace-specific kwargs
        model_path = model_kwargs.pop("local_model_path", model)
        device = model_kwargs.pop("device", "auto")
        torch_dtype = model_kwargs.pop("torch_dtype", "auto")
        quantization = model_kwargs.pop("quantization", "4bit")
        
        model_client = HuggingFaceLocalClient(
            model_path=model_path,
            model_info=model_info,
            device=device,
            torch_dtype=torch_dtype,
            quantization=quantization,
            **model_kwargs,
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

        model_client = OpenAIChatCompletionClient(
            model=model,
            api_key=api_key,
            model_info=model_info,
            **model_kwargs,
        )
    return model_client


class AutoGenAgent(Agent):
    """
    An AutoGen agent that interacts with MCP servers and runs tasks.


    Args:
        task (Task): The task to be performed by the agent.
        **kwargs: Additional keyword arguments.
    """

    def __init__(
        self,
        task: Task,
        model_client: Union[AsyncOpenAI, ChatCompletionClient],
        agent_name: str,
        memory: Optional[Any] = None,
        max_retries: int = 3,
        max_tool_calls: int = 30,
        timeout: int = 60,
        **kwargs,
    ) -> None:
        super().__init__(task=task, **kwargs)
        self.max_retries = max_retries
        self.max_tool_calls = max_tool_calls
        self.no_tools = False
        self.workbenches = []
        self.agent_name = agent_name
        self.model_client = model_client
        self.timeout = timeout
        self.memory = memory
        self.setup_kwargs = kwargs

    def create_servers(self, paths: List[str], urls: List[str]) -> List[Any]:
        """
        Creates MCP servers from the task's server paths.

        Returns:
            List[Any]: List of MCP server parameters.
        """
        mcp_servers = []

        for path in paths:
            mcp_servers.append(
                StdioServerParams(
                    command="python3",
                    args=[path],
                    read_timeout_seconds=self.timeout,
                )
            )
        for url in urls:
            mcp_servers.append(
                SseServerParams(
                    url=url,
                    timeout=self.timeout,
                    sse_read_timeout=self.timeout,
                )
            )
        return mcp_servers

    async def setup_mcp_workbenches(self) -> None:
        """
        Sets up MCP workbenches from the task's server paths.

        Returns:
            None
        """
        mcp_files = self.task.server_files
        mcp_urls = self.task.server_urls

        self.mcps = self.create_servers(mcp_files, mcp_urls)

        if len(self.mcps) == 0:
            self.no_tools = True
            return
        self.workbenches = [McpWorkbench(server) for server in self.mcps]

        await asyncio.gather(*[workbench.start() for workbench in self.workbenches])
        await _list_wb_tools(self.workbenches)

    async def close_workbenches(self) -> None:
        """
        Closes MCP workbenches.

        Returns:
            None
        """
        if self.no_tools:
            return
        await asyncio.gather(*[workbench.stop() for workbench in self.workbenches])

    async def run(self, **kwargs) -> str:
        """
        Runs the agent.


        Returns:
            str: The output content from the agent. If structured output is enabled,
                 the output will be checked with the task's formatting method and
                 the json string will be returned.
        """
        content = ""

        # set up workbenches from task server paths
        await self.setup_mcp_workbenches()
        try:
            agent = generate_agent(
                self.model_client,
                self.agent_name,
                self.task.get_system_prompt(),
                self.workbenches,
                max_tool_calls=self.max_tool_calls,
                memory=self.memory,
                **self.setup_kwargs,
            )
            user_prompt = self.task.get_user_prompt()
            if self.task.has_structured_output_schema():
                structured_out = self.task.get_structured_output_schema()
                assert structured_out is not None
                schema = structured_out.model_json_schema()
                keys = list(schema["properties"].keys())

                user_prompt += (
                    f"The output must be formatted correctly according to the schema {schema}"
                    + "Do not return the schema, only return the values as a JSON object."
                    + "\n\nPlease provide the answer as a JSON object with the following keys: "
                    + f"{keys}\n\n"
                )

            for i in range(self.max_retries):
                result = await agent.run(task=user_prompt)

                self.context_history.append(result)

                if isinstance(result.messages[-1], TextMessage):
                    proposed_content = result.messages[-1].content

                    if self.task.has_structured_output_schema():
                        # Use a new agent to convert the output to the structured format
                        try:
                            structured_output_agent = generate_agent(
                                self.model_client,
                                self.agent_name + "_STRUCTURED_OUTPUT_AGENT",
                                "You are an agent that converts model output to a structured format.",
                                [],
                                max_tool_calls=1,
                                memory=self.memory,
                                output_content_type=self.task.get_structured_output_schema(),
                            )
                            structured_prompt = (
                                "Convert the following output to the required structured format:\n\n"
                                + proposed_content
                            )
                            structured_result = await structured_output_agent.run(
                                task=structured_prompt
                            )

                            if structured_result:
                                if isinstance(
                                    structured_result.messages[-1], TextMessage
                                ):
                                    proposed_content = structured_result.messages[
                                        -1
                                    ].content
                                elif isinstance(
                                    structured_result.messages[-1], StructuredMessage
                                ):
                                    proposed_content = structured_result.messages[
                                        -1
                                    ].content.model_dump_json()
                                else:
                                    raise ValueError(
                                        "Structured output agent did not return a TextMessage."
                                    )
                        except Exception as e:
                            warnings.warn(
                                f"Error occurred while converting to structured format: {e}"
                            )
                    if self.task.check_output_formatting(proposed_content):
                        content = proposed_content

                        break
                    else:
                        warnings.warn(
                            f"Output formatting check failed. Retrying...\nProposed content: {proposed_content}\n"
                            + f"Remaining retries: {self.max_retries - i - 1}"
                        )

                        # TODO: Add feedback to the agent here - S.Z
                else:
                    warnings.warn(
                        f"Last message is not a TextMessage. Retrying... {result.messages[-1]}\n"
                        + f"Remaining retries: {self.max_retries - i - 1}"
                    )

        finally:
            await self.close_workbenches()
        return content

    async def chat(
        self,
        input_callback: Optional[Callable[[], str]] = None,
        output_callback: Optional[Callable] = cli_chat_callback,
        **kwargs,
    ) -> Any:
        """
        Starts a chat session with the agent.

        Args:
            output_callback (Optional[Callable], optional): Optional callback function to handle model output.
                                                            Defaults to the cli_chat_callback function. This allows capturing model outputs in a custom
                                                            callback such as printing to console or logging to a file
                                                            or websocket. Default is std.out.

        Returns:
            The state is returned as a nested dictionary: a dictionary with key agent_states,
            which is a dictionary the agent names as keys and the state as values.
        """
        agent_state = {}
        await self.setup_mcp_workbenches()
        try:
            agent = generate_agent(
                self.model_client,
                self.agent_name,
                self.task.get_system_prompt(),
                self.workbenches,
                max_tool_calls=self.max_tool_calls,
            )

            _input = (
                input_callback() if input_callback is not None else input("\nUser: ")
            )
            team = RoundRobinGroupChat(
                [agent],
                max_turns=1,
            )

            stop_signal = False

            while not stop_signal:
                stream = team.run_stream(task=_input, output_task_messages=False)
                await CustomConsole(
                    stream,
                    message_callback=(
                        cli_chat_callback
                        if output_callback is None
                        else output_callback
                    ),
                )
                print("\n" + "-" * 45)
                _input = (
                    input_callback()
                    if input_callback is not None
                    else input("\nUser: ")
                )
                if _input.lower().strip() in ["exit", "quit"]:
                    team_state = await team.save_state()
                    agent_state = team_state
                    stop_signal = True

        finally:
            await self.close_workbenches()
            await self.model_client.close()
        return agent_state

    def get_context_history(self) -> list:
        """
        Returns the context history of the agent.
        """
        return self.context_history


class AutoGenPool(AgentPool):
    """
    An AutoGen agent pool that creates AutoGen agents.
    Setup with a model client, backend, and model to spawn agents.

    Args:
        model_client (Union[AsyncOpenAI, ChatCompletionClient]): The model client to use.
        model (str): The model name to use.
        backend (str, optional): Backend to use: "openai", "gemini", "ollama", "huggingface", "liveai" or "livchat". Defaults to "openai".
    """

    AGENT_COUNT = 0

    @overload
    def __init__(
        self,
        model_client: Union[AsyncOpenAI, ChatCompletionClient],
    ) -> None: ...

    @overload
    def __init__(
        self,
        *,
        model: str,
        backend: str = "openai",
        model_kwargs: Optional[dict] = None,
    ) -> None: ...

    def __init__(
        self,
        model_client: Optional[Union[AsyncOpenAI, ChatCompletionClient]] = None,
        model: Optional[str] = None,
        backend: Optional[str] = "openai",
        model_kwargs: Optional[dict] = None,
    ):
        super().__init__()
        self.model_client = model_client

        if self.model_client is None:
            assert (
                model is not None
            ), "Model name must be provided if model_client is not given."
            assert (
                backend is not None
            ), "Backend must be provided if model_client is not given."

            model, backend, api_key, model_kwargs = model_configure(
                model=model, backend=backend
            )
            self.model_client = create_autogen_model_client(
                backend=backend, model=model, api_key=api_key, model_kwargs=model_kwargs
            )
        if self.model_client is None:
            raise ValueError("Failed to create model client.")

    def create_agent(
        self,
        task: Task,
        max_retries: int = 3,
        agent_name: Optional[str] = None,
        **kwargs,
    ):
        """Creates an AutoGen agent for the given task.
        Args:
            task (Task): The task to be performed by the agent.
            max_retries (int, optional): Maximum number of retries for failed tasks. Defaults to 3.
            agent_name (Optional[str], optional): Name of the agent. If None, a default name will be assigned. Defaults to None.
            **kwargs: Additional keyword arguments.
        Returns:
            AutoGenAgent: The created AutoGen agent.
        """
        self.max_retries = max_retries
        assert (
            self.model_client is not None
        ), "Model client must be initialized to create an agent."

        AutoGenPool.AGENT_COUNT += 1
        agent_name = (
            f"CHARGE_AUTOGEN_AGENT_{AutoGenPool.AGENT_COUNT}"
            if agent_name is None
            else agent_name
        )

        if agent_name in self.agent_list:
            warnings.warn(
                f"Agent with name {agent_name} already exists. Creating another agent with the same name."
            )
        else:
            self.agent_list.append(agent_name)

        agent = AutoGenAgent(
            task=task,
            model_client=self.model_client,
            agent_name=agent_name,
            max_retries=max_retries,
            **kwargs,
        )
        self.agent_dict[agent_name] = agent
        return agent

    def list_all_agents(self) -> list:
        """Lists all agents in the pool.

        Returns:
            list: List of agent names.
        """
        return self.agent_list

    def get_agent_by_name(self, name: str) -> AutoGenAgent:
        """Gets an agent by name.

        Args:
            name (str): The name of the agent.

        Returns:
            AutoGenAgent: The agent with the given name.
        """
        assert name in self.agent_dict, f"Agent with name {name} does not exist."
        return self.agent_dict[name]
