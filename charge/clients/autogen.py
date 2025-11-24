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
import re
import os
import warnings
from charge.clients.AgentPool import AgentPool, Agent
from charge.clients.Client import Client
from charge.clients.autogen_utils import (
    ChARGeListMemory,
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
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
) -> Tuple[str, str, Optional[str], Dict[str, str]]:
    """
    Raises:
        ValueError: If API key does not exist and is needed.

    """
    import httpx

    kwargs = {}
    default_model = None
    if backend in ["openai", "gemini", "livai", "livchat"]:
        if backend == "openai":
            if not api_key:
                api_key = os.getenv("OPENAI_API_KEY")
            if not base_url:
                kwargs["base_url"] = base_url
            default_model = "gpt-5"
            # kwargs["parallel_tool_calls"] = False
            kwargs["reasoning_effort"] = "high"
        elif backend == "livai" or backend == "livchat":
            if not api_key:
                api_key = os.getenv("LIVAI_API_KEY")
            if not base_url:
                base_url = os.getenv("LIVAI_BASE_URL")
                if base_url is None:
                    raise ValueError(f"LivAI Base URL must be set in environment variable for backend {backend}")
            default_model = "gpt-4.1"
            kwargs["base_url"] = base_url
            kwargs["http_client"] = httpx.AsyncClient(verify=False)
        elif backend == "LLamaMe":
            if not api_key:
                api_key = os.getenv("LLAMAME_API_KEY")
            if not base_url:
                base_url = os.getenv("LLAMAME_BASE_URL")
                if base_url is None:
                    raise ValueError(f"LLamaMe Base URL must be set in environment variable for backend {backend}")
            default_model = "openai/gpt-oss-120b "
            kwargs["base_url"] = base_url
            # kwargs["http_client"] = httpx.AsyncClient(verify=False)
        else:
            if not api_key:
                api_key = os.getenv("GOOGLE_API_KEY")
            default_model = "gemini-flash-latest"
            if not base_url:
                kwargs["base_url"] = base_url
            kwargs["parallel_tool_calls"] = False
            kwargs["reasoning_effort"] = "high"
        if api_key is None:
            raise ValueError(f"API key must be set for backend {backend}")
    elif backend in ["ollama"]:
        default_model = "gpt-oss:latest"

    if not model:
        model = default_model
    assert model is not None, "Model name must be provided."
    return (model, backend, api_key, kwargs)


def create_autogen_model_client(
    backend: str,
    model: str,
    api_key: Optional[str] = None,
    model_kwargs: Optional[dict[str, Any]] = None,
) -> Union[AsyncOpenAI, ChatCompletionClient]:
    """
    Creates an AutoGen model client based on the specified backend and model.

    Args:
        backend (str): The backend to use: "openai", "gemini", "ollama", "liveai" or "livchat".
        model (str): The model name to use.
        api_key (Optional[str], optional): API key for the model. Defaults to None.
        model_kwargs (Optional[dict], optional): Additional keyword arguments for the model client. Defaults to None.
    Returns:
        Union[AsyncOpenAI, ChatCompletionClient]: The created model client.
    """
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
            **model_kwargs if model_kwargs is not None else {},
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
        model: str,
        memory: Optional[Any] = None,
        max_retries: int = 3,
        max_tool_calls: int = 30,
        timeout: int = 60,
        backend: Optional[str] = None,
        model_kwargs: Optional[dict] = None,
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
        self.memory = self.setup_memory(memory)
        self.setup_kwargs = kwargs

        self.context_history = []
        self.model = model
        self.backend = backend
        self.model_kwargs = model_kwargs if model_kwargs is not None else {}
        self._structured_output_agent: Optional[Any] = None

    def setup_memory(self, memory: Optional[Any] = None) -> List[ChARGeListMemory]:
        """
        Sets up the memory for the agent if not already provided.
        Args:
            memory (Optional[Any], optional): Pre-initialized memory. Defaults to None.
        Returns:
            List[ChARGeListMemory]: The memory instance.
        """
        if memory is not None:
            return memory
        return [ChARGeListMemory()]

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

    def _create_agent(self, **kwargs) -> Any:
        """
        Creates an AutoGen agent with the given parameters.

        Returns:
            Any: The created AutoGen agent.
        """
        return generate_agent(
            self.model_client,
            self.model,
            self.task.get_system_prompt(),
            self.workbenches,
            max_tool_calls=self.max_tool_calls,
            name=self.agent_name,
            memory=self.memory,
            **self.setup_kwargs,
        )

    def _prepare_task_prompt(self, **kwargs) -> str:
        """
        Prepares the task prompt for the agent.

        Returns:
            str: The prepared task prompt.
        """
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
        return user_prompt

    async def _convert_to_structured_format(self, content: str) -> str:
        """
        Converts content to structured format using a dedicated agent.

        Args:
            content: The content to convert.

        Returns:
            The structured content as a JSON string.

        Raises:
            OutputValidationError: If conversion fails.
        """
        try:
            agent = self._create_structured_output_agent()
            prompt = (
                "Convert the following output to the required structured format:\n\n"
                f"{content}"
            )
            result = await agent.run(task=prompt)

            if not result or not result.messages:
                raise ValueError("Structured output agent returned empty result")

            last_message = result.messages[-1]

            if isinstance(last_message, StructuredMessage):
                return last_message.content.model_dump_json()
            elif isinstance(last_message, TextMessage):
                return last_message.content
            else:
                raise ValueError(f"Unexpected message type: {type(last_message)}")

        except Exception as e:
            logger.error(f"Failed to convert to structured format: {e}")
            raise ValueError(f"Structured output conversion failed: {e}") from e

    def _create_structured_output_agent(self) -> Any:
        """Creates an agent for structured output conversion."""
        if self._structured_output_agent is None:
            self._structured_output_agent = generate_agent(
                self.model_client,
                f"{self.agent_name}_structured_output",
                "You are an agent that converts model output to a structured format.",
                [],  # No tools needed
                max_tool_calls=1,
                memory=self.memory,
                output_content_type=self.task.get_structured_output_schema(),
            )
        return self._structured_output_agent

    async def _execute_with_retries(self, agent: Any, user_prompt: str) -> str:
        """
        Executes the agent with retry logic and output validation.

        Args:
            agent: The agent instance to run.
            user_prompt: The prompt to send to the agent.

        Returns:
            Valid output content as a string.

        Raises:
            OutputValidationError: If all retries fail to produce valid output.
        """
        last_error = None

        for attempt in range(1, self.max_retries + 1):
            try:
                logger.info(f"Attempt {attempt}/{self.max_retries}")
                result = await agent.run(task=user_prompt)
                self.context_history.append(result)

                if not result.messages:
                    logger.warning(f"Attempt {attempt}: No messages in result")
                    continue

                last_message = result.messages[-1]

                if not isinstance(last_message, TextMessage):
                    logger.warning(
                        f"Attempt {attempt}: Last message is {type(last_message)}, "
                        f"expected TextMessage"
                    )
                    continue

                proposed_content = last_message.content

                # Convert to structured format if needed
                if self.task.has_structured_output_schema():
                    try:
                        proposed_content = await self._convert_to_structured_format(
                            proposed_content
                        )
                    except Exception as e:
                        logger.warning(
                            f"Attempt {attempt}: Structured conversion failed: {e}"
                        )
                        last_error = e
                        continue

                # Validate output
                if self.task.check_output_formatting(proposed_content):
                    logger.info(f"Valid output obtained on attempt {attempt}")
                    return proposed_content
                else:
                    error_msg = (
                        f"Attempt {attempt}: Output validation failed. "
                        f"Content preview: {proposed_content[:200]}..."
                    )
                    logger.warning(error_msg)
                    last_error = ValueError("Output validation failed")

            except Exception as e:
                error_msg = f"Attempt {attempt}: Unexpected error: {e}"
                logger.error(error_msg)
                last_error = e

        # All retries exhausted
        raise ValueError(
            f"Failed to obtain valid output after {self.max_retries} attempts. "
            f"Last error: {last_error}"
        )

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
            agent = self._create_agent()
            user_prompt = self._prepare_task_prompt()
            result = await self._execute_with_retries(agent, user_prompt)
        finally:
            await self.close_workbenches()

        return result

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
            agent = self._create_agent()
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

    def load_context_history(self, history: list) -> None:
        """
        Loads the context history into the agent.
        """
        self.context_history = history

    def load_memory(self, jston_str: str) -> None:
        """
        Loads memory content into the agent's memory.
        """
        if self.memory is not None:
            if isinstance(self.memory, list):
                for mem in self.memory:
                    mem.load_memory_content(jston_str)
            else:
                self.memory.load_memory_content(jston_str)

    def save_memory(self) -> str:
        """
        Saves the agent's memory content to a JSON string.
        """
        if self.memory is not None:
            if isinstance(self.memory, list):
                combined_content = ""
                for mem in self.memory:
                    combined_content += mem.serialize_memory_content()
                return combined_content
            return self.memory.serialize_memory_content()
        return ""

    def get_model_info(self) -> Dict[str, Any]:
        """
        Returns the model information of the agent.
        """
        return {
            "model": self.model,
            "backend": self.backend,
            "model_kwargs": self.model_kwargs,
        }


class AutoGenPool(AgentPool):
    """
    An AutoGen agent pool that creates AutoGen agents.
    Setup with a model client, backend, and model to spawn agents.

    Args:
        model_client (Union[AsyncOpenAI, ChatCompletionClient]): The model client to use.
        model (str): The model name to use.
        backend (str, optional): Backend to use: "openai", "gemini", "ollama", "liveai" or "livchat". Defaults to "openai".
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
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
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
                model=model, backend=backend, api_key=api_key, base_url=base_url
            )
            self.model_client = create_autogen_model_client(
                backend=backend, model=model, api_key=api_key, model_kwargs=model_kwargs
            )
        self.model = model
        self.backend = backend
        self.model_kwargs = model_kwargs if model_kwargs is not None else {}
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

        model_name = self.model if self.model is not None else "default_model"
        backend_name = self.backend if self.backend is not None else "default_backend"

        default_name = f"_{backend_name}:{model_name}]_{AutoGenPool.AGENT_COUNT}"
        default_name = re.sub(r"[^a-zA-Z0-9_]", "_", default_name)
        agent_name = default_name if agent_name is None else agent_name

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
            model=self.model,  # type: ignore
            backend=self.backend,
            model_kwargs=self.model_kwargs,
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
            self.model_client = create_autogen_model_client(
                backend=backend,
                model=model,
                api_key=api_key,
                model_kwargs=self.model_kwargs,
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
    ) -> Tuple[str, str, Optional[str], Dict[str, str]]:
        return model_configure(model=model, backend=backend)

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
            structured_output_schema = self.task.get_structured_output_schema()

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
