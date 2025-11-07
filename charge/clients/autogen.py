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
        ModelInfo,
    )
    from openai import AsyncOpenAI

    # from autogen_ext.agents.openai import OpenAIAgent
    from autogen_ext.tools.mcp import StdioServerParams, McpWorkbench, SseServerParams
    from autogen_agentchat.messages import TextMessage
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

    async def run(self, **kwargs) -> Any:
        """
        Runs the agent.
        """

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
                user_prompt += (
                    "\n\n Please provide the answer in the following JSON format: "
                    + f"{self.task.get_structured_output_schema().model_json_schema()}\n\n"
                )

            for i in range(self.max_retries):
                result = await agent.run(task=user_prompt)

                self.context_history.append(result)

                if isinstance(result.messages[-1], TextMessage):
                    content = result.messages[-1].content
                    if self.task.check_output_formatting(content):
                        break
                else:
                    warnings.warn(
                        f"Last message is not a TextMessage. Retrying... {result.messages[-1]}\n"
                        + f"Remaining retries: {self.max_retries - i - 1}"
                    )

        finally:
            await self.close_workbenches()
        return result.messages[-1].content

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
