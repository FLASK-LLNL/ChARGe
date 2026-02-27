################################################################################
## Copyright 2025 Lawrence Livermore National Security, LLC. and Binghamton University.
## See the top-level LICENSE file for details.
##
## SPDX-License-Identifier: Apache-2.0
################################################################################
try:

    from autogen_core.models import (
        ModelFamily,
        ChatCompletionClient,
        ModelInfo,
    )
    from openai import AsyncOpenAI

    # from autogen_ext.agents.openai import OpenAIAgent
    from autogen_ext.tools.mcp import StdioServerParams, McpWorkbench, SseServerParams
    from autogen_agentchat.messages import TextMessage, StructuredMessage
    from autogen_agentchat.teams import RoundRobinGroupChat
    from autogen_core.memory import MemoryContent, MemoryMimeType


except ImportError:
    raise ImportError(
        "Please install the autogen-agentchat package to use this module."
    )

import asyncio
import re
import os
import warnings
from charge.clients.agent_factory import AgentBackend, AgentFactory, Agent
from charge.clients.client import Client
from charge.clients.huggingface_client import HuggingFaceLocalClient
from charge.clients.vllm_client import VLLMClient
from charge.clients.autogen_utils import (
    POSSIBLE_CONNECTION_ERRORS,
    ChARGeListMemory,
    _list_wb_tools,
    generate_agent,
    CustomConsole,
    cli_chat_callback,
)
from charge.utils.mcp_workbench_utils import (
    _setup_mcp_workbenches,
    _close_mcp_workbenches,
)
from typing import Any, Tuple, Optional, Dict, Union, List, Callable, overload
from charge.experiments.memory import Memory
from charge.tasks.task import Task
from loguru import logger

import logging
import httpx


# Configure httpx logging for network-level debugging
class LoggingTransport(httpx.AsyncHTTPTransport):
    """Custom transport that logs all HTTP requests and responses."""

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        # Log request details (mask API key)
        headers_copy = dict(request.headers)
        if "authorization" in headers_copy:
            auth_header = headers_copy["authorization"]
            if auth_header.startswith("Bearer "):
                masked_key = auth_header[:14] + "..." + auth_header[-4:]
                headers_copy["authorization"] = masked_key

        # logger.debug(f"HTTP Request: {request.method} {request.url}")
        # logger.debug(f"Request Headers: {headers_copy}")
        try:
            response = await super().handle_async_request(request)
            # logger.debug(f"HTTP Response: {response.status_code}")
            # logger.debug(f"Response Headers: {dict(response.headers)}")
            return response
        except Exception as e:
            logger.error(f"HTTP Request Failed: {type(e).__name__}: {e}")
            raise


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

    kwargs = {}
    default_model = None
    if backend in ["openai", "gemini", "livai", "livchat", "llamame", "alcf"]:
        if backend == "openai":
            if not api_key:
                api_key = os.getenv("OPENAI_API_KEY")
            if base_url:
                kwargs["base_url"] = base_url
            default_model = "gpt-5"
            # kwargs["parallel_tool_calls"] = False
            kwargs["reasoning_effort"] = "high"
            kwargs["http_client"] = httpx.AsyncClient(
                verify=False, transport=LoggingTransport()
            )
            logger.warning(
                f"BVE I am here in the openai backend with the kwargs {kwargs}"
            )
        elif backend == "livai" or backend == "livchat":
            if not api_key:
                api_key = os.getenv("LIVAI_API_KEY")
            if not base_url:
                base_url = os.getenv("LIVAI_BASE_URL")
                if base_url is None:
                    raise ValueError(
                        f"LivAI Base URL must be set in environment variable for backend {backend}"
                    )
            default_model = "gpt-4.1"
            kwargs["base_url"] = base_url
            kwargs["http_client"] = httpx.AsyncClient(
                verify=False, transport=LoggingTransport()
            )
        elif backend == "llamame":
            if not api_key:
                api_key = os.getenv("LLAMAME_API_KEY")
            if not base_url:
                base_url = os.getenv("LLAMAME_BASE_URL")
                if base_url is None:
                    raise ValueError(
                        f"LLamaMe Base URL must be set in environment variable for backend {backend}"
                    )
            default_model = "openai/gpt-oss-120b "
            kwargs["base_url"] = base_url
            kwargs["http_client"] = httpx.AsyncClient(
                verify=False, transport=LoggingTransport()
            )
        elif backend == "alcf":
            if not api_key:
                api_key = os.getenv("ALCF_API_KEY")
            if not base_url:
                base_url = os.getenv("ALCF_BASE_URL")
                if base_url is None:
                    raise ValueError(
                        f"ALCF Base URL must be set in environment variable for backend {backend}"
                    )
            default_model = "openai/gpt-oss-120b "
            kwargs["base_url"] = base_url
            kwargs["http_client"] = httpx.AsyncClient(
                verify=False, transport=LoggingTransport()
            )
        else:
            if not api_key:
                api_key = os.getenv("GOOGLE_API_KEY")
            default_model = "gemini-flash-latest"
            if base_url:
                kwargs["base_url"] = base_url
            kwargs["parallel_tool_calls"] = False
            kwargs["reasoning_effort"] = "high"
            kwargs["http_client"] = httpx.AsyncClient(
                verify=False, transport=LoggingTransport()
            )
        if api_key is None:
            raise ValueError(f"API key must be set for backend {backend}")
    elif backend in ["ollama"]:
        default_model = "gpt-oss:latest"
    elif backend in ["huggingface"]:
        default_model = "gpt-oss"  # Must be provided via model path
    elif backend in ["vllm"]:
        kwargs["reasoning_effort"] = os.getenv("OSS_REASONING", "medium")
        default_model = "gpt-oss"  # Default vLLM model name

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
        backend (str): The backend to use: "openai", "gemini", "ollama", "huggingface", "vllm", "liveai" or "livchat".
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
        model_path = os.getenv("LOCAL_MODEL_PATH", "./models/gpt-oss-20b")
        device = model_kwargs.pop("device", "auto")
        torch_dtype = model_kwargs.pop("torch_dtype", "auto")
        quantization = None  # model_kwargs.pop("quantization", "4bit")

        model_client = HuggingFaceLocalClient(
            model_path=model_path,
            model_info=model_info,
            device=device,
            torch_dtype=torch_dtype,
            quantization=quantization,
            **model_kwargs,
        )
    elif backend == "vllm":
        # Extract vLLM-specific kwargs
        vllm_url = model_kwargs.pop(
            "vllm_url", os.getenv("VLLM_URL", "http://localhost:8000/v1")
        )
        vllm_model = model_kwargs.pop("vllm_model", os.getenv("VLLM_MODEL", model))

        # Set reasoning effort level
        reasoning_effort = model_kwargs.pop("reasoning_effort", "medium")
        assert reasoning_effort in ["low", "medium", "high"]

        # Increase max_tokens for high (to avoid all tokens used on reasoning)
        max_tokens = 8192
        if reasoning_effort == "high":
            max_tokens = 16000

        logger.info(f"\n  ==> GPT-OSS reasoning effort: {reasoning_effort}")

        vllm_model_info = ModelInfo(
            vision=False,
            function_calling=True,
            json_output=False,
            family=ModelFamily.UNKNOWN,
            structured_output=False,
        )

        reasoning_client = False  # True to help with debugging
        if reasoning_client:
            from charge.clients.reasoning import ReasoningCaptureClient

            model_client = ReasoningCaptureClient(
                model=vllm_model,
                api_key="EMPTY",
                base_url=vllm_url,
                model_info=vllm_model_info,
                parallel_tool_calls=False,
                extra_body={
                    "reasoning_effort": reasoning_effort,
                    "max_tokens": max_tokens,
                },
            )
        else:
            from autogen_ext.models.openai import OpenAIChatCompletionClient

            # Use OpenAIChatCompletionClient
            model_client = OpenAIChatCompletionClient(
                model=vllm_model,
                api_key="EMPTY",
                base_url=vllm_url,
                model_info=vllm_model_info,
                parallel_tool_calls=False,  # False for gpt-oss
                extra_body={
                    "reasoning_effort": reasoning_effort,
                    "max_tokens": max_tokens,
                },
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
            **model_kwargs if model_kwargs is not None else {},  # type: ignore
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
        task: Optional[Task],
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
        self.workbenches: List[McpWorkbench] = []
        self.agent_name = agent_name
        self.model_client = model_client
        self.timeout = timeout
        self.memory: list[ChARGeListMemory] | ChARGeListMemory = self.setup_memory(
            memory
        )
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

    async def setup_mcp_workbenches(self) -> None:
        """
        Sets up MCP workbenches from the task's server paths.

        Returns:
            None
        """
        assert self.task is not None
        self.workbenches = await _setup_mcp_workbenches(
            self.task.server_files, self.task.server_urls
        )
        if len(self.workbenches) == 0:
            return
        await asyncio.gather(*[workbench.start() for workbench in self.workbenches])
        await _list_wb_tools(self.workbenches)

    async def close_workbenches(self) -> None:
        """
        Closes MCP workbenches.

        Returns:
            None
        """
        await _close_mcp_workbenches(self.workbenches)

    def _create_agent(self, **kwargs) -> Any:
        """
        Creates an AutoGen agent with the given parameters.

        Returns:
            Any: The created AutoGen agent.
        """
        assert self.task is not None
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
        assert self.task is not None
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
                f"{content}\n\n"
                "If the output is already correctly formatted, return the same output. "
                "Never modify the reasoning_summary field."
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
            assert self.task is not None
            self._structured_output_agent = generate_agent(
                self.model_client,
                f"{self.agent_name}_structured_output",
                "You are an agent that converts model output to a structured format.",
                [],  # No tools needed
                max_tool_calls=1,
                # memory=self.memory,
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
        assert self.task is not None

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
            except POSSIBLE_CONNECTION_ERRORS as api_err:
                error_msg = f"Attempt {attempt}: API connection error: {api_err}"
                logger.error(error_msg)
                raise ConnectionError(error_msg)
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
                logger.info("\n" + "-" * 45)
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

    def load_memory(self, json_str: str) -> None:
        """
        Loads memory content into the agent's memory.
        """
        if self.memory is not None:
            if isinstance(self.memory, list):
                for mem in self.memory:
                    mem.load_memory_content(json_str)
            else:
                self.memory.load_memory_content(json_str)

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

    async def add_to_context(self, task: Task, result: Any):
        # Implement adding the result to the context of the Autogen task

        # This is tricky and should be customized based on the use case
        # We set the default behavior to add the instruction
        #  result to the model context

        instruction = task.get_user_prompt()
        result = str(result)

        content = MemoryContent(
            content=f"Instruction: {instruction}\nResponse: {result}",
            mime_type=MemoryMimeType.TEXT,
        )

        name = self.agent_name or "Agent"
        if isinstance(self.memory, list):
            await self.memory[-1].add(content, source_agent=name)
        else:
            await self.memory.add(content, source_agent=name)

    def add_to_context_sync(self, task: Task, result: Any):
        # Implement adding the result to the context of the Autogen task

        # This is tricky and should be customized based on the use case
        # We set the default behavior to add the instruction
        #  result to the model context

        instruction = task.get_user_prompt()
        result = str(result)

        content = MemoryContent(
            content=f"Instruction: {instruction}\nResponse: {result}",
            mime_type=MemoryMimeType.TEXT,
        )

        name = self.agent_name or "Agent"
        if isinstance(self.memory, list):
            self.memory[-1].add_sync(content, source_agent=name)
        else:
            self.memory.add_sync(content, source_agent=name)


class AutoGenBackend(AgentBackend):
    """
    An AutoGen agent factory backend that creates AutoGen agents.
    Setup with a model client, backend, and model to spawn agents.

    Args:
        model_client (Union[AsyncOpenAI, ChatCompletionClient]): The model client to use.
        model (str): The model name to use.
        backend (str, optional): Backend to use: "openai", "gemini", "ollama", "huggingface", "vllm", "liveai" or "livchat". Defaults to "openai".
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
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model_kwargs: Optional[dict] = None,
    ) -> None: ...

    def __init__(
        self,
        model_client: Optional[Union[AsyncOpenAI, ChatCompletionClient]] = None,
        model: Optional[str] = None,
        backend: str = "openai",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model_kwargs: Optional[dict] = None,
        **kwargs,
    ):
        super().__init__(
            model=model,
            api_key=api_key,
            base_url=base_url,
            model_kwargs=model_kwargs,
            reasoning_effort=None,
            backend=backend,
            **kwargs,
        )
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
        task: Optional[Task],
        max_retries: int = 3,
        agent_name: Optional[str] = None,
        memory: Optional[Memory] = None,
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

        AutoGenBackend.AGENT_COUNT += 1

        default_name = f"autogen_agent_{AutoGenBackend.AGENT_COUNT}"
        agent_name = default_name if agent_name is None else agent_name

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
        # Add memory
        if memory:
            assert isinstance(agent.memory, list)
            memory_json = memory.to_list_of_tasks_and_results()
            for memory_task, memory_result in memory_json:
                agent.add_to_context_sync(memory_task, memory_result)

        return agent
