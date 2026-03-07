################################################################################
## Copyright 2025 Lawrence Livermore National Security, LLC. and Binghamton University.
## See the top-level LICENSE file for details.
##
## SPDX-License-Identifier: Apache-2.0
################################################################################
from __future__ import annotations

import json
from typing import Any, Optional, Union, List, overload

from loguru import logger

try:
    from agent_framework import Agent as AFAgent, AgentSession
    from agent_framework.openai import OpenAIChatClient

    try:
        from agent_framework.openai import OpenAIResponsesClient

        RESPONSES_API_AVAILABLE = True
    except ImportError:
        RESPONSES_API_AVAILABLE = False
        OpenAIResponsesClient = None  # type: ignore[assignment]
        logger.warning(
            "OpenAIResponsesClient not available in this version of agent-framework"
        )
except ImportError:
    raise ImportError(
        "Please install the agent-framework package to use this module. "
        "Install with: pip install 'charge[agentframework]'"
    )

from charge.clients.agent_factory import AgentBackend, Agent
from charge.clients.agentframework_utils import (
    POSSIBLE_CONNECTION_ERRORS,
    ChARGeListMemory,
    generate_agent,
    setup_mcp_tools,
    chargeConnectionError,
)
from charge.clients.openai_base import (
    model_configure,
    get_api_key_for_backend,
)
from charge.experiments.memory import Memory
from charge.tasks.task import Task


def create_agentframework_chat_client(
    backend: str,
    model: str,
    api_key: Optional[str] = None,
    model_kwargs: Optional[dict[str, Any]] = None,
    use_responses_api: bool = False,
) -> Union[OpenAIChatClient, OpenAIResponsesClient]:
    """
    Creates an Agent Framework chat client based on the specified backend and model.

    Args:
        backend (str): The backend to use: "openai", "gemini", "livai", "livchat", etc.
        model (str): The model name/ID to use.
        api_key (Optional[str], optional): API key for the model. Defaults to None.
        model_kwargs (Optional[dict], optional): Additional keyword arguments. Defaults to None.
        use_responses_api (bool, optional): Use OpenAI Responses API for hosted tools. Defaults to False.

    Returns:
        Union[OpenAIChatClient, OpenAIResponsesClient]: The created chat client.

    Raises:
        ValueError: If backend is not supported or configuration is invalid.
    """
    if model_kwargs is None:
        model_kwargs = {}

    if backend == "ollama":
        raise NotImplementedError(
            "Ollama support is planned but not yet available in Agent Framework. "
            "Use AutoGen implementation for Ollama support."
        )
    elif backend == "huggingface":
        raise NotImplementedError(
            "HuggingFace support requires custom implementation with Agent Framework. "
            "Use AutoGen implementation for HuggingFace support."
        )
    elif backend == "vllm":
        raise NotImplementedError(
            "vLLM support requires custom implementation with Agent Framework. "
            "Use AutoGen implementation for vLLM support."
        )
    else:
        # OpenAI or OpenAI-compatible endpoints only
        api_key = get_api_key_for_backend(backend, api_key)

        assert (
            api_key is not None
        ), "API key must be provided for OpenAI or Gemini backend"

        # Check if Responses API is requested
        if use_responses_api:
            if not RESPONSES_API_AVAILABLE:
                raise ImportError(
                    "OpenAIResponsesClient is not available in this version of agent-framework. "
                    "Update to a newer version or use the standard OpenAIChatClient."
                )

            logger.info("Creating OpenAIResponsesClient with hosted tools support")
            chat_client = OpenAIResponsesClient(
                model_id=model,
                api_key=api_key,
                **model_kwargs if model_kwargs is not None else {},
            )
        else:
            # Standard OpenAI or OpenAI-compatible client
            # Agent Framework reads OPENAI_API_KEY from environment by default
            chat_client = OpenAIChatClient(
                model_id=model,
                api_key=api_key,
                # Additional kwargs can be passed but Agent Framework has different options
                **model_kwargs if model_kwargs is not None else {},
            )

    return chat_client


class AgentFrameworkAgent(Agent):
    """
    An Agent Framework agent that interacts with MCP servers and runs tasks.

    Note: Agent Framework agents are stateless by default. Use AgentSession
    to maintain conversation state across multiple runs.

    Args:
        task (Task): The task to be performed by the agent.
        chat_client: The Agent Framework chat client.
        agent_name (str): Name of the agent.
        model (str): Model name.
        memory (Optional[Any], optional): Memory instance for conversation state. Defaults to None.
        max_retries (int, optional): Maximum number of retries. Defaults to 3.
        max_tool_calls (int, optional): Maximum tool call iterations. Defaults to 30.
        timeout (int, optional): Timeout in seconds. Defaults to 60.
        **kwargs: Additional keyword arguments.
    """

    def __init__(
        self,
        task: Optional[Task],
        chat_client: Union[OpenAIChatClient, OpenAIResponsesClient],
        agent_name: str,
        model: str,
        memory: Optional[Memory] = None,
        max_retries: int = 3,
        max_tool_calls: int = 30,
        timeout: int = 60,
        backend: Optional[str] = None,
        model_kwargs: Optional[dict] = None,
        **kwargs,
    ) -> None:
        super().__init__(task=task, **kwargs)
        if self.task is None:
            raise ValueError("AgentFrameworkAgent requires a task.")
        self.max_retries = max_retries
        self.max_tool_calls = max_tool_calls
        self.workbenches: List[Any] = []  # Will be MCP workbenches
        self.agent_name = agent_name
        self.chat_client = chat_client
        self.timeout = timeout
        self.memory: list[ChARGeListMemory] | ChARGeListMemory = self.setup_memory(
            memory
        )
        # self.memory = self.setup_memory(memory)
        self.setup_kwargs = kwargs

        self.context_history = []
        self.model = model
        self.backend = backend
        self.model_kwargs = model_kwargs if model_kwargs is not None else {}
        self._af_agent: Optional[AFAgent] = None
        self._agent_session: Optional[AgentSession] = None

    def setup_memory(self, memory: Optional[Memory] = None) -> Optional[Memory]:
        """
        Sets up the memory for the agent if not already provided.

        Args:
            memory (Optional[Any], optional): Pre-initialized memory. Defaults to None.

        Returns:
            Optional[Any]: The memory instance or None.
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
        # Agent Framework uses MCPStdioTool, MCPStreamableHTTPTool, MCPWebsocketTool
        assert self.task is not None
        if not self.task.server_files and not self.task.server_urls:
            return

        try:
            self.workbenches = await setup_mcp_tools(
                stdio_servers=self.task.server_files,
                sse_servers=self.task.server_urls,
            )
            logger.info(f"Set up {len(self.workbenches)} MCP tools")
        except Exception as e:
            logger.error(f"Failed to setup MCP tools: {e}")
            self.workbenches = []

    async def close_workbenches(self) -> None:
        """
        Closes MCP workbenches.

        Returns:
            None
        """
        # TODO: Implement MCP cleanup
        pass

    def _create_agent(self, **kwargs) -> AFAgent:
        """
        Creates an Agent Framework agent with the given parameters.

        Returns:
            AFAgent: The created Agent Framework agent.
        """
        # Agent Framework pattern:
        # agent = Agent(name="...", chat_client=..., instructions="...", tools=[...])

        af_agent = generate_agent(
            chat_client=self.chat_client,
            agent_name=self.agent_name,
            instructions=self.task.get_system_prompt() if self.task is not None else "",
            tools=self.workbenches,  # MCP tools
            max_tool_calls=self.max_tool_calls,
            **kwargs,
        )
        return af_agent

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

    def _prepend_experiment_memory(self, user_prompt: str) -> str:
        if not self.memory:
            return user_prompt

        try:
            items = self.memory.to_list_of_tasks_and_results()
        except Exception:
            return user_prompt

        if not items:
            return user_prompt

        context_str = "\n\n=== Previous task context ===\n"
        for prior_task, prior_result in items:
            instruction = prior_task.get_user_prompt()
            context_str += f"\nInstruction: {instruction}\nResponse: {prior_result}\n"
        context_str += "=== End of previous context ===\n\n"
        return context_str + user_prompt

    async def _execute_with_retries(
        self, agent: AFAgent, user_prompt: str, session: AgentSession
    ) -> str:
        """
        Executes the agent with retry logic and output validation.

        Args:
            agent: The agent instance to run.
            user_prompt: The prompt to send to the agent.
            session: The agent session for conversation state.

        Returns:
            Valid output content as a string.

        Raises:
            ValueError: If all retries fail to produce valid output.
        """
        last_error = None

        for attempt in range(1, self.max_retries + 1):
            try:
                logger.info(f"Attempt {attempt}/{self.max_retries}")

                # Run agent (Agent Framework returns AgentResponse)
                result = await agent.run(user_prompt, session=session)

                # Store in context history
                self.context_history.append(result)

                # Extract content from result
                proposed_content = ""
                if hasattr(result, "messages") and result.messages:
                    last_message = result.messages[-1]
                    if hasattr(last_message, "text"):
                        proposed_content = last_message.text
                    elif hasattr(last_message, "content"):
                        proposed_content = str(last_message.content)
                    else:
                        proposed_content = str(last_message)
                else:
                    proposed_content = str(result)

                if not proposed_content:
                    logger.warning(f"Attempt {attempt}: No content in result")
                    continue

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
                raise chargeConnectionError(error_msg)
            except Exception as e:
                error_msg = f"Attempt {attempt}: Unexpected error: {e}"
                logger.error(error_msg)
                last_error = e

        # All retries exhausted
        raise ValueError(
            f"Failed to obtain valid output after {self.max_retries} attempts. "
            f"Last error: {last_error}"
        )

    async def _convert_to_structured_format(self, content: str) -> str:
        """
        Converts content to structured format using a dedicated agent.

        Args:
            content: The content to convert.

        Returns:
            The structured content as a JSON string.

        Raises:
            ValueError: If conversion fails.
        """
        try:
            # Create a simple conversion agent
            structured_out = self.task.get_structured_output_schema()
            assert structured_out is not None
            schema = structured_out.model_json_schema()

            conversion_agent = AFAgent(
                name=f"{self.agent_name}_structured_output",
                client=self.chat_client,  # Fixed: parameter is 'client' not 'chat_client'
                instructions="You are an agent that converts model output to a structured JSON format.",
            )

            prompt = (
                f"Convert the following output to match this JSON schema:\n\n{json.dumps(schema, indent=2)}\n\n"
                f"Output to convert:\n{content}\n\n"
                f"Return ONLY a valid JSON object matching the schema, with no additional text."
            )

            result = await conversion_agent.run(prompt)

            # Extract JSON from result
            if hasattr(result, "messages") and result.messages:
                last_message = result.messages[-1]
                if hasattr(last_message, "text"):
                    return last_message.text
            return str(result)

        except Exception as e:
            logger.error(f"Failed to convert to structured format: {e}")
            raise ValueError(f"Structured output conversion failed: {e}") from e

    async def run(self, **kwargs) -> str:
        """
        Runs the agent.

        Returns:
            str: The output content from the agent. If structured output is enabled,
                 the output will be checked with the task's formatting method and
                 the json string will be returned.
        """
        logger.info(f"Running Agent Framework agent: {self.agent_name}")

        # Set up workbenches from task server paths
        await self.setup_mcp_workbenches()

        try:
            # Create agent
            if self._af_agent is None:
                self._af_agent = self._create_agent()

            # Create or reuse session for stateful conversation
            if self._agent_session is None:
                self._agent_session = self._af_agent.create_session()

            # Prepare prompt
            user_prompt = self._prepare_task_prompt()
            user_prompt = self._prepend_experiment_memory(user_prompt)

            # Execute with retries
            result = await self._execute_with_retries(
                self._af_agent, user_prompt, self._agent_session
            )

            return result

        finally:
            await self.close_workbenches()

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

    def get_model_info(self) -> dict[str, Any]:
        """
        Returns the model information of the agent.
        """
        return {
            "model": self.model,
            "backend": self.backend,
            "model_kwargs": self.model_kwargs,
        }


class AgentFrameworkBackend(AgentBackend):
    """
    An Agent Framework agent factory backend that creates Agent Framework agents.
    Setup with a model client, backend, and model to spawn agents.

    Args:
        chat_client: Optional pre-configured Agent Framework chat client.
        model: Model name/ID.
        backend: Backend name (OpenAI-compatible only).
        api_key: Optional API key.
        base_url: Optional base URL for custom endpoints.
        model_kwargs: Additional client kwargs.
        use_responses_api: Whether to use Responses API (hosted tools).
    """

    AGENT_COUNT = 0

    @overload
    def __init__(
        self,
        chat_client: OpenAIChatClient,
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
        use_responses_api: bool = False,
    ) -> None: ...

    def __init__(
        self,
        chat_client: Optional[OpenAIChatClient] = None,
        model: Optional[str] = None,
        backend: str = "openai",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model_kwargs: Optional[dict] = None,
        use_responses_api: bool = False,
        **kwargs,
    ):
        super().__init__(
            model=model,
            api_key=api_key,
            base_url=base_url,
            reasoning_effort=None,
            model_kwargs=model_kwargs,
            backend=backend,
            **kwargs,
        )
        self.chat_client = chat_client
        self.use_responses_api = use_responses_api

        if self.chat_client is None:
            assert (
                model is not None
            ), "Model name must be provided if chat_client is not given."

            model, backend, api_key, model_kwargs_configured = model_configure(
                model=model, backend=backend, api_key=api_key, base_url=base_url
            )
            if model_kwargs:
                model_kwargs_configured.update(model_kwargs)

            self.chat_client = create_agentframework_chat_client(
                backend=backend,
                model=model,
                api_key=api_key,
                model_kwargs=model_kwargs_configured,
                use_responses_api=use_responses_api,
            )

        self.model = model
        self.backend = backend
        self.model_kwargs = model_kwargs if model_kwargs is not None else {}

        if self.chat_client is None:
            raise ValueError("Failed to create chat client.")

    def create_agent(
        self,
        task: Optional[Task],
        max_retries: int = 3,
        agent_name: Optional[str] = None,
        memory: Optional[Memory] = None,
        **kwargs,
    ) -> AgentFrameworkAgent:
        self.max_retries = max_retries
        assert (
            self.chat_client is not None
        ), "Chat client must be initialized to create an agent."

        AgentFrameworkBackend.AGENT_COUNT += 1
        default_name = f"agentframework_agent_{AgentFrameworkBackend.AGENT_COUNT}"
        agent_name = default_name if agent_name is None else agent_name

        agent = AgentFrameworkAgent(
            task=task,
            chat_client=self.chat_client,
            agent_name=agent_name,
            max_retries=max_retries,
            model=self.model,  # type: ignore
            backend=self.backend,
            model_kwargs=self.model_kwargs,
            memory=memory,
            **kwargs,
        )
        return agent

    def get_hosted_tools(self) -> List[Any]:
        """
        Get hosted tools from an OpenAI Responses API client.

        Hosted tools are only available when `use_responses_api=True`.
        """
        if not self.use_responses_api:
            raise ValueError(
                "Hosted tools are only available with Responses API. "
                "Create the backend with use_responses_api=True"
            )

        if not hasattr(self.chat_client, "get_code_interpreter_tool"):
            raise AttributeError(
                "Client does not support hosted tools. Ensure you're using OpenAIResponsesClient."
            )

        tools: List[Any] = []

        try:
            if hasattr(self.chat_client, "get_code_interpreter_tool"):
                tools.append(self.chat_client.get_code_interpreter_tool())
        except Exception as e:
            logger.debug(f"code_interpreter tool not available: {e}")

        try:
            if hasattr(self.chat_client, "get_file_search_tool"):
                tools.append(self.chat_client.get_file_search_tool())
        except Exception as e:
            logger.debug(f"file_search tool not available: {e}")

        return tools
