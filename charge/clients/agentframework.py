################################################################################
## Copyright 2025 Lawrence Livermore National Security, LLC. and Binghamton University.
## See the top-level LICENSE file for details.
##
## SPDX-License-Identifier: Apache-2.0
################################################################################
from __future__ import annotations

import json
from typing import Any, Awaitable, Callable, Optional, Union, List, Literal

from loguru import logger

try:
    from agent_framework import Agent as AFAgent, AgentSession, InMemoryHistoryProvider
    from agent_framework.openai import (
        OpenAIChatClient,
        OpenAIResponsesClient,
        OpenAIResponsesOptions,
        OpenAIChatOptions,
    )
except ImportError:
    raise ImportError(
        "Please install the agent-framework package to use this module. "
        "Install with: pip install 'charge[agentframework]'"
    )

from charge.clients.agent_factory import AgentBackend, Agent, ReasoningCallbackType
from charge.clients.agentframework_utils import (
    POSSIBLE_CONNECTION_ERRORS,
    setup_mcp_tools,
)
from charge.clients.openai_base import (
    model_configure,
    get_api_key_for_backend,
)
from charge._utils import maybe_await_async
from charge.experiments.memory import Memory
from charge.tasks.task import Task


def create_agentframework_client(
    backend: str,
    model: str,
    api_key: Optional[str] = None,
    model_kwargs: Optional[dict[str, Any]] = None,
    use_responses_api: bool = True,
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
            logger.info("Creating OpenAIResponsesClient with hosted tools support")
            client = OpenAIResponsesClient(
                model_id=model,
                api_key=api_key,
                **model_kwargs if model_kwargs is not None else {},
            )
        else:
            # Standard OpenAI or OpenAI-compatible client
            # Agent Framework reads OPENAI_API_KEY from environment by default
            client = OpenAIChatClient(
                model_id=model,
                api_key=api_key,
                # Additional kwargs can be passed but Agent Framework has different options
                **model_kwargs if model_kwargs is not None else {},
            )

    return client


class AgentFrameworkAgent(Agent):
    """
    An Agent Framework agent that interacts with MCP servers and runs tasks.

    Note: Agent Framework agents are stateless by default. Use AgentSession
    to maintain conversation state across multiple runs.

    Args:
        task (Task): The task to be performed by the agent.
        client: The Agent Framework client.
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
        client: Union[OpenAIChatClient, OpenAIResponsesClient],
        agent_name: str,
        model: str | None,
        memory: Optional[Memory] = None,
        max_retries: int = 3,
        max_tool_calls: int = 30,
        timeout: int = 60,
        backend: Optional[str] = None,
        model_kwargs: Optional[dict] = None,
        reasoning_effort: Literal["low", "medium", "high"] = "medium",
        builtin_tools: Optional[list[Any]] = None,
        **kwargs,
    ) -> None:
        super().__init__(task=task, **kwargs)
        self.max_retries = max_retries
        self.max_tool_calls = max_tool_calls
        self.workbenches: List[Any] = []  # Will be MCP workbenches
        self.builtin_tools = builtin_tools or []
        self.workbenches.extend(self.builtin_tools)
        self.agent_name = agent_name
        self.client = client
        self.timeout = timeout
        self.setup_kwargs = kwargs
        self.reasoning_effort: Literal["low", "medium", "high"] = reasoning_effort

        self.model = model
        self.backend = backend
        self.model_kwargs = model_kwargs if model_kwargs is not None else {}

        self._af_agent: Optional[AFAgent] = None

        self._agent_session: Optional[AgentSession] = None
        self._session_seeded_from_experiment_memory = False

    def _create_agent(
        self,
        agent_name: str,
        reasoning_effort: Literal["low", "medium", "high"],
        instructions: str,
    ) -> AFAgent:
        if isinstance(self.client, OpenAIResponsesClient):
            af_agent = AFAgent[OpenAIResponsesOptions](
                client=self.client,
                name=agent_name,
                instructions=instructions,
                tools=self.workbenches,
                default_options={
                    "reasoning": {
                        "effort": reasoning_effort,
                        "summary": "detailed",
                    },
                    "include": ["reasoning.encrypted_content"],
                },
                context_providers=[
                    # This provider ensures session.state contains the history
                    InMemoryHistoryProvider(load_messages=True)
                ],
            )
        else:
            af_agent = AFAgent[OpenAIChatOptions](
                client=self.client,
                name=agent_name,
                instructions=instructions,
                tools=self.workbenches,
                context_providers=[
                    # This provider ensures session.state contains the history
                    InMemoryHistoryProvider(load_messages=True)
                ],
            )
        return af_agent

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
            self.workbenches = self.builtin_tools + await setup_mcp_tools(
                stdio_servers=self.task.server_files,
                mcp_servers=self.task.server_urls,
            )
            logger.info(f"Set up {len(self.workbenches)} MCP tools")
        except Exception as e:
            logger.error(f"Failed to setup MCP tools: {e}")
            self.workbenches = list(self.builtin_tools)

    async def close_workbenches(self) -> None:
        """
        Closes MCP workbenches.

        Returns:
            None
        """
        # TODO: Implement MCP cleanup
        pass

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

    async def _execute_with_retries(
        self,
        agent: AFAgent,
        user_prompt: str,
        session: AgentSession,
        reasoning_callback: ReasoningCallbackType,
    ) -> str:
        """
        Executes the agent with retry logic and output validation.

        Args:
            agent: The agent instance to run.
            user_prompt: The prompt to send to the agent.
            session: The agent session for conversation state.
            reasoning_callback: An optional function to be called whenever a
                                reasoning summary is generated.

        Returns:
            Valid output content as a string.

        Raises:
            ValueError: If all retries fail to produce valid output.
        """
        last_error = None
        assert self.task

        for attempt in range(1, self.max_retries + 1):
            try:
                logger.info(f"Attempt {attempt}/{self.max_retries}")

                # Run agent (Agent Framework returns AgentResponse)
                stream = await agent.run(user_prompt, session=session, stream=True)
                async for update in stream:
                    if (
                        update.contents
                        and update.contents[0].raw_representation
                        and update.contents[0].raw_representation.type
                        == "response.reasoning_summary_text.done"
                    ):
                        # Reasoning text - use callback to transmit back
                        if reasoning_callback:
                            await reasoning_callback(update.contents[0].text)
                result = await stream.get_final_response()

                # Extract content from result
                proposed_content = ""
                if hasattr(result, "messages") and result.messages:
                    last_message = result.messages[-1]
                    if hasattr(last_message, "text"):
                        proposed_content = last_message.text
                    elif hasattr(last_message, "content"):
                        proposed_content = str(last_message.contents)
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
            assert self.task
            # Create a simple conversion agent
            structured_out = self.task.get_structured_output_schema()
            assert structured_out is not None
            schema = structured_out.model_json_schema()

            conversion_agent = self._create_agent(
                f"{self.agent_name}_structured_output",
                reasoning_effort="low",
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

    async def run(
        self, reasoning_callback: ReasoningCallbackType = None, **kwargs
    ) -> str:
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
                instructions = (
                    self.task.get_system_prompt() if self.task is not None else ""
                )
                self._af_agent = self._create_agent(
                    self.agent_name, self.reasoning_effort, instructions=instructions
                )

            # Create or reuse session for stateful conversation
            if self._agent_session is None:
                self._agent_session = self._af_agent.create_session()

            # Prepare prompt
            user_prompt = self._prepare_task_prompt()

            # Execute with retries
            result = await self._execute_with_retries(
                self._af_agent, user_prompt, self._agent_session, reasoning_callback
            )

            return result

        finally:
            await self.close_workbenches()

    def get_model_info(self) -> dict[str, Any]:
        """
        Returns the model information of the agent.
        """
        return {
            "model": self.model,
            "backend": self.backend,
            "model_kwargs": self.model_kwargs,
        }

    def load_memory(self, json_str: str) -> None:
        """
        Loads memory content into the agent's memory.
        """
        self._agent_session = AgentSession.from_dict(json.loads(json_str))

    def save_memory(self) -> str:
        """
        Saves the agent's memory content to a JSON string.
        """
        if self._agent_session is None:
            return ""
        return json.dumps(self._agent_session.to_dict())


class AgentFrameworkBackend(AgentBackend):
    """
    An Agent Framework agent factory backend that creates Agent Framework agents.
    Setup with a model client, backend, and model to spawn agents.

    Args:
        client: Optional pre-configured Agent Framework chat client.
        model: Model name/ID.
        backend: Backend name (OpenAI-compatible only).
        api_key: Optional API key.
        base_url: Optional base URL for custom endpoints.
        model_kwargs: Additional client kwargs.
        use_responses_api: Whether to use Responses API (hosted tools).
    """

    AGENT_COUNT = 0

    def __init__(
        self,
        model: Optional[str] = None,
        backend: str = "openai",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model_kwargs: Optional[dict] = None,
        use_responses_api: bool = True,
        reasoning_effort: Literal["low", "medium", "high"] | None = "medium",
        client: Optional[OpenAIChatClient | OpenAIResponsesClient] = None,
        **kwargs,
    ):
        super().__init__(
            model=model,
            api_key=api_key,
            base_url=base_url,
            reasoning_effort=reasoning_effort,
            model_kwargs=model_kwargs,
            backend=backend,
            **kwargs,
        )
        self.client = client
        self.use_responses_api = use_responses_api
        self.reasoning_effort = reasoning_effort

        if self.client is None:
            assert (
                model is not None
            ), "Model name must be provided if client is not given."

            model, backend, api_key, model_kwargs_configured = model_configure(
                model=model, backend=backend, api_key=api_key, base_url=base_url
            )
            if model_kwargs:
                model_kwargs_configured.update(model_kwargs)

            self.client = create_agentframework_client(
                backend=backend,
                model=model,
                api_key=api_key,
                model_kwargs=model_kwargs_configured,
                use_responses_api=use_responses_api,
            )

        self.model = model
        self.backend = backend
        self.model_kwargs = model_kwargs if model_kwargs is not None else {}

        if self.client is None:
            raise ValueError("Failed to create OpenAI client.")

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
            self.client is not None
        ), "Chat client must be initialized to create an agent."

        AgentFrameworkBackend.AGENT_COUNT += 1
        default_name = f"agentframework_agent_{AgentFrameworkBackend.AGENT_COUNT}"
        agent_name = default_name if agent_name is None else agent_name

        agent = AgentFrameworkAgent(
            task=task,
            client=self.client,
            agent_name=agent_name,
            max_retries=max_retries,
            model=self.model,
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

        tools: List[Any] = []

        assert isinstance(self.client, OpenAIResponsesClient)

        try:
            if hasattr(self.client, "get_code_interpreter_tool"):
                tools.append(self.client.get_code_interpreter_tool())
        except Exception as e:
            logger.debug(f"code_interpreter tool not available: {e}")

        return tools
