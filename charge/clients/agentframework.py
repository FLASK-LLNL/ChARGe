################################################################################
## Copyright 2025 Lawrence Livermore National Security, LLC. and Binghamton University.
## See the top-level LICENSE file for details.
##
## SPDX-License-Identifier: Apache-2.0
################################################################################
try:
    from agent_framework import Agent as AFAgent, AgentSession
    from agent_framework.openai import OpenAIChatClient
    try:
        from agent_framework.openai import OpenAIResponsesClient
        RESPONSES_API_AVAILABLE = True
    except ImportError:
        RESPONSES_API_AVAILABLE = False
        logger.warning("OpenAIResponsesClient not available in this version of agent-framework")
except ImportError:
    raise ImportError(
        "Please install the agent-framework package to use this module. "
        "Install with: pip install 'charge[agentframework]'"
    )

import asyncio
import re
import os
import warnings
from charge.clients.AgentPool import AgentPool, Agent
from charge.clients.Client import Client
from charge.clients.agentframework_utils import (
    POSSIBLE_CONNECTION_ERRORS,
    AgentFrameworkMemory,
    generate_agent,
    setup_mcp_tools,
    chargeConnectionError,
)
from charge.clients.openai_base import (
    model_configure,
    LoggingTransport,
    create_http_client,
)
from typing import Any, Tuple, Optional, Dict, Union, List, Callable, overload
from charge.tasks.Task import Task
from loguru import logger

import logging
import httpx
import json


def create_agentframework_chat_client(
    backend: str,
    model: str,
    api_key: Optional[str] = None,
    model_kwargs: Optional[dict[str, Any]] = None,
    use_responses_api: bool = False,
) -> Union[OpenAIChatClient, "OpenAIResponsesClient"]:
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
        if api_key is None:
            if backend == "gemini":
                api_key = os.getenv("GOOGLE_API_KEY")
            else:
                api_key = os.getenv("OPENAI_API_KEY")

        assert api_key is not None, (
            "API key must be provided for OpenAI or Gemini backend"
        )

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
        task: Task,
        chat_client: OpenAIChatClient,
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
        self.workbenches: List[Any] = []  # Will be MCP workbenches
        self.agent_name = agent_name
        self.chat_client = chat_client
        self.timeout = timeout
        self.memory = self.setup_memory(memory)
        self.setup_kwargs = kwargs

        self.context_history = []
        self.model = model
        self.backend = backend
        self.model_kwargs = model_kwargs if model_kwargs is not None else {}
        self._af_agent: Optional[AFAgent] = None
        self._agent_session: Optional[AgentSession] = None

    def setup_memory(self, memory: Optional[Any] = None) -> Optional[Any]:
        """
        Sets up the memory for the agent if not already provided.

        Args:
            memory (Optional[Any], optional): Pre-initialized memory. Defaults to None.

        Returns:
            Optional[Any]: The memory instance or None.
        """
        # Agent Framework uses AgentSession for state management
        # Memory will be integrated with session
        return memory

    async def setup_mcp_workbenches(self) -> None:
        """
        Sets up MCP workbenches from the task's server paths.

        Returns:
            None
        """
        # Agent Framework uses MCPStdioTool, MCPStreamableHTTPTool, MCPWebsocketTool
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
            instructions=self.task.get_system_prompt(),
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

            # If we have memory from previous tasks, prepend it to the prompt
            if self.memory and isinstance(self.memory, list):
                for mem in self.memory:
                    if hasattr(mem, 'get_messages') and hasattr(mem, 'get_source_agents'):
                        messages = mem.get_messages()
                        sources = mem.get_source_agents()
                        if messages:
                            context_str = "\n\n=== Previous conversation context ===\n"
                            for msg, source in zip(messages, sources):
                                content = msg.get('content', '')
                                context_str += f"\n{content}\n"
                            context_str += "=== End of previous context ===\n\n"
                            user_prompt = context_str + user_prompt

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

    def load_memory(self, json_str: str) -> None:
        """
        Loads memory content into the agent's memory.
        """
        if self.memory is not None:
            if isinstance(self.memory, list):
                for mem in self.memory:
                    if isinstance(mem, AgentFrameworkMemory):
                        mem.load_memory_content(json_str)
            elif isinstance(self.memory, AgentFrameworkMemory):
                self.memory.load_memory_content(json_str)

    def save_memory(self) -> str:
        """
        Saves the agent's memory content to a JSON string.
        """
        if self.memory is not None:
            if isinstance(self.memory, list):
                combined_content = ""
                for mem in self.memory:
                    if isinstance(mem, AgentFrameworkMemory):
                        combined_content += mem.serialize_memory_content()
                return combined_content
            elif isinstance(self.memory, AgentFrameworkMemory):
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


class AgentFrameworkPool(AgentPool):
    """
    An Agent Framework agent pool that creates Agent Framework agents.
    Setup with a chat client, backend, and model to spawn agents.

    Args:
        chat_client: The Agent Framework chat client to use.
        model (str): The model name to use.
        backend (str, optional): Backend to use. Defaults to "openai".
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
        backend: Optional[str] = "openai",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model_kwargs: Optional[dict] = None,
        use_responses_api: bool = False,
    ):
        super().__init__()
        self.chat_client = chat_client
        self.use_responses_api = use_responses_api

        if self.chat_client is None:
            assert model is not None, (
                "Model name must be provided if chat_client is not given."
            )
            assert backend is not None, (
                "Backend must be provided if chat_client is not given."
            )

            model, backend, api_key, model_kwargs_configured = model_configure(
                model=model, backend=backend, api_key=api_key, base_url=base_url
            )
            # Merge configured kwargs with provided kwargs
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

    def get_hosted_tools(self) -> List[Any]:
        """
        Get hosted tools from the Responses API client.

        Hosted tools are provided by OpenAI and include:
        - code_interpreter: Execute Python code
        - file_search: Search through files
        - web_search: Search the web (if available)

        Returns:
            List of hosted tools available from the client.

        Raises:
            AttributeError: If the client is not a Responses API client.

        Example:
            >>> pool = AgentFrameworkPool(
            ...     model="gpt-4o",
            ...     backend="openai",
            ...     use_responses_api=True
            ... )
            >>> tools = pool.get_hosted_tools()
            >>> agent = pool.create_agent(task=task, tools=tools)
        """
        if not self.use_responses_api:
            raise ValueError(
                "Hosted tools are only available with Responses API. "
                "Create the pool with use_responses_api=True"
            )

        if not hasattr(self.chat_client, 'get_code_interpreter_tool'):
            raise AttributeError(
                "Client does not support hosted tools. Ensure you're using OpenAIResponsesClient."
            )

        tools = []

        # Try to get available hosted tools
        try:
            if hasattr(self.chat_client, 'get_code_interpreter_tool'):
                code_tool = self.chat_client.get_code_interpreter_tool()
                tools.append(code_tool)
                logger.info("Added code_interpreter hosted tool")
        except Exception as e:
            logger.debug(f"code_interpreter tool not available: {e}")

        try:
            if hasattr(self.chat_client, 'get_file_search_tool'):
                file_tool = self.chat_client.get_file_search_tool()
                tools.append(file_tool)
                logger.info("Added file_search hosted tool")
        except Exception as e:
            logger.debug(f"file_search tool not available: {e}")

        return tools

    def create_agent(
        self,
        task: Task,
        max_retries: int = 3,
        agent_name: Optional[str] = None,
        **kwargs,
    ) -> AgentFrameworkAgent:
        """Creates an Agent Framework agent for the given task.

        Args:
            task (Task): The task to be performed by the agent.
            max_retries (int, optional): Maximum number of retries for failed tasks. Defaults to 3.
            agent_name (Optional[str], optional): Name of the agent. If None, a default name will be assigned. Defaults to None.
            **kwargs: Additional keyword arguments.

        Returns:
            AgentFrameworkAgent: The created Agent Framework agent.
        """
        self.max_retries = max_retries
        assert self.chat_client is not None, (
            "Chat client must be initialized to create an agent."
        )

        AgentFrameworkPool.AGENT_COUNT += 1

        default_name = self.create_agent_name()
        agent_name = default_name if agent_name is None else agent_name

        if agent_name in self.agent_list:
            warnings.warn(
                f"Agent with name {agent_name} already exists. Creating another agent with the same name."
            )
        else:
            self.agent_list.append(agent_name)

        agent = AgentFrameworkAgent(
            task=task,
            chat_client=self.chat_client,
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

    def get_agent_by_name(self, name: str) -> AgentFrameworkAgent:
        """Gets an agent by name.

        Args:
            name (str): The name of the agent.

        Returns:
            AgentFrameworkAgent: The agent with the given name.
        """
        assert name in self.agent_dict, f"Agent with name {name} does not exist."
        return self.agent_dict[name]

    def create_agent_name(
        self, prefix: Optional[str] = None, suffix: Optional[str] = None
    ) -> str:
        """Create a unique agent name.

        Args:
            prefix (Optional[str], optional): Prefix for the name. Defaults to None.
            suffix (Optional[str], optional): Suffix for the name. Defaults to None.

        Returns:
            str: The generated agent name.
        """
        model_name = self.model if self.model is not None else "default_model"
        backend_name = self.backend if self.backend is not None else "default_backend"

        default_name = f"[{backend_name}:{model_name}]_{AgentFrameworkPool.AGENT_COUNT}"
        default_name = re.sub(r"[^a-zA-Z0-9_]", "_", default_name)

        if prefix:
            default_name = f"{prefix}{default_name}"
        if suffix:
            default_name = f"{default_name}{suffix}"

        return default_name
