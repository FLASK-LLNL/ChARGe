try:
    from autogen_agentchat.agents import AssistantAgent, UserProxyAgent
    from autogen_core.models import ModelFamily, ChatCompletionClient
    from autogen_ext.tools.mcp import StdioServerParams, McpWorkbench, SseServerParams
    from autogen_agentchat.messages import TextMessage
    from autogen_agentchat.conditions import HandoffTermination, TextMentionTermination
    from autogen_agentchat.teams import RoundRobinGroupChat
    from autogen_agentchat.ui import Console
    from autogen_agentchat.conditions import TextMentionTermination
except ImportError:
    raise ImportError(
        "Please install the autogen-agentchat package to use this module."
    )

from functools import partial
import os
from charge.clients.Client import Client
from typing import Type, Optional, Dict
from charge.Experiment import Experiment


class AutoGenClient(Client):
    def __init__(
        self,
        experiment_type: Experiment,
        path: str = ".",
        max_retries: int = 3,
        backend: str = "openai",
        model: str = "gpt-4",
        model_client: Optional[ChatCompletionClient] = None,
        api_key: Optional[str] = None,
        model_info: Optional[dict] = None,
        model_kwargs: Optional[dict] = None,
        server_path: Optional[str] = None,
        server_url: Optional[str] = None,
        server_kwargs: Optional[dict] = None,
        max_tool_calls: int = 15,
        check_response: bool = False,
        max_multi_turns: int = 100,
    ):
        """Initializes the AutoGenClient.

        Args:
            experiment_type (Type[Experiment]): The experiment class to use.
            path (str, optional): Path to save generated MCP server files. Defaults to ".".
            max_retries (int, optional): Maximum number of retries for failed tasks. Defaults to 3.
            backend (str, optional): Backend to use: "openai", "gemini", "ollama", "liveai" or "livchat". Defaults to "openai".
            model (str, optional): Model name to use. Defaults to "gpt-4".
            model_client (Optional[ChatCompletionClient], optional): Pre-initialized model client. If provided, `backend`, `model`, and `api_key` are ignored. Defaults to None.
            api_key (Optional[str], optional): API key for the model. Defaults to None.
            model_info (Optional[dict], optional): Additional model info. Defaults to None.
            model_kwargs (Optional[dict], optional): Additional keyword arguments for the model client.
                                                     Defaults to None.
            server_path (Optional[str], optional): Path to an existing MCP server script. If provided, this
                                                   server will be used instead of generating
                                                   new ones. Defaults to None.
            server_url (Optional[str], optional): URL of an existing MCP server over the SSE transport.
                                                  If provided, this server will be used instead of generating
                                                  new ones. Defaults to None.
            server_kwargs (Optional[dict], optional): Additional keyword arguments for the server client. Defaults to None.
            max_tool_calls (int, optional): Maximum number of tool calls per task. Defaults to 15.
            check_response (bool, optional): Whether to check the response using verifier methods.
                                             Defaults to False (Will be set to True in the future).
            max_multi_turns (int, optional): Maximum number of multi-turn interactions. Defaults to 100.
        Raises:
            ValueError: If neither `server_path` nor `server_url` is provided and MCP servers cannot be generated.
        """
        super().__init__(experiment_type, path, max_retries)
        self.backend = backend
        self.model = model
        self.api_key = api_key
        self.model_info = model_info
        self.model_kwargs = model_kwargs if model_kwargs is not None else {}
        self.max_tool_calls = max_tool_calls
        self.check_response = check_response
        self.max_multi_turns = max_multi_turns

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
                from autogen_ext.models.openai import OpenAIChatCompletionClient

                if api_key is None:
                    if backend == "gemini":
                        api_key = os.getenv("GOOGLE_API_KEY")
                    else:
                        api_key = os.getenv("OPENAI_API_KEY")
                assert (
                    api_key is not None
                ), "API key must be provided for OpenAI or Gemini backend"
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
                self.server = StdioServerParams(command="python3", args=[server_path])
            elif server_url is not None:
                self.server = SseServerParams(url=server_url, **(server_kwargs or {}))
        self.messages = []

    def configure(model: str, backend: str) -> (str, str, str, Dict[str, str]):
        import httpx

        kwargs = {}
        API_KEY = None
        if backend in ["openai", "gemini", "livai", "livchat"]:
            if backend == "openai":
                API_KEY = os.getenv("OPENAI_API_KEY")
                model = "gpt-4"
                kwargs["parallel_tool_calls"] = False
                kwargs["reasoning_effort"] = "high"
            elif backend == "livai" or backend == "livchat":
                API_KEY = os.getenv("OPENAI_API_KEY")
                BASE_URL = os.getenv("LIVAI_BASE_URL")
                assert (
                    BASE_URL is not None
                ), "LivAI Base URL must be set in environment variable"
                model = "gpt-4.1"
                kwargs["base_url"] = BASE_URL
                kwargs["http_client"] = httpx.AsyncClient(verify=False)
            else:
                API_KEY = os.getenv("GOOGLE_API_KEY")
                model = "gemini-flash-latest"
                kwargs["parallel_tool_calls"] = False
                kwargs["reasoning_effort"] = "high"
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
            return result.messages[-1].content

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
        system_prompt = self.experiment_type.get_system_prompt()
        user_prompt = self.experiment_type.get_user_prompt()
        async with McpWorkbench(self.server) as workbench:
            # TODO: Convert this to use custom agent in the future
            agent = AssistantAgent(
                name="Assistant",
                model_client=self.model_client,
                system_message=system_prompt,
                workbench=workbench,
                max_tool_iterations=self.max_tool_calls,
            )

            answer_invalid, result = await self.step(agent, user_prompt)

            if answer_invalid:
                raise ValueError(
                    "Failed to get a valid response after maximum retries."
                )
            else:
                return result.messages[-1].content

    async def chat(self):
        system_prompt = self.experiment_type.get_system_prompt()

        handoff_termination = HandoffTermination(target="user")
        # Define a termination condition that checks for a specific text mention.
        text_termination = TextMentionTermination("TERMINATE")

        async with McpWorkbench(self.server) as workbench:
            # TODO: Convert this to use custom agent in the future
            agent = AssistantAgent(
                name="Assistant",
                model_client=self.model_client,
                system_message=system_prompt,
                workbench=workbench,
                max_tool_iterations=1,
                reflect_on_tool_use=True,
            )
            # user = UserProxyAgent(
            #     "USER", input_func=partial(input, prompt="> (or 'exit' to quit): ")
            # )

            user = UserProxyAgent("USER", input_func=input)
            team = RoundRobinGroupChat(
                [agent, user],
                max_turns=self.max_multi_turns,
                # termination_condition=text_termination,
            )

            result = team.run_stream()
            await Console(result)

        await self.model_client.close()

    async def refine(self, feedback: str):
        raise NotImplementedError(
            "TODO: Multi-turn refine currently not supported. - S.Z."
        )
