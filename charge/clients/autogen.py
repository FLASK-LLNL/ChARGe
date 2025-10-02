try:
    from autogen_agentchat.agents import AssistantAgent
    from autogen_core.models import ModelFamily, ChatCompletionClient
    from autogen_ext.tools.mcp import StdioServerParams, McpWorkbench, SseServerParams
    from autogen_agentchat.messages import TextMessage
except ImportError:
    raise ImportError(
        "Please install the autogen-agentchat package to use this module."
    )
import asyncio
import os
from charge.clients.Client import Client
from typing import Type, Optional
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
    ):
        """Initializes the AutoGenClient.

        Args:
            experiment_type (Type[Experiment]): The experiment class to use.
            path (str, optional): Path to save generated MCP server files. Defaults to ".".
            max_retries (int, optional): Maximum number of retries for failed tasks. Defaults to 3.
            backend (str, optional): Backend to use: "openai", "gemini", "ollama", or "livchat". Defaults to "openai".
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

            result = await agent.run(task=user_prompt)

            for msg in result.messages:
                if isinstance(msg, TextMessage):
                    self.messages.append(msg.content)

            if not self.check_response:
                assert isinstance(result.messages[-1], TextMessage)
                return result.messages[-1].content

            def check_invalid_response(result) -> bool:
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

            answer_invalid = False
            if isinstance(result.messages[-1], TextMessage):
                answer_invalid = check_invalid_response(result.messages[-1].content)
            else:
                answer_invalid = True
            retries = 0
            while answer_invalid and retries < self.max_retries:
                new_user_prompt = (
                    "The previous response was invalid. Please try again.\n\n"
                    + user_prompt
                )
                # print("Retrying with new prompt...")
                result = await agent.run(task=new_user_prompt)
                if isinstance(result.messages[-1], TextMessage):
                    answer_invalid = check_invalid_response(result.messages[-1].content)
                else:
                    answer_invalid = True
                retries += 1
            if answer_invalid:
                raise ValueError(
                    "Failed to get a valid response after maximum retries."
                )
            else:
                return result.messages[-1].content

    async def refine(self, feedback: str):
        raise NotImplementedError(
            "TODO: Multi-turn refine currently not supported. - S.Z."
        )
