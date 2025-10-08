try:
    from autogen_agentchat.agents import AssistantAgent, UserProxyAgent
    from autogen_core.models import ModelFamily, ChatCompletionClient, CreateResult
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
from typing import Type, Optional, Dict, Union, List, Any
from charge.Experiment import Experiment


# Custom HuggingFace Model Client
class HuggingFaceLocalClient(ChatCompletionClient):
    """Custom ChatCompletionClient for local HuggingFace models"""
    
    def __init__(
        self,
        model_path: str,
        model_info: Optional[Dict[str, Any]] = None,
        device: str = "auto",
        torch_dtype: str = "auto",
        quantization: Optional[str] = "4bit",
        trust_remote_code: bool = True,
        **kwargs
    ):
        """
        Initialize a local HuggingFace model client.
        
        Args:
            model_path: Path to local model directory or HuggingFace model ID
            model_info: Model information dict
            device: Device to load model on ("auto", "cuda", "cpu")
            torch_dtype: Torch dtype for model ("auto", "float16", "bfloat16")
            quantization: Quantization method ("4bit", "8bit", None). Defaults to "4bit"
            trust_remote_code: Whether to trust remote code
            **kwargs: Additional arguments for model/tokenizer
        """
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
            import torch
        except ImportError:
            raise ImportError(
                "Please install transformers and torch: "
                "pip install transformers torch"
            )
        
        self._model_path = model_path
        self._model_info = model_info or {
            "vision": False,
            "function_calling": True,
            "json_output": True,
            "family": ModelFamily.UNKNOWN,
            "structured_output": True,
        }
        
        # Convert string dtype to torch dtype
        dtype_map = {
            "auto": "auto",
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        torch_dtype_obj = dtype_map.get(torch_dtype, "auto")
        
        quantization_config = None
        
        # Load tokenizer
        self._tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=trust_remote_code,
            **kwargs
        )
        # Prepare model loading arguments
        model_kwargs = {
            "device_map": device,
            "trust_remote_code": trust_remote_code,
        }
        
        ## Add quantization config if specified
        #if quantization_config is not None:
        #    model_kwargs["quantization_config"] = quantization_config
        #else:
        #    # Only set torch_dtype if not quantizing
        #    model_kwargs["torch_dtype"] = torch_dtype_obj
        model_kwargs["torch_dtype"] = torch_dtype_obj
        
        # Load model
        self._model = AutoModelForCausalLM.from_pretrained(
            model_path,
            **model_kwargs,
            **kwargs
        )

        # Set pad token if not set
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token
    
    async def create(
        self,
        messages: List[Any],
        **kwargs
    ) -> Any:
        """
        Create a completion from messages.
        
        Args:
            messages: List of message dicts with 'role' and 'content'
            **kwargs: Additional generation parameters
        """
        import asyncio
        
        # Run inference in thread pool to avoid blocking event loop
        def _generate():
            try:
                # Convert messages to prompt
                prompt = self._format_messages(messages)
                
                ## Tokenize
                #inputs = self._tokenizer(
                #    prompt,
                #    return_tensors="pt",
                #    padding=True,
                #    truncation=True,
                #).to(self._model.device)

                # Tokenize with proper max_length
                max_length = getattr(self._model.config, 'max_position_embeddings', 2048)
                inputs = self._tokenizer(
                    prompt,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=max_length - 512,  # Leave room for generation
                ).to(self._model.device)
        
                # Generate
                gen_kwargs = {
                    "max_new_tokens": kwargs.get("max_tokens", 8192),
                    "temperature": kwargs.get("temperature", 0.7),
                    "top_p": kwargs.get("top_p", 0.9),
                    "do_sample": kwargs.get("temperature", 0.7) > 0,
                    "pad_token_id": self._tokenizer.pad_token_id,
                    "eos_token_id": self._tokenizer.eos_token_id,
                }
                
                outputs = self._model.generate(**inputs, **gen_kwargs)
                
                # Decode - ensure we have valid output
                if len(outputs[0]) <= inputs['input_ids'].shape[1]:
                    # Model didn't generate anything new
                    response = "[Model produced no output]"
                else:
                    response = self._tokenizer.decode(
                        outputs[0][inputs['input_ids'].shape[1]:],
                        skip_special_tokens=True
                    )
                
                # Ensure we have some response
                if not response or not response.strip():
                    response = "[Empty response from model]"
                    
                return response.strip()
            except Exception as e:
                raise RuntimeError(f"Error during model generation: {e}")
        
        # Run in thread pool to not block event loop
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(None, _generate)

        ## Debug: see what we're returning
        #print(f"DEBUG: Generated response length: {len(response)}")
        #print(f"DEBUG: Generated response preview: {response[:200]}...")

        # Parse out the final answer if present
        if '<|start|>assistant<|channel|>final' in response or 'assistantfinal' in response:
            # Extract only the final channel content
            if 'assistantfinal' in response:
                final_content = response.split('assistantfinal', 1)[1].strip()
            else:
                final_content = response.split('<|channel|>final', 1)[1].strip()
            #print(f"DEBUG: Extracted 'final' channel content:")
            #print(final_content[:500] + "...\n")
            response_to_return = final_content
        else:
            response_to_return = response
            #print(f"DEBUG: No 'final' channel found, using full response\n")
        
        return CreateResult(
            content=response_to_return,
            usage=self.actual_usage(),
            finish_reason="stop",
            cached=False
        )

    def _format_messages(self, messages: List[Any]) -> str:
        """Format messages into a single prompt string"""
        # Convert AutoGen message objects to dicts
        formatted_messages = []
        for msg in messages:
            if hasattr(msg, 'content') and hasattr(msg, 'source'):
                # AutoGen message object
                role = 'system' if msg.source == 'system' else 'user' if msg.source == 'user' else 'assistant'
                formatted_messages.append({
                    'role': role,
                    'content': msg.content
                })
            elif isinstance(msg, dict):
                # Already a dict
                formatted_messages.append(msg)
            else:
                # Try to extract content
                content = getattr(msg, 'content', str(msg))
                formatted_messages.append({
                    'role': 'user',
                    'content': content
                })
        
        ## DEBUG: Print all messages
        #print(f"DEBUG: Formatting {len(formatted_messages)} messages:")
        #for i, msg in enumerate(formatted_messages):
        #    print(f"  Message {i}: role={msg['role']}, content_length={len(msg['content'])}")
        #    print(f"    Content preview: {msg['content'][:200]}...")
        
        # Try to use chat template if available
        if hasattr(self._tokenizer, 'apply_chat_template'):
            try:
                formatted = self._tokenizer.apply_chat_template(
                    formatted_messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
                #print(f"\nDEBUG: Full formatted prompt:\n{formatted}\n")
                return formatted
            except Exception as e:
                print(f"DEBUG: Chat template failed: {e}, using fallback")
        
        # Fallback to simple formatting
        formatted = []
        for msg in formatted_messages:
            role = msg['role']
            content = msg['content']
            if role == 'system':
                formatted.append(f"System: {content}")
            elif role == 'user':
                formatted.append(f"User: {content}")
            elif role == 'assistant':
                formatted.append(f"Assistant: {content}")
        
        formatted.append("Assistant:")
        result = "\n\n".join(formatted)
        #print(f"DEBUG: Using fallback format. Prompt preview:\n{result[:500]}...\n")
        return result
    
    @property
    def model_info(self) -> Dict[str, Any]:
        """Return model information"""
        return self._model_info
    
    async def close(self):
        """Clean up resources"""
        # Clean up model and tokenizer if needed
        if hasattr(self, '_model'):
            del self._model
        if hasattr(self, '_tokenizer'):
            del self._tokenizer
    
    def capabilities(self) -> dict:
        """Return model capabilities"""
        return self._model_info
    
    def count_tokens(self, messages: List[Dict[str, str]], **kwargs) -> int:
        """Count tokens in messages"""
        prompt = self._format_messages(messages)
        tokens = self._tokenizer.encode(prompt)
        return len(tokens)
    
    def remaining_tokens(self, messages: List[Dict[str, str]], **kwargs) -> int:
        """Return remaining tokens available"""
        # Most models have ~4096 context, but this varies
        # Return a safe estimate
        used = self.count_tokens(messages, **kwargs)
        return max(0, 4096 - used)
    
    def total_usage(self) -> dict:
        """Return total token usage"""
        # Simple implementation - could be enhanced to track actual usage
        return {"prompt_tokens": 0, "completion_tokens": 0}
    
    def actual_usage(self) -> dict:
        """Return actual token usage for last request"""
        # Simple implementation - could be enhanced to track actual usage
        return {"prompt_tokens": 0, "completion_tokens": 0}
    
    async def create_stream(self, messages: List[Dict[str, str]], **kwargs):
        """Stream completion - not implemented for local models"""
        raise NotImplementedError("Streaming not supported for local HuggingFace models")

class VLLMClient(ChatCompletionClient):
    """Client for vLLM served models via OpenAI-compatible API"""
    
    def __init__(
        self,
        base_url: str = "http://localhost:8000/v1",
        model_name: str = "gpt-oss",
        api_key: str = "EMPTY",
        model_info: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        """
        Initialize vLLM client.
        
        Args:
            base_url: vLLM server URL (default: http://localhost:8000/v1)
            model_name: Model name as registered in vLLM
            api_key: API key (usually "EMPTY" for local vLLM)
            model_info: Model information dict
            **kwargs: Additional arguments
        """
        try:
            from openai import AsyncOpenAI
        except ImportError:
            raise ImportError(
                "Please install openai: pip install openai"
            )
        
        self._base_url = base_url
        self._model_name = model_name
        self._client = AsyncOpenAI(
            base_url=base_url,
            api_key=api_key,
        )
        self._model_info = model_info or {
            "vision": False,
            "function_calling": True,
            "json_output": True,
            "family": ModelFamily.UNKNOWN,
            "structured_output": True,
        }
    
    async def create(
        self,
        messages: List[Any],
        **kwargs
    ) -> CreateResult:
        """
        Create a completion from messages using vLLM.
        
        Args:
            messages: List of message objects or dicts
            **kwargs: Additional generation parameters
        """
        # Convert AutoGen message objects to OpenAI format
        formatted_messages = []
        for msg in messages:
            if hasattr(msg, 'content') and hasattr(msg, 'source'):
                # AutoGen message object
                role = 'system' if msg.source == 'system' else 'user' if msg.source == 'user' else 'assistant'
                formatted_messages.append({
                    'role': role,
                    'content': msg.content
                })
            elif isinstance(msg, dict):
                formatted_messages.append(msg)
            else:
                content = getattr(msg, 'content', str(msg))
                formatted_messages.append({
                    'role': 'user',
                    'content': content
                })
        
        # Call vLLM API
        try:
            response = await self._client.chat.completions.create(
                model=self._model_name,
                messages=formatted_messages,
                max_tokens=kwargs.get("max_tokens", 8192),
                temperature=kwargs.get("temperature", 0.7),
                top_p=kwargs.get("top_p", 0.9),
                stream=False,
            )
            
            # Extract response content
            content = response.choices[0].message.content
            
            # Parse out final channel if present (for GPT-OSS)
            if 'assistantfinal' in content:
                content = content.split('assistantfinal', 1)[1].strip()
            elif '<|channel|>final' in content:
                content = content.split('<|channel|>final', 1)[1].strip()
            
            # Return in AutoGen format
            return CreateResult(
                content=content,
                usage={
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                },
                finish_reason=response.choices[0].finish_reason,
                cached=False,
            )
        except Exception as e:
            raise RuntimeError(f"Error calling vLLM: {e}")
    
    @property
    def model_info(self) -> Dict[str, Any]:
        """Return model information"""
        return self._model_info
    
    def capabilities(self) -> dict:
        """Return model capabilities"""
        return self._model_info
    
    def count_tokens(self, messages: List[Any], **kwargs) -> int:
        """Estimate token count (rough approximation)"""
        # Rough estimate: 4 chars per token
        total_chars = sum(len(str(m.get('content', '') if isinstance(m, dict) else getattr(m, 'content', ''))) for m in messages)
        return total_chars // 4
    
    def remaining_tokens(self, messages: List[Any], **kwargs) -> int:
        """Return remaining tokens available"""
        used = self.count_tokens(messages, **kwargs)
        return max(0, 8192 - used)  # Assume 8K context
    
    def total_usage(self) -> dict:
        """Return total token usage"""
        return {"prompt_tokens": 0, "completion_tokens": 0}
    
    def actual_usage(self) -> dict:
        """Return actual token usage for last request"""
        return {"prompt_tokens": 0, "completion_tokens": 0}
    
    async def create_stream(self, messages: List[Any], **kwargs):
        """Stream completion - not yet implemented"""
        raise NotImplementedError("Streaming not yet implemented for vLLM client")
    
    async def close(self):
        """Clean up resources"""
        await self._client.close()

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
        server_path: Optional[Union[str, list[str]]] = None,
        server_url: Optional[Union[str, list[str]]] = None,
        server_kwargs: Optional[dict] = None,
        max_tool_calls: int = 15,
        check_response: bool = False,
        max_multi_turns: int = 100,
        # New parameters for local HuggingFace models
        local_model_path: Optional[str] = "/p/vast1/flask/models/gpt-oss-120b",
        device: str = "auto",
        torch_dtype: str = "auto",
        quantization: Optional[str] = "4bit",
    ):
        """Initializes the AutoGenClient.

        Args:
            experiment_type (Type[Experiment]): The experiment class to use.
            path (str, optional): Path to save generated MCP server files. Defaults to ".".
            max_retries (int, optional): Maximum number of retries for failed tasks. Defaults to 3.
            backend (str, optional): Backend to use: "openai", "gemini", "ollama", "huggingface", "livai" or "livchat". Defaults to "openai".
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
            local_model_path (Optional[str], optional): Path to local HuggingFace model directory. 
                                                       Required when backend="huggingface". Defaults to local FLASK gpt-oss-120b.
            device (str, optional): Device to load model on ("auto", "cuda", "cpu"). Defaults to "auto".
            torch_dtype (str, optional): Torch dtype for model ("auto", "float16", "bfloat16"). Defaults to "auto".
            quantization (Optional[str], optional): Quantization method ("4bit", "8bit", None). Defaults to "4bit".
        
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
        
        # Initialize servers list if not already done by parent
        if not hasattr(self, 'servers'):
            self.servers = []

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
            
            if backend == "huggingface":
                # Use local HuggingFace model
                if local_model_path is None:
                    raise ValueError(
                        "local_model_path must be provided when backend='huggingface'"
                    )
                
                self.model_client = HuggingFaceLocalClient(
                    model_path=local_model_path,
                    model_info=model_info,
                    device=device,
                    torch_dtype=torch_dtype,
                    quantization=quantization,
                    **self.model_kwargs,
                )
            elif backend == "vllm":
                # Use vLLM server
                vllm_url = self.model_kwargs.get(
                    "vllm_url", 
                    os.getenv("VLLM_URL", "http://localhost:8000/v1")
                )
                vllm_model = self.model_kwargs.get(
                    "vllm_model", 
                    os.getenv("VLLM_MODEL", model or "/p/vast1/flask/models/gpt-oss-120b")
                )
                print(f"\n  ==> VLLM backend vllm_url: {vllm_url}")
                print(f"\n  ==> VLLM backend vllm_model: {vllm_model}")

                self.model_client = VLLMClient(
                    base_url=vllm_url,
                    model_name=vllm_model,
                    model_info=model_info,
                )
            elif backend == "ollama":
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
                if isinstance(server_path, str):
                    server_path = [server_path]
                for sp in server_path:
                    self.servers.append(StdioServerParams(command="python3", args=[sp]))
            if server_url is not None:
                if isinstance(server_url, str):
                    server_url = [server_url]
                for su in server_url:
                    self.servers.append(
                        SseServerParams(url=su, **(server_kwargs or {}))
                    )
        self.messages = []

    @staticmethod
    def configure(model: Optional[str], backend: str) -> tuple[str, str, Optional[str], Dict[str, str]]:
        import httpx

        kwargs = {}
        API_KEY = None
        default_model = None
        if backend in ["openai", "gemini", "livai", "livchat"]:
            if backend == "openai":
                API_KEY = os.getenv("OPENAI_API_KEY")
                default_model = "gpt-4"
                kwargs["parallel_tool_calls"] = False
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
        elif backend in ["huggingface"]:
            default_model = None  # Must be provided via local_model_path
        elif backend in ["vllm"]:
            default_model = "gpt-oss"  # Default vLLM model name

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
        system_prompt = self.experiment_type.get_system_prompt()
        user_prompt = self.experiment_type.get_user_prompt()
        assert (
            user_prompt is not None
        ), "User prompt must be provided for single-turn run."

        assert (
            len(self.servers) > 0
        ), "No MCP servers available. Please provide server_path or server_url."

        wokbenches = [McpWorkbench(server) for server in self.servers]

        # Start the servers
        for workbench in wokbenches:
            await workbench.start()

        # async with McpWorkbench(self.server) as workbench:
        #     # TODO: Convert this to use custom agent in the future
        agent = AssistantAgent(
            name="Assistant",
            model_client=self.model_client,
            system_message=system_prompt,
            workbench=wokbenches,
            max_tool_iterations=self.max_tool_calls,
        )

        answer_invalid, result = await self.step(agent, user_prompt)

        for workbench in wokbenches:
            await workbench.stop()

        if answer_invalid:
            raise ValueError("Failed to get a valid response after maximum retries.")
        else:
            return result.messages[-1].content

    async def chat(self):
        system_prompt = self.experiment_type.get_system_prompt()

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

        # TODO: Convert this to use custom agent in the future
        agent = AssistantAgent(
            name="Assistant",
            model_client=self.model_client,
            system_message=system_prompt,
            workbench=wokbenches,
            max_tool_iterations=self.max_tool_calls,
            reflect_on_tool_use=True,
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
