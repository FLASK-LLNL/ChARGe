try:
    from autogen_core.models import ModelFamily, ChatCompletionClient, CreateResult
except ImportError:
    raise ImportError(
        "Please install the autogen-agentchat package to use this module."
    )

from typing import Type, Optional, Dict, Union, List, Any

class VLLMClient(ChatCompletionClient):
    """Client for vLLM served models via OpenAI-compatible API"""
    
    def __init__(
        self,
        base_url: str = "http://localhost:8000/v1",
        model_name: str = "gpt-oss",
        api_key: str = "EMPTY",
        reasoning_effort: str = "medium",
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
            reasoning_effort: Reasoning level for GPT-OSS (low, medium, high)
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
        self._reasoning_effort = reasoning_effort
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
                extra_body={"reasoning_effort": self._reasoning_effort},
            )
            
            # Extract response content
            message = response.choices[0].message
            final_content = message.content
            
            # Get reasoning from reasoning_content field if available
            analysis_content = getattr(message, 'reasoning_content', None)

            ## Debug: Check reasoning and final content
            #print(f"\n=== DEBUG VLLMClient ===")
            #print(f"Analysis captured: {analysis_content is not None}")
            #if analysis_content:
            #    print(f"Analysis length: {len(analysis_content)}")
            #    print(f"Analysis preview: {analysis_content[:300]}...")
            #print(f"Final content length: {len(final_content)}")
            #print(f"======================\n")
            
            # Return in AutoGen format (with reasoning in thought field)
            return CreateResult(
                content=final_content,
                usage={
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                },
                finish_reason=response.choices[0].finish_reason,
                cached=False,
                thought=analysis_content,
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

