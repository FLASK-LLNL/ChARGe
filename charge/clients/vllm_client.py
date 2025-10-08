################################################################################
## Copyright 2025 Lawrence Livermore National Security, LLC. and Binghamton University.
## See the top-level LICENSE file for details.
##
## SPDX-License-Identifier: Apache-2.0
################################################################################

from typing import List, Dict, Any, Optional
try:
    from autogen_core.models import ModelFamily, ChatCompletionClient, CreateResult
    from openai import AsyncOpenAI
except ImportError:
    raise ImportError(
        "Please install the required packages: pip install autogen-agentchat openai"
    )


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
                max_tokens=kwargs.get("max_tokens", 4096),
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
