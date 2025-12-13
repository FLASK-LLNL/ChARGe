################################################################################
## Copyright 2025 Lawrence Livermore National Security, LLC. and Binghamton University.
## See the top-level LICENSE file for details.
##
## SPDX-License-Identifier: Apache-2.0
################################################################################

from typing import List, Dict, Any, Optional

try:
    from autogen_core.models import ModelFamily, ChatCompletionClient, CreateResult
except ImportError:
    raise ImportError(
        "Please install the autogen-agentchat package to use this module."
    )
from loguru import logger


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
        **kwargs,
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
            from transformers import (
                AutoModelForCausalLM,
                AutoTokenizer,
                BitsAndBytesConfig,
            )
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
            model_path, trust_remote_code=trust_remote_code, **kwargs
        )

        # Prepare model loading arguments
        model_kwargs = {
            "device_map": device,
            "trust_remote_code": trust_remote_code,
            "torch_dtype": torch_dtype_obj,
        }

        # Load model
        self._model = AutoModelForCausalLM.from_pretrained(
            model_path, **model_kwargs, **kwargs
        )

        # Set pad token if not set
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

    async def create(self, messages: List[Any], **kwargs) -> Any:
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

                # Tokenize with proper max_length
                max_length = getattr(
                    self._model.config, "max_position_embeddings", 2048
                )
                inputs = self._tokenizer(
                    prompt,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=max_length - 512,  # Leave room for generation
                ).to(self._model.device)

                # Generate
                gen_kwargs = {
                    "max_new_tokens": kwargs.get("max_tokens", 4096),
                    "temperature": kwargs.get("temperature", 0.7),
                    "top_p": kwargs.get("top_p", 0.9),
                    "do_sample": kwargs.get("temperature", 0.7) > 0,
                    "pad_token_id": self._tokenizer.pad_token_id,
                    "eos_token_id": self._tokenizer.eos_token_id,
                }

                outputs = self._model.generate(**inputs, **gen_kwargs)

                # Decode - ensure we have valid output
                if len(outputs[0]) <= inputs["input_ids"].shape[1]:
                    # Model didn't generate anything new
                    response = "[Model produced no output]"
                else:
                    response = self._tokenizer.decode(
                        outputs[0][inputs["input_ids"].shape[1] :],
                        skip_special_tokens=True,
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

        # Parse out the final answer if present
        if (
            "<|start|>assistant<|channel|>final" in response
            or "assistantfinal" in response
        ):
            # Extract only the final channel content
            if "assistantfinal" in response:
                final_content = response.split("assistantfinal", 1)[1].strip()
            else:
                final_content = response.split("<|channel|>final", 1)[1].strip()
            response_to_return = final_content
        else:
            response_to_return = response

        return CreateResult(
            content=response_to_return,
            usage=self.actual_usage(),
            finish_reason="stop",
            cached=False,
        )

    def _format_messages(self, messages: List[Any]) -> str:
        """Format messages into a single prompt string"""
        # Convert AutoGen message objects to dicts
        formatted_messages = []
        for msg in messages:
            if hasattr(msg, "content") and hasattr(msg, "source"):
                # AutoGen message object
                role = (
                    "system"
                    if msg.source == "system"
                    else "user" if msg.source == "user" else "assistant"
                )
                formatted_messages.append({"role": role, "content": msg.content})
            elif isinstance(msg, dict):
                # Already a dict
                formatted_messages.append(msg)
            else:
                # Try to extract content
                content = getattr(msg, "content", str(msg))
                formatted_messages.append({"role": "user", "content": content})

        # Try to use chat template if available
        if hasattr(self._tokenizer, "apply_chat_template"):
            try:
                formatted = self._tokenizer.apply_chat_template(
                    formatted_messages, tokenize=False, add_generation_prompt=True
                )
                return formatted
            except Exception as e:
                logger.debug(f"DEBUG: Chat template failed: {e}, using fallback")

        # Fallback to simple formatting
        formatted = []
        for msg in formatted_messages:
            role = msg["role"]
            content = msg["content"]
            if role == "system":
                formatted.append(f"System: {content}")
            elif role == "user":
                formatted.append(f"User: {content}")
            elif role == "assistant":
                formatted.append(f"Assistant: {content}")

        formatted.append("Assistant:")
        return "\n\n".join(formatted)

    @property
    def model_info(self) -> Dict[str, Any]:
        """Return model information"""
        return self._model_info

    async def close(self):
        """Clean up resources"""
        if hasattr(self, "_model"):
            del self._model
        if hasattr(self, "_tokenizer"):
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
        used = self.count_tokens(messages, **kwargs)
        return max(0, 4096 - used)

    def total_usage(self) -> dict:
        """Return total token usage"""
        return {"prompt_tokens": 0, "completion_tokens": 0}

    def actual_usage(self) -> dict:
        """Return actual token usage for last request"""
        return {"prompt_tokens": 0, "completion_tokens": 0}

    async def create_stream(self, messages: List[Dict[str, str]], **kwargs):
        """Stream completion - not implemented for local models"""
        raise NotImplementedError(
            "Streaming not supported for local HuggingFace models"
        )
