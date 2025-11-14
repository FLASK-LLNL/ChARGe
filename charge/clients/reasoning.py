from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_core.models import CreateResult, RequestUsage
import sys
import json

class ReasoningCaptureClient(OpenAIChatCompletionClient):
    """Wrapper that captures reasoning from raw API responses"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.reasoning_history = []
    
    async def create(self, messages, **kwargs):
        """Intercept the raw API call to capture reasoning_content"""
        
        sys.stderr.write("\n" + "="*80 + "\n")
        sys.stderr.write("ReasoningCaptureClient.create() called\n")
        sys.stderr.write(f"kwargs keys: {kwargs.keys()}\n")
        sys.stderr.flush()
        
        # The model is passed in the request, extract it from kwargs if present
        # OR get it from the parent class's stored configuration
        model_name = kwargs.get('model')
        if not model_name:
            # Try to access parent's model info
            if hasattr(self, '_model_info'):
                sys.stderr.write(f"Found _model_info\n")
            # Just call parent - it knows how to handle it
            sys.stderr.write("Calling parent's create method\n")
            sys.stderr.flush()
            
            # Call parent but we need to intercept the actual HTTP response
            # The problem is the parent converts it already
            # So we need a different approach - let's override at HTTP level
            
        # Actually, let's just call parent and extract from the result
        response = await super().create(messages, **kwargs)
        
        sys.stderr.write(f"Got response, type: {type(response)}\n")
        sys.stderr.write(f"Response has thought: {hasattr(response, 'thought')}\n")
        if hasattr(response, 'thought'):
            sys.stderr.write(f"Thought value: {response.thought}\n")
            if response.thought:
                self.reasoning_history.append(response.thought)
                sys.stderr.write("[CAPTURED REASONING]\n")
                sys.stderr.write(f"{response.thought}\n")
        
        sys.stderr.write("="*80 + "\n\n")
        sys.stderr.flush()
        
        return response
    
    def get_reasoning_history(self):
        """Get all captured reasoning"""
        return self.reasoning_history
    
    def clear_reasoning_history(self):
        """Clear reasoning history"""
        self.reasoning_history = []

import logging
import json
import sys
from autogen_agentchat.messages import ThoughtEvent

class ReasoningCapture(logging.Handler):
    """Custom log handler to capture reasoning from autogen_core.events"""
    
    def __init__(self):
        super().__init__()
        self.reasoning_history = []
    
    def emit(self, record):
        """Called for each log message"""
        if record.name == "autogen_core.events":
            try:
                log_data = json.loads(record.getMessage())
                
                if log_data.get("type") == "LLMCall" and "response" in log_data:
                    response = log_data["response"]
                    if "choices" in response:
                        for choice in response["choices"]:
                            message = choice.get("message", {})
                            reasoning = message.get("reasoning_content")
                            if reasoning:
                                self.reasoning_history.append(reasoning)
                                sys.stderr.write(f"\n[CAPTURED INTERMEDIATE REASONING]\n{reasoning}\n\n")
                                sys.stderr.flush()
            except (json.JSONDecodeError, KeyError):
                pass
    
    def get_reasoning_history(self):
        return self.reasoning_history
    
    def clear_reasoning_history(self):
        self.reasoning_history = []
    
    def inject_into_result(self, result):
        """Inject captured reasoning as ThoughtEvents into the result messages"""
        if not hasattr(result, 'messages'):
            return result
        
        # Find existing ThoughtEvents to see the pattern
        existing_thoughts = [msg for msg in result.messages if hasattr(msg, 'type') and msg.type == 'ThoughtEvent']
        
        # If we captured more reasoning than what's in the result, add them
        if len(self.reasoning_history) > len(existing_thoughts):
            # We need to insert ThoughtEvents between the tool calls
            # This is tricky because we need to figure out where to insert them
            
            # Simpler approach: append all missing reasoning at the end before final message
            new_messages = list(result.messages[:-1])  # All except last
            
            # Add ThoughtEvents for each captured reasoning
            for reasoning in self.reasoning_history[len(existing_thoughts):]:
                thought_event = type(result.messages[0])(
                    source='Assistant',
                    content=reasoning,
                    type='ThoughtEvent'
                )
                new_messages.append(thought_event)
            
            # Add the final message back
            new_messages.append(result.messages[-1])
            
            # Replace messages
            result.messages = new_messages
        
        return result


### Usage:
##reasoning_capture = ReasoningCapture()
##reasoning_capture.setLevel(logging.INFO)
##logging.getLogger("autogen_core.events").addHandler(reasoning_capture)
##
### Run agent
##agent = AssistantAgent(...)
##answer_invalid, result = await self.step(agent, user_prompt)
##
### Inject reasoning into result
##result = reasoning_capture.inject_into_result(result)
##
### Now result.messages contains all ThoughtEvents with intermediate reasoning!
##all_reasoning = reasoning_capture.get_reasoning_history()
##print("\n==>> All reasoning steps (including intermediate):")
##for i, reasoning in enumerate(all_reasoning, 1):
##    print(f"{i}. {reasoning}")
##
### And you can also access them from result.messages:
##thought_messages = [msg for msg in result.messages if hasattr(msg, 'type') and msg.type == 'ThoughtEvent']
##print(f"\nThoughtEvents in result: {len(thought_messages)}")
