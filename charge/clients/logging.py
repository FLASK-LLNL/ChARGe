from autogen_ext.models.openai import OpenAIChatCompletionClient

## Create a custom wrapper to log responses
#class LoggingModelClient(OpenAIChatCompletionClient):
#    async def create(self, messages, **kwargs):
#        response = await super().create(messages, **kwargs)
#        
#        # Log the full response
#        print("Full response:", response)
#        if hasattr(response, 'choices'):
#            for choice in response.choices:
#                if hasattr(choice, 'message'):
#                    print("Message:", choice.message)
#                    # Check for reasoning/analysis content
#                    if hasattr(choice.message, 'content'):
#                        print("Content:", choice.message.content)
#        
#        return response

import sys
from openai.types.chat import ChatCompletion

class LoggingModelClient(OpenAIChatCompletionClient):
    async def create(self, messages, **kwargs):
        sys.stderr.write("!!! LoggingModelClient.create() WAS CALLED !!!\n")
        sys.stderr.flush()
        
        # Call the parent to get the response
        response = await super().create(messages, **kwargs)
        
        # CRITICAL: Access the raw API response before AutoGen processes it
        # The reasoning_content is in the raw response but might not be mapped to CreateResult
        
        # Try to extract reasoning_content from the internal state
        # Option 1: Check if we can access the raw completion object
        reasoning_content = None
        
        # The response object might have internal attributes we can access
        if hasattr(response, '__dict__'):
            # Check for thought attribute (which AutoGen might populate)
            if hasattr(response, 'thought') and response.thought:
                reasoning_content = response.thought
        
        # If reasoning_content is not in the CreateResult, we need to intercept earlier
        # We'll need to override at a lower level to catch the raw API response
        
        # Log what we have
        sys.stderr.write("\n" + "="*80 + "\n")
        sys.stderr.write("GPT-OSS MODEL RESPONSE\n")
        sys.stderr.write("="*80 + "\n")
        
        if reasoning_content:
            sys.stderr.write("\n[ANALYSIS CHANNEL]\n")
            sys.stderr.write("-" * 80 + "\n")
            sys.stderr.write(str(reasoning_content) + "\n")
            sys.stderr.write("-" * 80 + "\n")
        
        if hasattr(response, 'content') and response.content:
            sys.stderr.write("\n[FINAL CHANNEL]\n")
            sys.stderr.write("-" * 80 + "\n")
            if isinstance(response.content, list):
                for item in response.content:
                    sys.stderr.write(str(item) + "\n")
            else:
                sys.stderr.write(str(response.content) + "\n")
            sys.stderr.write("-" * 80 + "\n")
        
        if hasattr(response, 'usage') and response.usage:
            sys.stderr.write("\n[USAGE]\n")
            sys.stderr.write("-" * 80 + "\n")
            sys.stderr.write(f"Prompt tokens: {response.usage.prompt_tokens}\n")
            sys.stderr.write(f"Completion tokens: {response.usage.completion_tokens}\n")
            sys.stderr.write("-" * 80 + "\n")
        
        sys.stderr.write("\n" + "="*80 + "\n\n")
        sys.stderr.flush()
        
        return response

#class LoggingModelClient(OpenAIChatCompletionClient):
#    async def create(self, messages, **kwargs):
#        # Verify the client is being called
#        sys.stderr.write("!!! LoggingModelClient.create() WAS CALLED !!!\n")
#        sys.stderr.flush()
#        
#        response = await super().create(messages, **kwargs)
#        
#        # Format and display the response
#        sys.stderr.write("\n" + "="*80 + "\n")
#        sys.stderr.write("GPT-OSS MODEL RESPONSE\n")
#        sys.stderr.write("="*80 + "\n")
#        
#        # Analysis/Reasoning Channel (thought)
#        if hasattr(response, 'thought') and response.thought:
#            sys.stderr.write("\n[ANALYSIS CHANNEL]\n")
#            sys.stderr.write("-" * 80 + "\n")
#            sys.stderr.write(str(response.thought) + "\n")
#            sys.stderr.write("-" * 80 + "\n")
#        
#        # Final Channel (content) - handle both string and list
#        if hasattr(response, 'content') and response.content:
#            sys.stderr.write("\n[FINAL CHANNEL]\n")
#            sys.stderr.write("-" * 80 + "\n")
#            if isinstance(response.content, list):
#                for item in response.content:
#                    sys.stderr.write(str(item) + "\n")
#            else:
#                sys.stderr.write(str(response.content) + "\n")
#            sys.stderr.write("-" * 80 + "\n")
#        
#        # Usage information
#        if hasattr(response, 'usage') and response.usage:
#            sys.stderr.write("\n[USAGE]\n")
#            sys.stderr.write("-" * 80 + "\n")
#            sys.stderr.write(f"Prompt tokens: {response.usage.prompt_tokens}\n")
#            sys.stderr.write(f"Completion tokens: {response.usage.completion_tokens}\n")
#            sys.stderr.write("-" * 80 + "\n")
#        
#        sys.stderr.write("\n" + "="*80 + "\n\n")
#        sys.stderr.flush()
#        
#        return response


#class LoggingModelClient(OpenAIChatCompletionClient):
#    async def create(self, messages, **kwargs):
#        # Verify the client is being called
#        sys.stderr.write("!!! LoggingModelClient.create() WAS CALLED !!!\n")
#        sys.stderr.flush()
#        
#        response = await super().create(messages, **kwargs)
#        
#        # Debug: What is the response object?
#        sys.stderr.write("\n" + "="*80 + "\n")
#        sys.stderr.write("DEBUG: Response type and attributes\n")
#        sys.stderr.write("="*80 + "\n")
#        sys.stderr.write(f"Response type: {type(response)}\n")
#        sys.stderr.write(f"Response attributes: {dir(response)}\n")
#        sys.stderr.write(f"Has 'choices': {hasattr(response, 'choices')}\n")
#        
#        # Try to print the actual response object
#        sys.stderr.write(f"\nResponse object: {response}\n")
#        
#        # Check for different possible attributes
#        if hasattr(response, '__dict__'):
#            sys.stderr.write(f"\nResponse.__dict__: {response.__dict__}\n")
#        
#        if hasattr(response, 'content'):
#            sys.stderr.write(f"\nResponse.content: {response.content}\n")
#        
#        if hasattr(response, 'thought'):
#            sys.stderr.write(f"\nResponse.thought: {response.thought}\n")
#        
#        sys.stderr.write("\n" + "="*80 + "\n\n")
#        sys.stderr.flush()
#        
#        # Original code
#        if hasattr(response, 'choices'):
#            for i, choice in enumerate(response.choices):
#                sys.stderr.write("\n" + "="*80 + "\n")
#                sys.stderr.write(f"CHOICE {i}\n")
#                sys.stderr.write("="*80 + "\n")
#                
#                if hasattr(choice.message, 'reasoning_content') and choice.message.reasoning_content:
#                    sys.stderr.write("\n[ANALYSIS CHANNEL]\n")
#                    sys.stderr.write("-" * 80 + "\n")
#                    sys.stderr.write(choice.message.reasoning_content + "\n")
#                    sys.stderr.write("-" * 80 + "\n")
#                
#                if hasattr(choice.message, 'tool_calls') and choice.message.tool_calls:
#                    sys.stderr.write(f"\n[TOOL CALLS] ({len(choice.message.tool_calls)} total)\n")
#                    sys.stderr.write("-" * 80 + "\n")
#                    for idx, tc in enumerate(choice.message.tool_calls, 1):
#                        sys.stderr.write(f"{idx}. {tc.function.name}\n")
#                        sys.stderr.write(f"   Arguments: {tc.function.arguments}\n")
#                        sys.stderr.write(f"   ID: {tc.id}\n")
#                    sys.stderr.write("-" * 80 + "\n")
#                
#                if hasattr(choice.message, 'content') and choice.message.content:
#                    sys.stderr.write("\n[FINAL CHANNEL]\n")
#                    sys.stderr.write("-" * 80 + "\n")
#                    sys.stderr.write(choice.message.content + "\n")
#                    sys.stderr.write("-" * 80 + "\n")
#                
#                sys.stderr.write("\n" + "="*80 + "\n\n")
#                sys.stderr.flush()
#        
#        return response
class InspectingModelClient(OpenAIChatCompletionClient):
    async def create(self, messages, **kwargs):
        # Make the request and inspect
        response = await super().create(messages, **kwargs)
        
        # The response object should contain all channels
        # gpt-oss typically puts reasoning in response.choices[0].message.reasoning_content
        # or similar field
        for choice in response.choices:
            if hasattr(choice.message, 'reasoning_content'):
                print("Analysis channel:", choice.message.reasoning_content)
            if hasattr(choice.message, 'content'):
                print("Final channel:", choice.message.content)
            if hasattr(choice.message, 'tool_calls'):
                print("Tool calls:", choice.message.tool_calls)
                
        return response
