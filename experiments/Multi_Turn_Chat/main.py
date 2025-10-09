import argparse
import asyncio
from charge.Experiment import Experiment
from charge.clients.Client import Client
from charge.clients.autogen import AutoGenClient

parser = argparse.ArgumentParser()

# Add standard CLI arguments
Client.add_std_parser_arguments(parser)

class ChargeChatExperiment(Experiment):
    def __init__(
        self,
    ):
        system_prompt = (
            "You are a world-class chemist. Your task is to generate unique molecules "
            "based on the lead molecule provided by the user. The generated molecules "
            "should be chemically valid and diverse, exploring different chemical spaces "
            "while maintaining some structural similarity to the lead molecule. "
            "Provide the final answer in a clear and concise manner."
        )

        super().__init__(system_prompt=system_prompt, user_prompt=None)
        print("ChargeChatExperiment initialized with the provided prompts.")

        self.system_prompt = system_prompt
        self.user_prompt = None


if __name__ == "__main__":

    args = parser.parse_args()
    server_url = args.server_urls[0]
    assert server_url is not None, "Server URL must be provided"
    assert server_url.endswith("/sse"), "Server URL must end with /sse"
    myexperiment = ChargeChatExperiment()

    from openai import AsyncOpenAI
    from autogen_ext.agents.openai import OpenAIAgent
    from autogen_core import CancellationToken
    from autogen_agentchat.messages import TextMessage
    import os
    import httpx
    async def example():
        cancellation_token = CancellationToken()
        client = AsyncOpenAI(
            # api_key = os.getenv("OPENAI_API_KEY"),
            # base_url = os.getenv("LIVAI_BASE_URL"),
            # http_client = httpx.AsyncClient(verify=False),
        )
        client.api_key = os.getenv("OPENAI_API_KEY")
        client.base_url = os.getenv("LIVAI_BASE_URL")
        client.http_client = httpx.AsyncClient(verify=False)

        agent = OpenAIAgent(
            name="foo",
            description='A simple OpenAI agent using the Responses API',
            client=client,
            model='gpt-4.1',
            instructions='You are a helpful assistant.',
#            parallel_tool_calls = False,
#            reasoning_effort = "high",
        )
        response = await agent.on_messages([TextMessage(source="user", content="Hello!")], cancellation_token)
        print(response)
        response = await agent.on_messages([TextMessage(source="user", content="I need to design a new molecule!")], cancellation_token)
        print(response)
        
    asyncio.run(example())
##    exit()
    
    (model, backend, API_KEY, kwargs) = AutoGenClient.configure(
        args.model, args.backend
    )

    runner = AutoGenClient(
        experiment_type=myexperiment,
        backend=backend,
        model=model,
        api_key=API_KEY,
        model_kwargs=kwargs,
        # server_path=server_path,
        server_url=server_url,
    )

    results = asyncio.run(runner.chat())

    print(f"Experiment completed. Results: {results}")
