import pytest
import os


@pytest.mark.skipif(
    os.getenv("OPENAI_API_KEY", None) is None, reason="OPENAI_API_KEY not set"
)
class TestOpenAISimpleTask:
    @pytest.fixture(autouse=True)
    def setup_fixture(self):
        from charge.clients.autogen import AutoGenPool

        self.agent_pool = AutoGenPool(model="gpt-5", backend="openai")

    @pytest.mark.asyncio
    async def test_openai_simple_task(self):
        from charge.tasks.Task import Task

        task = Task(
            system_prompt="You are a helpful assistant.",
            user_prompt="What is the capital of France?",
        )

        agent = self.agent_pool.create_agent(task=task)

        response = await agent.run()
        print("Response from Agent:", response)
        assert "Paris" in response
