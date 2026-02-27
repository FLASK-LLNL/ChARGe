import pytest
import os


@pytest.mark.skipif(
    os.getenv("OPENAI_API_KEY", None) is None, reason="OPENAI_API_KEY not set"
)
class TestOpenAISimpleTask:
    @pytest.fixture(autouse=True)
    def setup_fixture(self):
        from charge.clients.autogen import AutoGenBackend

        self.agent_backend = AutoGenBackend(model="gpt-5", backend="openai")

    @pytest.mark.asyncio
    async def test_openai_simple_task(self):
        from charge.tasks.task import Task

        task = Task(
            system_prompt="You are a helpful assistant.",
            user_prompt="What is the capital of France?",
        )

        agent = self.agent_backend.create_agent(task=task)

        response = await agent.run()
        print("Response from Agent:", response)
        assert "Paris" in response

    @pytest.mark.asyncio
    async def test_openai_math_task(self):
        from charge.tasks.task import Task

        task = Task(
            system_prompt="You are a helpful assistant.",
            user_prompt="What is 15 multiplied by 12?",
        )

        agent = self.agent_backend.create_agent(task=task)

        response = await agent.run()
        print("Response from Agent:", response)
        assert "180" in response

    @pytest.mark.asyncio
    async def test_openai_structured_output_task(self):
        from charge.tasks.task import Task
        from pydantic import BaseModel
        import re

        class MathAnswerSchema(BaseModel):
            answer: int

        task = Task(
            system_prompt="You are a helpful assistant.",
            user_prompt="What is 20 plus 22?",
            structured_output_schema=MathAnswerSchema,
        )

        agent = self.agent_backend.create_agent(task=task)

        response = await agent.run()
        print("Response from Agent:", response)
        assert re.search(r'"answer":\s*42', response)
