################################################################################
## Copyright 2025 Lawrence Livermore National Security, LLC. and Binghamton University.
## See the top-level LICENSE file for details.
##
## SPDX-License-Identifier: Apache-2.0
################################################################################
import pytest
import os


@pytest.mark.skipif(
    os.getenv("OPENAI_API_KEY", None) is None, reason="OPENAI_API_KEY not set"
)
class TestAgentFrameworkIntegration:
    """Integration tests for Agent Framework implementation."""

    @pytest.fixture(autouse=True)
    def setup_fixture(self):
        """Set up test fixtures with tasks and agent factory."""
        from charge.tasks.task import Task
        from charge.clients.agentframework import AgentFrameworkBackend
        from charge.clients.agent_factory import AgentFactory
        from charge.experiments.experiment import Experiment
        from pydantic import BaseModel

        # Create and register agent backend with OpenAI backend
        self.agent_backend = AgentFrameworkBackend(
            model="gpt-4o-mini", backend="openai"
        )
        AgentFactory.register_backend("agentframework", self.agent_backend)

        # First task: Simple arithmetic
        first_task = Task(
            system_prompt="You are a helpful assistant that is capable of "
            + "doing arithmetic and returning an explanation of how you arrived at the "
            + "answer. Provide concise and fast responses.",
            user_prompt="What is 10 plus 5?",
        )

        # Define structured output schema
        class MathExplanationSchema(BaseModel):
            answer: int
            explanation: str

        self.schema = MathExplanationSchema

        # Second task: Convert to structured JSON
        second_task = Task(
            system_prompt="You are a helpful assistant that is capable of "
            + "taking an answer and explanation and converting it to a structured JSON format. "
            + "Provide concise and fast responses.",
            user_prompt="Take the previous answer and explanation "
            + "and convert it into a JSON",
            structured_output_schema=MathExplanationSchema,
        )

        # Create experiment with two tasks
        self.experiment = Experiment(task=[first_task, second_task])

        # Third task: Use structured output from previous task
        third_task = Task(
            system_prompt="You are a helpful assistant that can parse JSON from text "
            + "that can extract fields and do arithmetic.",
            user_prompt="Extract the 'answer' field from the previous JSON and "
            + "multiply it by 3.",
        )
        self.third_task = third_task

        # Alternate third task for state reload testing
        alternate_third_task = Task(
            system_prompt="You are a helpful assistant that can parse JSON from text "
            + "that can extract fields and do arithmetic.",
            user_prompt="Extract the 'answer' field from the previous JSON and "
            + "multiply it by 4.",
        )
        self.alternate_third_task = alternate_third_task

    @pytest.mark.asyncio
    async def test_simple_task_execution(self):
        """Test basic task execution with Agent Framework."""
        from charge.tasks.task import Task
        from charge.clients.agent_factory import AgentFactory

        task = Task(
            system_prompt="You are a helpful assistant.",
            user_prompt="What is the capital of France? Answer in one word.",
        )

        agent = AgentFactory.create_agent(task=task, backend="agentframework")
        result = await agent.run()

        print(f"Simple task result: {result}")
        assert "Paris" in result or "paris" in result.lower()

    @pytest.mark.asyncio
    async def test_structured_output(self):
        """Test structured output validation with Pydantic schemas."""
        from charge.tasks.task import Task
        from charge.clients.agent_factory import AgentFactory
        from pydantic import BaseModel

        class CityInfo(BaseModel):
            city: str
            country: str

        task = Task(
            system_prompt="You are a helpful assistant that provides city information.",
            user_prompt="Provide information about Paris. Return as JSON with 'city' and 'country' fields.",
            structured_output_schema=CityInfo,
        )

        agent = AgentFactory.create_agent(task=task, backend="agentframework")
        result = await agent.run()

        print(f"Structured output result: {result}")

        # Validate against schema
        assert task.check_output_formatting(result)
        parsed = CityInfo.model_validate_json(result)
        assert parsed.city.lower() == "paris"
        assert parsed.country.lower() == "france"

    @pytest.mark.asyncio
    async def test_linear_experiment_run(self, mocker):
        """Test sequential multi-task experiment execution with context sharing."""
        import re

        # Run the experiment with two tasks
        await self.experiment.run_async(backend="agentframework")
        finished_tasks = self.experiment.get_finished_tasks()
        assert len(finished_tasks) == 2

        first_task, first_task_result = finished_tasks[0]
        second_task, second_task_result = finished_tasks[1]

        print("First Task Result:", first_task_result)
        print("Second Task Result:", second_task_result)

        # Verify first task result contains the answer
        assert re.search(r"15", first_task_result)

        # Verify second task result is valid structured output
        assert second_task.check_output_formatting(second_task_result)

        # Parse and validate the JSON output
        parse_output = self.schema.model_validate_json(second_task_result)
        assert parse_output.answer == 15

        # Test state serialization
        self.state = self.experiment.save_state()
        print("Serialized State:", self.state)

        # Test state loading
        self.experiment.load_state(self.state)

        # Add third task and run
        self.experiment.add_task(self.third_task)
        assert self.experiment.remaining_tasks() == 1

        await self.experiment.run_async(backend="agentframework")
        finished_tasks = self.experiment.get_finished_tasks()
        assert len(finished_tasks) == 3

        third_task, third_task_result = finished_tasks[2]
        print("Third Task Result:", third_task_result)
        assert "45" in third_task_result

        # Test state reload and alternate path
        save_file_content = self.state

        mocker.patch("builtins.open", mocker.mock_open(read_data=save_file_content))

        with open("mock_save_file.json", "r") as f:
            content = f.read()

        assert content == save_file_content

        # Reload state (should be back to 2 finished tasks)
        self.experiment.load_state(content)
        assert self.experiment.remaining_tasks() == 0

        print("Experiment State Loaded Successfully")

        # Add alternate third task and run
        self.experiment.add_task(self.alternate_third_task)
        assert self.experiment.remaining_tasks() == 1

        await self.experiment.run_async(backend="agentframework")
        finished_tasks = self.experiment.get_finished_tasks()

        third_task, third_task_result = finished_tasks[-1]
        print("Third Task Result After Reloading State:", third_task_result)
        assert "60" in third_task_result

    @pytest.mark.asyncio
    async def test_experiment_memory_persistence(self):
        """Test that experiment memory persists across agent creation."""
        from charge.tasks.task import Task

        task1 = Task(
            system_prompt="You are a helpful assistant.",
            user_prompt="Remember this number: 42",
        )

        task2 = Task(
            system_prompt="You are a helpful assistant.",
            user_prompt="What number did I ask you to remember?",
        )

        # Reuse existing experiment but reset it properly
        experiment = self.experiment
        experiment.reset()  # Clears finished_tasks, tasks, and memory

        # Add the new tasks
        experiment.add_task(task1)
        experiment.add_task(task2)

        await experiment.run_async(backend="agentframework")
        finished_tasks = experiment.get_finished_tasks()

        assert len(finished_tasks) == 2
        _, result1 = finished_tasks[0]
        _, result2 = finished_tasks[1]

        print(f"Task 1 result: {result1}")
        print(f"Task 2 result: {result2}")

        # Second task should reference the number from first task
        assert "42" in result2

    @pytest.mark.asyncio
    async def test_agent_retry_logic(self):
        """Test that agent retry logic works correctly."""
        from charge.tasks.task import Task
        from charge.clients.agent_factory import AgentFactory
        from pydantic import BaseModel

        class NumberSchema(BaseModel):
            number: int

        # Task that requires specific output format
        task = Task(
            system_prompt="You are a helpful assistant.",
            user_prompt="Return the number 7 as JSON with field 'number'.",
            structured_output_schema=NumberSchema,
        )

        agent = AgentFactory.create_agent(
            task=task, backend="agentframework", max_retries=3
        )
        result = await agent.run()

        print(f"Retry test result: {result}")

        # Should succeed and return valid JSON
        assert task.check_output_formatting(result)
        parsed = NumberSchema.model_validate_json(result)
        assert parsed.number == 7


# @pytest.mark.skipif(
#     os.getenv("OPENAI_API_KEY", None) is None, reason="OPENAI_API_KEY not set"
# )
# class TestAgentFrameworkMCPIntegration:
#     """Integration tests for MCP tool support."""

#     @pytest.fixture(autouse=True)
#     def setup_fixture(self):
#         from charge.clients.agentframework import AgentFrameworkPool

#         self.agent_pool = AgentFrameworkPool(model="gpt-4o-mini", backend="openai")

#     @pytest.mark.asyncio
#     @pytest.mark.skip(reason="Requires actual MCP servers to be running")
#     async def test_mcp_sse_integration(self):
#         """Test MCP SSE server integration (requires running MCP server)."""
#         from charge.tasks.Task import Task

#         task = Task(
#             system_prompt="You are a helpful assistant with access to tools.",
#             user_prompt="Use the available tools.",
#             server_urls=["http://localhost:8000/sse"],
#         )

#         agent = self.agent_pool.create_agent(task=task)
#         result = await agent.run()

#         print(f"MCP SSE result: {result}")
#         assert len(result) > 0
