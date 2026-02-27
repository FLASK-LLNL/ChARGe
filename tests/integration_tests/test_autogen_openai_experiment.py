import pytest
import os


@pytest.mark.skipif(
    os.getenv("OPENAI_API_KEY", None) is None, reason="OPENAI_API_KEY not set"
)
class TestOpenAISimpleTask:
    @pytest.fixture(autouse=True)
    def setup_fixture(self):
        from charge.tasks.task import Task
        from charge.clients.autogen import AutoGenBackend
        from charge.clients.agent_factory import AgentFactory
        from charge.experiments.experiment import Experiment
        from pydantic import BaseModel

        self.agent_backend = AutoGenBackend(model="gpt-5-nano")
        AgentFactory.register_backend("autogen", self.agent_backend)

        first_task = Task(
            system_prompt="You are a helpful assistant, that is capable of"
            + "doing arithmetic and returning an explanation of how you arrived at the"
            + "answer. Provide concise and fast responses.",
            user_prompt="What is 10 plus 5?",
        )

        class MathExplanationSchema(BaseModel):
            answer: int
            explanation: str

        self.schema = MathExplanationSchema

        second_task = Task(
            system_prompt="You are a helpful assistant, that is capable of "
            + "take an answer and explanation and convert it a structured JSON format."
            + " Provide concise and fast responses.",
            user_prompt="Take the previous answer and explanation "
            + "and convert it into a JSON",
            structured_output_schema=MathExplanationSchema,
        )

        self.experiment = Experiment(task=[first_task, second_task])

        third_task = Task(
            system_prompt="You are a helpful assistant that can parse JSON from text"
            + " that can extract fields and do arithmetic.",
            user_prompt="Extract the 'answer' field from the previous JSON and "
            + "multiply it by 3.",
        )

        # Don't add the third task to the experiment yet
        self.third_task = third_task

        alternate_third_task = Task(
            system_prompt="You are a helpful assistant that can parse JSON from text"
            + " that can extract fields and do arithmetic.",
            user_prompt="Extract the 'answer' field from the previous JSON and "
            + "multiply it by 4.",
        )
        self.alternate_third_task = alternate_third_task

    @pytest.mark.asyncio
    async def test_linear_experiment_run(self, mocker):
        import re

        await self.experiment.run_async()
        finished_tasks = self.experiment.get_finished_tasks()
        assert len(finished_tasks) == 2

        first_task, first_task_result = finished_tasks[0]
        second_task, second_task_result = finished_tasks[1]

        print("First Task Result:", first_task_result)
        print("Second Task Result:", second_task_result)

        assert re.search(r"15", first_task_result)
        assert second_task.check_output_formatting(second_task_result)

        parse_output = self.schema.model_validate_json(second_task_result)
        assert parse_output.answer == 15

        # assert re.search(r'"answer":\s*15', second_task_result)

        self.state = self.experiment.save_state()

        print("Serialized State:", self.state)

        self.experiment.load_state(self.state)

        self.experiment.add_task(self.third_task)
        assert self.experiment.remaining_tasks() == 1

        await self.experiment.run_async()
        finished_tasks = self.experiment.get_finished_tasks()
        assert len(finished_tasks) == 3

        third_task, third_task_result = finished_tasks[2]
        print("Third Task Result:", third_task_result)
        assert "45" in third_task_result

        # Redo loading state and ensure we go back to 2 finished tasks
        save_file_content = self.state

        mocker.patch("builtins.open", mocker.mock_open(read_data=save_file_content))

        with open("mock_save_file.json", "r") as f:
            content = f.read()

        assert content == save_file_content

        self.experiment.load_state(content)

        assert self.experiment.remaining_tasks() == 0

        print("Experiment State Loaded Successfully")

        self.experiment.add_task(self.alternate_third_task)
        assert self.experiment.remaining_tasks() == 1
        await self.experiment.run_async()
        finished_tasks = self.experiment.get_finished_tasks()

        third_task, third_task_result = finished_tasks[-1]
        print("Third Task Result After Reloading State:", third_task_result)
        assert "60" in third_task_result
