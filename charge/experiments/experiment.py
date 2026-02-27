from typing import Any, List, Union, Optional
from charge.tasks.task import Task
from charge.clients.agent_factory import Agent, AgentFactory
from charge.experiments.memory import Memory, ListMemory
from charge._utils import maybe_await_async
import asyncio


class Experiment:
    def __init__(
        self,
        task: Optional[Union[Task, List[Task]]],
        *args,
        memory: Optional[Memory] = None,
        **kwargs,
    ):
        if task is None:
            task = []
        self.tasks = task if isinstance(task, list) else [task]
        self.finished_tasks = []
        self.memory = memory or ListMemory()

        self.args = args
        self.kwargs = kwargs

    def create_agent_with_experiment_state(self, task, **kwargs):
        # Create an agent that incorporates the experiment state
        # NOTE: Should self.context be passed into the agent?
        return AgentFactory.create_agent(task=task, memory=self.memory, **kwargs)

    def add_to_context(self, agent: Agent, task: Task, result: Any):
        # Add the result to the context of the experiment
        self.memory.add_to_context(task, result)

    def save_state(self):
        # Save the state of the experiment
        return self.memory.to_json()

    def load_state(self, state):
        # Load the state of the experiment
        self.memory = ListMemory.from_json(state)

    def num_finished_tasks(self) -> int:
        """Returns the number of finished tasks.

        Returns:
            int: Number of finished tasks.
        """
        return len(self.finished_tasks)

    def remaining_tasks(self) -> int:
        """Returns the number of remaining tasks.

        Returns:
            int: Number of remaining tasks.
        """
        return len(self.tasks)

    def add_task(self, task: Task):
        """Adds a new task to the experiment.
        Args:
            task (Task): The task to be added.
        """
        self.tasks.append(task)

    def get_finished_tasks(self) -> List[Any]:
        """Returns the list of finished tasks.

        Returns:
            List[Any]: List of finished tasks.
        """
        return self.finished_tasks

    async def run_async(self, **kwargs) -> None:
        while self.tasks:
            current_task = self.tasks.pop(0)
            agent = self.create_agent_with_experiment_state(task=current_task, **kwargs)
            result = await maybe_await_async(agent.run)
            await maybe_await_async(self.add_to_context, agent, current_task, result)
            self.finished_tasks.append((current_task, result))

    def run(self) -> None:
        asyncio.run(self.run_async())

    def reset(self):
        """
        Resets the experiment state.
        """
        self.finished_tasks = []
        self.memory = ListMemory()
