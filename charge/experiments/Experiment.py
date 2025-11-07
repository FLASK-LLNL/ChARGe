from abc import abstractmethod
from typing import Any, List, Union
from charge.tasks.Task import Task
from charge.clients.AgentPool import Agent, AgentPool
from charge._utils import maybe_await_async
import asyncio


class Experiment(object):
    def __init__(
        self, task: Union[Task, List[Task]], agent_pool: AgentPool, *args, **kwargs
    ):
        self.tasks = task if isinstance(task, list) else [task]
        self.finished_tasks = []

        self.agent_pool = agent_pool
        self.args = args
        self.kwargs = kwargs

    @abstractmethod
    def create_agent_with_experiment_state(self, task):
        # Create an agent that incorporates the experiment state

        # Default implementation is no context is shared across agents
        return self.agent_pool.create_agent(task=task)

    @abstractmethod
    def save_agent_state(self, agent):
        # Save the state of the agent
        raise NotImplementedError("Subclasses must implement save_agent_state method")

    @abstractmethod
    def save_agent_state_async(self, agent):
        # Save the state of the agent
        raise NotImplementedError("Subclasses must implement save_agent_state method")

    @abstractmethod
    def add_to_context(self, agent: Agent, task: Task, result):
        # Add the result to the context of the task
        raise NotImplementedError("Subclasses must implement add_to_context method")

    @abstractmethod
    def save_state(self):
        # Save the state of the experiment
        raise NotImplementedError("Subclasses must implement save_state method")

    @abstractmethod
    def load_state(self, state):
        # Load the state of the experiment
        raise NotImplementedError("Subclasses must implement load_state method")

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

    async def run_async(self) -> None:
        while self.tasks:
            current_task = self.tasks.pop(0)
            agent = self.create_agent_with_experiment_state(task=current_task)
            result = await maybe_await_async(agent.run)
            await maybe_await_async(self.add_to_context, agent, current_task, result)
            self.finished_tasks.append((current_task, result))
            self.save_agent_state(agent)

    def run(self) -> None:
        asyncio.run(self.run_async())
