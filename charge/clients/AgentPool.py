from charge.tasks.Task import Task
from abc import abstractmethod
from typing import Any


class Agent:
    """
    Base class for an Agent that performs Tasks.
    """

    def __init__(self, task: Task, **kwargs):
        self.task = task
        self.kwargs = kwargs

    @abstractmethod
    def run(self, **kwargs) -> Any:
        """
        Abstract method to run the Agent's task.
        """
        raise NotImplementedError("Method 'run' is not implemented.")


class AgentPool:
    """
    Base class for an Agent Pool that manages multiple Agents.
    """

    def __init__(self, **kwargs):
        pass

    @abstractmethod
    def create_agent(
        self,
        task: Task,
        **kwargs,
    ):
        """
        Abstract method to create and return an Agent instance.
        """
        raise NotImplementedError("Method 'create_agent' is not implemented.")
