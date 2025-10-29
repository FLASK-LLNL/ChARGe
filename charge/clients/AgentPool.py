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
        self.context_history = []

    @abstractmethod
    def run(self, **kwargs) -> Any:
        """
        Abstract method to run the Agent's task.
        """
        raise NotImplementedError("Method 'run' is not implemented.")

    @abstractmethod
    def get_context_history(self) -> list:
        """
        Abstract method to get the Agent's context history.
        """
        raise NotImplementedError("Method 'get_context_history' is not implemented.")


class AgentPool:
    """
    Base class for an Agent Pool that manages multiple Agents.
    """

    def __init__(self, **kwargs):
        self.agent_list = []
        self.agent_dict = {}
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

    @abstractmethod
    def list_all_agents(self) -> list:
        """
        Abstract method to get a list of all Agents in the pool.
        """
        raise NotImplementedError("Method 'list_all_agents' is not implemented.")

    @abstractmethod
    def get_agent_by_name(self, name: str) -> Agent:
        """
        Abstract method to get an Agent by name.
        """
        raise NotImplementedError("Method 'get_agent_by_name' is not implemented.")
