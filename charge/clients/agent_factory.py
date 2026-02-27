from charge.tasks.task import Task
from charge.experiments.memory import Memory
from abc import abstractmethod
from typing import Any, Literal, Optional

DEFAULT_BACKEND = "autogen"
"""
Default backend to use for agent factory
"""


class Agent:
    """
    Base class for an Agent that performs Tasks.
    """

    def __init__(self, task: Optional[Task], **kwargs):
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


class AgentBackend:
    def __init__(
        self,
        model: Optional[str],
        api_key: Optional[str],
        base_url: Optional[str],
        reasoning_effort: Optional[Literal["low", "medium", "high"]],
        model_kwargs: Optional[dict[str, Any]],
        backend: str,
        **kwargs,
    ) -> None:
        self.model = model
        self.api_key = api_key
        self.base_url = base_url
        self.reasoning_effort = reasoning_effort
        self.model_kwargs = model_kwargs
        self.backend = backend

    @abstractmethod
    def create_agent(
        self,
        task: Optional[Task],
        *,
        agent_name: Optional[str] = None,
        memory: Optional[Memory] = None,
        **kwargs,
    ) -> Agent:
        raise NotImplementedError


class AgentFactory:
    """
    Base class for an Agent Factory that manages multiple Agents.
    """

    backends: dict[str, AgentBackend] = {}

    @classmethod
    def create_agent(
        cls,
        task: Optional[Task],
        backend: str = DEFAULT_BACKEND,
        agent_name: Optional[str] = None,
        memory: Optional[Memory] = None,
        **kwargs,
    ):
        """
        Abstract method to create and return an Agent instance.
        """
        return cls.backends[backend.lower()].create_agent(
            task, agent_name=agent_name, memory=memory, **kwargs
        )

    @classmethod
    def list_all_backends(cls) -> list[str]:
        """
        Abstract method to get a list of all Agent backends in the pool.
        """
        return list(cls.backends.keys())

    @classmethod
    def default_backend(cls) -> AgentBackend:
        return cls.backends[DEFAULT_BACKEND]

    @classmethod
    def register_backend(cls, name: str, backend: AgentBackend):
        """
        Registers an agent creation backend with the given name.
        """
        cls.backends[name.lower()] = backend
