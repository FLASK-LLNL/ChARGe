from charge.tasks.task import Task
from charge.experiments.memory import Memory
from abc import abstractmethod
from dataclasses import dataclass
import warnings
from typing import Any, Awaitable, Callable, Literal, Optional, Protocol, TypeAlias

DEFAULT_BACKEND = "agentframework"
"""
Default backend to use for agent factory
"""

ReasoningCallbackType: TypeAlias = Optional[Callable[[str], Awaitable[None]]]


class AgentCallback(Protocol):
    async def on_tool_call(
        self,
        tool_name: str,
        arguments: Any,
        *,
        source: Optional[str] = None,
        call_id: Optional[str] = None,
    ) -> None: ...

    async def on_tool_result(
        self,
        tool_name: str,
        result: Any,
        *,
        is_error: bool = False,
        source: Optional[str] = None,
        call_id: Optional[str] = None,
    ) -> None: ...


AgentCallbackType: TypeAlias = Optional[AgentCallback]


class Agent:
    """
    Base class for an Agent that performs Tasks.
    """

    task: Optional[Task]
    callback: AgentCallbackType

    def __init__(
        self,
        task: Optional[Task],
        *,
        callback: AgentCallbackType = None,
        **kwargs,
    ):
        self.task = task
        self.callback: AgentCallbackType = callback
        self.kwargs = kwargs

    @abstractmethod
    def run(self, reasoning_callback: ReasoningCallbackType = None, **kwargs) -> Any:
        """
        Abstract method to run the Agent's task.
        """
        raise NotImplementedError("Method 'run' is not implemented.")

    @abstractmethod
    def load_memory(self, json_str: str) -> None:
        """
        Abstract method to load the Agent's serialized memory.
        """
        raise NotImplementedError("Method 'load_memory' is not implemented.")

    @abstractmethod
    def save_memory(self) -> str:
        """
        Abstract method to serialize the Agent's memory.
        """
        raise NotImplementedError("Method 'save_memory' is not implemented.")

    def get_model_info(self) -> dict[str, Any]:
        """
        Returns the model information of the agent.
        """
        return {}


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
        agent_key: Optional[str] = None,
        memory: Optional[Memory] = None,
        callback: AgentCallbackType = None,
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
        agent_key: Optional[str] = None,
        memory: Optional[Memory] = None,
        callback: AgentCallbackType = None,
        **kwargs,
    ):
        """
        Abstract method to create and return an Agent instance.
        """
        return cls.backends[backend.lower()].create_agent(
            task, agent_key=agent_key, memory=memory, callback=callback, **kwargs
        )

    @classmethod
    def list_all_backends(cls) -> list[str]:
        """
        Abstract method to get a list of all Agent backends in the factory.
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


@dataclass
class AgentRuntimeConfig:
    agent_key: str
    backend: str = DEFAULT_BACKEND
    model: Optional[str] = None

    @classmethod
    def from_agent(cls, *, agent_key: str, agent: Agent) -> "AgentRuntimeConfig":
        model_info = agent.get_model_info()
        backend = model_info.get("backend") if isinstance(model_info, dict) else None
        model = model_info.get("model") if isinstance(model_info, dict) else None
        backend_obj = AgentFactory.default_backend()
        return cls(
            agent_key=agent_key,
            backend=str(backend or backend_obj.backend),
            model=model or backend_obj.model,
        )

    @classmethod
    def from_record(
        cls, agent_key: str, record: dict[str, Any]
    ) -> "AgentRuntimeConfig":
        runtime_config = record.get("runtimeConfig")
        if not isinstance(runtime_config, dict):
            runtime_config = {}
        model_info = record.get("modelInfo")
        if not isinstance(model_info, dict):
            model_info = {}
        backend = str(
            runtime_config.get("backend")
            or model_info.get("backend")
            or DEFAULT_BACKEND
        )
        return cls(
            agent_key=agent_key,
            backend=backend,
            model=runtime_config.get("model") or model_info.get("model"),
        )

    def to_json(self) -> dict[str, Any]:
        return {
            "agentKey": self.agent_key,
            "backend": self.backend,
            "model": self.model,
        }


def create_agent_for_runtime_config(
    *,
    task: Optional[Task],
    memory: Optional[Memory],
    runtime_config: AgentRuntimeConfig,
    create_kwargs: Optional[dict[str, Any]] = None,
    callback: AgentCallbackType = None,
) -> tuple[Agent, AgentRuntimeConfig]:
    agent = AgentFactory.create_agent(
        task=task,
        memory=memory,
        agent_key=runtime_config.agent_key,
        callback=callback,
        **(create_kwargs or {}),
    )
    return agent, AgentRuntimeConfig.from_agent(
        agent_key=runtime_config.agent_key,
        agent=agent,
    )


def verify_restored_agent_config(
    saved_config: AgentRuntimeConfig, current_config: AgentRuntimeConfig
) -> None:
    if saved_config.backend != current_config.backend:
        raise RuntimeError(
            "Cannot restore agent session with mismatched backend for "
            f"{saved_config.agent_key!r}: saved {saved_config.backend!r}, "
            f"current {current_config.backend!r}."
        )
    if (
        saved_config.model
        and current_config.model
        and saved_config.model != current_config.model
    ):
        warnings.warn(
            "Restoring agent session with a different model for "
            f"{saved_config.agent_key!r}: saved {saved_config.model!r}, "
            f"current {current_config.model!r}.",
            RuntimeWarning,
            stacklevel=2,
        )


def serialize_agent_session(
    agent_key: str, agent: Agent, runtime_config: AgentRuntimeConfig
) -> dict[str, Any]:
    task_json = None
    task = agent.task
    if task is not None:
        task_json = task.to_json()
    return {
        "agentKey": agent_key,
        "runtimeConfig": runtime_config.to_json(),
        "memory": agent.save_memory(),
        "modelInfo": agent.get_model_info(),
        "task": task_json,
    }


def restore_agent_session(
    agent_key: str,
    record: dict[str, Any],
    *,
    memory: Optional[Memory],
) -> tuple[Agent, AgentRuntimeConfig]:
    task = None
    task_json = record.get("task")
    if isinstance(task_json, dict):
        try:
            task = Task.from_json(task_json)
        except Exception as exc:
            warnings.warn(
                f"Failed to restore task for agent {agent_key!r}; "
                f"creating the agent with no task. Original error: {exc}",
                RuntimeWarning,
                stacklevel=2,
            )

    saved_config = AgentRuntimeConfig.from_record(agent_key, record)
    agent, current_config = create_agent_for_runtime_config(
        task=task,
        memory=memory,
        runtime_config=saved_config,
    )
    verify_restored_agent_config(saved_config, current_config)

    memory_json = record.get("memory")
    if memory_json:
        try:
            agent.load_memory(str(memory_json))
        except Exception as exc:
            warnings.warn(
                f"Failed to restore memory for agent {agent_key!r}; "
                "continuing with an empty live agent session. "
                f"Original error: {exc}",
                RuntimeWarning,
                stacklevel=2,
            )
    return agent, current_config
