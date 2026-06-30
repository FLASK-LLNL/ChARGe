from charge.tasks.task import Task
from charge.experiments.memory import Memory
from abc import abstractmethod
from dataclasses import dataclass
import json
import warnings
from typing import Any, Literal, Optional, Protocol, TypeAlias

DEFAULT_BACKEND = "agentframework"
"""
Default backend to use for agent factory
"""


class AgentCallback(Protocol):
    async def on_task_start(self) -> None: ...

    async def on_task_finish(self) -> None: ...

    async def on_reasoning_update(
        self,
        text: str,
        *,
        source: Optional[str] = None,
    ) -> None: ...

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


@dataclass(frozen=True)
class AgentInstructionSnapshot:
    message_count: int
    instructions: str

    @classmethod
    def from_json(cls, record: dict[str, Any]) -> "AgentInstructionSnapshot":
        message_count = record["messageCount"]
        instructions = record["instructions"]
        if not isinstance(message_count, int):
            raise TypeError("instructionHistory messageCount must be an integer")
        if not isinstance(instructions, str) or not instructions:
            raise TypeError(
                "instructionHistory instructions must be a non-empty string"
            )
        return cls(message_count=message_count, instructions=instructions)

    def to_json(self) -> dict[str, Any]:
        return {
            "messageCount": self.message_count,
            "instructions": self.instructions,
        }


class Agent:
    """
    Base class for an Agent that performs Tasks.
    """

    task: Optional[Task]
    callback: AgentCallbackType
    instruction_history: list[AgentInstructionSnapshot]

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
        self.instruction_history = []
        self.pending_user_message: Optional[dict[str, Any]] = None

    def _message_count(self) -> int:
        try:
            memory = json.loads(self.save_memory() or "{}")
        except (TypeError, ValueError):
            return 0
        if not isinstance(memory, dict):
            return 0
        state = memory.get("state")
        if not isinstance(state, dict):
            return 0
        in_memory = state.get("in_memory")
        if not isinstance(in_memory, dict):
            return 0
        messages = in_memory.get("messages")
        return len(messages) if isinstance(messages, list) else 0

    def begin_task_run(self) -> None:
        if self.task is None:
            self.pending_user_message = None
            return
        text = self.task.get_user_prompt().strip()
        if not text:
            self.pending_user_message = None
            return
        pending: dict[str, Any] = {
            "text": text,
            "afterMessageCount": self._message_count(),
        }
        attachments = getattr(self.task, "attachments", []) or []
        images = []
        for attachment in attachments:
            if not isinstance(attachment, dict):
                continue
            images.append(
                {
                    "id": str(attachment.get("id") or f"image_{len(images) + 1}"),
                    "name": str(attachment.get("name") or f"Image {len(images) + 1}"),
                    "mimeType": str(attachment.get("mimeType") or ""),
                    "sizeBytes": int(attachment.get("sizeBytes") or 0),
                    "dataUrl": attachment.get("dataUrl"),
                }
            )
        if images:
            pending["images"] = images
        self.pending_user_message = pending

    def finish_task_run(self) -> None:
        self._capture_instruction_snapshot()
        self.pending_user_message = None

    def _capture_instruction_snapshot(self) -> None:
        if self.task is None:
            return
        instructions = self.task.get_system_prompt().strip()
        if not instructions:
            return
        message_count = self._message_count()
        if message_count <= 0:
            return
        snapshot = AgentInstructionSnapshot(
            message_count=message_count,
            instructions=instructions,
        )
        if (
            self.instruction_history
            and self.instruction_history[-1].message_count == message_count
        ):
            self.instruction_history[-1] = snapshot
        else:
            self.instruction_history.append(snapshot)

    async def notify_task_start(self) -> None:
        if self.callback is not None:
            await self.callback.on_task_start()

    async def notify_task_finish(self) -> None:
        if self.callback is not None:
            await self.callback.on_task_finish()

    @abstractmethod
    def run(self, **kwargs) -> Any:
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


@dataclass
class AgentRuntimeConfig:
    backend: str = DEFAULT_BACKEND
    model: Optional[str] = None

    @classmethod
    def from_agent(
        cls, *, agent: Agent, backend: AgentBackend
    ) -> "AgentRuntimeConfig":
        model_info = agent.get_model_info()
        backend_name = (
            model_info.get("backend") if isinstance(model_info, dict) else None
        )
        model = model_info.get("model") if isinstance(model_info, dict) else None
        return cls(
            backend=str(backend_name or backend.backend),
            model=model or backend.model,
        )

    @classmethod
    def from_record(cls, record: dict[str, Any]) -> "AgentRuntimeConfig":
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
            backend=backend,
            model=runtime_config.get("model") or model_info.get("model"),
        )

    def to_json(self) -> dict[str, Any]:
        return {
            "backend": self.backend,
            "model": self.model,
        }


def create_agent_for_runtime_config(
    *,
    task: Optional[Task],
    agent_key: str,
    memory: Optional[Memory],
    runtime_config: AgentRuntimeConfig,
    backend: AgentBackend,
    create_kwargs: Optional[dict[str, Any]] = None,
    callback: AgentCallbackType = None,
) -> tuple[Agent, AgentRuntimeConfig]:
    agent = backend.create_agent(
        task=task,
        memory=memory,
        agent_key=agent_key,
        callback=callback,
        **(create_kwargs or {}),
    )
    return agent, AgentRuntimeConfig.from_agent(agent=agent, backend=backend)


def verify_restored_agent_config(
    agent_key: str,
    saved_config: AgentRuntimeConfig,
    current_config: AgentRuntimeConfig,
) -> None:
    if saved_config.backend != current_config.backend:
        raise RuntimeError(
            "Cannot restore agent session with mismatched backend for "
            f"{agent_key!r}: saved {saved_config.backend!r}, "
            f"current {current_config.backend!r}."
        )
    if (
        saved_config.model
        and current_config.model
        and saved_config.model != current_config.model
    ):
        warnings.warn(
            "Restoring agent session with a different model for "
            f"{agent_key!r}: saved {saved_config.model!r}, "
            f"current {current_config.model!r}.",
            RuntimeWarning,
            stacklevel=2,
        )


def serialize_agent_session(
    agent: Agent,
    runtime_config: AgentRuntimeConfig,
) -> dict[str, Any]:
    task_json = None
    task = agent.task
    if task is not None:
        task_json = task.to_json()
    record = {
        "runtimeConfig": runtime_config.to_json(),
        "memory": agent.save_memory(),
        "modelInfo": agent.get_model_info(),
        "task": task_json,
    }
    if agent.instruction_history:
        record["instructionHistory"] = [
            snapshot.to_json() for snapshot in agent.instruction_history
        ]
    if agent.pending_user_message:
        record["pendingUserMessage"] = agent.pending_user_message
    return record


def restore_agent_session(
    agent_key: str,
    record: dict[str, Any],
    *,
    memory: Optional[Memory],
    backend: AgentBackend,
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

    saved_config = AgentRuntimeConfig.from_record(record)
    agent, current_config = create_agent_for_runtime_config(
        task=task,
        agent_key=agent_key,
        memory=memory,
        runtime_config=saved_config,
        backend=backend,
    )
    verify_restored_agent_config(agent_key, saved_config, current_config)

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
    instruction_history = record.get("instructionHistory")
    if isinstance(instruction_history, list):
        agent.instruction_history = [
            AgentInstructionSnapshot.from_json(snapshot)
            for snapshot in instruction_history
            if isinstance(snapshot, dict)
        ]
    return agent, current_config
