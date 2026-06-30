from dataclasses import dataclass
from typing import Any, List, Union, Optional
from charge.tasks.task import Task
from charge.clients.agent import (
    Agent,
    AgentBackend,
    AgentCallbackType,
    AgentRuntimeConfig,
    create_agent_for_runtime_config,
    restore_agent_session,
    serialize_agent_session,
)
from charge.experiments.memory import Memory, ListMemory
from charge._utils import maybe_await_async
import asyncio
import json


@dataclass
class AgentRegistryEntry:
    agent: Agent
    runtime_config: AgentRuntimeConfig


class Experiment:
    def __init__(
        self,
        task: Optional[Union[Task, List[Task]]],
        *args,
        memory: Optional[Memory] = None,
        backend: Optional[AgentBackend] = None,
        **kwargs,
    ):
        if task is None:
            task = []
        self.tasks = task if isinstance(task, list) else [task]
        self.finished_tasks = []
        self.memory = memory or ListMemory()
        # The agent backend (which carries per-user credentials) is supplied by
        # the caller and is scoped to that caller's session, so configuration is
        # never shared across sessions.
        self.backend = backend
        self.agent_registry: dict[str, AgentRegistryEntry] = {}
        self.args = args
        self.kwargs = kwargs

    def _require_backend(self) -> AgentBackend:
        if self.backend is None:
            raise RuntimeError(
                "Experiment has no agent backend configured; pass backend= when "
                "constructing the Experiment before creating agents."
            )
        return self.backend

    def create_agent_with_experiment_state(
        self,
        task,
        *,
        agent_key: Optional[str] = None,
        callback: AgentCallbackType = None,
        **kwargs,
    ):
        # Create an agent that incorporates the experiment state
        # NOTE: Should self.context be passed into the agent?
        if agent_key and agent_key in self.agent_registry:
            registry_item = self.agent_registry[agent_key]
            agent = registry_item.agent
            if task is not None:
                agent.task = task
            if callback is not None:
                agent.callback = callback
            return agent

        backend = self._require_backend()

        if agent_key:
            runtime_config = AgentRuntimeConfig()
            agent, runtime_config = create_agent_for_runtime_config(
                task=task,
                agent_key=agent_key,
                memory=self.memory,
                runtime_config=runtime_config,
                backend=backend,
                create_kwargs=kwargs,
                callback=callback,
            )
            self.agent_registry[agent_key] = AgentRegistryEntry(
                agent=agent,
                runtime_config=runtime_config,
            )
            return agent

        agent = backend.create_agent(
            task=task,
            memory=self.memory,
            callback=callback,
            **kwargs,
        )
        return agent

    def add_to_context(self, agent: Agent, task: Task, result: Any):
        # Add the result to the context of the experiment
        self.memory.add_to_context(task, result)

    def save_state(self):
        # Save the state of the experiment
        state = self.memory.to_json()
        agent_sessions = {}
        for agent_key, registry_item in self.agent_registry.items():
            agent_sessions[agent_key] = serialize_agent_session(
                registry_item.agent, registry_item.runtime_config
            )
        if agent_sessions:
            state["agentSessions"] = agent_sessions
        return state

    def load_state(self, state):
        # Load the state of the experiment
        if isinstance(state, str):
            state = json.loads(state)
        self.memory = ListMemory.from_json(state)
        self.agent_registry = {}
        agent_sessions = state.get("agentSessions", {}) or {}
        if not isinstance(agent_sessions, dict):
            return
        for raw_agent_key, record in agent_sessions.items():
            if not isinstance(record, dict):
                continue
            agent_key = str(raw_agent_key)
            agent, runtime_config = restore_agent_session(
                agent_key, record, memory=self.memory, backend=self._require_backend()
            )
            self.agent_registry[agent_key] = AgentRegistryEntry(
                agent=agent,
                runtime_config=runtime_config,
            )

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

    async def run_async(
        self,
        *,
        agent_key: Optional[str] = None,
        callback: AgentCallbackType = None,
        **kwargs,
    ) -> None:
        while self.tasks:
            current_task = self.tasks.pop(0)
            agent = self.create_agent_with_experiment_state(
                task=current_task,
                agent_key=agent_key,
                callback=callback,
                **kwargs,
            )
            result = await maybe_await_async(agent.run)
            await maybe_await_async(self.add_to_context, agent, current_task, result)
            self.finished_tasks.append((current_task, result))

    def run(
        self,
        *,
        agent_key: Optional[str] = None,
        callback: AgentCallbackType = None,
        **kwargs,
    ) -> None:
        asyncio.run(
            self.run_async(
                agent_key=agent_key,
                callback=callback,
                **kwargs,
            )
        )

    def reset(self):
        """
        Resets the experiment state, clearing both finished and pending tasks.
        """
        self.finished_tasks = []
        self.memory = ListMemory()
        self.tasks = []
        self.agent_registry = {}
