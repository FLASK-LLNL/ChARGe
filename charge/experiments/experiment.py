from typing import Any, Callable, List, Union, Optional
from charge.tasks.task import Task
from charge.clients.agent_factory import Agent, AgentFactory, ReasoningCallbackType
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
        self.agent_registry: dict[str, dict[str, Any]] = {}
        self.saved_agent_sessions: dict[str, dict[str, Any]] = {}

        self.args = args
        self.kwargs = kwargs

    def create_agent_with_experiment_state(
        self,
        task,
        *,
        agent_key: Optional[str] = None,
        agent_metadata: Optional[dict[str, Any]] = None,
        **kwargs,
    ):
        # Create an agent that incorporates the experiment state
        # NOTE: Should self.context be passed into the agent?
        if agent_key and agent_key in self.agent_registry:
            registry_item = self.agent_registry[agent_key]
            agent = registry_item["agent"]
            if task is not None:
                agent.task = task
            if "callback" in kwargs:
                setattr(agent, "callback", kwargs["callback"])
            if agent_metadata:
                registry_item["metadata"] = {
                    **registry_item.get("metadata", {}),
                    **agent_metadata,
                }
            return agent

        agent = AgentFactory.create_agent(task=task, memory=self.memory, **kwargs)
        if agent_key:
            saved_session = self.saved_agent_sessions.get(agent_key, {})
            saved_memory = saved_session.get("memory")
            if saved_memory and hasattr(agent, "load_memory"):
                agent.load_memory(saved_memory)
            self.agent_registry[agent_key] = {
                "agent": agent,
                "metadata": {
                    **saved_session.get("metadata", {}),
                    **(agent_metadata or {}),
                },
                "messageMetadata": saved_session.get("messageMetadata", {}),
            }
        return agent

    def add_to_context(self, agent: Agent, task: Task, result: Any):
        # Add the result to the context of the experiment
        self.memory.add_to_context(task, result)

    def save_state(self):
        # Save the state of the experiment
        state = self.memory.to_json()
        agent_sessions = dict(self.saved_agent_sessions)
        for agent_key, registry_item in self.agent_registry.items():
            agent = registry_item["agent"]
            memory = ""
            if hasattr(agent, "save_memory"):
                memory = agent.save_memory()
            model_info = {}
            if hasattr(agent, "get_model_info"):
                model_info = agent.get_model_info()
            task_json = None
            task = getattr(agent, "task", None)
            if task is not None and hasattr(task, "to_json"):
                task_json = task.to_json()
            agent_sessions[agent_key] = {
                "agentKey": agent_key,
                "metadata": registry_item.get("metadata", {}),
                "memory": memory,
                "modelInfo": model_info,
                "task": task_json,
                "messageMetadata": registry_item.get("messageMetadata", {}),
            }
        if agent_sessions:
            state["agentSessions"] = agent_sessions
        return state

    def load_state(self, state):
        # Load the state of the experiment
        self.memory = ListMemory.from_json(state)
        self.saved_agent_sessions = state.get("agentSessions", {}) or {}
        self.agent_registry = {}

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
        self, reasoning_callback: ReasoningCallbackType, **kwargs
    ) -> None:
        while self.tasks:
            current_task = self.tasks.pop(0)
            agent = self.create_agent_with_experiment_state(task=current_task, **kwargs)
            result = await maybe_await_async(agent.run, reasoning_callback)
            await maybe_await_async(self.add_to_context, agent, current_task, result)
            self.finished_tasks.append((current_task, result))

    def run(self, reasoning_callback: ReasoningCallbackType) -> None:
        asyncio.run(self.run_async(reasoning_callback))

    def reset(self):
        """
        Resets the experiment state, clearing both finished and pending tasks.
        """
        self.finished_tasks = []
        self.memory = ListMemory()
        self.tasks = []
        self.agent_registry = {}
        self.saved_agent_sessions = {}
