from typing import Any, Optional

import pytest

from charge.clients.agent_factory import (
    Agent,
    AgentBackend,
    AgentCallbackType,
    AgentFactory,
    AgentInstructionSnapshot,
    DEFAULT_BACKEND,
)
from charge.experiments.experiment import Experiment
from charge.experiments.memory import Memory
from charge.tasks.task import Task


class DummyAgent(Agent):
    def __init__(
        self,
        task: Optional[Task],
        *,
        agent_key: Optional[str] = None,
        callback: AgentCallbackType = None,
        **kwargs,
    ):
        super().__init__(task, callback=callback, **kwargs)
        self.agent_key = agent_key
        self.saved_memory = ""
        self.loaded_memory = ""

    def run(self, **kwargs) -> str:
        return "ok"

    def load_memory(self, json_str: str) -> None:
        self.loaded_memory = json_str
        self.saved_memory = json_str

    def save_memory(self) -> str:
        return self.saved_memory

    def get_model_info(self) -> dict[str, Any]:
        return {"backend": "dummy", "model": "dummy-model"}


class DummyBackend(AgentBackend):
    def __init__(self):
        super().__init__(
            model="dummy-model",
            api_key=None,
            base_url=None,
            reasoning_effort=None,
            model_kwargs=None,
            backend="dummy",
        )

    def create_agent(
        self,
        task: Optional[Task],
        *,
        agent_key: Optional[str] = None,
        memory: Optional[Memory] = None,
        callback: AgentCallbackType = None,
        **kwargs,
    ) -> Agent:
        return DummyAgent(task=task, agent_key=agent_key, callback=callback, **kwargs)


def test_experiment_saves_and_rehydrates_raw_agent_sessions():
    original_default_backend = AgentFactory.backends.get(DEFAULT_BACKEND)
    AgentFactory.register_backend(DEFAULT_BACKEND, DummyBackend())
    experiment = Experiment(task=None)
    task = Task(system_prompt="System", user_prompt="Hello")

    try:
        with pytest.raises(TypeError, match="does not accept backend"):
            experiment.create_agent_with_experiment_state(
                task,
                agent_key="molecule:node_1",
                backend="dummy_sessions",
            )

        agent = experiment.create_agent_with_experiment_state(
            task,
            agent_key="molecule:node_1",
        )
        assert isinstance(agent, DummyAgent)
        assert agent.agent_key == "molecule:node_1"
        agent.saved_memory = '{"state":{"in_memory":{"messages":[{"role":"user"}]}}}'
        agent.instruction_history = [
            AgentInstructionSnapshot(message_count=1, instructions="System")
        ]

        state = experiment.save_state()

        assert state["agentSessions"]["molecule:node_1"]["memory"] == agent.saved_memory
        assert (
            state["agentSessions"]["molecule:node_1"]["task"]["system_prompt"]
            == "System"
        )
        assert state["agentSessions"]["molecule:node_1"]["instructionHistory"] == [
            {"messageCount": 1, "instructions": "System"}
        ]
        assert "metadata" not in state["agentSessions"]["molecule:node_1"]
        assert "messageMetadata" not in state["agentSessions"]["molecule:node_1"]
        assert "agentKey" not in state["agentSessions"]["molecule:node_1"]
        runtime_config = state["agentSessions"]["molecule:node_1"]["runtimeConfig"]
        assert "agentName" not in runtime_config
        assert runtime_config["backend"] == "dummy"
        assert set(runtime_config) == {"backend", "model"}
        assert (
            state["agentSessions"]["molecule:node_1"]["modelInfo"]["model"]
            == "dummy-model"
        )

        restored = Experiment(task=None)
        restored.load_state(state)
        restored_agent = restored.agent_registry["molecule:node_1"].agent
        assert isinstance(restored_agent, DummyAgent)
        assert restored_agent.loaded_memory == agent.saved_memory
        assert restored_agent.instruction_history == [
            AgentInstructionSnapshot(message_count=1, instructions="System")
        ]

        restored_agent = restored.create_agent_with_experiment_state(
            Task(system_prompt="System", user_prompt="Continue"),
            agent_key="molecule:node_1",
        )

        assert isinstance(restored_agent, DummyAgent)
        assert restored_agent.loaded_memory == agent.saved_memory
        restored_state = restored.save_state()
        assert "metadata" not in restored_state["agentSessions"]["molecule:node_1"]
    finally:
        if original_default_backend is None:
            AgentFactory.backends.pop(DEFAULT_BACKEND, None)
        else:
            AgentFactory.register_backend(DEFAULT_BACKEND, original_default_backend)


def test_experiment_load_state_rejects_backend_mismatch():
    original_default_backend = AgentFactory.backends.get(DEFAULT_BACKEND)
    AgentFactory.register_backend(DEFAULT_BACKEND, DummyBackend())
    state = {
        "items": [],
        "agentSessions": {
            "molecule:node_1": {
                "runtimeConfig": {
                    "backend": "other",
                    "model": "dummy-model",
                },
                "memory": "",
                "modelInfo": {"backend": "other", "model": "dummy-model"},
                "task": Task(system_prompt="System", user_prompt="Hello").to_json(),
            }
        },
    }

    try:
        with pytest.raises(RuntimeError, match="mismatched backend"):
            Experiment(task=None).load_state(state)
    finally:
        if original_default_backend is None:
            AgentFactory.backends.pop(DEFAULT_BACKEND, None)
        else:
            AgentFactory.register_backend(DEFAULT_BACKEND, original_default_backend)
