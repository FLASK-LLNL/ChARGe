from typing import Any, Optional

from charge.clients.agent_factory import Agent, AgentBackend, AgentFactory
from charge.experiments.experiment import Experiment
from charge.experiments.memory import Memory
from charge.tasks.task import Task


class DummyAgent(Agent):
    def __init__(self, task: Optional[Task], **kwargs):
        super().__init__(task, **kwargs)
        self.saved_memory = ""
        self.loaded_memory = ""

    def run(self, reasoning_callback=None, **kwargs) -> str:
        return "ok"

    def get_context_history(self) -> list:
        return []

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
        agent_name: Optional[str] = None,
        memory: Optional[Memory] = None,
        **kwargs,
    ) -> Agent:
        return DummyAgent(task=task, **kwargs)


def test_experiment_saves_and_rehydrates_raw_agent_sessions():
    AgentFactory.register_backend("dummy_sessions", DummyBackend())
    experiment = Experiment(task=None)
    task = Task(system_prompt="System", user_prompt="Hello")

    agent = experiment.create_agent_with_experiment_state(
        task,
        backend="dummy_sessions",
        agent_key="molecule:node_1",
        agent_metadata={"title": "Molecule node_1"},
    )
    assert isinstance(agent, DummyAgent)
    agent.saved_memory = '{"state":{"in_memory":{"messages":[{"role":"user"}]}}}'

    state = experiment.save_state()

    assert state["agentSessions"]["molecule:node_1"]["memory"] == agent.saved_memory
    assert (
        state["agentSessions"]["molecule:node_1"]["task"]["system_prompt"] == "System"
    )
    experiment.agent_registry["molecule:node_1"]["messageMetadata"] = {
        "0": {"label": "Chat message"}
    }
    state = experiment.save_state()
    assert (
        state["agentSessions"]["molecule:node_1"]["messageMetadata"]["0"]["label"]
        == "Chat message"
    )
    assert (
        state["agentSessions"]["molecule:node_1"]["metadata"]["title"]
        == "Molecule node_1"
    )
    assert (
        state["agentSessions"]["molecule:node_1"]["modelInfo"]["model"] == "dummy-model"
    )

    restored = Experiment(task=None)
    restored.load_state(state)
    restored_agent = restored.create_agent_with_experiment_state(
        Task(system_prompt="System", user_prompt="Continue"),
        backend="dummy_sessions",
        agent_key="molecule:node_1",
        agent_metadata={"subtitle": "CCO"},
    )

    assert isinstance(restored_agent, DummyAgent)
    assert restored_agent.loaded_memory == agent.saved_memory
    restored_state = restored.save_state()
    restored_metadata = restored_state["agentSessions"]["molecule:node_1"]["metadata"]
    assert restored_metadata["title"] == "Molecule node_1"
    assert restored_metadata["subtitle"] == "CCO"
