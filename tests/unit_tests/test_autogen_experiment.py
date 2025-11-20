import pytest


@pytest.fixture
def autogen_agentpool_module():
    import charge.clients.autogen

    return charge.clients.autogen


@pytest.fixture
def autogen_experiment_module():
    import charge.experiments.AutoGenExperiment

    return charge.experiments.AutoGenExperiment


@pytest.fixture
def autoget_mock_agent_pool(autogen_agentpool_module):
    from charge.tasks.Task import Task

    class MockAutoGenPool(autogen_agentpool_module.AutoGenPool):
        def create_agent(self, task: Task, *args, **kwargs):
            class MockAgent:
                def __init__(self, task):
                    self.task = task

                def run(self):
                    return f"Mock response for task: {self.task}"

            return MockAgent(task)

    return MockAutoGenPool(model="gpt-5")


@pytest.fixture
def setup_autogen_experiment(autogen_experiment_module, autoget_mock_agent_pool):
    from charge.tasks.Task import Task

    # Create a mock task
    class MockTask(Task):
        def __init__(self):
            super().__init__(name="MockTask")

    task = MockTask()
    agent_pool = autoget_mock_agent_pool

    # Initialize AutoGenExperiment
    experiment = autogen_experiment_module.AutoGenExperiment(
        task=task, agent_pool=agent_pool
    )

    return experiment


def test_autogen_experiment_initialization(setup_autogen_experiment):
    experiment = setup_autogen_experiment
    assert experiment.num_finished_tasks() == 0
    assert experiment.remaining_tasks() == 1


def test_autogen_experiment_run(setup_autogen_experiment):
    experiment = setup_autogen_experiment
    experiment.run()
    assert experiment.num_finished_tasks() == 1
    assert experiment.remaining_tasks() == 0


@pytest.mark.asyncio
async def test_save_and_load_state_methods(setup_autogen_experiment):
    experiment = setup_autogen_experiment
    state = await experiment.save_state()
    assert state is not None
    await experiment.load_state(state)
    assert experiment is not None
