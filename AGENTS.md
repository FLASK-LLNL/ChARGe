# AGENTS.md

This file provides repository-specific guidance to AI agents working in this codebase.

## Project Overview

ChARGe / SARGe is a scientific agent framework for running LLM-driven tasks with MCP tool augmentation. The main user-facing abstraction is a `Task`, which is executed by a backend-specific agent created from either the AutoGen or Agent Framework integrations.

## Current Architecture

The codebase currently centers on these pieces:

1. `Task`
   - Defined in `charge/tasks/task.py`
   - Holds prompts, MCP server references, and optional structured output schema
   - Can expose tool methods via `@hypothesis` and verifier methods via `@verifier`

2. Backend factories
   - `charge/clients/autogen.py` exposes `AutoGenBackend` and `AutoGenAgent`
   - `charge/clients/agentframework.py` exposes `AgentFrameworkBackend` and `AgentFrameworkAgent`
   - Backends are configured once, then used to create agents for tasks

3. Global backend registry
   - `charge/clients/agent_factory.py` defines `AgentFactory`
   - `AgentFactory.register_backend(name, backend_instance)` registers a backend
   - `AgentFactory.create_agent(task=..., backend="...")` creates an agent from the registry

4. Sequential experiments
   - `charge/experiments/experiment.py` defines `Experiment`
   - `Experiment.run_async(...)` creates one agent per task and carries forward shared memory
   - `charge/experiments/memory.py` provides `Memory` and `ListMemory`

## Important Files

- `charge/tasks/task.py`: base `Task` class
- `charge/_tags.py`: `@hypothesis` and `@verifier`
- `charge/_to_mcp.py`: task method to MCP conversion
- `charge/clients/agent_factory.py`: `Agent`, `AgentBackend`, `AgentFactory`
- `charge/clients/autogen.py`: AutoGen implementation
- `charge/clients/agentframework.py`: Agent Framework implementation
- `charge/clients/openai_base.py`: shared backend/model configuration
- `charge/experiments/experiment.py`: sequential experiment runner
- `charge/experiments/memory.py`: experiment memory abstractions
- `charge/utils/mcp_workbench_utils.py`: MCP setup helpers

Use lowercase paths exactly as they exist in the repo. Several older docs still refer to names like `Task.py`, `Client.py`, `AgentPool.py`, or `AutoGenExperiment.py`; those are stale.

## Installation

```bash
pip install -e .
pip install -e ".[autogen]"
pip install -e ".[agentframework]"
pip install -e ".[ollama]"
pip install -e ".[gemini]"
pip install -e ".[test]"
pip install -e ".[all]"
```

Python requirement: `>=3.11, <3.13`

## Backend Model Configuration

Shared model configuration lives in `charge/clients/openai_base.py`.


## Framework Choice

Use `AutoGenBackend` when:

- You need `ollama`, `vllm`, or local Hugging Face support
- You want the existing AutoGen MCP workbench integration
- You are working with the examples in `examples/SSE_MCP/`, `examples/Multi_Turn_Chat/`, or `examples/Multi_Server_Experiments/`

Use `AgentFrameworkBackend` when:

- You want Microsoft Agent Framework orchestration
- You are using OpenAI-compatible endpoints only
- You want optional OpenAI Responses API support via `use_responses_api=True`

Current support in code:

- AutoGen: `openai`, `gemini`, `livai`, `livchat`, `llamame`, `alcf`, `ollama`, `huggingface`, `vllm`
- Agent Framework: OpenAI-compatible backends only; `ollama`, `huggingface`, and `vllm` explicitly raise `NotImplementedError`

## Defining a Task

Use `charge.tasks.task.Task`.

```python
from charge import hypothesis, verifier
from charge.tasks.task import Task


class MyTask(Task):
    def __init__(self):
        super().__init__(
            system_prompt="You are a chemistry expert.",
            user_prompt="Propose a molecule.",
        )

    @hypothesis
    def score_molecule(self, smiles: str) -> float:
        """Return a score for a SMILES string."""
        return 0.0

    @verifier
    def is_valid(self, smiles: str) -> bool:
        """Return whether the candidate is acceptable."""
        return True
```

Important details from the current implementation:

- `Task` accepts `system_prompt`, `user_prompt`, `verification_prompt`, `refinement_prompt`, `server_urls`, `server_files`, and `structured_output_schema`
- `structured_output_schema` should be a Pydantic `BaseModel` subclass
- `Task.check_output_formatting(...)` validates JSON output against that schema when one is configured
- Decorated methods should have type annotations and docstrings
- External MCP servers should be passed as `server_urls` for SSE and `server_files` for STDIO

## Running a Task

### Direct backend usage

```python
import asyncio
from charge.tasks.task import Task
from charge.clients.autogen import AutoGenBackend

task = Task(
    system_prompt="You are a helpful assistant.",
    user_prompt="What is the capital of France?",
)

backend = AutoGenBackend(model="gpt-5", backend="openai")
agent = backend.create_agent(task=task)
result = asyncio.run(agent.run())
```

Agent Framework is analogous:

```python
import asyncio
from charge.tasks.task import Task
from charge.clients.agentframework import AgentFrameworkBackend

task = Task(
    system_prompt="You are a helpful assistant.",
    user_prompt="Return Paris as JSON.",
)

backend = AgentFrameworkBackend(model="gpt-4o-mini", backend="openai")
agent = backend.create_agent(task=task)
result = asyncio.run(agent.run())
```

### Using the backend registry

```python
from charge.clients.agent_factory import AgentFactory
from charge.clients.autogen import AutoGenBackend

AgentFactory.register_backend("autogen", AutoGenBackend(model="gpt-5", backend="openai"))
agent = AgentFactory.create_agent(task=task, backend="autogen")
```

## Running Sequential Experiments

Use `charge.experiments.experiment.Experiment`.

```python
import asyncio
from charge.tasks.task import Task
from charge.experiments.experiment import Experiment
from charge.clients.agent_factory import AgentFactory
from charge.clients.agentframework import AgentFrameworkBackend

AgentFactory.register_backend(
    "agentframework",
    AgentFrameworkBackend(model="gpt-4o-mini", backend="openai"),
)

experiment = Experiment(
    task=[
        Task(system_prompt="You are helpful.", user_prompt="Remember 42."),
        Task(system_prompt="You are helpful.", user_prompt="What number did I ask you to remember?"),
    ]
)

asyncio.run(experiment.run_async(backend="agentframework"))
```

Current experiment behavior:

- Each task gets a new agent instance
- Shared state is carried in `Memory`
- `Experiment.save_state()` and `load_state(...)` serialize that memory
- Agent Framework seeds prior task/result pairs into the session from experiment memory
- AutoGen rehydrates prior task/result pairs into its memory object

## Responses API

`AgentFrameworkBackend` supports `use_responses_api=True` when the installed `agent-framework` version exposes `OpenAIResponsesClient`.

```python
from charge.clients.agentframework import AgentFrameworkBackend

backend = AgentFrameworkBackend(
    model="gpt-5",
    backend="openai",
    use_responses_api=True,
)

hosted_tools = backend.get_hosted_tools()
```

Notes:

- Hosted tools are only available when `use_responses_api=True`
- `get_hosted_tools()` currently attempts to expose code interpreter and file search when supported by the installed client

## CLI and Examples

Examples currently live in:

- `examples/SSE_MCP/`
- `examples/Multi_Turn_Chat/`
- `examples/Multi_Server_Experiments/`

The examples use:

- `Client.add_std_parser_arguments(...)` from `charge/clients/client.py`
- a backend instance such as `AutoGenBackend(...)`
- `backend.create_agent(task=...)`

## Testing

Useful commands:

```bash
pytest
pytest -v
pytest tests/unit_tests/test_tasks.py -v
pytest tests/unit_tests/test_autogen_configure.py -v
pytest tests/unit_tests/test_agentframework_configure.py -v
pytest tests/integration_tests/test_openai_simple_task.py -v
pytest tests/integration_tests/test_autogen_openai_experiment.py -v
pytest tests/integration_tests/test_agentframework_integration.py -v
```

Test layout:

- `tests/unit_tests/`: unit coverage for tasks, backend configuration, and memory behavior
- `tests/integration_tests/`: live backend tests that require credentials

## Code Quality

The repo includes `.pre-commit-config.yaml` and uses Black plus standard formatting hooks.

Run:

```bash
pre-commit run --all-files
```

## Practical Guidance For Agents

- Prefer the current API names from code over older README-style names
- Use `server_files`, not `server_paths`, when targeting the current `Task` constructor
- Import `Task` from `charge.tasks.task` for accuracy; `charge.tasks` is currently empty
- Import backend classes from their concrete modules rather than from `charge.clients`, because `charge/clients/__init__.py` is currently empty
- If you touch experiment behavior, verify both `Experiment` and `Memory` serialization paths
- If you touch structured output behavior, verify `Task.check_output_formatting(...)` and both backend implementations
