# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

ChARGe/SARGe is a **Scientific Automated Reasoning and Generation** agentic framework that enables LLM-based agents to solve complex scientific tasks (particularly chemistry-related) using tool augmentation via the Model Context Protocol (MCP).

## Installation & Setup

```bash
# Basic installation
pip install -e .

# With specific backends
pip install -e ".[autogen]"    # AutoGen backend with OpenAI support
pip install -e ".[ollama]"     # Ollama backend
pip install -e ".[gemini]"     # Google Gemini backend
pip install -e ".[all]"        # All backends

# With test dependencies
pip install -e ".[test]"
```

Python versions: 3.11 or 3.12 (specified in pyproject.toml: `>=3.11, <3.13`)

## Testing

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/unit_tests/test_tasks.py

# Run with verbose output
pytest -v
```

Test files are organized in:
- `tests/unit_tests/` - Unit tests
- `tests/integration_tests/` - Integration tests

## Code Quality

The project uses pre-commit hooks (`.pre-commit-config.yaml`):
- Black for code formatting (Python 3.11)
- Standard pre-commit hooks (end-of-file-fixer, trailing-whitespace, etc.)

Run hooks manually:
```bash
pre-commit run --all-files
```

## Core Architecture

### Three-Layer Design

1. **Task Layer** (`charge/tasks/Task.py`)
   - Users define tasks by inheriting from `Task` base class
   - Tasks specify prompts (system, user, verification, refinement)
   - Methods decorated with `@hypothesis` become tools available to the LLM
   - Methods decorated with `@verifier` are used for automatic verification
   - Tasks can reference external MCP servers via `server_urls` (SSE) or `server_files` (STDIO)

2. **Client Layer** (`charge/clients/`)
   - `Client` is the base abstract class
   - Main implementation: `AutoGenClient` (in `autogen.py`) using Microsoft AutoGen framework
   - Other implementations: `VLLMClient`, `HuggingFaceLocalClient`
   - Clients handle LLM communication, MCP workbench setup, and execution flow
   - Support multiple backends: openai, gemini, ollama, vllm, huggingface, livai, livchat

3. **Agent/AgentPool Layer** (`charge/clients/AgentPool.py`)
   - `Agent`: Wraps a task and provides execution context
   - `AgentPool`: Factory for creating and managing multiple agents
   - Used by the `Experiment` framework for sequential task execution

### Experiments Framework

`charge/experiments/Experiment.py` provides infrastructure for running sequences of related tasks with shared context. The `AutoGenExperiment` implementation shows how to maintain conversation history across multiple tasks.

### MCP (Model Context Protocol) Integration

ChARGe uses MCP to expose tools to LLM agents:

- **Auto-generation**: Methods decorated with `@hypothesis` in Task classes are automatically converted to MCP servers
- **External servers**: Tasks can connect to standalone MCP servers via SSE (HTTP) or STDIO protocols
- **Server types**:
  - SSE servers: Network-accessible, persistent servers (see `examples/SSE_MCP/`)
  - STDIO servers: Local process-based servers (see `examples/Multi_Server_Experiments/`)

MCP utilities are in `charge/utils/mcp_workbench_utils.py` and `charge/_to_mcp.py`.

## Key Patterns and Conventions

### Defining a Task

```python
from charge.tasks import Task
from charge import hypothesis, verifier

class MyChemistryTask(Task):
    def __init__(self, system_prompt, user_prompt, **kwargs):
        super().__init__(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            **kwargs
        )

    @hypothesis
    def calculate_property(self, smiles: str) -> float:
        """Calculate molecular property from SMILES.

        Args:
            smiles: A SMILES string representing a molecule
        Returns:
            float: The calculated property value
        """
        # Implementation
        pass

    @verifier
    def verify_result(self, smiles: str) -> bool:
        """Verify if the result meets criteria."""
        # Verification logic
        pass
```

**Important**: All decorated methods must have:
- Proper type annotations for parameters and return values
- Clear docstrings (used in MCP tool descriptions)
- Static/stateless implementation (they become RPC endpoints)

### Running a Task

```python
from charge.clients.autogen import AutoGenPool

# Create task
task = MyChemistryTask(
    system_prompt="You are a chemistry expert...",
    user_prompt="Generate a molecule with...",
    server_urls=["http://127.0.0.1:8000/sse"],  # Optional external MCP servers
)

# Create agent pool and agent
pool = AutoGenPool(model="gpt-4", backend="openai")
agent = pool.create_agent(task=task)

# Run (async)
import asyncio
result = asyncio.run(agent.run())
```

### Backend Configuration

The `Client.add_std_parser_arguments()` method sets up standard CLI args:
- `--model`: Model name (backend-specific)
- `--backend`: One of: ollama, openai, gemini, livai, livchat, huggingface, vllm
- `--server-urls`: List of SSE MCP server URLs
- `--history`: Path for command history file

Environment variables used by backends:
- `OPENAI_API_KEY`: For OpenAI backend
- `GOOGLE_API_KEY`: For Gemini backend
- `VLLM_URL`, `VLLM_MODEL`, `OSS_REASONING`: For VLLM backend

## Project Structure Notes

- `charge/__init__.py` exports: `verifier`, `hypothesis`, `enable_cmd_history_and_shell_integration`
- `charge/_tags.py`: Defines the `@verifier` and `@hypothesis` decorators
- `charge/inspector.py`: Introspection utilities for extracting class information
- `charge/clients/autogen_utils.py`: Helper functions for AutoGen integration (custom memory, console, error handling)
- `charge/clients/reasoning.py`: Reasoning trace utilities
- `charge/utils/`: System utilities, MCP workbench setup, logging helpers

## Examples

The `examples/` directory demonstrates various usage patterns:
- `Multi_Server_Experiments/`: Using multiple MCP servers (both SSE and STDIO)
- `Multi_Turn_Chat/`: Interactive multi-turn conversations
- `SSE_MCP/`: Persistent SSE MCP server setup and usage

Each example includes its own README with specific instructions.

## Type Checking

The project uses Pyright for type checking (see `pyrightconfig.json`). The configuration:
- Uses Python 3.11
- Virtual environment: `charge_venv_311` (developer-specific, may vary)
- Allows untyped libraries like torch
- Suppresses some type warnings (unused call results, Any types)

## Debugging and Logging

The project uses `loguru` for logging. Key logging locations:
- `charge/clients/logging.py`: Custom HTTP transport with request/response logging
- AutoGen clients support verbose logging for debugging MCP interactions
- Use `logger.debug()` for detailed execution traces
