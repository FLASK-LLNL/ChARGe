# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

ChARGe/SARGe is a **Scientific Automated Reasoning and Generation** agentic framework that enables LLM-based agents to solve complex scientific tasks (particularly chemistry-related) using tool augmentation via the Model Context Protocol (MCP).

## Installation & Setup

```bash
# Basic installation
pip install -e .

# With specific frameworks
pip install -e ".[autogen]"         # AutoGen framework (full backend support)
pip install -e ".[agentframework]"  # Microsoft Agent Framework (OpenAI-compatible only)
pip install -e ".[ollama]"          # Ollama backend
pip install -e ".[gemini]"          # Google Gemini backend
pip install -e ".[all]"             # All frameworks and backends

# With test dependencies
pip install -e ".[test]"
```

Python versions: 3.11 or 3.12 (specified in pyproject.toml: `>=3.11, <3.13`)

### Framework Choice

ChARGe supports two agent orchestration frameworks:

- **AutoGen** (`charge.clients.autogen`): Full support for all backends (OpenAI, Gemini, Ollama, vLLM, HuggingFace, Azure, LivAI, LLamaMe, ALCF)
- **Agent Framework** (`charge.clients.agentframework`): Microsoft's newer framework with OpenAI Responses API support (hosted tools: code interpreter, file search)

Both frameworks share the same Task-based API and can be used interchangeably for OpenAI-compatible backends.

**When to use AutoGen**:
- Need support for Ollama, Azure OpenAI, vLLM, or HuggingFace backends
- Working with existing AutoGen-based code
- Require specific AutoGen features or customizations

**When to use Agent Framework**:
- Want access to OpenAI Responses API with hosted tools (code interpreter, file search)
- Using OpenAI or OpenAI-compatible endpoints (Gemini, LivAI, etc.)
- Prefer Microsoft's newer agent orchestration framework
- Building new projects with latest Microsoft agent tooling

## Testing

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/unit_tests/test_tasks.py

# Run with verbose output
pytest -v

# Test specific frameworks
pytest tests/unit_tests/test_autogen_configure.py -v
pytest tests/unit_tests/test_agentframework_configure.py -v

# Integration tests (require API keys)
export OPENAI_API_KEY="your-key"
pytest tests/integration_tests/test_autogen_openai_experiment.py -v
pytest tests/integration_tests/test_agentframework_integration.py -v
```

Test files are organized in:
- `tests/unit_tests/` - Unit tests for both frameworks
  - `test_autogen_*.py` - AutoGen framework tests
  - `test_agentframework_*.py` - Agent Framework tests
- `tests/integration_tests/` - Integration tests requiring API keys

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
   - **Framework Implementations**:
     - `autogen.py`: AutoGenAgent and AutoGenPool using Microsoft AutoGen framework
     - `agentframework.py`: AgentFrameworkAgent and AgentFrameworkPool using Microsoft Agent Framework
   - **Shared Infrastructure**:
     - `openai_base.py`: Common OpenAI API configuration and utilities shared between frameworks
     - `autogen_utils.py`: AutoGen-specific utilities (memory, console, error handling)
     - `agentframework_utils.py`: Agent Framework-specific utilities (memory, MCP adapters)
   - **Other implementations**: `VLLMClient`, `HuggingFaceLocalClient`
   - **Backend support**: openai, gemini, ollama, vllm, huggingface, livai, livchat, llamame, alcf
   - Clients handle LLM communication, MCP workbench setup, and execution flow

3. **Agent/AgentPool Layer** (`charge/clients/AgentPool.py`)
   - `Agent`: Wraps a task and provides execution context
   - `AgentPool`: Factory for creating and managing multiple agents
   - Used by the `Experiment` framework for sequential task execution

### Experiments Framework

`charge/experiments/Experiment.py` provides infrastructure for running sequences of related tasks with shared context. Framework-specific implementations:
- `AutoGenExperiment.py`: Sequential task execution with AutoGen framework
- `AgentFrameworkExperiment.py`: Sequential task execution with Agent Framework

Both implementations maintain conversation history across multiple tasks and support state serialization for checkpointing.

### MCP (Model Context Protocol) Integration

ChARGe uses MCP to expose tools to LLM agents:

- **Auto-generation**: Methods decorated with `@hypothesis` in Task classes are automatically converted to MCP servers
- **External servers**: Tasks can connect to standalone MCP servers via SSE (HTTP) or STDIO protocols
- **Server types**:
  - SSE servers: Network-accessible, persistent servers (see `examples/SSE_MCP/`)
  - STDIO servers: Local process-based servers (see `examples/Multi_Server_Experiments/`)

MCP utilities are in `charge/utils/mcp_workbench_utils.py` and `charge/_to_mcp.py`.

### OpenAI Responses API and Hosted Tools (Agent Framework Only)

Agent Framework provides exclusive access to OpenAI's Responses API, which includes hosted tools:

- **Code Interpreter**: Execute Python code directly in conversations with automatic sandboxing
- **File Search**: Search through uploaded files for retrieval-augmented generation
- **Web Search**: Search the web (when available)

These tools are hosted by OpenAI and don't require custom implementation:

```python
from charge.clients.agentframework import AgentFrameworkPool

# Enable Responses API
pool = AgentFrameworkPool(
    model="gpt-4o",
    backend="openai",
    use_responses_api=True
)

# Hosted tools are automatically available to agents
agent = pool.create_agent(task=task)
```

**Note**: Responses API requires Agent Framework. AutoGen does not support this feature.

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

#### Using AutoGen Framework

```python
from charge.clients.autogen import AutoGenPool
import asyncio

# Create task
task = MyChemistryTask(
    system_prompt="You are a chemistry expert...",
    user_prompt="Generate a molecule with...",
    server_urls=["http://127.0.0.1:8000/sse"],  # Optional external MCP servers
)

# Create agent pool and agent
pool = AutoGenPool(model="gpt-4", backend="openai")
agent = pool.create_agent(task=task)

# Run
result = asyncio.run(agent.run())
```

#### Using Agent Framework

```python
from charge.clients.agentframework import AgentFrameworkPool
import asyncio

# Create task
task = MyChemistryTask(
    system_prompt="You are a chemistry expert...",
    user_prompt="Generate a molecule with...",
    server_urls=["http://127.0.0.1:8000/sse"],  # Optional external MCP servers
)

# Create agent pool and agent
pool = AgentFrameworkPool(model="gpt-4o", backend="openai")
agent = pool.create_agent(task=task)

# Run
result = asyncio.run(agent.run())
```

#### Using Responses API with Hosted Tools (Agent Framework only)

```python
from charge.clients.agentframework import AgentFrameworkPool

# Enable Responses API for hosted tools
pool = AgentFrameworkPool(
    model="gpt-4o",
    backend="openai",
    use_responses_api=True  # Enables code interpreter, file search
)

# Get hosted tools
hosted_tools = pool.get_hosted_tools()

# Create and run agent (tools automatically available)
agent = pool.create_agent(task=task)
result = asyncio.run(agent.run())
```

### Backend Configuration

The `Client.add_std_parser_arguments()` method sets up standard CLI args:
- `--model`: Model name (backend-specific)
- `--backend`: Backend name (see support matrix below)
- `--server-urls`: List of SSE MCP server URLs
- `--history`: Path for command history file

Environment variables used by backends:
- `OPENAI_API_KEY`: For OpenAI backend
- `GOOGLE_API_KEY`: For Gemini backend
- `LIVAI_API_KEY`, `LIVAI_BASE_URL`: For LivAI/LivChat backend
- `LLAMAME_API_KEY`, `LLAMAME_BASE_URL`: For LLamaMe backend
- `ALCF_API_KEY`, `ALCF_BASE_URL`: For ALCF backend
- `VLLM_URL`, `VLLM_MODEL`, `OSS_REASONING`: For VLLM backend

#### Backend Support Matrix

| Backend | AutoGen | Agent Framework | Notes |
|---------|---------|-----------------|-------|
| OpenAI | ✅ Full | ✅ Full + Responses API | Agent Framework has hosted tools |
| Gemini | ✅ Full | ✅ Full | Via OpenAI-compatible endpoint |
| LivAI/LivChat | ✅ Full | ✅ Full | Custom OpenAI-compatible endpoints |
| LLamaMe | ✅ Full | ✅ Full | Custom OpenAI-compatible endpoints |
| ALCF | ✅ Full | ✅ Full | Custom OpenAI-compatible endpoints |
| Azure OpenAI | ✅ Full | ❌ Not Supported | Use AutoGen for Azure |
| Ollama | ✅ Full | ❌ Not Supported | Use AutoGen for Ollama |
| vLLM | ✅ Full | ❌ Not Supported | Use AutoGen for vLLM |
| HuggingFace | ✅ Full | ❌ Not Supported | Use AutoGen for HuggingFace |

**Configuration is shared**: Both frameworks use the same configuration functions from `charge/clients/openai_base.py` for OpenAI-compatible backends.

## Project Structure Notes

### Core Modules
- `charge/__init__.py` exports: `verifier`, `hypothesis`, `enable_cmd_history_and_shell_integration`
- `charge/_tags.py`: Defines the `@verifier` and `@hypothesis` decorators
- `charge/inspector.py`: Introspection utilities for extracting class information
- `charge/tasks/Task.py`: Base Task class for defining scientific workflows
- `charge/utils/`: System utilities, MCP workbench setup, logging helpers

### Client Layer
- `charge/clients/autogen.py`: AutoGen framework implementation (AutoGenAgent, AutoGenPool)
- `charge/clients/agentframework.py`: Agent Framework implementation (AgentFrameworkAgent, AgentFrameworkPool)
- `charge/clients/openai_base.py`: **Shared OpenAI API abstraction** - common configuration, API key handling, logging transport
- `charge/clients/autogen_utils.py`: AutoGen-specific utilities (ChARGeListMemory, console, error handling)
- `charge/clients/agentframework_utils.py`: Agent Framework-specific utilities (AgentFrameworkMemory, MCP adapters)
- `charge/clients/reasoning.py`: Reasoning trace utilities
- `charge/clients/AgentPool.py`: Abstract base classes (Agent, AgentPool)
- `charge/clients/Client.py`: Base client class

### Experiment Framework
- `charge/experiments/Experiment.py`: Base experiment class
- `charge/experiments/AutoGenExperiment.py`: Multi-task sequential execution with AutoGen
- `charge/experiments/AgentFrameworkExperiment.py`: Multi-task sequential execution with Agent Framework

### MCP Integration
- `charge/utils/mcp_workbench_utils.py`: MCP server setup and management
- `charge/_to_mcp.py`: Auto-conversion of decorated methods to MCP tools

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
- `charge/clients/openai_base.py`: `LoggingTransport` class with HTTP request/response logging and API key masking (shared by both frameworks)
- `charge/clients/logging.py`: Additional logging utilities
- Both AutoGen and Agent Framework support verbose logging for debugging MCP interactions
- Use `logger.debug()` for detailed execution traces
- HTTP requests are automatically logged with masked API keys for security
