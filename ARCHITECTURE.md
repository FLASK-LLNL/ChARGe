# Architecture Specification

## Purpose

ChARGe provides the generic `charge` Python framework for defining LLM tasks, exposing task methods as tools, configuring agent backends, and running sequential experiments. It is intentionally domain-neutral and is consumed by FLASK Copilot through the `externals/ChARGe` submodule.

Keep generic task, backend, MCP, and experiment behavior here. Domain-specific chemistry prompts, graph payloads, and web UI behavior belong in the parent repository.

## Package Layout

- `charge/tasks/task.py`: base `Task` abstraction.
- `charge/_tags.py`: `@hypothesis` and `@verifier` decorators.
- `charge/_to_mcp.py`: conversion of decorated task methods into MCP server definitions.
- `charge/clients/agent.py`: common `Agent` and `AgentBackend` interfaces, plus agent-session (de)serialization helpers.
- `charge/clients/autogen.py`: AutoGen backend implementation.
- `charge/clients/agentframework.py`: Microsoft Agent Framework backend implementation.
- `charge/clients/openai_base.py`: shared backend/model/API-key configuration.
- `charge/clients/autogen_utils.py` and `agentframework_utils.py`: backend-specific tool and MCP setup.
- `charge/experiments/experiment.py`: sequential task runner.
- `charge/experiments/memory.py`: memory serialization and replay abstractions.
- `charge/utils/mcp_workbench_utils.py`: direct MCP session setup, tool listing, and tool calls.
- `examples/`: small runnable examples for SSE, multi-turn, and multi-server usage.
- `tests/`: unit and integration tests.

## Core Data Model

`Task` is the central user-facing model. It stores system/user prompts, optional verification/refinement prompts, MCP server URLs or STDIO files, per-server tool allowlists, structured output schema, bearer token, and attachments. Subclasses may define static tool methods decorated with `@hypothesis` or `@verifier`; those methods must include type annotations and docstrings because their signatures are converted into tool metadata.

Structured output is represented by a Pydantic `BaseModel` subclass. `Task.check_output_formatting()` validates generated JSON against that schema and is used by backends when structured responses are requested.

## Agent Backend Model

Backends implement `AgentBackend.create_agent(task=...)` and produce `Agent` instances with an async `run()` method. An `Experiment` is constructed with the `AgentBackend` it should use; callers that serve multiple users should give each session its own backend instance, since a backend carries that user's credentials (API key, base URL, model).

[DEPRECATED] The AutoGen path builds model clients, MCP workbenches, and tool wrappers around AutoGen agent primitives. It supports a broader set of providers, including OpenAI-compatible endpoints and local-style backends such as Ollama, Hugging Face, and vLLM where implemented.

The Agent Framework path builds Microsoft Agent Framework clients and tool adapters. It is primarily for OpenAI-compatible endpoints. When changing shared model configuration, verify both backend implementations because they translate settings differently.

`openai_base.py` is the provider configuration authority. Add or change backend names, base URLs, default models, and capability flags there before adapting backend-specific clients.

## MCP and Tool Flow

External tools enter through `Task.server_urls` or `Task.server_files`. `mcp_workbench_utils.py` opens HTTP/SSE or STDIO sessions, lists tools, and calls selected MCP tools directly when needed.

Task-local Python tools enter through decorated methods or registered built-in callables. Backend implementations wrap those callables into the tool format expected by AutoGen or Agent Framework.

Per-server allowlists should be applied before exposing tools to the model. When debugging missing or extra tools, inspect task construction, MCP session setup, backend utility adapters, and the final agent creation call.

## Experiment Flow

`Experiment` runs one or more tasks sequentially. Each task gets a new agent instance while shared state is carried through a `Memory` implementation, usually `ListMemory`. Save/load behavior serializes memory so parent applications can persist context across sessions.

Changing experiment behavior requires testing task ordering, memory serialization, and backend-specific memory replay in both AutoGen and Agent Framework paths.

## Debugging Guide

- Bad prompts or missing tools: inspect `Task` construction, decorators, and MCP allowlists.
- Backend configuration failure: inspect `openai_base.py`, then the selected backend module.
- Structured output failures: inspect the Pydantic schema and `Task.check_output_formatting()`.
- MCP connection failure: inspect `mcp_workbench_utils.py` and provider bearer token handling.
- Memory/context regression: inspect `Experiment`, `Memory`, and backend replay utilities.
