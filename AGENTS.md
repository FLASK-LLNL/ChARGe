# Repository Guidelines

## Project Structure & Module Organization

ChARGe provides the `charge` Python package used by FLASK Copilot for LLM-backed tasks, tool augmentation, backend clients, and sequential experiments. Core task behavior lives in `charge/tasks/task.py`; decorators and MCP conversion helpers are in `charge/_tags.py` and `charge/_to_mcp.py`.

Backend integrations live in `charge/clients/`, including AutoGen, Agent Framework, OpenAI-compatible configuration, and the `AgentFactory`. Experiment orchestration and memory are in `charge/experiments/`. Examples are under `examples/`, and tests are split between `tests/unit_tests/` and `tests/integration_tests/`.

## Build, Test, and Development Commands

- `pip install -e .`: install the package in editable mode.
- `pip install -e ".[test]"`: install pytest and test helpers.
- `pip install -e ".[all]"`: install all optional backend integrations.
- `pytest tests/unit_tests`: run the fast unit test suite.
- `pytest tests/integration_tests`: run live integration tests; credentials may be required.
- `pre-commit run --all-files`: run formatting and basic file checks.

## Coding Style & Naming Conventions

Python requires `>=3.11, <3.13` and is formatted with Black through pre-commit. Use 4-space indentation, `snake_case` modules/functions, `PascalCase` classes, and `test_*.py` test files. Decorated task methods using `@hypothesis` or `@verifier` should include type annotations and docstrings because they are exposed as tool metadata.

Prefer current lowercase module paths from code, such as `charge.tasks.task`, over stale documentation references. Import backend classes from their concrete modules when `__init__.py` does not re-export them.

## Testing Guidelines

Use `pytest`. Add unit tests for task formatting, structured output validation, backend configuration, memory serialization, and MCP tool setup. Integration tests in `tests/integration_tests/` exercise live providers; document required credentials and avoid making them mandatory for routine local validation.

## Commit & Pull Request Guidelines

Use concise, scoped commit subjects, for example `fix(agentframework): preserve memory context`. Pull requests should include a summary, test commands, linked issues when available, and notes about backend/provider behavior changes. Because this repository is consumed as a submodule, commit changes here first, then update the parent repository's submodule pointer.

## Security & Configuration Tips

Do not commit provider credentials, tokens, local endpoint files, virtualenvs, caches, or generated logs. Keep model/provider configuration explicit and avoid changing default backend behavior without tests for both AutoGen and Agent Framework paths.
