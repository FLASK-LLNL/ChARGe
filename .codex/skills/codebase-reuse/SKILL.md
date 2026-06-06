---
name: codebase-reuse
description: Use when making code changes in an existing repository to discover and reuse existing APIs, helpers, types, components, tests, and object-oriented extension points before writing new code from primitives.
---

# Codebase Reuse

Use this skill before implementing code changes in an existing repository.

## Workflow

1. Read the nearest `AGENTS.md`, `ARCHITECTURE.md`, and relevant README files.
2. Identify the owning module or abstraction before editing.
3. Search for existing functions, classes, types, hooks, components, tests, and helpers related to the requested behavior.
4. Prefer extending existing abstractions over creating parallel implementations.
5. Reuse existing data models, validators, serializers, callbacks, UI components, and service APIs.
6. Add a new abstraction only when existing ones cannot represent the behavior cleanly.
7. Add tests near the existing behavior being changed.

## Required Pre-Edit Check

Before editing, briefly identify:

- Existing APIs found.
- The abstraction or module being reused.
- Why the change belongs in the chosen file.
- What validation will be run.

## Design Rules

- Do not duplicate parsing, validation, transport, persistence, or formatting logic.
- Prefer composition, subclassing, adapters, or existing extension points.
- Keep domain behavior in domain modules and shared behavior in shared libraries.
- Preserve local naming, async, error handling, and message-shape conventions.
- Avoid new dependencies when an existing project dependency already solves the problem.
