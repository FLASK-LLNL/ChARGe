# ChARGe RSA Algorithm

Recursive Self-Aggregation (RSA): generate N diverse proposals, then for T-1 stages aggregate K-subsets of the current pool into improved candidates. Reference: <https://arxiv.org/pdf/2509.26626>.

The algorithm is exposed as a single `RSATask` class that extends `charge.tasks.Task`. It is domain-agnostic out of the box and can be specialized by subclassing.

## Minimal Working Example (no subclassing)

```python
import asyncio
from charge.algorithms import RSATask
from charge.experiments.experiment import Experiment

async def main():
    experiment = Experiment(name="rsa_example")
    runner = experiment.create_agent_with_experiment_state(
        task=None,
        agent_name="rsa_agent",
    )

    task = RSATask(
        n=4, k=2, t=2,
        user_prompt="What are three solutions to reduce traffic congestion?",
    )

    output, result = await task.run_rsa(runner)
    print(f"Final solution: {result.solution}")

asyncio.run(main())
```

That's it. RSA works out of the box with sensible defaults. `result` is a `GenericRSAOutput` with `reasoning` and `solution` fields.

## Cross-Domain Reuse

Because every hook on `RSATask` has a working default, the same class fits any prompt-and-aggregate problem with no chemistry-specific code:

```python
# Short-story brainstorming
task = RSATask(
    n=6, k=3, t=3,
    system_prompt="You are an award-winning fiction editor.",
    proposal_prompt=(
        "Pitch one original short story premise. Keep it under 80 words. "
        "Specify protagonist, setting, central conflict, and a single "
        "irresistible hook."
    ),
    aggregation_prompt=(
        "Original brief: {original_prompt}\n\n"
        "Candidate premises (Step {step} of {total_steps}):\n{candidates}\n\n"
        "Combine the strongest elements into one improved premise."
    ),
    user_prompt="A story about an unexpected return.",
)
output, result = await task.run_rsa(runner)
```

```python
# Math problem solving with a custom output schema
from pydantic import BaseModel

class MathSolution(BaseModel):
    reasoning: str
    answer: str
    confidence: float

task = RSATask(
    n=8, k=4, t=2,
    user_prompt="Find x: 3*x**2 - 12*x + 7 = 0.",
    structured_output_schema=MathSolution,
)
output, result = await task.run_rsa(runner)
print(result.answer, result.confidence)
```

No subclassing required for either. The default `format_candidates` introspects any Pydantic schema; the default validator accepts all results.

## Customization

Subclass `RSATask` only when you need domain-specific behavior. Override one or more hooks:

| Hook | Default behavior | Override to |
|---|---|---|
| `format_candidates(subset)` | Introspects Pydantic schema, formats every field | Emit a domain-shaped candidates block (e.g. chemistry with reactants/products) |
| `validate_proposal(result)` | Accept all | Reject proposals missing required structure |
| `build_proposal_task()` | Combine `proposal_prompt` + `user_prompt` into a `Task` | Inject per-call context (RAG, prior turns) before building the Task |
| `build_aggregation_task(text, step)` | Format `aggregation_prompt` template | Use a different Task subclass or aggregation strategy |

```python
from charge.algorithms import RSATask
from pydantic import BaseModel

class ChemistryOutput(BaseModel):
    reasoning_summary: str
    reactants_smiles_list: list[str]
    products_smiles_list: list[str]

class RetroRSATask(RSATask):
    def __init__(self, **kw):
        kw.setdefault("structured_output_schema", ChemistryOutput)
        super().__init__(**kw)

    def format_candidates(self, subset):
        out = ""
        for idx, prop in enumerate(subset, 1):
            r = prop["result"]
            out += f"\n---- Candidate {idx} ----\n"
            out += f"Reasoning: {r.reasoning_summary}\n"
            out += f"Reactants: {', '.join(r.reactants_smiles_list)}\n"
            out += f"Products: {', '.join(r.products_smiles_list)}\n"
        return out

    def validate_proposal(self, result):
        return len(getattr(result, "reactants_smiles_list", [])) > 0
```

### Adding Tools

`Task` keyword arguments (e.g. `server_urls`, `builtin_tools`, `bearer_token`) are forwarded through `RSATask.__init__` and reused for every proposal / aggregation Task constructed by the defaults:

```python
task = RSATask(
    n=4, k=2, t=2,
    user_prompt="...",
    server_urls=["http://localhost:8000/mcp"],
    builtin_tools=[verify_smiles, canonicalize_smiles],
    bearer_token="...",
)
```

### Parallel Execution

Pass a `runner_factory` to `run_rsa()` to enable parallel proposals and parallel per-stage aggregations:

```python
def make_runner():
    return experiment.create_agent_with_experiment_state(
        task=None,
        agent_name=f"proposal_runner",
    )

output, result = await task.run_rsa(runner, runner_factory=make_runner)
```

Without `runner_factory`, the loop runs sequentially even when `parallel=True`.

### Custom Progress / Logging

```python
async def info(msg): await my_ui.send(f"[INFO] {msg}")
async def warn(msg): await my_ui.send(f"[WARN] {msg}")

output, result = await task.run_rsa(
    runner,
    log_progress=my_streaming_callback,
    logger_info=info,
    logger_warning=warn,
)
```

When omitted, RSA falls back to `print()` for info/warn/error.

## RSATask Reference

```python
RSATask(
    n: int,                      # number of initial proposals (>=2)
    k: int,                      # aggregation subset size (>=2, <=n)
    t: int,                      # total stages (>=2)
    *,
    user_prompt: str | None,           # problem to solve (required for default build_*_task)
    system_prompt: str | None,         # default: generic expert
    proposal_prompt: str | None,       # default: loaded from prompts/default_proposal_system.txt
    aggregation_prompt: str | None,    # default: loaded from prompts/default_aggregation_template.txt
    structured_output_schema: type | None,  # default: GenericRSAOutput
    parallel: bool = True,             # parallel proposals/aggregations when runner_factory given
    log_dir: str | None,               # default: /tmp/rsa_execution_<timestamp>
    disable_validation: bool = False,  # skip schema validation; also via CHARGE_DISABLE_OUTPUT_VALIDATION=1
    **task_kwargs,                     # forwarded to underlying Task
)

await task.run_rsa(
    runner,
    *,
    runner_factory=None,         # required for true parallel execution
    log_progress=None,           # per-token streaming callback
    logger_info=None,            # async info logger; default print() to stdout
    logger_warning=None,         # async warning logger; default print() to stderr
    logger_error=None,           # reserved
    callback_handler=None,       # optional handler with awaitable .drain()
) -> tuple[str, BaseModel]       # (raw_output_json, validated_result)
```

## Default Prompts

ChARGe ships generic defaults in `charge/algorithms/prompts/`:

- `default_proposal_system.txt` — generic problem-solver prompt
- `default_aggregation_template.txt` — generic aggregation template with `{original_prompt}`, `{candidates}`, `{step}`, `{total_steps}` placeholders

These work for any problem but are usually replaced with domain-specific prompts for best results.

## Standalone helpers

`create_default_proposal_task(user_prompt, ...)` and `create_default_aggregation_task(original_user_prompt, candidates_text, step, total_steps, ...)` are also exported for callers that want a single Task without running the loop.
