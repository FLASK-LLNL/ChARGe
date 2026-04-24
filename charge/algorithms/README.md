# ChARGe RSA Algorithm

Recursive Self-Aggregation (RSA): Generate N diverse proposals, recursively aggregate K-subset samples across T stages.

## Minimal Working Example

```python
import asyncio
from charge.algorithms import run_rsa_loop, RSAConfig, RSACallbacks, RSATaskFactories
from charge.experiment import Experiment

async def main():
    # Create experiment and agent
    experiment = Experiment(name="rsa_example")
    runner = experiment.create_agent_with_experiment_state(
        task=None,
        agent_name="rsa_agent"
    )

    # Configure RSA (N=4 proposals, K=2 subset size, T=2 stages)
    config = RSAConfig(n=4, k=2, t=2)

    # Create task factories with defaults (no customization needed)
    factories = RSATaskFactories(
        user_prompt="What are three solutions to reduce traffic congestion?"
    )

    # Setup callbacks
    callbacks = RSACallbacks()

    # Run RSA
    output, result = await run_rsa_loop(
        config=config,
        factories=factories,
        callbacks=callbacks,
        runner=runner,
    )

    print(f"Final result: {result.solution}")

asyncio.run(main())
```

That's it. RSA works out-of-the-box with sensible defaults.

## Customization

### 3-Part Prompt Structure

RSA uses three prompts:
1. **`system_prompt`**: Domain expert definition (same for proposals and aggregations)
2. **`proposal_prompt`**: Task instructions for generating solutions
3. **`aggregation_prompt`**: Task instructions for evaluating/synthesizing solutions

```python
from charge.algorithms import RSAPrompts
from pathlib import Path

# Load custom prompts
prompts = RSAPrompts(
    system_prompt="You are an expert chemist specializing in retrosynthesis.",
    proposal_prompt=Path("proposal_task.txt").read_text(),
    aggregation_prompt=Path("aggregation_task.txt").read_text(),
)

factories = RSATaskFactories(
    user_prompt="Synthesize aspirin from basic precursors",
    prompts=prompts,
)
```

### Custom Output Schema

```python
from pydantic import BaseModel
from typing import List

class ChemistryOutput(BaseModel):
    reasoning_summary: str
    reactants_smiles_list: List[str]
    products_smiles_list: List[str]

factories = RSATaskFactories(
    user_prompt="Provide retrosynthesis for CC(=O)Oc1ccccc1C(=O)O",
    output_schema=ChemistryOutput,
    prompts=prompts,
)
```

### Custom Formatter and Validator

```python
def format_candidates(subset):
    """Format proposals for aggregation"""
    text = ""
    for idx, prop in enumerate(subset, 1):
        result = prop["result"]
        text += f"\n---- Candidate {idx} ----\n"
        text += f"Reasoning: {result.reasoning_summary}\n"
        text += f"Reactants: {', '.join(result.reactants_smiles_list)}\n"
    return text

def validate_proposal(result):
    """Check if proposal is valid"""
    return (hasattr(result, 'reactants_smiles_list') and
            len(result.reactants_smiles_list) > 0)

factories = RSATaskFactories(
    user_prompt="...",
    output_schema=ChemistryOutput,
    prompts=prompts,
    format_candidates=format_candidates,
    validate_proposal=validate_proposal,
)
```

### Adding Tools

```python
from your_tools import verify_smiles, canonicalize_smiles

factories = RSATaskFactories(
    user_prompt="...",
    output_schema=ChemistryOutput,
    prompts=prompts,
    builtin_tools=[verify_smiles, canonicalize_smiles],  # Direct tool functions
    server_urls=["http://localhost:8000/mcp"],           # MCP server URLs
)
```

## Components

### RSAConfig
- `n`: Number of initial proposals (default: 8)
- `k`: Subset size for aggregation (default: 4)
- `t`: Number of stages (default: 3)
- `parallel`: Run proposals in parallel (default: True)
- `log_dir`: Directory for execution logs (default: auto-generated)

### RSAPrompts
- `system_prompt`: Domain expert (e.g., "You are an expert chemist")
- `proposal_prompt`: Generation task instructions
- `aggregation_prompt`: Evaluation task instructions (supports `{original_prompt}`, `{candidates}`, `{step}`, `{total_steps}`)

### RSACallbacks
- `log_progress`: Progress callback for UI updates
- `logger_info`, `logger_warning`, `logger_error`: Logging functions

### RSATaskFactories
- `user_prompt`: The problem to solve (required)
- `prompts`: Custom RSAPrompts (optional, uses generic defaults)
- `output_schema`: Pydantic model for validation (optional, uses GenericRSAOutput)
- `format_candidates`: Function to format proposals for aggregation (optional)
- `validate_proposal`: Function to check proposal validity (optional)
- `builtin_tools`: Direct tool functions (optional)
- `server_urls`: MCP server URLs (optional)
- `**task_kwargs`: Additional Task parameters

## Default Prompts

ChARGe provides generic defaults in `charge/algorithms/prompts/`:
- `default_proposal_system.txt` - Generic expert definition
- `default_proposal_prompt.txt` - Generic generation task
- `default_aggregation_prompt.txt` - Generic evaluation task

These work for any problem but should be replaced with domain-specific prompts for best results.
