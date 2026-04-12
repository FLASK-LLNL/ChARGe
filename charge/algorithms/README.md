# ChARGe Algorithms

This module provides generic algorithms that work with ChARGe tasks.

## Recursive Self-Aggregation (RSA)

The RSA algorithm generates N diverse proposals and recursively aggregates them T times using K-subset sampling to find high-quality solutions.

### Basic Usage with Default Prompts

```python
from charge.algorithms import run_rsa_loop, RSAConfig, RSACallbacks, RSATaskFactories

# Configure RSA
config = RSAConfig(n=8, k=4, t=3)

# Setup callbacks (uses defaults if omitted)
callbacks = RSACallbacks()

# Create task factories
factories = RSATaskFactories(
    create_proposal_task=lambda: YourTask(...),
    create_aggregation_task=lambda candidates, subset, step, total: YourAggTask(...),
    format_candidates=lambda subset: format_to_text(subset),
    output_schema=YourOutputSchema,
)

# Run RSA
output, result = await run_rsa_loop(
    config=config,
    factories=factories,
    callbacks=callbacks,
    runner=your_runner,
)
```

```python
from charge.algorithms import RSAPrompts
from pydantic import BaseModel
from typing import List

# Example: Chemistry retrosynthesis customization

# 1. Define custom output schema
class ChemistryOutput(BaseModel):
    reasoning_summary: str
    reactants_smiles_list: List[str]
    products_smiles_list: List[str]

# 2. Load custom prompts
chemistry_prompts = RSAPrompts(
    proposal_system_prompt=Path("chemistry_system.txt").read_text(),
    aggregation_template=Path("chemistry_aggregation.txt").read_text(),
)

# 3. Create custom formatter
def format_chemistry_candidates(proposals):
    text = ""
    for idx, prop in enumerate(proposals, 1):
        result = prop["result"]
        text += f"\n---- Candidate {idx} ----\n"
        text += f"Reasoning: {result.reasoning_summary}\n"
        text += f"Reactants: {', '.join(result.reactants_smiles_list)}\n"
        text += f"Products: {', '.join(result.products_smiles_list)}\n"
    return text

# 4. Create custom validator
def validate_chemistry(result):
    return (hasattr(result, 'reactants_smiles_list') and
            len(result.reactants_smiles_list) > 0)

# 5. Use with RSA
factories = RSATaskFactories(
    user_prompt="Synthesize aspirin from...",
    output_schema=ChemistryOutput,
    format_candidates=format_chemistry_candidates,
    validate_proposal=validate_chemistry,
    prompts=chemistry_prompts,
    builtin_tools=[verify_smiles, ...],  # Domain tools
)
```

Or provide fully custom task factories:

```python
# For maximum control, provide custom task creation functions
def create_my_proposal_task():
    return MyDomainTask(...)

def create_my_aggregation_task(candidates, subset, step, total):
    return MyAggregationTask(...)

factories = RSATaskFactories(
    create_proposal_task=create_my_proposal_task,
    create_aggregation_task=create_my_aggregation_task,
    format_candidates=my_formatter,
    output_schema=MySchema,
)
```

### Default Prompts

ChARGe provides task-agnostic default prompts in `charge/algorithms/prompts/`:
- `default_proposal_system.txt` - Generic problem-solving system prompt
- `default_aggregation_template.txt` - Generic aggregation template

These work out-of-the-box for any task but can be swapped for domain-specific prompts.

### Aggregation Template Placeholders

The aggregation template supports these placeholders:
- `{original_prompt}` - The original user prompt
- `{candidates}` - Formatted candidate solutions
- `{step}` - Current aggregation step (2..T)
- `{total_steps}` - Total number of steps (T)

## Components

### RSAConfig
Algorithm parameters: `n` (proposals), `k` (subset size), `t` (stages), `parallel` (bool), `log_dir` (path)

### RSAPrompts
Prompt configuration: `proposal_system_prompt`, `aggregation_template`

### RSACallbacks
Logging hooks: `log_progress`, `logger_info`, `logger_warning`, `logger_error`

### RSATaskFactories
Task creation logic: `create_proposal_task`, `create_aggregation_task`, `format_candidates`, `output_schema`, `validate_proposal`, `prompts`
