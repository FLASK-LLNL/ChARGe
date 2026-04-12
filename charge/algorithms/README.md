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

### Using Custom Domain-Specific Prompts

The default prompts are task-agnostic. For domain-specific tasks (chemistry, biology, etc.), you can provide custom prompts:

```python
from charge.algorithms import RSAPrompts

# Option 1: Load from files
chemistry_prompts = RSAPrompts(
    proposal_system_prompt=Path("chemistry_system.txt").read_text(),
    aggregation_template=Path("chemistry_aggregation.txt").read_text(),
)

# Option 2: Provide strings directly
chemistry_prompts = RSAPrompts(
    proposal_system_prompt="You are a chemistry expert...",
    aggregation_template="Synthesize {candidates} into a better solution...",
)

# Pass to factories
factories = RSATaskFactories(
    ...,
    prompts=chemistry_prompts,  # Use domain-specific prompts
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
