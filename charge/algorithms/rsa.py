"""
Recursive Self-Aggregation (RSA) Algorithm

This module provides the core N-K-T RSA loop that can be used by any task type
(retrosynthesis, LMO, etc.) through customizable factory functions and formatters.

RSA Algorithm:
    Stage 1: Generate N diverse proposals
    Stages 2-T: Recursively aggregate K-subset proposals
    Output: Single best proposal from final stage
"""

import random
import json
import os
import asyncio
from typing import Any, Callable, Optional, TypeVar, Generic, List
from pathlib import Path
from dataclasses import dataclass, field
from pydantic import BaseModel


# Type variable for output schemas
T = TypeVar("T")

# Default prompt paths
_PROMPTS_DIR = Path(__file__).parent / "prompts"
_DEFAULT_PROPOSAL_SYSTEM = _PROMPTS_DIR / "default_proposal_system.txt"
_DEFAULT_AGGREGATION_TEMPLATE = _PROMPTS_DIR / "default_aggregation_template.txt"


# Generic default output schema
class GenericRSAOutput(BaseModel):
    """Generic output schema for RSA proposals and aggregations.

    This provides a sensible default that works for any task.
    Domain-specific implementations can define their own schemas.

    Attributes:
        reasoning: Explanation of the approach and solution
        solution: The proposed solution as a string
    """

    reasoning: str
    solution: str


@dataclass
class RSAConfig:
    """Configuration for RSA algorithm execution.

    Attributes:
        n: Number of initial proposals to generate
        k: Size of subsets for aggregation (K <= N)
        t: Total number of stages (including initial proposals)
        parallel: If True, generate proposals/aggregations in parallel (default: True)
        log_dir: Directory to save execution logs (default: /tmp/rsa_execution_{timestamp})
        disable_validation: If True, skip output schema validation (default: False)
    """

    n: int
    k: int
    t: int
    parallel: bool = True
    log_dir: Optional[str] = None
    disable_validation: bool = False

    def __post_init__(self):
        """Validate configuration parameters and set defaults."""
        if self.n < 2 or self.k < 2 or self.t < 2:
            raise ValueError(
                f"Invalid RSA parameters: N={self.n}, K={self.k}, T={self.t}. "
                f"All must be >= 2. (T=1 would skip aggregation, defeating RSA's purpose)"
            )

        if self.k > self.n:
            raise ValueError(
                f"Invalid RSA parameters: K ({self.k}) cannot be greater than N ({self.n}). "
                f"K is the subset size for aggregation and must be <= N (number of proposals)."
            )

        if self.log_dir is None:
            import datetime

            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            self.log_dir = f"/tmp/rsa_execution_{timestamp}"

        os.makedirs(self.log_dir, exist_ok=True)


@dataclass
class RSAPrompts:
    """Prompts for RSA proposal and aggregation tasks.

    Provides default task-agnostic prompts that can be overridden for domain-specific use.

    Attributes:
        system_prompt: System prompt defining domain expert (used for both proposals and aggregations)
        proposal_prompt: Task-specific instructions for proposal generation
        aggregation_prompt: Task-specific instructions for aggregation (uses {original_prompt},
                           {candidates}, {step}, {total_steps} placeholders)
    """

    system_prompt: Optional[str] = None
    proposal_prompt: Optional[str] = None
    aggregation_prompt: Optional[str] = None

    def __post_init__(self):
        """Load default prompts if not provided."""
        if self.system_prompt is None:
            self.system_prompt = "You are an expert problem solver using systematic reasoning and available tools."

        if self.proposal_prompt is None:
            if _DEFAULT_PROPOSAL_SYSTEM.exists():
                # Load old default file for backward compatibility
                self.proposal_prompt = _DEFAULT_PROPOSAL_SYSTEM.read_text()
            else:
                # Fallback if file doesn't exist
                self.proposal_prompt = (
                    "Your task is to generate a high-quality solution to the problem. "
                    "Use available tools and provide clear reasoning."
                )

        if self.aggregation_prompt is None:
            if _DEFAULT_AGGREGATION_TEMPLATE.exists():
                self.aggregation_prompt = _DEFAULT_AGGREGATION_TEMPLATE.read_text()
            else:
                # Fallback if file doesn't exist
                self.aggregation_prompt = (
                    "You are aggregating multiple solutions.\n\n"
                    "Original problem:\n{original_prompt}\n\n"
                    "Candidates (Step {step} of {total_steps}):\n{candidates}\n\n"
                    "Synthesize these into a single improved solution."
                )


class RSACallbacks:
    """Callbacks for RSA algorithm progress and logging.

    These provide hooks for custom logging and UI updates during RSA execution.
    All callbacks are async functions.
    """

    def __init__(
        self,
        log_progress: Optional[Callable[[str], Any]] = None,
        logger_info: Optional[Callable[[str], Any]] = None,
        logger_warning: Optional[Callable[[str], Any]] = None,
        logger_error: Optional[Callable[[str], Any]] = None,
    ):
        """Initialize callbacks with optional custom implementations.

        Args:
            log_progress: Callback for reasoning progress (for LLM streaming)
            logger_info: Info level logging callback
            logger_warning: Warning level logging callback
            logger_error: Error level logging callback
        """
        self.log_progress = log_progress or self._default_log_progress
        self.logger_info = logger_info or self._default_logger_info
        self.logger_warning = logger_warning or self._default_logger_warning
        self.logger_error = logger_error or self._default_logger_error

    async def _default_log_progress(self, message: str):
        """Default progress logger - print to stdout."""
        print(f"[RSA Progress] {message}")

    async def _default_logger_info(self, message: str):
        """Default info logger - print to stdout."""
        print(f"[RSA Info] {message}")

    async def _default_logger_warning(self, message: str):
        """Default warning logger - print to stderr."""
        import sys

        print(f"[RSA Warning] {message}", file=sys.stderr)

    async def _default_logger_error(self, message: str):
        """Default error logger - print to stderr."""
        import sys

        print(f"[RSA Error] {message}", file=sys.stderr)


def default_format_candidates(proposals: list[dict]) -> str:
    """Generic formatter for RSA candidates.

    Works with any Pydantic schema by extracting all fields and formatting them.
    Domain-specific implementations can provide custom formatters.

    Args:
        proposals: List of proposal dicts with 'result' key containing validated output

    Returns:
        Formatted text suitable for aggregation prompts
    """
    candidates_text = ""
    for idx, prop in enumerate(proposals, 1):
        prop_result = prop["result"]
        candidates_text += f"\n---- Candidate {idx} ----\n"

        # Format all fields from the schema
        for field_name, field_value in prop_result.model_dump().items():
            # Format field name nicely (snake_case -> Title Case)
            display_name = field_name.replace("_", " ").title()
            candidates_text += f"{display_name}: {field_value}\n"

    return candidates_text


def create_default_proposal_task(
    user_prompt: str,
    system_prompt: Optional[str] = None,
    proposal_prompt: Optional[str] = None,
    output_schema: Optional[type[BaseModel]] = None,
    **task_kwargs,
) -> Any:
    """Create a generic proposal task using default prompts.

    This is a production-ready default that works for any problem.
    Domain-specific implementations can provide custom task creation.

    Args:
        user_prompt: The problem/question to solve
        system_prompt: Domain expert definition (e.g., "You are an expert chemist")
        proposal_prompt: Task instructions for generating proposals
        output_schema: Optional output schema (uses GenericRSAOutput if None)
        **task_kwargs: Additional arguments passed to Task (server_urls, builtin_tools, etc.)

    Returns:
        Task instance configured for proposal generation
    """
    from charge.tasks.task import Task

    if system_prompt is None:
        system_prompt = "You are an expert problem solver using systematic reasoning and available tools."

    if proposal_prompt is None:
        proposal_prompt = "Your task is to generate a high-quality solution to the problem. Use available tools and provide clear reasoning."

    # Combine proposal instructions with user's problem
    combined_user_prompt = f"{proposal_prompt}\n\n{user_prompt}"

    if output_schema is None:
        output_schema = GenericRSAOutput

    task = Task(
        system_prompt=system_prompt,
        user_prompt=combined_user_prompt,
        structured_output_schema=output_schema,
        **task_kwargs,
    )

    return task


def create_default_aggregation_task(
    original_user_prompt: str,
    candidates_text: str,
    subset: list[dict],
    step: int,
    total_steps: int,
    aggregation_prompt: Optional[str] = None,
    system_prompt: Optional[str] = None,
    output_schema: Optional[type[BaseModel]] = None,
    **task_kwargs,
) -> Any:
    """Create a generic aggregation task using default template.

    This is a production-ready default that works for any problem.
    Domain-specific implementations can provide custom task creation.

    Args:
        original_user_prompt: The original problem/question
        candidates_text: Formatted candidate solutions
        subset: list of candidate proposal dicts (unused in generic version)
        step: Current aggregation step (2..T)
        total_steps: Total number of steps (T)
        aggregation_prompt: Task instructions template for aggregation (uses {original_prompt},
                           {candidates}, {step}, {total_steps} placeholders)
        system_prompt: Domain expert definition (same as proposals)
        output_schema: Optional output schema (uses GenericRSAOutput if None)
        **task_kwargs: Additional arguments passed to Task

    Returns:
        Task instance configured for aggregation
    """
    from charge.tasks.task import Task

    if system_prompt is None:
        system_prompt = "You are an expert problem solver using systematic reasoning and available tools."

    if aggregation_prompt is None:
        # Load default aggregation template
        if _DEFAULT_AGGREGATION_TEMPLATE.exists():
            aggregation_prompt = _DEFAULT_AGGREGATION_TEMPLATE.read_text()
        else:
            aggregation_prompt = (
                "Original problem:\n{original_prompt}\n\n"
                "Candidates (Step {step} of {total_steps}):\n{candidates}\n\n"
                "Synthesize these into a single improved solution."
            )

    if output_schema is None:
        output_schema = GenericRSAOutput

    # Format the aggregation prompt
    user_prompt = aggregation_prompt.format(
        original_prompt=original_user_prompt,
        candidates=candidates_text,
        step=step,
        total_steps=total_steps,
    )

    task = Task(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        structured_output_schema=output_schema,
        **task_kwargs,
    )

    return task


class RSATaskFactories(Generic[T]):
    """Factory functions for creating RSA tasks.

    This class encapsulates task creation logic. All parameters are optional
    with production-ready defaults, allowing RSA to work out-of-box.
    Domain-specific implementations can override any component.
    """

    def __init__(
        self,
        user_prompt: Optional[str] = None,
        create_proposal_task: Optional[Callable[[], Any]] = None,
        create_aggregation_task: Optional[
            Callable[[str, list[dict], int, int], Any]
        ] = None,
        format_candidates: Optional[Callable[[list[dict]], str]] = None,
        output_schema: Optional[type[T]] = None,
        validate_proposal: Optional[Callable[[Any], bool]] = None,
        prompts: Optional[RSAPrompts] = None,
        **task_kwargs,
    ):
        """Initialize task factories with optional overrides.

        All parameters have sensible defaults for out-of-box usage.

        Args:
            user_prompt: The problem/question to solve (required for default factories).
                        Not needed if providing custom factories.
            create_proposal_task: Factory that returns a Task for generating proposals.
                                 If None, uses create_default_proposal_task.
            create_aggregation_task: Factory that returns a Task for aggregating proposals.
                                    If None, uses create_default_aggregation_task.
            format_candidates: Function to format proposals into text for aggregation.
                              If None, uses default_format_candidates.
            output_schema: Pydantic model class for validating task outputs.
                          If None, uses GenericRSAOutput.
            validate_proposal: Optional function to validate a proposal result.
                              If None, accepts all proposals (returns True).
            prompts: Optional RSAPrompts instance with custom prompts.
                    If None, uses default task-agnostic prompts from ChARGe.
            **task_kwargs: Additional arguments passed to Task instances
                          (server_urls, builtin_tools, bearer_token, etc.)
        """
        self.output_schema = output_schema or GenericRSAOutput
        self.validate_proposal = validate_proposal or self._default_validator
        self.prompts = prompts or RSAPrompts()
        self.task_kwargs = task_kwargs
        self.user_prompt = user_prompt

        # Use provided factories or create defaults
        if create_proposal_task is not None:
            self.create_proposal_task = create_proposal_task
        else:
            # Create default factory using user_prompt
            if user_prompt is None:
                raise ValueError(
                    "user_prompt is required when using default task factories. "
                    "Either provide user_prompt or custom create_proposal_task/create_aggregation_task."
                )
            self.create_proposal_task = self._create_default_proposal_factory()

        if create_aggregation_task is not None:
            self.create_aggregation_task = create_aggregation_task
        else:
            # Create default factory
            if user_prompt is None:
                raise ValueError(
                    "user_prompt is required when using default task factories. "
                    "Either provide user_prompt or custom create_proposal_task/create_aggregation_task."
                )
            self.create_aggregation_task = self._create_default_aggregation_factory()

        self.format_candidates = format_candidates or default_format_candidates

    def _create_default_proposal_factory(self) -> Callable[[], Any]:
        """Create a default proposal task factory."""

        def factory():
            return create_default_proposal_task(
                user_prompt=self.user_prompt,
                system_prompt=self.prompts.system_prompt,
                proposal_prompt=self.prompts.proposal_prompt,
                output_schema=self.output_schema,
                **self.task_kwargs,
            )

        return factory

    def _create_default_aggregation_factory(
        self,
    ) -> Callable[[str, list[dict], int, int], Any]:
        """Create a default aggregation task factory."""

        def factory(
            candidates_text: str, subset: list[dict], step: int, total_steps: int
        ):
            return create_default_aggregation_task(
                original_user_prompt=self.user_prompt,
                candidates_text=candidates_text,
                subset=subset,
                step=step,
                total_steps=total_steps,
                aggregation_prompt=self.prompts.aggregation_prompt,
                system_prompt=self.prompts.system_prompt,  # Same expert as proposals
                output_schema=self.output_schema,
                **self.task_kwargs,
            )

        return factory

    def _default_validator(self, result: Any) -> bool:
        """Default validator - always returns True."""
        return True


async def run_rsa_loop(
    config: RSAConfig,
    factories: RSATaskFactories,
    callbacks: RSACallbacks,
    runner: Any,
    runner_factory: Optional[Callable[[], Any]] = None,
    callback_handler: Optional[Any] = None,
) -> tuple[str, Any]:
    """
    Execute the generic N-K-T RSA algorithm.

    Args:
        config: RSA configuration (n, k, t, parallel, log_dir)
        factories: Task factory functions for proposal and aggregation
        callbacks: Logging and progress callbacks
        runner: ChARGe agent runner with .task and .run() interface
        runner_factory: Factory to create independent runner instances for parallel execution.
                       If None and parallel=True, falls back to sequential execution.
        callback_handler: Optional callback handler to drain after each task

    Returns:
        tuple: (final_output_json, final_result_object)

    Raises:
        ValueError: If all proposals fail or invalid parameters
    """
    # K validation happens in RSAConfig.__post_init__
    k = config.k

    # Helper function to run a single proposal
    async def run_single_proposal(proposal_index: int, proposal_runner: Any):
        """Run a single proposal and return result or None if failed"""
        try:
            await callbacks.logger_info(
                f"Generating proposal {proposal_index+1}/{config.n}"
            )

            # Create proposal task
            proposal_task = factories.create_proposal_task()
            proposal_runner.task = proposal_task

            # Disable validation if requested
            if (
                config.disable_validation
                or os.getenv("CHARGE_DISABLE_OUTPUT_VALIDATION", "0") == "1"
            ):
                proposal_task.structured_output_schema = None

            # Save proposal prompt
            proposer_log = {
                "proposal_index": proposal_index + 1,
                "system_prompt": proposal_task.get_system_prompt(),
                "user_prompt": proposal_task.get_user_prompt(),
            }
            with open(
                f"{config.log_dir}/proposer_{proposal_index+1:02d}_prompt.json", "w"
            ) as f:
                json.dump(proposer_log, f, indent=2)

            # Run proposal
            proposal_output = await proposal_runner.run(callbacks.log_progress)
            if callback_handler:
                await callback_handler.drain()

            # Validate output
            proposal_result = factories.output_schema.model_validate_json(
                proposal_output
            )

            # Check if proposal is valid using custom validator
            if not factories.validate_proposal(proposal_result):
                await callbacks.logger_warning(
                    f"Proposal {proposal_index+1} failed validation (empty or invalid), skipping"
                )
                return None

            # Save proposal output
            proposer_output_log = {
                "proposal_index": proposal_index + 1,
                "result": proposal_result.model_dump(),
                "full_output": json.loads(proposal_output),
            }
            with open(
                f"{config.log_dir}/proposer_{proposal_index+1:02d}_output.json", "w"
            ) as f:
                json.dump(proposer_output_log, f, indent=2)

            await callbacks.logger_info(
                f"Proposal {proposal_index+1} completed successfully"
            )

            return {
                "output": proposal_output,
                "result": proposal_result,
                "index": proposal_index,
            }

        except asyncio.CancelledError:
            await callbacks.logger_warning(f"Proposal {proposal_index+1} was cancelled")
            return None
        except Exception as e:
            await callbacks.logger_warning(
                f"Proposal {proposal_index+1} failed: {str(e)}"
            )
            return None

    # Stage 1: Generate N initial proposals
    await callbacks.logger_info(
        f"RSA Step 1/{config.t}: Generating {config.n} initial proposals"
        + (" (parallel mode)" if config.parallel else " (sequential mode)")
    )

    if config.parallel and runner_factory:
        # Parallel mode: generate all proposals concurrently
        proposal_tasks = []
        for i in range(config.n):
            # Create independent runner for each proposal
            proposal_runner = runner_factory()
            task = run_single_proposal(i, proposal_runner)
            proposal_tasks.append(task)

        # Run all proposals in parallel
        proposal_results = await asyncio.gather(*proposal_tasks, return_exceptions=True)

        # Filter valid proposals and handle exceptions
        proposals = []
        for i, result in enumerate(proposal_results):
            if isinstance(result, Exception):
                await callbacks.logger_warning(
                    f"Proposal {i+1} failed with exception: {str(result)}"
                )
            elif result is not None:
                proposals.append(result)
    else:
        # Sequential mode: generate proposals one by one
        if config.parallel and not runner_factory:
            await callbacks.logger_warning(
                "Parallel mode requested but no runner_factory provided, falling back to sequential"
            )

        proposals = []
        for i in range(config.n):
            result = await run_single_proposal(i, runner)
            if result is not None:
                proposals.append(result)

    if not proposals:
        raise ValueError("All RSA proposals failed")

    await callbacks.logger_info(f"Generated {len(proposals)} valid proposals")

    # Helper function to run a single aggregation
    async def run_single_aggregation(
        agg_index: int, step: int, current_proposals: list, agg_runner: Any
    ):
        """Run a single aggregation and return result or None if failed"""
        try:
            # Adjust K if needed
            current_k = k
            if len(current_proposals) < k:
                current_k = len(current_proposals)

            # Select K random proposals
            if len(current_proposals) <= current_k:
                subset = current_proposals
            else:
                subset = random.sample(current_proposals, current_k)

            # Format candidates using task-specific formatter
            candidates_text = factories.format_candidates(subset)
            subset_indices = [prop["index"] + 1 for prop in subset]

            # Create aggregation task
            agg_task = factories.create_aggregation_task(
                candidates_text, subset, step, config.t
            )
            agg_runner.task = agg_task

            if (
                config.disable_validation
                or os.getenv("CHARGE_DISABLE_OUTPUT_VALIDATION", "0") == "1"
            ):
                agg_task.structured_output_schema = None

            # Save aggregation prompt
            aggregator_log = {
                "step": step,
                "aggregation_index": agg_index + 1,
                "k_subset_indices": subset_indices,
                "system_prompt": agg_task.get_system_prompt(),
                "user_prompt": agg_task.get_user_prompt(),
                "candidates_text": candidates_text,
            }
            with open(
                f"{config.log_dir}/aggregator_step{step}_{agg_index+1:02d}_prompt.json",
                "w",
            ) as f:
                json.dump(aggregator_log, f, indent=2)

            # Run aggregation
            agg_output = await agg_runner.run(callbacks.log_progress)
            if callback_handler:
                await callback_handler.drain()

            # Validate output
            agg_result = factories.output_schema.model_validate_json(agg_output)

            # Check if aggregation is valid using custom validator
            if not factories.validate_proposal(agg_result):
                await callbacks.logger_warning(
                    f"Aggregation {agg_index+1} (Step {step}) failed validation (empty or invalid), skipping"
                )
                return None

            # Save aggregation output
            aggregator_output_log = {
                "step": step,
                "aggregation_index": agg_index + 1,
                "k_subset_indices": subset_indices,
                "result": agg_result.model_dump(),
                "full_output": json.loads(agg_output),
            }
            with open(
                f"{config.log_dir}/aggregator_step{step}_{agg_index+1:02d}_output.json",
                "w",
            ) as f:
                json.dump(aggregator_output_log, f, indent=2)

            await callbacks.logger_info(
                f"Aggregation {agg_index+1} (Step {step}) completed successfully"
            )

            return {
                "output": agg_output,
                "result": agg_result,
                "index": agg_index,
                "step": step,
            }

        except asyncio.CancelledError:
            await callbacks.logger_warning(
                f"Aggregation {agg_index+1} (Step {step}) was cancelled"
            )
            return None
        except Exception as e:
            await callbacks.logger_warning(
                f"Aggregation {agg_index+1} (Step {step}) failed: {str(e)}"
            )
            return None

    # Stages 2-T: Recursive aggregation
    # BARRIER: Wait for all Stage 1 proposals to complete before starting Stage 2
    current_proposals = proposals
    await callbacks.logger_info(
        f"Stage 1 complete. Generated {len(proposals)} valid proposals."
    )

    for step in range(2, config.t + 1):
        await callbacks.logger_info(
            f"RSA Step {step}/{config.t}: Aggregating {len(current_proposals)} proposals into {k}-subsets"
            + (" (parallel mode)" if config.parallel else " (sequential mode)")
        )

        # Log if K needs adjustment due to insufficient proposals
        if len(current_proposals) < k:
            await callbacks.logger_warning(
                f"Step {step}: Only {len(current_proposals)} proposals available, K will be adjusted from {k} to {len(current_proposals)} for this stage"
            )

        # Generate aggregations
        num_aggregations = max(config.n, len(current_proposals))

        if config.parallel and runner_factory:
            # Parallel mode: run all aggregations in this stage concurrently
            agg_tasks = []
            for i in range(num_aggregations):
                # Create independent runner for each aggregation
                agg_runner = runner_factory()
                task = run_single_aggregation(i, step, current_proposals, agg_runner)
                agg_tasks.append(task)

            # Run all aggregations in parallel and wait for all to complete
            # BARRIER: asyncio.gather waits for all aggregations in this stage
            agg_results = await asyncio.gather(*agg_tasks, return_exceptions=True)

            # Filter valid aggregations and handle exceptions
            next_proposals = []
            for i, result in enumerate(agg_results):
                if isinstance(result, Exception):
                    await callbacks.logger_warning(
                        f"Aggregation {i+1} (Step {step}) failed with exception: {str(result)}"
                    )
                elif result is not None:
                    next_proposals.append(result)

        else:
            # Sequential mode: run aggregations one by one
            if config.parallel and not runner_factory:
                await callbacks.logger_warning(
                    "Parallel mode requested but no runner_factory provided, falling back to sequential"
                )

            next_proposals = []
            for i in range(num_aggregations):
                result = await run_single_aggregation(
                    i, step, current_proposals, runner
                )
                if result is not None:
                    next_proposals.append(result)

        if not next_proposals:
            await callbacks.logger_warning(
                f"No successful aggregations in step {step}, using previous proposals"
            )
            break

        # BARRIER: All aggregations in current stage complete before moving to next stage
        current_proposals = next_proposals
        await callbacks.logger_info(
            f"Stage {step} complete. Generated {len(current_proposals)} valid aggregations."
        )

    # Select final proposal (first one from final stage)
    if not current_proposals:
        raise ValueError("RSA failed to produce any valid proposals")

    final_proposal = current_proposals[0]
    final_output = final_proposal["output"]
    final_result = final_proposal["result"]

    # Save final output
    final_log = {
        "final_step": step if step <= config.t else config.t,
        "n_proposals": config.n,
        "k_subset_size": k,
        "t_stages": config.t,
        "final_result": final_result.model_dump(),
    }
    log_path = Path(config.log_dir) / "FINAL_OUTPUT.json"
    log_path.write_text(json.dumps(final_log, indent=2))

    await callbacks.logger_info(
        f"RSA completed! Final output saved to {config.log_dir}"
    )

    return final_output, final_result
