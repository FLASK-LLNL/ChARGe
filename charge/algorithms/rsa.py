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
from typing import Any, Callable, Optional, TypeVar, Generic
from pathlib import Path
from dataclasses import dataclass, field


# Type variable for output schemas
T = TypeVar('T')

# Default prompt paths
_PROMPTS_DIR = Path(__file__).parent / "prompts"
_DEFAULT_PROPOSAL_SYSTEM = _PROMPTS_DIR / "default_proposal_system.txt"
_DEFAULT_AGGREGATION_TEMPLATE = _PROMPTS_DIR / "default_aggregation_template.txt"


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
        if self.n < 1 or self.k < 1 or self.t < 1:
            raise ValueError(f"Invalid RSA parameters: N={self.n}, K={self.k}, T={self.t} (all must be >= 1)")

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
        proposal_system_prompt: System prompt for proposal generation tasks
        aggregation_template: Template for aggregation prompts (uses {original_prompt},
                             {candidates}, {step}, {total_steps} placeholders)
    """
    proposal_system_prompt: Optional[str] = None
    aggregation_template: Optional[str] = None

    def __post_init__(self):
        """Load default prompts if not provided."""
        if self.proposal_system_prompt is None:
            if _DEFAULT_PROPOSAL_SYSTEM.exists():
                self.proposal_system_prompt = _DEFAULT_PROPOSAL_SYSTEM.read_text()
            else:
                # Fallback if file doesn't exist
                self.proposal_system_prompt = (
                    "You are an expert problem solver. Generate a high-quality solution "
                    "to the problem provided by the user. Use available tools and provide "
                    "clear reasoning."
                )

        if self.aggregation_template is None:
            if _DEFAULT_AGGREGATION_TEMPLATE.exists():
                self.aggregation_template = _DEFAULT_AGGREGATION_TEMPLATE.read_text()
            else:
                # Fallback if file doesn't exist
                self.aggregation_template = (
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


class RSATaskFactories(Generic[T]):
    """Factory functions for creating RSA tasks.

    This class encapsulates the domain-specific task creation logic,
    allowing RSA to work with any task type.
    """

    def __init__(
        self,
        create_proposal_task: Callable[[], Any],
        create_aggregation_task: Callable[[str, list[dict], int, int], Any],
        format_candidates: Callable[[list[dict]], str],
        output_schema: type[T],
        validate_proposal: Optional[Callable[[Any], bool]] = None,
        prompts: Optional[RSAPrompts] = None,
    ):
        """Initialize task factories.

        Args:
            create_proposal_task: Factory that returns a Task for generating proposals.
                                 Should return a fresh Task instance each time.
            create_aggregation_task: Factory that returns a Task for aggregating proposals.
                                    Takes (candidates_text, subset, step, total_steps).
            format_candidates: Function to format proposals into text for aggregation.
                              Takes list of proposal dicts, returns formatted string.
            output_schema: Pydantic model class for validating task outputs.
            validate_proposal: Optional function to validate a proposal result.
                              Takes result object, returns True if valid.
                              Used to filter out empty/invalid proposals.
            prompts: Optional RSAPrompts instance with custom prompts.
                    If None, uses default task-agnostic prompts from ChARGe.
        """
        self.create_proposal_task = create_proposal_task
        self.create_aggregation_task = create_aggregation_task
        self.format_candidates = format_candidates
        self.output_schema = output_schema
        self.validate_proposal = validate_proposal or self._default_validator
        self.prompts = prompts or RSAPrompts()

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

    # Validate K <= N
    k = config.k
    if k > config.n:
        await callbacks.logger_warning(f"K ({k}) > N ({config.n}), adjusting K to N")
        k = config.n

    # Helper function to run a single proposal
    async def run_single_proposal(proposal_index: int, proposal_runner: Any):
        """Run a single proposal and return result or None if failed"""
        try:
            await callbacks.logger_info(f"Generating proposal {proposal_index+1}/{config.n}")

            # Create proposal task
            proposal_task = factories.create_proposal_task()
            proposal_runner.task = proposal_task

            # Disable validation if requested
            if config.disable_validation or os.getenv("CHARGE_DISABLE_OUTPUT_VALIDATION", "0") == "1":
                proposal_task.structured_output_schema = None

            # Save proposal prompt
            proposer_log = {
                "proposal_index": proposal_index + 1,
                "system_prompt": proposal_task.get_system_prompt(),
                "user_prompt": proposal_task.get_user_prompt(),
            }
            with open(f"{config.log_dir}/proposer_{proposal_index+1:02d}_prompt.json", "w") as f:
                json.dump(proposer_log, f, indent=2)

            # Run proposal
            proposal_output = await proposal_runner.run(callbacks.log_progress)
            if callback_handler:
                await callback_handler.drain()

            # Validate output
            proposal_result = factories.output_schema.model_validate_json(proposal_output)

            # Check if proposal is valid using custom validator
            if not factories.validate_proposal(proposal_result):
                await callbacks.logger_warning(f"Proposal {proposal_index+1} failed validation (empty or invalid), skipping")
                return None

            # Save proposal output
            proposer_output_log = {
                "proposal_index": proposal_index + 1,
                "result": proposal_result.model_dump(),
                "full_output": json.loads(proposal_output)
            }
            with open(f"{config.log_dir}/proposer_{proposal_index+1:02d}_output.json", "w") as f:
                json.dump(proposer_output_log, f, indent=2)

            await callbacks.logger_info(f"Proposal {proposal_index+1} completed successfully")

            return {
                "output": proposal_output,
                "result": proposal_result,
                "index": proposal_index
            }

        except Exception as e:
            await callbacks.logger_warning(f"Proposal {proposal_index+1} failed: {str(e)}")
            return None

    # Stage 1: Generate N initial proposals
    await callbacks.logger_info(f"RSA Step 1/{config.t}: Generating {config.n} initial proposals" +
                      (" (parallel mode)" if config.parallel else " (sequential mode)"))

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
                await callbacks.logger_warning(f"Proposal {i+1} failed with exception: {str(result)}")
            elif result is not None:
                proposals.append(result)
    else:
        # Sequential mode: generate proposals one by one
        if config.parallel and not runner_factory:
            await callbacks.logger_warning("Parallel mode requested but no runner_factory provided, falling back to sequential")

        proposals = []
        for i in range(config.n):
            result = await run_single_proposal(i, runner)
            if result is not None:
                proposals.append(result)

    if not proposals:
        raise ValueError("All RSA proposals failed")

    await callbacks.logger_info(f"Generated {len(proposals)} valid proposals")

    # Helper function to run a single aggregation
    async def run_single_aggregation(agg_index: int, step: int, current_proposals: list, agg_runner: Any):
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
                candidates_text,
                subset,
                step,
                config.t
            )
            agg_runner.task = agg_task

            if config.disable_validation or os.getenv("CHARGE_DISABLE_OUTPUT_VALIDATION", "0") == "1":
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
            with open(f"{config.log_dir}/aggregator_step{step}_{agg_index+1:02d}_prompt.json", "w") as f:
                json.dump(aggregator_log, f, indent=2)

            # Run aggregation
            agg_output = await agg_runner.run(callbacks.log_progress)
            if callback_handler:
                await callback_handler.drain()

            # Validate output
            agg_result = factories.output_schema.model_validate_json(agg_output)

            # Check if aggregation is valid using custom validator
            if not factories.validate_proposal(agg_result):
                await callbacks.logger_warning(f"Aggregation {agg_index+1} (Step {step}) failed validation (empty or invalid), skipping")
                return None

            # Save aggregation output
            aggregator_output_log = {
                "step": step,
                "aggregation_index": agg_index + 1,
                "k_subset_indices": subset_indices,
                "result": agg_result.model_dump(),
                "full_output": json.loads(agg_output)
            }
            with open(f"{config.log_dir}/aggregator_step{step}_{agg_index+1:02d}_output.json", "w") as f:
                json.dump(aggregator_output_log, f, indent=2)

            await callbacks.logger_info(f"Aggregation {agg_index+1} (Step {step}) completed successfully")

            return {
                "output": agg_output,
                "result": agg_result,
                "index": agg_index,
                "step": step
            }

        except Exception as e:
            await callbacks.logger_warning(f"Aggregation {agg_index+1} (Step {step}) failed: {str(e)}")
            return None

    # Stages 2-T: Recursive aggregation
    # BARRIER: Wait for all Stage 1 proposals to complete before starting Stage 2
    current_proposals = proposals
    await callbacks.logger_info(f"Stage 1 complete. Generated {len(proposals)} valid proposals.")

    for step in range(2, config.t + 1):
        await callbacks.logger_info(
            f"RSA Step {step}/{config.t}: Aggregating {len(current_proposals)} proposals into {k}-subsets" +
            (" (parallel mode)" if config.parallel else " (sequential mode)")
        )

        # Adjust K if needed
        current_k = k
        if len(current_proposals) < k:
            await callbacks.logger_warning(
                f"Not enough proposals ({len(current_proposals)}) for K={k}, using all available"
            )
            current_k = len(current_proposals)

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
                    await callbacks.logger_warning(f"Aggregation {i+1} (Step {step}) failed with exception: {str(result)}")
                elif result is not None:
                    next_proposals.append(result)

        else:
            # Sequential mode: run aggregations one by one
            if config.parallel and not runner_factory:
                await callbacks.logger_warning("Parallel mode requested but no runner_factory provided, falling back to sequential")

            next_proposals = []
            for i in range(num_aggregations):
                result = await run_single_aggregation(i, step, current_proposals, runner)
                if result is not None:
                    next_proposals.append(result)

        if not next_proposals:
            await callbacks.logger_warning(f"No successful aggregations in step {step}, using previous proposals")
            break

        # BARRIER: All aggregations in current stage complete before moving to next stage
        current_proposals = next_proposals
        await callbacks.logger_info(f"Stage {step} complete. Generated {len(current_proposals)} valid aggregations.")

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

    await callbacks.logger_info(f"RSA completed! Final output saved to {config.log_dir}")

    return final_output, final_result
