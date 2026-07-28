"""Recursive Self-Aggregation (RSA) algorithm.

Reference: https://arxiv.org/pdf/2509.26626

Implements an N-K-T orchestration loop on top of the ChARGe Task interface:

    Stage 1     : Generate N independent proposals from the same problem.
    Stages 2..T : Repeatedly draw K-subsets of the current proposal pool and
                  aggregate each subset into a single improved proposal.
    Output      : Best proposal from the final stage.

The algorithm is domain-agnostic. ``RSATask`` works out of the box for any
prompt-and-aggregate problem using ``GenericRSAOutput`` as its schema. To
specialize for a domain, subclass ``RSATask`` and override one or more of:

    format_candidates(subset)        -> str
    validate_proposal(result)        -> bool
    build_proposal_task()            -> Task
    build_aggregation_task(text, step) -> Task
"""

from __future__ import annotations

import os
import json
import random
import asyncio
import datetime
from pathlib import Path
from typing import Any, Awaitable, Callable, Optional, Type

import sys
from pydantic import BaseModel

from charge.tasks.task import Task


# ----- module-level constants & defaults -----

_PROMPTS_DIR = Path(__file__).parent / "prompts"
_DEFAULT_PROPOSAL_SYSTEM = _PROMPTS_DIR / "default_proposal_system.txt"
_DEFAULT_AGGREGATION_TEMPLATE = _PROMPTS_DIR / "default_aggregation_template.txt"

_DEFAULT_SYSTEM_PROMPT = (
    "You are an expert problem solver using systematic reasoning and "
    "available tools."
)
_FALLBACK_PROPOSAL_PROMPT = (
    "Your task is to generate a high-quality solution to the problem. "
    "Use available tools and provide clear reasoning."
)
_FALLBACK_AGGREGATION_PROMPT = (
    "You are aggregating multiple solutions.\n\n"
    "Original problem:\n{original_prompt}\n\n"
    "Candidates (Step {step} of {total_steps}):\n{candidates}\n\n"
    "Synthesize these into a single improved solution."
)


class GenericRSAOutput(BaseModel):
    """Default output schema for RSA proposals and aggregations.

    Domain-specific code can supply a richer Pydantic model via the
    ``structured_output_schema`` parameter of :class:`RSATask`.
    """

    reasoning: str
    solution: str


def default_format_candidates(proposals: list[dict]) -> str:
    """Format a list of proposal dicts into a prompt-ready candidates block.

    Introspects whatever Pydantic model populates ``proposal["result"]`` so
    that the helper works for any output schema without modification.
    """
    out = ""
    for idx, prop in enumerate(proposals, 1):
        out += f"\n---- Candidate {idx} ----\n"
        for field_name, field_value in prop["result"].model_dump().items():
            display = field_name.replace("_", " ").title()
            out += f"{display}: {field_value}\n"
    return out


def create_default_proposal_task(
    user_prompt: str,
    system_prompt: Optional[str] = None,
    proposal_prompt: Optional[str] = None,
    output_schema: Optional[Type[BaseModel]] = None,
    **task_kwargs,
) -> Task:
    """Construct a generic proposal :class:`Task`.

    Combines ``proposal_prompt`` (instructions) with ``user_prompt`` (the
    specific problem) and wraps the result in a Task with the supplied
    schema. Used as the default body of :meth:`RSATask.build_proposal_task`
    and exposed for callers that want one-off Task construction without an
    RSA loop.
    """
    if system_prompt is None:
        system_prompt = _DEFAULT_SYSTEM_PROMPT
    if proposal_prompt is None:
        proposal_prompt = _FALLBACK_PROPOSAL_PROMPT
    if output_schema is None:
        output_schema = GenericRSAOutput

    combined = f"{proposal_prompt}\n\n{user_prompt}"
    return Task(
        system_prompt=system_prompt,
        user_prompt=combined,
        structured_output_schema=output_schema,
        **task_kwargs,
    )


def create_default_aggregation_task(
    original_user_prompt: str,
    candidates_text: str,
    step: int,
    total_steps: int,
    aggregation_prompt: Optional[str] = None,
    system_prompt: Optional[str] = None,
    output_schema: Optional[Type[BaseModel]] = None,
    **task_kwargs,
) -> Task:
    """Construct a generic aggregation :class:`Task`.

    Formats ``aggregation_prompt`` with ``{original_prompt}``,
    ``{candidates}``, ``{step}``, ``{total_steps}`` placeholders and wraps
    the result in a Task with the supplied schema.
    """
    if system_prompt is None:
        system_prompt = _DEFAULT_SYSTEM_PROMPT
    if aggregation_prompt is None:
        aggregation_prompt = _FALLBACK_AGGREGATION_PROMPT
    if output_schema is None:
        output_schema = GenericRSAOutput

    user_prompt = aggregation_prompt.format(
        original_prompt=original_user_prompt,
        candidates=candidates_text,
        step=step,
        total_steps=total_steps,
    )
    return Task(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        structured_output_schema=output_schema,
        **task_kwargs,
    )


# Type aliases
_AsyncStr = Callable[[str], Awaitable[Any]]


class RSATask(Task):
    """A :class:`Task` that orchestrates the N-K-T RSA loop.

    Domain-agnostic by construction: every hook method (``format_candidates``,
    ``validate_proposal``, ``build_proposal_task``, ``build_aggregation_task``)
    has a working default. Subclass to customize per-domain behavior.

    Parameters
    ----------
    n, k, t
        RSA hyperparameters. ``n`` is the number of initial proposals,
        ``k`` is the aggregation subset size, ``t`` is the total number of
        stages. All must be ``>= 2`` and ``k <= n``.
    user_prompt
        The specific problem to solve. Required when using the default
        ``build_proposal_task``; subclasses that override the hook may pass
        ``None``.
    system_prompt
        Domain-expert system message. Defaults to a generic problem-solver.
    proposal_prompt
        Instructions appended above ``user_prompt`` when building proposal
        tasks. Defaults to a generic template loaded from the prompts/
        directory.
    aggregation_prompt
        Template for aggregation tasks. Receives ``{original_prompt}``,
        ``{candidates}``, ``{step}``, ``{total_steps}`` placeholders.
    structured_output_schema
        Pydantic model for parsing/validating LLM output. Defaults to
        :class:`GenericRSAOutput`.
    parallel
        If True (default), runs proposals and aggregations concurrently when
        a ``runner_factory`` is supplied to :meth:`run_rsa`. Falls back to
        sequential execution otherwise.
    log_dir
        Directory to write per-stage JSON logs. Auto-generated under
        ``/tmp/rsa_execution_<timestamp>`` when ``None``.
    disable_validation
        If True, skip structured-output schema validation. Also enabled when
        the ``CHARGE_DISABLE_OUTPUT_VALIDATION=1`` environment variable is set.
    **task_kwargs
        Extra keyword arguments forwarded to the underlying :class:`Task`
        (e.g. ``server_urls``, ``server_files``, ``bearer_token``,
        ``builtin_tools``).
    """

    def __init__(
        self,
        n: int,
        k: int,
        t: int,
        *,
        user_prompt: Optional[str] = None,
        system_prompt: Optional[str] = None,
        proposal_prompt: Optional[str] = None,
        aggregation_prompt: Optional[str] = None,
        structured_output_schema: Optional[Type[BaseModel]] = None,
        parallel: bool = True,
        log_dir: Optional[str] = None,
        disable_validation: bool = False,
        **task_kwargs,
    ):
        if n < 2 or k < 2 or t < 2:
            violators = [
                name for name, val in (("N", n), ("K", k), ("T", t)) if val < 2
            ]
            raise ValueError(
                f"RSA parameter(s) {', '.join(violators)} must be >= 2 "
                f"(got N={n}, K={k}, T={t})."
            )
        if k > n:
            raise ValueError(
                f"RSA parameter K ({k}) cannot exceed N ({n}); "
                "K is the subset size and must fit inside the proposal pool."
            )

        sys_p = system_prompt or _DEFAULT_SYSTEM_PROMPT
        prop_p = proposal_prompt or self._load_or_fallback(
            _DEFAULT_PROPOSAL_SYSTEM, _FALLBACK_PROPOSAL_PROMPT
        )
        agg_p = aggregation_prompt or self._load_or_fallback(
            _DEFAULT_AGGREGATION_TEMPLATE, _FALLBACK_AGGREGATION_PROMPT
        )
        schema = structured_output_schema or GenericRSAOutput

        # Initialize the underlying Task: user_prompt is the problem statement
        # so subclasses inherit access to it via self.user_prompt.
        super().__init__(
            system_prompt=sys_p,
            user_prompt=user_prompt,
            structured_output_schema=schema,
            **task_kwargs,
        )

        self.n = n
        self.k = k
        self.t = t
        self.proposal_prompt = prop_p
        self.aggregation_prompt = agg_p
        self.parallel = parallel
        self.disable_validation = disable_validation

        if log_dir is None:
            ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            log_dir = f"/tmp/rsa_execution_{ts}"
        self.log_dir = log_dir
        os.makedirs(self.log_dir, exist_ok=True)

        # Stash so default builders can reconstruct child Tasks with the same
        # tool / server / token configuration.
        self._task_kwargs = task_kwargs

    @staticmethod
    def _load_or_fallback(path: Path, fallback: str) -> str:
        return path.read_text() if path.exists() else fallback

    # -------- Hooks (override in subclasses for domain customization) --------

    def format_candidates(self, subset: list[dict]) -> str:
        """Format a K-subset of proposals as candidates text for aggregation."""
        return default_format_candidates(subset)

    def validate_proposal(self, result: BaseModel) -> bool:
        """Return True if ``result`` is acceptable. Defaults to accept-all."""
        return True

    def build_proposal_task(self) -> Task:
        """Return a fresh :class:`Task` for generating one proposal.

        Default delegates to :func:`create_default_proposal_task` using this
        RSATask's prompts, schema, and ``**task_kwargs``.
        """
        if self.user_prompt is None:
            raise ValueError(
                "user_prompt is required when using the default "
                "build_proposal_task; either pass user_prompt to __init__ or "
                "override build_proposal_task in a subclass."
            )
        return create_default_proposal_task(
            user_prompt=self.user_prompt,
            system_prompt=self.system_prompt,
            proposal_prompt=self.proposal_prompt,
            output_schema=self.structured_output_schema,
            **self._task_kwargs,
        )

    def build_aggregation_task(self, candidates_text: str, step: int) -> Task:
        """Return a fresh :class:`Task` for aggregating one K-subset.

        Default delegates to :func:`create_default_aggregation_task`.
        """
        if self.user_prompt is None:
            raise ValueError(
                "user_prompt is required when using the default "
                "build_aggregation_task; either pass user_prompt to __init__ "
                "or override build_aggregation_task in a subclass."
            )
        return create_default_aggregation_task(
            original_user_prompt=self.user_prompt,
            candidates_text=candidates_text,
            step=step,
            total_steps=self.t,
            aggregation_prompt=self.aggregation_prompt,
            system_prompt=self.system_prompt,
            output_schema=self.structured_output_schema,
            **self._task_kwargs,
        )

    # ---------------------- The N-K-T orchestrator ----------------------

    async def run(self, runner, **kwargs):
        """Polymorphic :meth:`Task.run` entry point for RSA.

        Delegates to :meth:`run_rsa`, which drives the N-K-T loop. Exists so
        callers can invoke RSA through the generic Task API
        (``await rsa_task.run(runner, ...)``) without knowing it is RSA.
        """
        return await self.run_rsa(runner, **kwargs)

    async def run_rsa(
        self,
        runner: Any,
        *,
        runner_factory: Optional[Callable[[], Any]] = None,
        log_progress: Optional[Callable[[str], Any]] = None,
        logger_info: Optional[_AsyncStr] = None,
        logger_warning: Optional[_AsyncStr] = None,
        logger_error: Optional[_AsyncStr] = None,
        callback_handler: Optional[Any] = None,
    ) -> tuple[str, BaseModel]:
        """Execute the N-K-T RSA loop.

        Parameters
        ----------
        runner
            ChARGe agent runner with a ``.task`` attribute and an awaitable
            ``run(log_progress)`` method. Used directly in sequential mode and
            as a fallback when ``runner_factory`` is omitted.
        runner_factory
            Factory that returns a fresh runner per parallel proposal. Required
            for parallel execution; without it, the loop runs sequentially.
        log_progress
            Per-token / per-chunk streaming callback handed to ``runner.run``.
        logger_info, logger_warning, logger_error
            Async loggers for stage-level events. Default to loguru wrappers.
        callback_handler
            Optional handler with an awaitable ``drain()`` method, called after
            each runner invocation to flush UI updates.

        Returns
        -------
        (final_output_json, final_result)
            ``final_output_json`` is the raw JSON string returned by the LLM
            for the chosen final proposal; ``final_result`` is the same data
            already validated against ``structured_output_schema``.

        Raises
        ------
        ValueError
            If no proposal survives Stage 1 or no aggregation survives any
            subsequent stage.
        """
        info = logger_info or self._default_info
        warning = logger_warning or self._default_warning
        # logger_error reserved for future use; keep parameter to preserve API.
        del logger_error

        # Stage 1: N proposals
        await info(
            f"RSA Step 1/{self.t}: Generating {self.n} initial proposals"
            + (" (parallel mode)" if self.parallel else " (sequential mode)")
        )
        proposals = await self._stage_proposals(
            runner, runner_factory, log_progress, info, warning, callback_handler
        )
        if not proposals:
            raise ValueError("All RSA proposals failed")
        await info(
            f"Stage 1 complete. Generated {len(proposals)} valid proposals."
        )

        # Stages 2..T: recursive aggregation
        current = proposals
        step = 1  # defensive default; range below always assigns at least once
        for step in range(2, self.t + 1):
            await info(
                f"RSA Step {step}/{self.t}: Aggregating {len(current)} "
                f"proposals into {self.k}-subsets"
                + (" (parallel mode)" if self.parallel else " (sequential mode)")
            )
            if len(current) < self.k:
                await warning(
                    f"Step {step}: only {len(current)} proposals available; "
                    f"K reduced from {self.k} to {len(current)} for this stage."
                )

            num_aggs = max(self.n, len(current))
            next_props = await self._stage_aggregations(
                num_aggs, step, current, runner, runner_factory,
                log_progress, info, warning, callback_handler,
            )
            if not next_props:
                await warning(
                    f"No successful aggregations in step {step}; "
                    "falling back to previous stage's proposals."
                )
                break
            current = next_props
            await info(
                f"Stage {step} complete. Generated {len(current)} valid aggregations."
            )

        if not current:
            raise ValueError("RSA failed to produce any valid proposals")

        final = current[0]
        final_output = final["output"]
        final_result = final["result"]

        log_path = Path(self.log_dir) / "FINAL_OUTPUT.json"
        log_path.write_text(
            json.dumps(
                {
                    "final_step": step if step <= self.t else self.t,
                    "n_proposals": self.n,
                    "k_subset_size": self.k,
                    "t_stages": self.t,
                    "final_result": final_result.model_dump(),
                },
                indent=2,
            )
        )
        await info(f"RSA completed! Final output saved to {self.log_dir}")

        return final_output, final_result

    # ----------------------- private orchestration -------------------------

    def _validation_disabled(self) -> bool:
        return (
            self.disable_validation
            or os.getenv("CHARGE_DISABLE_OUTPUT_VALIDATION", "0") == "1"
        )

    def _write_json(self, name: str, payload: dict) -> None:
        with open(f"{self.log_dir}/{name}", "w") as f:
            json.dump(payload, f, indent=2)

    async def _run_single_proposal(
        self,
        index: int,
        runner: Any,
        log_progress: Optional[Callable],
        info: _AsyncStr,
        warning: _AsyncStr,
        callback_handler: Optional[Any],
    ) -> Optional[dict]:
        try:
            await info(f"Generating proposal {index + 1}/{self.n}")
            task = self.build_proposal_task()
            runner.task = task
            if self._validation_disabled():
                task.structured_output_schema = None

            self._write_json(
                f"proposer_{index + 1:02d}_prompt.json",
                {
                    "proposal_index": index + 1,
                    "system_prompt": task.get_system_prompt(),
                    "user_prompt": task.get_user_prompt(),
                },
            )

            output = await runner.run(log_progress)
            if callback_handler:
                await callback_handler.drain()

            schema = self.structured_output_schema or GenericRSAOutput
            result = schema.model_validate_json(output)

            if not self.validate_proposal(result):
                await warning(
                    f"Proposal {index + 1} failed validation (empty or invalid), skipping"
                )
                return None

            self._write_json(
                f"proposer_{index + 1:02d}_output.json",
                {
                    "proposal_index": index + 1,
                    "result": result.model_dump(),
                    "full_output": json.loads(output),
                },
            )
            await info(f"Proposal {index + 1} completed successfully")
            return {"output": output, "result": result, "index": index}

        except asyncio.CancelledError:
            await warning(f"Proposal {index + 1} was cancelled")
            return None
        except Exception as e:
            await warning(f"Proposal {index + 1} failed: {str(e)}")
            return None

    async def _run_single_aggregation(
        self,
        agg_index: int,
        step: int,
        current_proposals: list[dict],
        runner: Any,
        log_progress: Optional[Callable],
        info: _AsyncStr,
        warning: _AsyncStr,
        callback_handler: Optional[Any],
    ) -> Optional[dict]:
        try:
            current_k = min(self.k, len(current_proposals))
            if len(current_proposals) <= current_k:
                subset = current_proposals
            else:
                subset = random.sample(current_proposals, current_k)

            candidates_text = self.format_candidates(subset)
            subset_indices = [p["index"] + 1 for p in subset]

            agg_task = self.build_aggregation_task(candidates_text, step)
            runner.task = agg_task
            if self._validation_disabled():
                agg_task.structured_output_schema = None

            self._write_json(
                f"aggregator_step{step}_{agg_index + 1:02d}_prompt.json",
                {
                    "step": step,
                    "aggregation_index": agg_index + 1,
                    "k_subset_indices": subset_indices,
                    "system_prompt": agg_task.get_system_prompt(),
                    "user_prompt": agg_task.get_user_prompt(),
                    "candidates_text": candidates_text,
                },
            )

            output = await runner.run(log_progress)
            if callback_handler:
                await callback_handler.drain()

            schema = self.structured_output_schema or GenericRSAOutput
            result = schema.model_validate_json(output)

            if not self.validate_proposal(result):
                await warning(
                    f"Aggregation {agg_index + 1} (Step {step}) failed validation, skipping"
                )
                return None

            self._write_json(
                f"aggregator_step{step}_{agg_index + 1:02d}_output.json",
                {
                    "step": step,
                    "aggregation_index": agg_index + 1,
                    "k_subset_indices": subset_indices,
                    "result": result.model_dump(),
                    "full_output": json.loads(output),
                },
            )
            await info(
                f"Aggregation {agg_index + 1} (Step {step}) completed successfully"
            )
            return {
                "output": output,
                "result": result,
                "index": agg_index,
                "step": step,
            }

        except asyncio.CancelledError:
            await warning(f"Aggregation {agg_index + 1} (Step {step}) was cancelled")
            return None
        except Exception as e:
            await warning(
                f"Aggregation {agg_index + 1} (Step {step}) failed: {str(e)}"
            )
            return None

    async def _stage_proposals(
        self,
        runner: Any,
        runner_factory: Optional[Callable[[], Any]],
        log_progress: Optional[Callable],
        info: _AsyncStr,
        warning: _AsyncStr,
        callback_handler: Optional[Any],
    ) -> list[dict]:
        if self.parallel and runner_factory:
            coros = [
                self._run_single_proposal(
                    i, runner_factory(), log_progress, info, warning, callback_handler
                )
                for i in range(self.n)
            ]
            results = await asyncio.gather(*coros, return_exceptions=True)
            out: list[dict] = []
            for i, r in enumerate(results):
                if isinstance(r, Exception):
                    await warning(f"Proposal {i + 1} failed with exception: {str(r)}")
                elif r is not None:
                    out.append(r)
            return out

        if self.parallel and not runner_factory:
            await warning(
                "Parallel mode requested but no runner_factory provided; "
                "falling back to sequential."
            )
        out: list[dict] = []
        for i in range(self.n):
            r = await self._run_single_proposal(
                i, runner, log_progress, info, warning, callback_handler
            )
            if r is not None:
                out.append(r)
        return out

    async def _stage_aggregations(
        self,
        num_aggs: int,
        step: int,
        current: list[dict],
        runner: Any,
        runner_factory: Optional[Callable[[], Any]],
        log_progress: Optional[Callable],
        info: _AsyncStr,
        warning: _AsyncStr,
        callback_handler: Optional[Any],
    ) -> list[dict]:
        if self.parallel and runner_factory:
            coros = [
                self._run_single_aggregation(
                    i, step, current, runner_factory(),
                    log_progress, info, warning, callback_handler,
                )
                for i in range(num_aggs)
            ]
            results = await asyncio.gather(*coros, return_exceptions=True)
            out: list[dict] = []
            for i, r in enumerate(results):
                if isinstance(r, Exception):
                    await warning(
                        f"Aggregation {i + 1} (Step {step}) failed with exception: {str(r)}"
                    )
                elif r is not None:
                    out.append(r)
            return out

        if self.parallel and not runner_factory:
            await warning(
                "Parallel mode requested but no runner_factory provided; "
                "falling back to sequential."
            )
        out: list[dict] = []
        for i in range(num_aggs):
            r = await self._run_single_aggregation(
                i, step, current, runner,
                log_progress, info, warning, callback_handler,
            )
            if r is not None:
                out.append(r)
        return out

    # ------------------- default loggers (stdout/stderr) --------------------

    async def _default_info(self, message: str) -> None:
        print(f"[RSA Info] {message}")

    async def _default_warning(self, message: str) -> None:
        print(f"[RSA Warning] {message}", file=sys.stderr)

    async def _default_error(self, message: str) -> None:
        print(f"[RSA Error] {message}", file=sys.stderr)
