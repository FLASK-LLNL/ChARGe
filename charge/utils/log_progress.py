################################################################################
## Copyright 2025 Lawrence Livermore National Security, LLC. and Binghamton University.
## See the top-level LICENSE file for details.
##
## SPDX-License-Identifier: Apache-2.0
################################################################################


from loguru import logger

LOG_PROGRESS_SYSTEM_PROMPT = "At each step of your reasoning use the log_progress tool to report your current prograss, current thinking, and plan."


def log_progress(log_msg: str) -> None:
    """
    Log the progress of the model to the ChARGe infrastructure.

    Args:
        log_msg (str): The model's current progress.
    Returns:
        None: returns None empty object.
    """

    logger.info(f"[ChARGe Orchestrator Inner Monologue] {log_msg}")
