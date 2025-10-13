LOG_PROGRESS_SYSTEM_PROMPT="At each step of your reasoning use the log_progress tool to report your current prograss, current thinking, and plan."


def log_progress(msg: str) -> None:
    """
    Log the progress of the model to the ChARGe infrastructure.

    Args:
        msg (str): The model's current progress.
    Returns:
        None: returns None empty object.
    """

    print(f"[ChARGe Orchestrator Inner Monologue] {msg}")
    
