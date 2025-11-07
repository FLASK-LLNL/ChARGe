import atexit
import readline
import inspect
import asyncio


def enable_cmd_history_and_shell_integration(history: str):
    """Enable persistent command-line history and integrate with the interactive shell.
    Attempts to load an existing readline history file, sets the in-memory history
    length to 1000, and registers an atexit handler to persist history on exit.

    Args:
        history (str): Path to the history file to read from and write to.

    Returns:
        None

    - If the history file exists, it is loaded into readline.
    - If the history file does not exist, the function continues silently.
    - Other exceptions raised by readline.read_history_file may be propagated.
    """
    try:
        readline.read_history_file(history)
        readline.set_history_length(1000)
    except FileNotFoundError:
        pass

    atexit.register(readline.write_history_file, history)


async def maybe_await_async(var, *args, **kwargs):
    """Utility function to handle both synchronous and asynchronous callables or values.
    Args:
        var: A value, callable, or awaitable.
        *args: Positional arguments to pass if var is callable.
        **kwargs: Keyword arguments to pass if var is callable.
    Returns:
        The result of the callable or awaitable, or the value itself.
    """

    if inspect.isawaitable(var):
        # If var is an awaitable, like a coroutine
        result = var
    elif callable(var):
        # If var is a callable function or coroutine function
        result = var(*args, **kwargs)
    else:
        # If var is a regular value
        result = var

    if inspect.isawaitable(result):
        result = await result
    return result
