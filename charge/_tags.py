def verifier(func):
    """Decorator to mark a method as a verifier."""
    func.__verifier_tag = True
    return func


def is_verifier(func):
    """Check if a method is marked as a verifier."""
    return hasattr(func, "__verifier_tag")


def hypothesis(func):
    """Decorator to mark a method as a hypothesis."""
    func.__hypothesis_tag = True
    return func


def is_hypothesis(func):
    """Check if a method is marked as a hypothesis."""
    return hasattr(func, "__hypothesis_tag")
