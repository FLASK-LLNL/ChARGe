"""ChARGe algorithms.

Currently provides:

- RSA (Recursive Self-Aggregation): an N-K-T proposal-and-aggregation loop,
  exposed as a :class:`Task` subclass for direct use with any ChARGe runner.
"""

from charge.algorithms.rsa import (
    RSATask,
    GenericRSAOutput,
    default_format_candidates,
    create_default_proposal_task,
    create_default_aggregation_task,
)

__all__ = [
    "RSATask",
    "GenericRSAOutput",
    "default_format_candidates",
    "create_default_proposal_task",
    "create_default_aggregation_task",
]
