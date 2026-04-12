"""
ChARGe Algorithms Module

This module provides generic algorithms that work with ChARGe tasks.
Currently includes:
- RSA (Recursive Self-Aggregation): N-K-T algorithm for proposal generation and aggregation
"""

from charge.algorithms.rsa import (
    run_rsa_loop,
    RSAConfig,
    RSAPrompts,
    RSACallbacks,
    RSATaskFactories,
    GenericRSAOutput,
    default_format_candidates,
    create_default_proposal_task,
    create_default_aggregation_task,
)

__all__ = [
    "run_rsa_loop",
    "RSAConfig",
    "RSAPrompts",
    "RSACallbacks",
    "RSATaskFactories",
    "GenericRSAOutput",
    "default_format_candidates",
    "create_default_proposal_task",
    "create_default_aggregation_task",
]
