"""
ε-differential privacy for **aggregated** cooperative stats pushed upstream (not raw rows).
Documented budget: default ε per learner per week for one scalar export.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import numpy as np

# Documented in README: one export = this ε (tune for product policy)
DEFAULT_EPSILON_PER_LEARNER_WEEK: float = float(
    os.environ.get("TUTOR_DP_EPSILON", "0.8")
)
"""Privacy budget (ε) for a single weekly cooperative aggregate push. Lower = more privacy."""


@dataclass
class DpExport:
    """Noisy aggregate + metadata for audit."""

    raw_value: float
    noisy_value: float
    epsilon: float
    mechanism: str


def laplace_noise(scale: float, rng: np.random.Generator | None = None) -> float:
    r = rng or np.random.default_rng()
    return float(r.laplace(0.0, scale))


def dp_count(
    true_count: int, epsilon: float, sensitivity: int = 1, rng: np.random.Generator | None = None
) -> DpExport:
    """Laplace mechanism: sensitivity 1 for a count (add-one DP)."""
    if epsilon <= 0:
        raise ValueError("epsilon must be > 0")
    scale = sensitivity / epsilon
    n = true_count + laplace_noise(scale, rng)
    return DpExport(
        raw_value=float(true_count),
        noisy_value=max(0.0, n),
        epsilon=epsilon,
        mechanism="laplace",
    )


def dp_rate(
    sum_correct: int, n: int, epsilon: float, rng: np.random.Generator | None = None
) -> DpExport:
    """Laplace noise on numerator and denominator counts (sensitivity 1 each, split budget)."""
    if n <= 0:
        return DpExport(0.0, 0.0, epsilon, "laplace")
    e1 = epsilon / 2.0
    c = dp_count(sum_correct, e1, 1, rng)
    t = dp_count(n, e1, 1, rng)
    rate = c.noisy_value / max(1.0, t.noisy_value)
    return DpExport(
        raw_value=sum_correct / n,
        noisy_value=min(1.0, max(0.0, rate)),
        epsilon=epsilon,
        mechanism="laplace_rate",
    )


def build_coop_payload(
    num_learners: int,
    total_correct: int,
    total_items: int,
    epsilon: float | None = None,
) -> dict[str, Any]:
    """JSON-safe dict for a fake 'upstream' sync body (defense + audit)."""
    e = epsilon if epsilon is not None else DEFAULT_EPSILON_PER_LEARNER_WEEK
    c = dp_count(num_learners, e, 1)
    r = dp_rate(total_correct, total_items, e)
    return {
        "coop_learner_count_noisy": round(c.noisy_value, 2),
        "coop_accuracy_noisy": round(r.noisy_value, 4),
        "epsilon_used": e,
        "dp_mechanism": "laplace",
        "raw_learner_count": num_learners,
        "raw_total_correct": total_correct,
        "raw_total_items": total_items,
    }
