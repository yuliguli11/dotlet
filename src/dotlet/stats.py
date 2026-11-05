"""Two-sample statistics comparing your material ("candidates") against controls.

Convention everywhere: a HIGHER score means "looks more like it was trained on /
memorized". The one-sided alternative is: candidates score higher than controls.
"""

from __future__ import annotations

import math
import random
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from typing import Sequence


def auc(pos: Sequence[float], neg: Sequence[float]) -> float:
    """P(random candidate score > random control score), ties count 1/2.

    Computed via ranks in O((n+m) log(n+m)).
    """
    n, m = len(pos), len(neg)
    if n == 0 or m == 0:
        return float("nan")
    ranks = _ranks(list(pos) + list(neg))
    r_pos = sum(ranks[:n])
    u = r_pos - n * (n + 1) / 2
    return u / (n * m)


def _ranks(values: Sequence[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        avg = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks


def mann_whitney_greater(pos: Sequence[float], neg: Sequence[float]) -> float:
    """One-sided Mann-Whitney U p-value (normal approximation, tie-corrected)."""
    n, m = len(pos), len(neg)
    if n == 0 or m == 0:
        return float("nan")
    combined = list(pos) + list(neg)
    ranks = _ranks(combined)
    u = sum(ranks[:n]) - n * (n + 1) / 2
    N = n + m
    counts: dict[float, int] = defaultdict(int)
    for v in combined:
        counts[v] += 1
    tie_term = sum(t**3 - t for t in counts.values())
    var = n * m / 12 * ((N + 1) - tie_term / (N * (N - 1))) if N > 1 else 0.0
    if var <= 0:
        return 1.0
    z = (u - n * m / 2 - 0.5) / math.sqrt(var)  # continuity correction
    return 0.5 * math.erfc(z / math.sqrt(2))


def permutation_test(
    pos: Sequence[float], neg: Sequence[float], n_resamples: int = 10000, seed: int = 0
) -> float:
    """One-sided permutation p-value for mean(pos) - mean(neg) > 0.

    Uses the (count + 1) / (n_resamples + 1) estimator, so p is never exactly 0.
    """
    n, m = len(pos), len(neg)
    if n == 0 or m == 0:
        return float("nan")
    rng = random.Random(seed)
    pooled = list(pos) + list(neg)
    total = sum(pooled)
    observed = sum(pos) / n - sum(neg) / m
    hits = 0
    for _ in range(n_resamples):
        rng.shuffle(pooled)
        s = sum(pooled[:n])
        diff = s / n - (total - s) / m
        if diff >= observed - 1e-12:
            hits += 1
    return (hits + 1) / (n_resamples + 1)


def bootstrap_auc_ci(
    pos: Sequence[float], neg: Sequence[float], n_resamples: int = 2000, alpha: float = 0.05, seed: int = 0
) -> tuple[float, float]:
    if not pos or not neg:
        return (float("nan"), float("nan"))
    rng = random.Random(seed)
    vals = []
    for _ in range(n_resamples):
        p = [pos[rng.randrange(len(pos))] for _ in pos]
        q = [neg[rng.randrange(len(neg))] for _ in neg]
        vals.append(auc(p, q))
    vals.sort()
    lo = vals[int(math.floor(alpha / 2 * (n_resamples - 1)))]
    hi = vals[int(math.ceil((1 - alpha / 2) * (n_resamples - 1)))]
    return (lo, hi)


def aggregate_by_group(ids: Sequence[str], scores: Sequence[float]) -> tuple[list[str], list[float]]:
    """Average chunk scores per source document (ids look like 'path#index')."""
    groups: dict[str, list[float]] = defaultdict(list)
    for i, s in zip(ids, scores):
        groups[i.rsplit("#", 1)[0]].append(s)
    names = sorted(groups)
    return names, [sum(groups[k]) / len(groups[k]) for k in names]


@dataclass
class TestResult:
    metric: str
    n_candidates: int
    n_controls: int
    mean_candidates: float
    mean_controls: float
    auc: float
    auc_ci: tuple[float, float]
    p_permutation: float
    p_mann_whitney: float
    alpha: float
    verdict: str
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def compare(
    pos: Sequence[float],
    neg: Sequence[float],
    metric: str,
    alpha: float = 0.05,
    n_permutations: int = 10000,
    n_bootstrap: int = 2000,
    seed: int = 0,
) -> TestResult:
    pos, neg = list(pos), list(neg)
    notes = []
    if len(pos) < 10 or len(neg) < 10:
        notes.append(
            f"Small sample ({len(pos)} candidate vs {len(neg)} control units). "
            "Results are low-power; add more material/controls for a reliable test."
        )
    a = auc(pos, neg)
    ci = bootstrap_auc_ci(pos, neg, n_bootstrap, alpha, seed)
    p_perm = permutation_test(pos, neg, n_permutations, seed)
    p_mw = mann_whitney_greater(pos, neg)
    p = max(p_perm, p_mw)  # be conservative: require both tests to agree
    if p < alpha / 5 and ci[0] > 0.5:
        verdict = "STRONG statistical evidence your material is treated differently from controls (consistent with training exposure)"
    elif p < alpha:
        verdict = "SOME statistical evidence your material is treated differently from controls"
    else:
        verdict = "NO significant evidence at this sample size (this is not proof of absence)"
    return TestResult(
        metric=metric,
        n_candidates=len(pos),
        n_controls=len(neg),
        mean_candidates=sum(pos) / len(pos) if pos else float("nan"),
        mean_controls=sum(neg) / len(neg) if neg else float("nan"),
        auc=a,
        auc_ci=ci,
        p_permutation=p_perm,
        p_mann_whitney=p_mw,
        alpha=alpha,
        verdict=verdict,
        notes=notes,
    )
