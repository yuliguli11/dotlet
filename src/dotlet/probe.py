"""Test 2: prefix-completion probing (black-box, works with any chat model).

Each chunk is cut in two. The model sees the first part and is asked to
continue it. A model that memorized your text tends to reproduce the real
continuation; for control text it can only guess. We score similarity between
the model's continuation and the true one and compare the two groups.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from .stats import TestResult, aggregate_by_group, compare
from .text import Chunk, ngrams, words

PROMPT_TEMPLATE = (
    "Continue the following passage exactly as it appears in its original source. "
    "Output only the continuation, with no commentary.\n\n{prefix}"
)

METRICS = ("ngram", "lcs", "prefix")


def make_probes(candidates: list[Chunk], controls: list[Chunk], prefix_frac: float = 0.5) -> list[dict]:
    probes = []
    for group, chunks in (("candidate", candidates), ("control", controls)):
        for c in chunks:
            k = max(1, int(len(c.tokens) * prefix_frac))
            prefix = " ".join(c.tokens[:k])
            probes.append(
                {
                    "id": c.id,
                    "group": group,
                    "prefix": prefix,
                    "reference": " ".join(c.tokens[k:]),
                    "prompt": PROMPT_TEMPLATE.format(prefix=prefix),
                }
            )
    return probes


def write_jsonl(rows: Iterable[dict], path: str | Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def read_jsonl(path: str | Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _lcs_len(a: list[str], b: list[str]) -> int:
    if not a or not b:
        return 0
    prev = [0] * (len(b) + 1)
    for x in a:
        cur = [0] * (len(b) + 1)
        for j, y in enumerate(b, 1):
            cur[j] = prev[j - 1] + 1 if x == y else max(prev[j], cur[j - 1])
        prev = cur
    return prev[-1]


def score_continuation(generated: str, reference: str, metric: str = "ngram", n: int = 4) -> float:
    """Similarity in [0, 1] between a model continuation and the true continuation.

    The generation is truncated to the reference length so verbosity is not rewarded.
      ngram : fraction of the reference's n-grams present in the generation
      lcs   : longest common (word) subsequence / reference length
      prefix: fraction of reference words reproduced exactly from the start
    """
    ref = words(reference)
    gen = words(generated)[: len(ref)]
    if not ref:
        return 0.0
    if metric == "ngram":
        rg = ngrams(ref, n)
        return len(rg & ngrams(gen, n)) / len(rg) if rg else 0.0
    if metric == "lcs":
        return _lcs_len(gen, ref) / len(ref)
    if metric == "prefix":
        k = 0
        for a, b in zip(gen, ref):
            if a != b:
                break
            k += 1
        return k / len(ref)
    raise ValueError(f"unknown metric {metric!r}; choose from {METRICS}")


def analyze(
    probes: list[dict],
    responses: list[dict],
    metric: str = "ngram",
    unit: str = "chunk",
    **compare_kwargs,
) -> tuple[TestResult, list[dict]]:
    by_id = {r["id"]: r for r in responses}
    rows, missing = [], 0
    for p in probes:
        r = by_id.get(p["id"])
        if r is None or r.get("response") is None:
            missing += 1
            continue
        rows.append({**p, "response": r["response"], "score": score_continuation(r["response"], p["reference"], metric)})

    def collect(group):
        sel = [r for r in rows if r["group"] == group]
        ids, sc = [r["id"] for r in sel], [r["score"] for r in sel]
        return aggregate_by_group(ids, sc)[1] if unit == "document" else sc

    result = compare(collect("candidate"), collect("control"), metric=f"continuation similarity ({metric})", **compare_kwargs)
    if missing:
        result.notes.append(f"{missing} probe(s) had no response and were skipped.")
    return result, rows
