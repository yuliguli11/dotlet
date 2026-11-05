"""Test 1: verbatim n-gram overlap between AI outputs and your material.

For every equal-sized chunk of your material and of the controls, we compute the
fraction of its word n-grams that also appear somewhere in the AI outputs.
N-grams found in an optional background corpus (common phrases, boilerplate,
quotes everyone uses) are ignored. If the AI reproduces your text, your chunks
should score higher than control chunks.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .stats import TestResult, aggregate_by_group, compare
from .text import Chunk, Doc, ngrams, words


@dataclass
class Span:
    chunk_id: str
    group: str  # "candidate" or "control"
    length_words: int
    text: str


@dataclass
class OverlapReport:
    result: TestResult
    candidate_scores: dict[str, float]
    control_scores: dict[str, float]
    top_spans: list[Span] = field(default_factory=list)
    longest_candidate_run: int = 0
    longest_control_run: int = 0
    long_runs: list[Span] = field(default_factory=list)  # candidate runs >= long_run words
    long_run: int = 20


def _index(docs: list[Doc], n: int) -> set[tuple[str, ...]]:
    idx: set[tuple[str, ...]] = set()
    for d in docs:
        idx |= ngrams(words(d.text), n)
    return idx


def score_chunk(
    chunk: Chunk, output_idx: set, background_idx: set, n: int
) -> tuple[float, list[tuple[int, int]]]:
    """Return (score, matched word spans [start, end))."""
    w = chunk.words
    positions = []
    total = 0
    for i in range(len(w) - n + 1):
        g = tuple(w[i : i + n])
        if g in background_idx:
            continue
        total += 1
        if g in output_idx:
            positions.append(i)
    spans: list[tuple[int, int]] = []
    for i in positions:
        if spans and i <= spans[-1][1] - n + 1:
            spans[-1] = (spans[-1][0], i + n)
        else:
            spans.append((i, i + n))
    return (len(positions) / total if total else 0.0), spans


def analyze(
    candidates: list[Chunk],
    controls: list[Chunk],
    outputs: list[Doc],
    background: list[Doc] | None = None,
    n: int = 8,
    unit: str = "chunk",
    top_k: int = 10,
    long_run: int = 20,
    **compare_kwargs,
) -> OverlapReport:
    out_idx = _index(outputs, n)
    bg_idx = _index(background or [], n)

    def run(chunks: list[Chunk], group: str):
        scores, spans = {}, []
        for c in chunks:
            s, sp = score_chunk(c, out_idx, bg_idx, n)
            scores[c.id] = s
            w = c.words
            spans.extend(Span(c.id, group, b - a, " ".join(w[a:b])) for a, b in sp)
        return scores, spans

    cand_scores, cand_spans = run(candidates, "candidate")
    ctrl_scores, ctrl_spans = run(controls, "control")

    pos, neg = list(cand_scores.values()), list(ctrl_scores.values())
    if unit == "document":
        _, pos = aggregate_by_group(list(cand_scores), pos)
        _, neg = aggregate_by_group(list(ctrl_scores), neg)

    result = compare(pos, neg, metric=f"fraction of {n}-grams reproduced in AI outputs", **compare_kwargs)
    if not bg_idx:
        result.notes.append(
            "No --background corpus given: common stock phrases may inflate both groups. "
            "Usually fine (controls absorb it), but a background corpus sharpens the test."
        )
    spans = sorted(cand_spans + ctrl_spans, key=lambda s: -s.length_words)[:top_k]
    longest = lambda sp: max((s.length_words for s in sp), default=0)
    return OverlapReport(
        result, cand_scores, ctrl_scores, spans,
        longest_candidate_run=longest(cand_spans),
        longest_control_run=longest(ctrl_spans),
        long_runs=[s for s in cand_spans if s.length_words >= long_run],
        long_run=long_run,
    )
