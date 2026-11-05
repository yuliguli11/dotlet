"""Image test 1: are AI-generated images near-copies of your images?

Each of your images and each control image gets a score: its highest
similarity to any AI output (mirrored outputs included). If the generator
regurgitates your work, your images should score higher than controls. The
best-matching pairs are listed so you can inspect them side by side.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from ..stats import TestResult, compare
from .features import featurize_paths


@dataclass
class Match:
    image: str
    group: str  # "candidate" or "control"
    best_output: str
    similarity: float


@dataclass
class ImageOverlapReport:
    result: TestResult
    matches: list[Match]  # every image, sorted by similarity (desc)
    threshold: float
    near_duplicates: list[Match] = field(default_factory=list)  # yours, >= threshold
    max_control_similarity: float = 0.0


def analyze(
    candidates: list[Path],
    controls: list[Path],
    outputs: list[Path],
    feat,
    threshold: float | None = None,
    **compare_kwargs,
) -> ImageOverlapReport:
    threshold = feat.default_threshold if threshold is None else threshold
    O = featurize_paths(feat, outputs, with_mirror=True)
    out_names = [str(p) for p in outputs] * 2

    def run(paths, group):
        if not paths:
            return []
        S = feat.similarity(featurize_paths(feat, paths), O)
        best = S.argmax(1)
        return [Match(str(p), group, out_names[j], float(S[i, j])) for i, (p, j) in enumerate(zip(paths, best))]

    cand, ctrl = run(candidates, "candidate"), run(controls, "control")
    result = compare([m.similarity for m in cand], [m.similarity for m in ctrl],
                     metric=f"max {feat.name} similarity to any AI output", **compare_kwargs)
    return ImageOverlapReport(
        result=result,
        matches=sorted(cand + ctrl, key=lambda m: -m.similarity),
        threshold=threshold,
        near_duplicates=[m for m in cand if m.similarity >= threshold],
        max_control_similarity=max((m.similarity for m in ctrl), default=0.0),
    )
