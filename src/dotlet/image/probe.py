"""Image test 2: caption probing of a text-to-image model.

Each image (yours and controls) has a caption: a sidecar `<image>.txt` file or
an entry in a captions CSV/JSONL. The model generates from the caption alone
(several samples each). Score = the closest sample's similarity to the
ORIGINAL image. A model that memorized your images reproduces them from their
captions; for unseen controls it can only produce something generic.

Captions must be equally specific for both groups (same style, length,
level of detail). Otherwise you're testing the captions, not the model.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

from ..stats import TestResult, compare
from .features import featurize_paths


def load_captions(images: list[Path], captions_file: str | Path | None = None) -> dict[str, str]:
    """Map image path -> caption. Captions file rows need `image` and `prompt` fields;
    `image` may be a full path or just a file name."""
    table: dict[str, str] = {}
    if captions_file:
        cf = Path(captions_file)
        if cf.suffix.lower() == ".csv":
            with open(cf, newline="", encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
        else:
            rows = [json.loads(l) for l in cf.read_text(encoding="utf-8").splitlines() if l.strip()]
        for r in rows:
            table[str(r["image"])] = r["prompt"]
            table.setdefault(Path(r["image"]).name, r["prompt"])
    out, missing = {}, []
    for p in images:
        side = p.with_suffix(".txt")
        cap = table.get(str(p)) or table.get(p.name) or (side.read_text(encoding="utf-8").strip() if side.exists() else None)
        if cap:
            out[str(p)] = cap
        else:
            missing.append(str(p))
    if missing:
        raise SystemExit(
            f"{len(missing)} image(s) have no caption (e.g. {missing[0]}). Add a sidecar .txt "
            "next to each image or pass --captions file.csv with columns image,prompt."
        )
    return out


def make_probes(candidates: list[Path], controls: list[Path], captions: dict[str, str]) -> list[dict]:
    return [
        {"id": str(p), "group": g, "image": str(p), "prompt": captions[str(p)]}
        for g, paths in (("candidate", candidates), ("control", controls))
        for p in paths
    ]


def analyze(probes: list[dict], responses: list[dict], feat, **compare_kwargs) -> tuple[TestResult, list[dict]]:
    by_id = {r["id"]: r for r in responses}
    rows, missing = [], 0
    for p in probes:
        files = (by_id.get(p["id"]) or {}).get("files") or []
        if not files:
            missing += 1
            continue
        F_orig = featurize_paths(feat, [Path(p["image"])])
        F_gen = featurize_paths(feat, [Path(f) for f in files], with_mirror=True)
        sims = feat.similarity(F_orig, F_gen)[0]
        j = int(sims.argmax())
        rows.append({**p, "best_file": ([*files, *files])[j], "score": float(sims[j]), "n_samples": len(files)})
    pos = [r["score"] for r in rows if r["group"] == "candidate"]
    neg = [r["score"] for r in rows if r["group"] == "control"]
    result = compare(pos, neg, metric=f"max {feat.name} similarity of generations to the original", **compare_kwargs)
    if missing:
        result.notes.append(f"{missing} probe(s) had no generated images and were skipped.")
    return result, rows
