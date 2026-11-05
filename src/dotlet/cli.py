"""Command-line interface: `dotlet overlap | probe make/run/score | mia`."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict

from . import overlap, probe
from .stats import TestResult
from .text import chunk_docs, load_docs


def format_result(r: TestResult) -> str:
    lines = [
        f"Metric:            {r.metric}",
        f"Units compared:    {r.n_candidates} of yours vs {r.n_controls} controls",
        f"Mean score:        yours {r.mean_candidates:.4f}   controls {r.mean_controls:.4f}",
        f"AUC:               {r.auc:.3f}  ({int((1 - r.alpha) * 100)}% CI {r.auc_ci[0]:.3f} to {r.auc_ci[1]:.3f})"
        "   [0.5 = indistinguishable, 1.0 = perfectly separated]",
        f"p (permutation):   {r.p_permutation:.4g}",
        f"p (Mann-Whitney):  {r.p_mann_whitney:.4g}",
        f"Verdict:           {r.verdict}",
    ]
    lines += [f"Note: {n}" for n in r.notes]
    return "\n".join(lines)


def _common(p: argparse.ArgumentParser, need_materials: bool = True) -> None:
    if need_materials:
        p.add_argument("--materials", nargs="+", required=True, help="Your work: files or folders")
        p.add_argument("--controls", nargs="+", required=True,
                       help="Comparable text the model could NOT have seen (e.g. written after its training cutoff)")
        p.add_argument("--chunk-size", type=int, default=200, help="Words per chunk (default 200)")
        p.add_argument("--max-chunks-per-doc", type=int, default=None,
                       help="Cap chunks per document so one long file can't dominate")
    p.add_argument("--unit", choices=["chunk", "document"], default="chunk",
                   help="Statistical unit. 'document' averages chunks per file: fewer units but more independent")
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--permutations", type=int, default=10000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--json", metavar="PATH", help="Also write full results as JSON")


def _cmp_kwargs(a) -> dict:
    return {"alpha": a.alpha, "n_permutations": a.permutations, "seed": a.seed}


def _chunks(a):
    mats = chunk_docs(load_docs(a.materials), a.chunk_size, a.max_chunks_per_doc)
    ctrls = chunk_docs(load_docs(a.controls), a.chunk_size, a.max_chunks_per_doc)
    if not mats or not ctrls:
        sys.exit("Need at least one chunk of materials and of controls (are the files long enough for --chunk-size?)")
    return mats, ctrls


def _dump(path, obj):
    if path:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(obj, f, indent=2, ensure_ascii=False)
        print(f"\nWrote {path}")


def cmd_overlap(a):
    mats, ctrls = _chunks(a)
    outputs = load_docs(a.outputs)
    bg = load_docs(a.background) if a.background else None
    rep = overlap.analyze(mats, ctrls, outputs, bg, n=a.n, unit=a.unit, long_run=a.long_run, **_cmp_kwargs(a))
    print("== Overlap test: do AI outputs reproduce your text more than control text? ==\n")
    print("-- Frequency test (is reuse systematic across your material?) --")
    print(format_result(rep.result))
    print("\n-- Verbatim evidence (are there individual matches too long to be chance?) --")
    print(f"Longest verbatim run from your text:   {rep.longest_candidate_run} words")
    print(f"Longest verbatim run from controls:    {rep.longest_control_run} words  (the chance baseline)")
    if rep.long_runs and rep.longest_candidate_run > 2 * max(rep.longest_control_run, a.n):
        print(f"FLAG: {len(rep.long_runs)} run(s) of >= {rep.long_run} words reproduce your text, far beyond the control\n"
              "      baseline. Individually, matches this long are rarely coincidence. Before treating them as\n"
              "      evidence, rule out the obvious: the passage was in the prompt, is widely quoted/public\n"
              "      domain, or the output came from a tool with web search/retrieval.")
    if rep.top_spans:
        print("\nLongest verbatim runs found in AI outputs:")
        for s in rep.top_spans:
            tag = "YOURS  " if s.group == "candidate" else "control"
            print(f"  [{tag}] {s.length_words:3d} words  {s.chunk_id}\n           \"{s.text[:200]}{'...' if len(s.text) > 200 else ''}\"")
    _dump(a.json, {"result": rep.result.to_dict(), "longest_candidate_run": rep.longest_candidate_run,
                   "longest_control_run": rep.longest_control_run, "candidate_scores": rep.candidate_scores,
                   "control_scores": rep.control_scores, "top_spans": [asdict(s) for s in rep.top_spans]})


def cmd_probe_make(a):
    mats, ctrls = _chunks(a)
    probes = probe.make_probes(mats, ctrls, a.prefix_frac)
    probe.write_jsonl(probes, a.out)
    print(f"Wrote {len(probes)} probes ({len(mats)} yours, {len(ctrls)} controls) to {a.out}")
    print("Next: `dotlet probe run` to query a model, or paste each 'prompt' into a chat UI and save\n"
          'responses as JSONL lines: {"id": "...", "response": "..."}')


def cmd_probe_run(a):
    from .backends import BackendError, get_backend

    temp = None if a.temperature.lower() == "none" else float(a.temperature)
    be = get_backend(a.backend, model=a.model, base_url=a.base_url, max_tokens=a.max_tokens, temperature=temp)
    probes = probe.read_jsonl(a.probes)
    done = {}
    try:
        done = {r["id"]: r for r in probe.read_jsonl(a.out)}  # resume support
    except FileNotFoundError:
        pass
    with open(a.out, "a", encoding="utf-8") as f:
        for i, p in enumerate(probes, 1):
            if p["id"] in done:
                continue
            try:
                resp = be.complete(p["prompt"])
            except BackendError as e:
                sys.exit(f"\nStopped at probe {i}/{len(probes)}: {e}\nRe-run the same command to resume.")
            f.write(json.dumps({"id": p["id"], "response": resp}, ensure_ascii=False) + "\n")
            f.flush()
            print(f"\r{i}/{len(probes)} probes done", end="", file=sys.stderr)
    print(f"\nResponses in {a.out}. Next: dotlet probe score --probes {a.probes} --responses {a.out}")


def cmd_probe_score(a):
    result, rows = probe.analyze(probe.read_jsonl(a.probes), probe.read_jsonl(a.responses),
                                 metric=a.metric, unit=a.unit, **_cmp_kwargs(a))
    print("== Completion probe: does the model continue your text more faithfully than controls? ==\n")
    print(format_result(result))
    best = sorted((r for r in rows if r["group"] == "candidate"), key=lambda r: -r["score"])[:5]
    if best:
        print("\nYour passages the model continued most faithfully:")
        for r in best:
            print(f"  {r['score']:.2f}  {r['id']}\n        expected: \"{r['reference'][:120]}...\"\n"
                  f"        model:    \"{r['response'][:120]}...\"")
    _dump(a.json, {"result": result.to_dict(), "rows": rows})


def cmd_mia(a):
    from . import mia

    mats, ctrls = _chunks(a)
    results = mia.analyze(mats, ctrls, a.model, device=a.device, k=a.k, unit=a.unit, **_cmp_kwargs(a))
    print(f"== Membership inference against {a.model} ==\n")
    for name, r in results.items():
        print(format_result(r) + "\n")
    print("The 'mink' score is primary. If you rely on the other two as well, compare p-values to alpha/3.")
    _dump(a.json, {k: v.to_dict() for k, v in results.items()})


# ---------------------------------------------------------------- images

def _img_sets(a):
    from .image.features import find_images

    mats, ctrls = find_images(a.materials), find_images(a.controls)
    if not mats or not ctrls:
        sys.exit("Need at least one image in --materials and in --controls.")
    return mats, ctrls


def _feat(a):
    from .image.features import get_features

    return get_features(a.features, getattr(a, "device", None))


def cmd_image_overlap(a):
    from .image import overlap as io
    from .image.features import find_images

    mats, ctrls = _img_sets(a)
    outputs = find_images(a.outputs)
    if not outputs:
        sys.exit("No images found in --outputs.")
    rep = io.analyze(mats, ctrls, outputs, _feat(a), a.threshold, **_cmp_kwargs(a))
    print("== Image overlap: are AI images closer to yours than to controls? ==\n")
    print("-- Frequency test --")
    print(format_result(rep.result))
    print("\n-- Near-duplicate evidence --")
    print(f"Your images with a near-copy among outputs (similarity >= {rep.threshold:.2f}): {len(rep.near_duplicates)}")
    print(f"Closest any CONTROL image came to an output: {rep.max_control_similarity:.3f}  (the chance baseline)")
    if rep.near_duplicates and min(m.similarity for m in rep.near_duplicates) > rep.max_control_similarity:
        print("FLAG: these near-copies exceed every control match. Inspect them side by side, and rule out\n"
              "      image-to-image use, reference uploads, or retrieval before treating them as evidence.")
    print("\nClosest pairs (inspect these):")
    for m in rep.matches[: a.top]:
        tag = "YOURS  " if m.group == "candidate" else "control"
        print(f"  [{tag}] {m.similarity:.3f}  {m.image}\n                    ~ {m.best_output}")
    _dump(a.json, {"result": rep.result.to_dict(), "threshold": rep.threshold,
                   "max_control_similarity": rep.max_control_similarity,
                   "matches": [asdict(m) for m in rep.matches]})


def cmd_image_probe_make(a):
    from .image import probe as ip

    mats, ctrls = _img_sets(a)
    caps = ip.load_captions(mats + ctrls, a.captions)
    rows = ip.make_probes(mats, ctrls, caps)
    probe.write_jsonl(rows, a.out)
    print(f"Wrote {len(rows)} image probes ({len(mats)} yours, {len(ctrls)} controls) to {a.out}")
    print("Next: `dotlet image probe run`, or generate images yourself from each 'prompt' and list them as\n"
          'JSONL lines: {"id": "<image path>", "files": ["gen1.png", "gen2.png"]}')


def cmd_image_probe_run(a):
    import os

    from .backends import BackendError, OpenAIImages

    be = OpenAIImages(a.model, a.base_url, size=None if a.size.lower() == "none" else a.size)
    rows = probe.read_jsonl(a.probes)
    os.makedirs(a.images_dir, exist_ok=True)
    done = {}
    try:
        done = {r["id"]: r for r in probe.read_jsonl(a.out)}
    except FileNotFoundError:
        pass
    with open(a.out, "a", encoding="utf-8") as f:
        for i, p in enumerate(rows, 1):
            if p["id"] in done:
                continue
            files = []
            for k in range(a.samples):
                try:
                    data = be.generate(p["prompt"])
                except BackendError as e:
                    sys.exit(f"\nStopped at probe {i}/{len(rows)}: {e}\nRe-run the same command to resume.")
                path = os.path.join(a.images_dir, f"{i:05d}_{k}.png")
                with open(path, "wb") as g:
                    g.write(data)
                files.append(path)
            f.write(json.dumps({"id": p["id"], "files": files}) + "\n")
            f.flush()
            print(f"\r{i}/{len(rows)} probes done", end="", file=sys.stderr)
    print(f"\nGenerations listed in {a.out}. Next: dotlet image probe score --probes {a.probes} --responses {a.out}")


def cmd_image_probe_score(a):
    from .image import probe as ip

    result, rows = ip.analyze(probe.read_jsonl(a.probes), probe.read_jsonl(a.responses), _feat(a), **_cmp_kwargs(a))
    print("== Image caption probe: does the model recreate your images from their captions? ==\n")
    print(format_result(result))
    best = sorted((r for r in rows if r["group"] == "candidate"), key=lambda r: -r["score"])[:5]
    if best:
        print("\nYour images the model came closest to recreating:")
        for r in best:
            print(f"  {r['score']:.3f}  {r['image']}\n         ~ {r['best_file']}")
    _dump(a.json, {"result": result.to_dict(), "rows": rows})


def cmd_image_mia(a):
    from .image import mia as im
    from .image.probe import load_captions

    mats, ctrls = _img_sets(a)
    prompts = load_captions(mats + ctrls, a.captions) if a.captions else None
    loss_fn = lambda paths, ps, model: im.denoising_losses(paths, ps, model, a.device, a.timesteps, a.seed)
    result = im.analyze(mats, ctrls, a.model, prompts, loss_fn=loss_fn, **_cmp_kwargs(a))
    print(f"== Image membership inference against {a.model} (experimental) ==\n")
    print(format_result(result))
    _dump(a.json, {"result": result.to_dict()})


def _img_common(p, need_sets=True, features=True):
    if need_sets:
        p.add_argument("--materials", nargs="+", required=True, help="Your images: files or folders")
        p.add_argument("--controls", nargs="+", required=True,
                       help="Comparable images the model could NOT have seen (e.g. made after its training cutoff)")
    if features:
        p.add_argument("--features", default="hash",
                       help="'hash' (near-duplicates, default), 'clip', or 'clip:<hf-model-id>' (semantic similarity)")
        p.add_argument("--device")
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--permutations", type=int, default=10000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--json", metavar="PATH", help="Also write full results as JSON")


def _add_image_parser(sub):
    img = sub.add_parser("image", help="The same tests for images").add_subparsers(dest="icmd", required=True)

    p = img.add_parser("overlap", help="Test existing AI images for near-copies of your images")
    _img_common(p)
    p.add_argument("--outputs", nargs="+", required=True, help="AI-generated images/folders to examine")
    p.add_argument("--threshold", type=float, help="Near-duplicate similarity cut-off (default: 0.85 hash, 0.95 clip)")
    p.add_argument("--top", type=int, default=10, help="How many closest pairs to list")
    p.set_defaults(fn=cmd_image_overlap)

    pr = img.add_parser("probe", help="Caption probe against a text-to-image model").add_subparsers(dest="ipcmd", required=True)
    p = pr.add_parser("make", help="Create probes from images + captions")
    _img_common(p, features=False)
    p.add_argument("--captions", help="CSV/JSONL with image,prompt (default: sidecar <image>.txt files)")
    p.add_argument("--out", default="image_probes.jsonl")
    p.set_defaults(fn=cmd_image_probe_make)
    p = pr.add_parser("run", help="Generate images via an OpenAI-compatible images API")
    p.add_argument("--probes", default="image_probes.jsonl")
    p.add_argument("--model", required=True)
    p.add_argument("--base-url")
    p.add_argument("--size", default="1024x1024", help="'none' to omit")
    p.add_argument("--samples", type=int, default=4, help="Generations per prompt (score uses the closest)")
    p.add_argument("--images-dir", default="generated_images")
    p.add_argument("--out", default="image_responses.jsonl")
    p.set_defaults(fn=cmd_image_probe_run)
    p = pr.add_parser("score", help="Run the statistical test on generated images")
    _img_common(p, need_sets=False)
    p.add_argument("--probes", default="image_probes.jsonl")
    p.add_argument("--responses", default="image_responses.jsonl")
    p.set_defaults(fn=cmd_image_probe_score)

    p = img.add_parser("mia", help="Denoising-loss membership inference on an open diffusion model (experimental)")
    _img_common(p, features=False)
    p.add_argument("--model", required=True, help="diffusers model id, e.g. a Stable Diffusion checkpoint")
    p.add_argument("--captions", help="Optional captions to condition on (CSV/JSONL or sidecar .txt)")
    p.add_argument("--device")
    p.add_argument("--timesteps", type=int, nargs="+", default=[100, 200, 300, 400, 500])
    p.set_defaults(fn=cmd_image_mia)


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="dotlet",
                                 description="Statistical tests for whether an AI model was trained on your material (text or images).")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("overlap", help="Test existing AI outputs for verbatim reuse of your text")
    _common(p)
    p.add_argument("--outputs", nargs="+", required=True, help="AI-generated text files/folders to examine")
    p.add_argument("--background", nargs="+", help="Generic corpus whose n-grams are ignored (stock phrases)")
    p.add_argument("--n", type=int, default=8, help="N-gram length in words (default 8)")
    p.add_argument("--long-run", type=int, default=20, help="Flag verbatim runs at least this many words long")
    p.set_defaults(fn=cmd_overlap)

    pr = sub.add_parser("probe", help="Prefix-completion test against a chat model").add_subparsers(dest="pcmd", required=True)
    p = pr.add_parser("make", help="Create probe prompts from your material and controls")
    _common(p)
    p.add_argument("--prefix-frac", type=float, default=0.5, help="Share of each chunk shown to the model")
    p.add_argument("--out", default="probes.jsonl")
    p.set_defaults(fn=cmd_probe_make, chunk_size=120)
    p = pr.add_parser("run", help="Send probes to a model API")
    p.add_argument("--probes", default="probes.jsonl")
    p.add_argument("--backend", choices=["openai", "anthropic"], required=True,
                   help="'openai' = any OpenAI-compatible endpoint (use --base-url for Ollama, vLLM, etc.)")
    p.add_argument("--model", required=True)
    p.add_argument("--base-url")
    p.add_argument("--max-tokens", type=int, default=400)
    p.add_argument("--temperature", default="0", help="'none' to omit (for APIs that reject it)")
    p.add_argument("--out", default="responses.jsonl")
    p.set_defaults(fn=cmd_probe_run)
    p = pr.add_parser("score", help="Run the statistical test on model responses")
    _common(p, need_materials=False)
    p.add_argument("--probes", default="probes.jsonl")
    p.add_argument("--responses", default="responses.jsonl")
    p.add_argument("--metric", choices=probe.METRICS, default="ngram")
    p.set_defaults(fn=cmd_probe_score)

    p = sub.add_parser("mia", help="Likelihood-based membership inference on an open-weight model (needs torch)")
    _common(p)
    p.add_argument("--model", required=True, help="Hugging Face model id or local path")
    p.add_argument("--device")
    p.add_argument("--k", type=float, default=0.2, help="Fraction of lowest-probability tokens for Min-K%%")
    p.set_defaults(fn=cmd_mia)

    _add_image_parser(sub)
    return ap


def main(argv=None) -> None:
    a = build_parser().parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
