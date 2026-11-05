"""Test 3: likelihood-based membership inference for open-weight models.

Needs token probabilities, so it runs a local Hugging Face causal LM
(`pip install dotlet[mia]`). Models tend to assign higher probability to text
they were trained on. Three standard scores (higher = more member-like):

  loss  : negative mean token NLL                (Yeom et al. 2018; Carlini et al. 2021)
  mink  : mean log-prob of the k% least likely tokens (Min-K% Prob, Shi et al. 2024)
  zlib  : -NLL / zlib-compressed length          (Carlini et al. 2021)

`mink` is the primary score; the others are reported for context. If you look
at all three, mentally apply a multiple-comparison correction (e.g. alpha/3).
"""

from __future__ import annotations

import zlib

from .stats import TestResult, aggregate_by_group, compare
from .text import Chunk

METHODS = ("mink", "loss", "zlib")


def _load(model_name: str, device: str | None):
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as e:  # pragma: no cover
        raise SystemExit("The mia test needs PyTorch + transformers: pip install 'dotlet[mia]'") from e
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device).eval()
    return torch, tok, model, device


def token_logprobs(texts: list[str], model_name: str, device: str | None = None, max_tokens: int = 512) -> list[list[float]]:
    torch, tok, model, device = _load(model_name, device)
    out = []
    with torch.no_grad():
        for t in texts:
            ids = tok(t, return_tensors="pt", truncation=True, max_length=max_tokens).input_ids.to(device)
            if ids.shape[1] < 2:
                out.append([])
                continue
            logits = model(ids).logits[0, :-1].float()
            lp = torch.log_softmax(logits, dim=-1).gather(1, ids[0, 1:].unsqueeze(1)).squeeze(1)
            out.append(lp.cpu().tolist())
    return out


def scores_from_logprobs(text: str, lps: list[float], k: float = 0.2) -> dict[str, float]:
    if not lps:
        return {m: float("nan") for m in METHODS}
    mean_lp = sum(lps) / len(lps)
    lowest = sorted(lps)[: max(1, int(len(lps) * k))]
    return {
        "loss": mean_lp,
        "mink": sum(lowest) / len(lowest),
        "zlib": mean_lp / len(zlib.compress(text.encode())),  # mean_lp<0, so larger (closer to 0) = more member-like
    }


def analyze(
    candidates: list[Chunk],
    controls: list[Chunk],
    model_name: str,
    device: str | None = None,
    k: float = 0.2,
    unit: str = "chunk",
    logprob_fn=token_logprobs,
    **compare_kwargs,
) -> dict[str, TestResult]:
    chunks = candidates + controls
    lps = logprob_fn([c.text for c in chunks], model_name, device)
    per = [scores_from_logprobs(c.text, lp, k) for c, lp in zip(chunks, lps)]
    n = len(candidates)
    results = {}
    for m in METHODS:
        pos = [s[m] for s in per[:n]]
        neg = [s[m] for s in per[n:]]
        if unit == "document":
            pos = aggregate_by_group([c.id for c in candidates], pos)[1]
            neg = aggregate_by_group([c.id for c in controls], neg)[1]
        label = {"mink": f"Min-{int(k * 100)}% log-prob (primary)", "loss": "mean log-prob", "zlib": "log-prob / zlib size"}[m]
        results[m] = compare(pos, neg, metric=f"{label} under {model_name}", **compare_kwargs)
    return results
