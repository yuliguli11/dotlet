import json
import math
import random

import pytest

from dotlet import cli, mia, overlap, probe
from dotlet.stats import auc, compare, mann_whitney_greater, permutation_test
from dotlet.text import Doc, chunk_docs, words

VOCAB = [f"w{i}" for i in range(400)]


def fake_text(rng, n_words):
    return " ".join(rng.choice(VOCAB) for _ in range(n_words))


def corpus(seed, n_docs=6, n_words=1200, prefix="doc"):
    rng = random.Random(seed)
    return [Doc(f"{prefix}{i}", fake_text(rng, n_words)) for i in range(n_docs)]


# ---------- stats ----------

def test_auc_basic():
    assert auc([3, 4], [1, 2]) == 1.0
    assert auc([1, 2], [3, 4]) == 0.0
    assert auc([1, 1], [1, 1]) == 0.5


def test_tests_detect_shift_and_not_null():
    rng = random.Random(1)
    a = [rng.gauss(1, 1) for _ in range(60)]
    b = [rng.gauss(0, 1) for _ in range(60)]
    assert permutation_test(a, b, 2000) < 0.01
    assert mann_whitney_greater(a, b) < 0.01
    assert permutation_test(b, a, 2000) > 0.5  # wrong direction -> not significant


def test_null_false_positive_rate_is_controlled():
    rng = random.Random(7)
    rejections = 0
    for _ in range(200):
        a = [rng.random() for _ in range(25)]
        b = [rng.random() for _ in range(25)]
        rejections += mann_whitney_greater(a, b) < 0.05
    assert rejections / 200 < 0.09


def test_compare_verdicts():
    strong = compare([1.0] * 30, [0.0] * 30, "m", n_permutations=500, n_bootstrap=200)
    assert strong.verdict.startswith("STRONG")
    null = compare([0.0] * 30, [0.0] * 30, "m", n_permutations=500, n_bootstrap=200)
    assert null.verdict.startswith("NO")
    assert not math.isnan(null.p_mann_whitney)


# ---------- text ----------

def test_words_and_chunking():
    assert words("Hello, World! It’s fine.") == ["hello", "world", "it's", "fine"]
    chunks = chunk_docs([Doc("a", " ".join(["x"] * 450))], 200)
    assert [len(c.tokens) for c in chunks] == [200, 200]  # 50-word tail dropped
    chunks = chunk_docs([Doc("a", " ".join(["x"] * 2000))], 100, max_per_doc=4)
    assert len(chunks) == 4


# ---------- overlap ----------

def test_overlap_detects_regurgitation():
    mats, ctrls = corpus(1, prefix="mine"), corpus(2, prefix="ctrl")
    rng = random.Random(3)
    outputs = []
    for d in mats:  # AI outputs that paste 40-word fragments of the material into filler
        toks = d.text.split()
        for start in range(0, len(toks) - 40, 150):
            outputs.append(Doc("out", fake_text(rng, 50) + " " + " ".join(toks[start:start + 40])))
    rep = overlap.analyze(chunk_docs(mats, 200), chunk_docs(ctrls, 200), outputs,
                          n_permutations=2000, n_bootstrap=300)
    assert rep.result.p_permutation < 0.01
    assert rep.result.auc > 0.9
    assert rep.top_spans[0].group == "candidate" and rep.top_spans[0].length_words >= 40
    assert rep.longest_candidate_run >= 40 and rep.longest_control_run < 10 and rep.long_runs


def test_overlap_null_when_outputs_unrelated():
    mats, ctrls = corpus(1, prefix="mine"), corpus(2, prefix="ctrl")
    outputs = corpus(9, n_docs=10, prefix="out")
    rep = overlap.analyze(chunk_docs(mats, 200), chunk_docs(ctrls, 200), outputs,
                          n_permutations=2000, n_bootstrap=300)
    assert rep.result.verdict.startswith("NO")


def test_background_ngrams_are_ignored():
    phrase = "at the end of the day we all know that"
    c = chunk_docs([Doc("m", phrase + " zz")], 12)[0]
    idx = {tuple(words(phrase)[i:i + 8]) for i in range(3)}
    s, _ = overlap.score_chunk(c, idx, set(), 8)
    s_bg, _ = overlap.score_chunk(c, idx, idx, 8)
    assert s > 0 and s_bg == 0


# ---------- probe ----------

def test_continuation_metrics():
    ref = "the quick brown fox jumps over the lazy dog"
    assert probe.score_continuation(ref, ref, "ngram") == 1.0
    assert probe.score_continuation(ref, ref, "lcs") == 1.0
    assert probe.score_continuation("the quick red", ref, "prefix") == pytest.approx(2 / 9)
    assert probe.score_continuation("completely unrelated words here", ref, "ngram") == 0.0
    # verbosity is not rewarded: extra text after the reference length is cut
    assert probe.score_continuation("nope " * 9 + ref, ref, "ngram") == 0.0


def test_probe_end_to_end_with_memorizing_model():
    mats, ctrls = corpus(1, prefix="mine"), corpus(2, prefix="ctrl")
    probes = probe.make_probes(chunk_docs(mats, 120), chunk_docs(ctrls, 120))
    rng = random.Random(5)
    responses = []
    for p in probes:
        if p["group"] == "candidate":  # model recalls ~60% of the real continuation
            ref = p["reference"].split()
            resp = " ".join(ref[: int(len(ref) * 0.6)]) + " " + fake_text(rng, 20)
        else:
            resp = fake_text(rng, 60)
        responses.append({"id": p["id"], "response": resp})
    result, rows = probe.analyze(probes, responses, n_permutations=2000, n_bootstrap=300)
    assert result.p_permutation < 0.01 and result.auc > 0.95
    result_doc, _ = probe.analyze(probes, responses, unit="document", n_permutations=2000, n_bootstrap=300)
    assert result_doc.n_candidates == len(mats)


# ---------- mia (with a fake log-prob function, no torch needed) ----------

def test_mia_with_fake_model():
    mats, ctrls = corpus(1, prefix="mine"), corpus(2, prefix="ctrl")
    member_texts = {c.text for c in chunk_docs(mats, 100)}
    rng = random.Random(0)

    def fake_logprobs(texts, model_name, device):
        return [[rng.gauss(-2.0 if t in member_texts else -3.0, 1.0) for _ in range(100)] for t in texts]

    res = mia.analyze(chunk_docs(mats, 100), chunk_docs(ctrls, 100), "fake",
                      logprob_fn=fake_logprobs, n_permutations=2000, n_bootstrap=300)
    assert res["mink"].p_permutation < 0.01
    assert res["loss"].auc > 0.9


# ---------- CLI ----------

def test_cli_overlap_and_probe_score(tmp_path, capsys):
    mats, ctrls = corpus(1, prefix="mine"), corpus(2, prefix="ctrl")
    for name, docs in (("mats", mats), ("ctrls", ctrls)):
        (tmp_path / name).mkdir()
        for i, d in enumerate(docs):
            (tmp_path / name / f"{i}.txt").write_text(d.text)
    (tmp_path / "out.txt").write_text(" ".join(mats[0].text.split()[:300]))
    cli.main(["overlap", "--materials", str(tmp_path / "mats"), "--controls", str(tmp_path / "ctrls"),
              "--outputs", str(tmp_path / "out.txt"), "--permutations", "500",
              "--json", str(tmp_path / "r.json")])
    out = capsys.readouterr().out
    assert "Overlap test" in out and "YOURS" in out
    assert json.loads((tmp_path / "r.json").read_text())["result"]["n_candidates"] > 0

    probes_path = tmp_path / "p.jsonl"
    cli.main(["probe", "make", "--materials", str(tmp_path / "mats"), "--controls", str(tmp_path / "ctrls"),
              "--out", str(probes_path)])
    probes = probe.read_jsonl(probes_path)
    probe.write_jsonl([{"id": p["id"], "response": p["reference"] if p["group"] == "candidate" else "x y z"}
                       for p in probes], tmp_path / "r.jsonl")
    cli.main(["probe", "score", "--probes", str(probes_path), "--responses", str(tmp_path / "r.jsonl"),
              "--permutations", "500"])
    assert "STRONG" in capsys.readouterr().out
