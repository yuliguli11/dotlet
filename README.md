# dotlet

Statistical tests for one question: **was this AI model trained on my material?**
Works for **text** and **images**.

No test can prove this by itself. What dotlet can do is measure whether a model
treats **your text** differently from **comparable text it could not have seen**
(the *controls*). It then tells you whether that difference is larger than chance.
That kind of comparative evidence is how published membership-inference and
copyright-audit studies work.

```
pip install -e .            # core: pure Python, no dependencies
pip install -e '.[mia]'     # + PyTorch/transformers for the open-model test
pip install -e '.[images]'  # image tests (Pillow + numpy)
pip install -e '.[image-clip]' / '.[image-mia]'   # + CLIP features / diffusion-model test
```

## Website

`web/index.html` is a single-file web app with the text overlap, image overlap
and completion-probe tests. It reads .txt, .md, .html, Word (.docx) and PDF
files, and images can be pasted in. **Everything runs in your browser and no file is
uploaded**, which matters if you're checking unpublished work. Open the file
directly, or host it anywhere static. For GitHub Pages: Settings > Pages >
deploy from branch, folder `/web`.

The statistics are ported from the Python package. On identical input the
per-chunk scores, AUC and Mann-Whitney p-values match exactly. Permutation and
bootstrap results differ slightly because they use a different random number
generator. Tests that need model weights or automated API calls (`mia`,
`image probe`, `image mia`, `probe run`) are only in the CLI.

## Text: the three tests

| Test | What you need | Works on | Detects |
|---|---|---|---|
| `overlap` | AI outputs you already have | any AI, even screenshots you transcribed | verbatim reuse of your text in outputs |
| `probe` | API access (or patience with a chat window) | any chat model | the model can *continue* your passages from memory |
| `mia` | model weights | open models (Llama, Mistral, Pythia, ...) | the model finds your text unusually *likely* |

Each test splits your material and the controls into equal-sized chunks and gives
each chunk a score (higher = more "memorized"-looking). It then reports:

- **AUC** with a bootstrap 95% CI: the probability that one of your chunks
  outscores a control chunk. 0.5 means no difference.
- **One-sided permutation test** and **Mann-Whitney U** p-values. The verdict
  uses the larger of the two (conservative).
- A plain-language verdict: *strong*, *some* or *no significant* evidence.

### 1. `overlap`: do AI outputs reproduce your text?

```
dotlet overlap --materials my_book/ --controls new_writing/ \
               --outputs chatgpt_outputs/ [--background generic_corpus/]
```

This counts how many rare 8-word sequences from each chunk appear in the AI
outputs. The report has two parts:

- **Frequency test**: is reuse systematic across your material?
- **Verbatim evidence**: the longest run copied from your text compared with the
  longest run "copied" from controls. A single 60-word verbatim match can be
  strong evidence even when it is too rare to show up in a frequency test.

Do not include your text in the prompts that produced the outputs. Otherwise the
model is just copying from its input.

### 2. `probe`: can the model continue your text?

```
dotlet probe make  --materials my_book/ --controls new_writing/ --out probes.jsonl
dotlet probe run   --backend openai --model <model> --probes probes.jsonl --out responses.jsonl
dotlet probe score --probes probes.jsonl --responses responses.jsonl
```

The model sees the first half of each passage and is asked to continue it.
dotlet scores how closely the continuation matches the real text (`--metric
ngram|lcs|prefix`). `--backend openai` works with any OpenAI-compatible server
(OpenAI, Ollama, vLLM, LM Studio, OpenRouter...) via `--base-url`.
`--backend anthropic` uses `ANTHROPIC_API_KEY`. Runs can be resumed. Without an
API, paste each `prompt` into a chat window and save the replies as
`{"id": ..., "response": ...}` lines.

Many deployed chat models are trained *not* to reproduce copyrighted text, so a
null result from `probe` is weak evidence of absence.

### 3. `mia`: does the model find your text unusually likely?

```
dotlet mia --materials my_book/ --controls new_writing/ --model EleutherAI/pythia-1.4b
```

This reports Min-K% Prob (primary), loss, and zlib-normalized loss. It only
works where you can run the model and read its token probabilities.

## Images: `dotlet image ...`

These are the same three ideas and the same statistics. One image is one unit.

| Test | What you need | Works on | Detects |
|---|---|---|---|
| `image overlap` | AI images you already have | any image generator | near-copies of your images among outputs |
| `image probe` | an images API (or generate by hand) | any text-to-image model | the model recreates your image from its caption |
| `image mia` *(experimental)* | model weights | Stable Diffusion-family models (`diffusers`) | lower denoising loss on your images |

**Similarity features** (`--features`):

- `hash` (default): a perceptual hash plus a difference hash, 128 bits in all.
  It survives resizing, JPEG re-compression, brightness shifts and mirroring.
  On our calibration set, edited copies score at least 0.94 and unrelated images
  at most 0.74. Near-duplicate cut-off: 0.85.
- `clip` or `clip:<hf-model-id>`: CLIP embeddings. Use this for copies that are
  cropped, re-composed or re-styled. Be careful: CLIP also scores images that
  merely share a subject as close. Near-duplicate cut-off: 0.95.

### `image overlap`

```
dotlet image overlap --materials my_art/ --controls new_art/ --outputs ai_images/
```

Each of your images and each control is scored by its closest AI output
(mirrored outputs included). You get the frequency test, a near-duplicate count
compared against the closest any control came (the chance baseline), and a
list of the closest pairs to inspect side by side. This is the approach
Carlini et al. (2023) used to pull training images out of Stable Diffusion.

### `image probe`

```
dotlet image probe make  --materials my_art/ --controls new_art/   # captions: sidecar img.txt, or --captions caps.csv
dotlet image probe run   --model <images-model> --samples 4        # any OpenAI-compatible /images/generations API
dotlet image probe score
```

The model gets only the caption. The score is how close its best sample comes to
**your original**. Captions must be equally detailed for your images and for
controls, or you end up testing the captions. Don't put titles or your name
in one group only, unless that is exactly what you mean to test (for example
"Untitled 3 by <you>"). Without an API, generate images yourself and list them
as `{"id": "<image path>", "files": [...]}` lines.

### `image mia` (experimental)

```
dotlet image mia --materials my_art/ --controls new_art/ --model <diffusers SD checkpoint> [--captions caps.csv]
```

This adds the same seeded noise at several timesteps and measures how well the
model predicts it. Training images tend to get lower error. The error also
depends heavily on image complexity, resolution and source. Controls must
closely match your images (same medium, camera, style and size), and you need a
GPU. This test has not been validated against a real model in this repo yet.

### Reading image results

If only some of your images leaked, the AUC is capped. For example, with 12 of
30 leaked the best possible AUC is about 0.70. A modest AUC with a very small
p-value can still be meaningful, so look at the near-duplicate list.
Image-to-image modes, reference uploads ("make it like this") and generators
with web retrieval all produce near-copies without any training. Rule these out
before you treat a match as evidence.

## Choosing controls (the part that matters most)

The test is only as good as the controls. They should be:

1. **Unseen by the model.** The best option is text written or first published
   *after the model's training cutoff*, or text that was never published. If
   the model also trained on the controls, the test loses power.
2. **Matched to your material.** Use the same genre, style, language, era and
   format. If your material is 19th-century poetry and the controls are 2025
   blog posts, you are measuring style rather than training exposure. This
   matters especially for `mia`, where "memorization" scores are known to
   pick up date and topic shifts (Duan et al. 2024; Das et al. 2024).
   **Your own newer writing is often the ideal control.**
3. **Not too few.** Aim for 30+ chunks on each side. dotlet warns below 10.

Other confounders to rule out before drawing conclusions:

- Your text (or image) was widely quoted, reposted or copied elsewhere online. Then the model may
  have learned it from those copies.
- The model has web search or retrieval. Then it is reading your text, not
  remembering it.
- Chunks from one long document are not independent. Use `--unit document`
  (average per file) or `--max-chunks-per-doc` for a more conservative test.

## Interpreting results

- **Strong evidence** (p < α/5 and AUC CI above 0.5): the model systematically
  favors your text over matched controls. That is consistent with training
  exposure. It is not proof of how the text got there.
- **No significant evidence** does not mean "not trained on". The model may have
  seen your text too few times to memorize it, or the sample may be too small.
- Running several tests or metrics raises the false-positive rate. Decide on
  your primary test in advance, or divide α by the number of tests.

This is a research tool, not legal advice.

## Development

```
pip install -e '.[dev]' && pytest
```

The tests use synthetic data where the answer is known in advance. They check
that each test detects planted memorization and does not fire on unrelated data,
and that the p-values are calibrated under the null.
