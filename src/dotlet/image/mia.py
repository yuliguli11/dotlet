"""Image test 3 (experimental): denoising-loss membership inference for open
latent-diffusion models (Stable Diffusion family, via `diffusers`).

Diffusion models learn to predict the noise added to training images; on
images they were trained on, that prediction error tends to be lower
(Matsumoto et al. 2023; Carlini et al. 2023; Duan et al. 2023, "SecMI").
For each image we add noise at fixed timesteps with a fixed seed (identical
for every image), ask the model to predict it, and record the error.
Score = -mean error (higher = more member-like).

Needs: pip install 'dotlet[image-mia]' and a GPU for any reasonable speed.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

from ..stats import TestResult, compare
from .features import load_image

DEFAULT_TIMESTEPS = (100, 200, 300, 400, 500)


def denoising_losses(
    paths: Sequence[Path],
    prompts: Sequence[str],
    model_name: str,
    device: str | None = None,
    timesteps: Sequence[int] = DEFAULT_TIMESTEPS,
    seed: int = 0,
    resolution: int = 512,
) -> list[float]:
    try:
        import numpy as np
        import torch
        from diffusers import StableDiffusionPipeline
        from PIL import Image, ImageOps
    except ImportError as e:  # pragma: no cover
        raise SystemExit("Image mia needs torch + diffusers + transformers: pip install 'dotlet[image-mia]'") from e

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device == "cuda" else torch.float32
    pipe = StableDiffusionPipeline.from_pretrained(model_name, torch_dtype=dtype).to(device)
    vae, unet, sched = pipe.vae, pipe.unet, pipe.scheduler
    pred_type = getattr(sched.config, "prediction_type", "epsilon")

    losses = []
    with torch.no_grad():
        for path, prompt in zip(paths, prompts):
            im = ImageOps.fit(load_image(path), (resolution, resolution), Image.BICUBIC)
            x = torch.from_numpy(np.asarray(im, dtype=np.float32) / 127.5 - 1).permute(2, 0, 1)[None].to(device, dtype)
            latents = vae.encode(x).latent_dist.mean * vae.config.scaling_factor
            ids = pipe.tokenizer(prompt or "", padding="max_length", truncation=True,
                                 max_length=pipe.tokenizer.model_max_length, return_tensors="pt").input_ids.to(device)
            emb = pipe.text_encoder(ids)[0]
            errs = []
            for t in timesteps:
                g = torch.Generator(device=device).manual_seed(seed + int(t))
                noise = torch.randn(latents.shape, generator=g, device=device, dtype=dtype)
                tt = torch.tensor([int(t)], device=device)
                noisy = sched.add_noise(latents, noise, tt)
                pred = unet(noisy, tt, encoder_hidden_states=emb).sample
                target = sched.get_velocity(latents, noise, tt) if pred_type == "v_prediction" else noise
                errs.append(torch.mean((pred.float() - target.float()) ** 2).item())
            losses.append(sum(errs) / len(errs))
    return losses


def analyze(
    candidates: list[Path],
    controls: list[Path],
    model_name: str,
    prompts: dict[str, str] | None = None,
    loss_fn=denoising_losses,
    **compare_kwargs,
) -> TestResult:
    paths = candidates + controls
    prompts = prompts or {}
    losses = loss_fn(paths, [prompts.get(str(p), "") for p in paths], model_name)
    scores = [-l for l in losses]
    n = len(candidates)
    result = compare(scores[:n], scores[n:], metric=f"negative denoising loss under {model_name}", **compare_kwargs)
    result.notes.append(
        "Experimental. Denoising loss also depends on image complexity, resolution and source; "
        "controls must match your images closely (same camera/medium/style/size) for this to mean anything."
    )
    return result
