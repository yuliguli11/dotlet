"""Image loading and similarity features.

Two featurizers, both giving similarities in roughly [0, 1] (higher = more alike):

- HashFeatures (default, needs only Pillow + numpy): 64-bit perceptual hash
  (pHash, DCT-based) + 64-bit difference hash (dHash). Robust to resizing,
  re-compression, mild color/brightness changes. Good at near-duplicates
  ("did it spit out my image?"), blind to crops, re-compositions and style.
- ClipFeatures (optional, needs torch + transformers): CLIP image embeddings,
  cosine similarity. Catches semantic/compositional copies a hash misses, at
  the cost of also rating merely "similar subject" images as close.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif", ".tif", ".tiff"}


def _np():
    try:
        import numpy as np
        from PIL import Image, ImageOps  # noqa: F401
    except ImportError as e:  # pragma: no cover
        raise SystemExit("Image tests need Pillow and numpy: pip install 'dotlet[images]'") from e
    return np


def find_images(paths: Iterable[str | Path]) -> list[Path]:
    out: list[Path] = []
    for p in paths:
        p = Path(p)
        if p.is_dir():
            out.extend(sorted(f for f in p.rglob("*") if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS))
        elif p.is_file():
            out.append(p)
        else:
            raise FileNotFoundError(f"No such file or directory: {p}")
    return out


def load_image(path: str | Path):
    _np()
    from PIL import Image, ImageOps

    with Image.open(path) as im:
        im = ImageOps.exif_transpose(im)
        if im.mode in ("RGBA", "LA", "P"):
            im = im.convert("RGBA")
            bg = Image.new("RGBA", im.size, (255, 255, 255, 255))
            im = Image.alpha_composite(bg, im)
        return im.convert("RGB")


def _dct_matrix(n: int):
    np = _np()
    k = np.arange(n)[:, None]
    i = np.arange(n)[None, :]
    m = np.cos(np.pi * (2 * i + 1) * k / (2 * n)) * np.sqrt(2 / n)
    m[0] /= np.sqrt(2)
    return m


class HashFeatures:
    name = "hash"
    default_threshold = 0.85  # <= 19 of 128 bits differ: typical near-duplicate cut-off

    def __init__(self):
        self._dct = _dct_matrix(32)

    def _one(self, im):
        np = _np()
        from PIL import Image

        g = im.convert("L")
        a = np.asarray(g.resize((32, 32), Image.LANCZOS), dtype=np.float64)
        d = self._dct @ a @ self._dct.T
        low = d[:8, :8].flatten()
        phash = low > np.median(low[1:])
        b = np.asarray(g.resize((9, 8), Image.LANCZOS), dtype=np.float64)
        dhash = (b[:, 1:] > b[:, :-1]).flatten()
        return np.concatenate([phash, dhash])

    def __call__(self, images: Sequence) -> "np.ndarray":  # noqa: F821
        np = _np()
        return np.stack([self._one(im) for im in images]).astype(np.float32) if images else np.zeros((0, 128), np.float32)

    @staticmethod
    def similarity(A, B):
        """1 - normalized Hamming distance, all pairs: (len A, len B)."""
        np = _np()
        if len(A) == 0 or len(B) == 0:
            return np.zeros((len(A), len(B)))
        a1 = A.sum(1)[:, None]
        b1 = B.sum(1)[None, :]
        hamming = a1 + b1 - 2 * (A @ B.T)
        return 1.0 - hamming / A.shape[1]


class ClipFeatures:
    name = "clip"
    default_threshold = 0.95

    def __init__(self, model_name: str = "openai/clip-vit-base-patch32", device: str | None = None):
        try:
            import torch
            from transformers import CLIPModel, CLIPProcessor
        except ImportError as e:  # pragma: no cover
            raise SystemExit("CLIP features need torch + transformers: pip install 'dotlet[image-clip]'") from e
        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = CLIPModel.from_pretrained(model_name).to(self.device).eval()
        self.proc = CLIPProcessor.from_pretrained(model_name)
        self.name = f"clip ({model_name})"

    def __call__(self, images: Sequence):
        np = _np()
        out = []
        with self.torch.no_grad():
            for i in range(0, len(images), 32):
                x = self.proc(images=list(images[i : i + 32]), return_tensors="pt").to(self.device)
                e = self.model.get_image_features(**x)
                out.append(self.torch.nn.functional.normalize(e, dim=-1).cpu().numpy())
        return np.concatenate(out) if out else np.zeros((0, 1))

    @staticmethod
    def similarity(A, B):
        return A @ B.T


def get_features(name: str, device: str | None = None):
    if name == "hash":
        return HashFeatures()
    if name.startswith("clip"):
        _, _, model = name.partition(":")
        return ClipFeatures(model or "openai/clip-vit-base-patch32", device)
    raise ValueError(f"unknown features {name!r}: use 'hash', 'clip' or 'clip:<hf-model-id>'")


def featurize_paths(feat, paths: Sequence[Path], with_mirror: bool = False):
    """Features for each path; with_mirror also appends features of horizontally flipped copies."""
    from PIL import ImageOps

    imgs = [load_image(p) for p in paths]
    F = feat(imgs)
    if with_mirror and imgs:
        np = _np()
        F = np.concatenate([F, feat([ImageOps.mirror(im) for im in imgs])])
    return F
