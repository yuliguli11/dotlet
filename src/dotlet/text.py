"""Loading, normalizing, chunking and n-gram helpers."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

TEXT_EXTENSIONS = {".txt", ".md", ".markdown", ".rst", ".html", ".htm", ".tex", ".csv", ".json"}

_WORD_RE = re.compile(r"[^\W_]+(?:'[^\W_]+)*", re.UNICODE)
_TAG_RE = re.compile(r"<[^>]+>")


@dataclass(frozen=True)
class Doc:
    name: str
    text: str


@dataclass(frozen=True)
class Chunk:
    source: str
    index: int
    tokens: tuple[str, ...]  # raw whitespace tokens (original casing/punctuation)

    @property
    def id(self) -> str:
        return f"{self.source}#{self.index}"

    @property
    def text(self) -> str:
        return " ".join(self.tokens)

    @property
    def words(self) -> list[str]:
        return words(self.text)


def normalize(s: str) -> str:
    s = unicodedata.normalize("NFKC", s)
    s = s.replace("’", "'").replace("‘", "'")
    return s.lower()


def words(s: str) -> list[str]:
    """Lowercased word tokens with punctuation stripped."""
    return _WORD_RE.findall(normalize(s))


def ngrams(toks: Sequence[str], n: int) -> set[tuple[str, ...]]:
    return {tuple(toks[i : i + n]) for i in range(len(toks) - n + 1)}


def load_docs(paths: Iterable[str | Path]) -> list[Doc]:
    """Load text files. Directories are walked recursively for known text extensions."""
    docs: list[Doc] = []
    for p in paths:
        p = Path(p)
        if p.is_dir():
            files = sorted(f for f in p.rglob("*") if f.is_file() and f.suffix.lower() in TEXT_EXTENSIONS)
        elif p.is_file():
            files = [p]
        else:
            raise FileNotFoundError(f"No such file or directory: {p}")
        for f in files:
            text = f.read_text(encoding="utf-8", errors="replace")
            if f.suffix.lower() in {".html", ".htm"}:
                text = _TAG_RE.sub(" ", text)
            if text.strip():
                docs.append(Doc(name=str(f), text=text))
    return docs


def chunk_docs(docs: Iterable[Doc], size: int, max_per_doc: int | None = None) -> list[Chunk]:
    """Split documents into non-overlapping chunks of `size` whitespace tokens.

    A trailing remainder is kept only if it is at least half a chunk long, so
    chunks stay comparable in size (important: overlap scores depend on length).
    """
    if size < 1:
        raise ValueError("chunk size must be positive")
    out: list[Chunk] = []
    for doc in docs:
        toks = doc.text.split()
        pieces = []
        for start in range(0, len(toks), size):
            piece = toks[start : start + size]
            if len(piece) >= max(1, size // 2):
                pieces.append(piece)
        if max_per_doc is not None and len(pieces) > max_per_doc:
            # spread picks evenly across the document instead of taking the start
            step = len(pieces) / max_per_doc
            pieces = [pieces[int(i * step)] for i in range(max_per_doc)]
        out.extend(Chunk(doc.name, i, tuple(p)) for i, p in enumerate(pieces))
    return out
