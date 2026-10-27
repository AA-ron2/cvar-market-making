"""File hashes, repository containment and git output for the adapter's command-line tools."""

from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def file_sha256(path: str | Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def is_inside(path: str | Path, root: str | Path) -> bool:
    """Whether path, resolved, lies under root: data never goes into the repository (WO-HB1)."""

    try:
        Path(path).resolve().relative_to(Path(root).resolve())
    except ValueError:
        return False
    return True


def git_output(repo_root: str | Path, *args: str) -> subprocess.CompletedProcess:
    """git's output decoded as UTF-8 (not the Windows locale codec, which fails on some commit subjects)."""

    return subprocess.run(
        ["git", *args], cwd=repo_root, capture_output=True, encoding="utf-8", errors="replace"
    )
