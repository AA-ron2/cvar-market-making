"""Verify the published results and the release assets of v1.0.0 (standard library only).

    python scripts/analysis/verify_release.py [--assets DIR]

1. Every file in results/final/ against results/final/SHA256SUMS (and no file missing from it or unlisted).
2. With --assets, the downloaded release assets in DIR against results/final/release_assets.sha256.
3. For every file the thesis's profile record lists (results/final/profiling/profile_stage.sha256), matched by file
   name: "identical to the thesis output" if it has the recorded SHA-256, "documented change" if results/final/
   REDACTIONS.md lists that file with the recorded SHA-256 as the original and its current SHA-256 as the published
   one, and a failure otherwise. Release assets are checked only with --assets.

Exit status 0 when every check passes, 1 otherwise.
"""
from __future__ import annotations

import argparse
import hashlib
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SHA256 = re.compile(r"^[0-9a-f]{64}$")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_sums(path: Path) -> list[tuple[str, str]]:
    """The (SHA-256, relative path) lines of a sha256sum file."""

    entries = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        digest, name = line.split(maxsplit=1)
        name = name.strip().lstrip("*")
        if not SHA256.match(digest):
            raise ValueError(f"{path}: not a sha256sum line: {line!r}")
        entries.append((digest, name))
    return entries


def documented_changes(redactions: Path) -> dict[str, list[tuple[str, str]]]:
    """File name -> [(original SHA-256, published SHA-256)] from the first table of REDACTIONS.md."""

    changes: dict[str, list[tuple[str, str]]] = {}
    in_table = False
    for line in redactions.read_text(encoding="utf-8").splitlines():
        if line.startswith("| path | original SHA-256 | published SHA-256 | change"):
            in_table = True
            continue
        if in_table and line.startswith("|---"):
            continue
        if in_table and not line.startswith("|"):
            break
        if in_table:
            cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
            path = re.search(r"`([^`]+)`", cells[0])
            original, published = (re.search(r"`([0-9a-f]{64})`", cell) for cell in cells[1:3])
            if path and original and published:
                changes.setdefault(Path(path.group(1)).name, []).append((original.group(1), published.group(1)))
    return changes


def check_results(final: Path, report: list[str]) -> bool:
    ok = True
    listed = read_sums(final / "SHA256SUMS")
    names = set()
    for digest, name in listed:
        names.add(name)
        path = final / name
        if not path.is_file():
            report.append(f"MISSING  results/final/{name}")
            ok = False
        elif file_sha256(path) != digest:
            report.append(f"CHANGED  results/final/{name}")
            ok = False
    for path in sorted(final.rglob("*")):
        relative = path.relative_to(final).as_posix()
        if path.is_file() and relative != "SHA256SUMS" and relative not in names:
            report.append(f"UNLISTED results/final/{relative}")
            ok = False
    report.append(f"results/final/: {len(listed)} files listed in SHA256SUMS, {'all OK' if ok else 'FAILED'}")
    return ok


def check_assets(final: Path, assets: Path, report: list[str]) -> bool:
    ok = True
    listed = read_sums(final / "release_assets.sha256")
    for digest, name in listed:
        path = assets / name
        if not path.is_file():
            report.append(f"MISSING  asset {name}")
            ok = False
        elif file_sha256(path) != digest:
            report.append(f"CHANGED  asset {name}")
            ok = False
    report.append(f"release assets: {len(listed)} files, {'all OK' if ok else 'FAILED'}")
    return ok


def check_profile_record(final: Path, assets: Path | None, report: list[str]) -> bool:
    ok = True
    changes = documented_changes(final / "REDACTIONS.md")
    by_name = {path.name: path for path in final.rglob("*") if path.is_file()}
    for recorded, name in read_sums(final / "profiling" / "profile_stage.sha256"):
        base = Path(name).name
        if name.startswith("models/"):
            if assets is None:
                report.append(f"skipped  {base}: a release asset (pass --assets)")
                continue
            path = assets / base
        else:
            path = by_name.get(base)
        if path is None or not path.is_file():
            report.append(f"FAIL     {base}: not found")
            ok = False
            continue
        current = file_sha256(path)
        if current == recorded:
            report.append(f"OK       {base}: identical to the thesis output")
        elif (recorded, current) in changes.get(base, []):
            report.append(f"OK       {base}: documented change (results/final/REDACTIONS.md)")
        else:
            report.append(f"FAIL     {base}: SHA-256 {current} is neither the recorded one nor a documented change")
            ok = False
    return ok


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--assets", type=Path, help="a folder with the downloaded release assets")
    parser.add_argument("--root", type=Path, default=REPO, help="the repository root (default: this checkout)")
    args = parser.parse_args(argv)
    final = args.root / "results" / "final"
    report: list[str] = []
    ok = check_results(final, report)
    if args.assets is not None:
        ok &= check_assets(final, args.assets, report)
    ok &= check_profile_record(final, args.assets, report)
    print("\n".join(report))
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
