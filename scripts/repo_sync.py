#!/usr/bin/env python3
"""Commit and push only this project's own code and source configuration."""

from __future__ import annotations

import argparse
import fcntl
import os
from pathlib import Path
import re
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOTS = ("scripts", "experiments", "research", "patches")
ROOT_FILES = ("README.md", "AGENTS.md", ".gitignore", ".vscode/settings.json")
SOURCE_EXTENSIONS = {".py", ".sh", ".bash"}
CONFIG_EXTENSIONS = {".yaml", ".yml", ".toml", ".ini", ".cfg", ".gin"}
MAX_FILE_BYTES = 512_000
BLOCKED_PARTS = {
    ".git", ".venv", "venv", "envs", "site-packages", "node_modules",
    "__pycache__", "third_party", "data", "datasets", "models", "model",
    "checkpoints", "weights", "weight", "output", "outputs", "tmp",
    "logs", "cache", "assets", "figures", "visualizations", "images",
    "evidence", "source_snapshots", "root_document_snapshots",
    "portable_report", "structured_hoi_20260923_跨主机阅读包",
    "repo", "deps", "local_data", "megapose", "MoSca", "GVHMR",
    "sam2", "pytorch3d", "UniDepth", "Point4D", "co-tracker",
    "final_script_snapshot", "upstream_preview", "source_versions",
}
SECRET_MARKERS = (
    re.compile(rb"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
    re.compile(rb"\bghp_[A-Za-z0-9]{20,}\b"),
    re.compile(rb"\bgithub_pat_[A-Za-z0-9_]{20,}\b"),
)
PATCH_SOURCES = {
    "mosca_runtime.patch": "third_party/MoSca",
    "mosca_validation.patch": "experiments/mosca_validation_20260923/code/MoSca",
    "mosca_interface.patch": "experiments/mosca_interface_validation_20260923/code/MoSca",
    "gvhmr_validation.patch": "experiments/gvhmr_validation_20260923/code/GVHMR",
    "megapose_pose.patch": "experiments/object_pose_refinement_20260924/run01/megapose/repo",
}


def git(*args: str, input_data: bytes | None = None) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(
        ["git", "-C", str(ROOT), *args], input=input_data,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False,
    )


def fail(message: str) -> None:
    print(message, file=sys.stderr)
    raise SystemExit(1)


def allowed(path: Path) -> bool:
    relative = path.relative_to(ROOT)
    name = relative.as_posix()
    if name in ROOT_FILES:
        return True
    if relative.parts[0] == "patches":
        return len(relative.parts) == 2 and (path.suffix == ".patch" or path.name == "README.md")
    if name.startswith("scripts/systemd/") and path.suffix in {".service", ".timer"}:
        return True
    if not relative.parts or relative.parts[0] not in SOURCE_ROOTS:
        return False
    if any(part in BLOCKED_PARTS or part.startswith(".") for part in relative.parts[:-1]):
        return False
    if path.suffix in SOURCE_EXTENSIONS or path.suffix in CONFIG_EXTENSIONS:
        return True
    if path.name.startswith("requirements") and path.suffix == ".txt":
        return True
    if path.suffix == ".json" and ("code" in relative.parts or "protocol" in relative.parts):
        return bool(re.search(r"(^|_)(config|protocol|settings|params)(_|$)", path.stem))
    return False


def generate_patches() -> None:
    destination = ROOT / "patches"
    destination.mkdir(exist_ok=True)
    for filename, source in PATCH_SOURCES.items():
        repo = ROOT / source
        if not (repo / ".git").exists():
            continue
        diff = subprocess.run(
            ["git", "-C", str(repo), "diff", "--diff-filter=ACMRT", "--binary"],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False,
        )
        if diff.returncode:
            fail(diff.stderr.decode(errors="replace"))
        content = diff.stdout
        if filename == "mosca_interface.patch":
            new_code = repo / "lib_moca/pixel_geometry.py"
            if new_code.is_file() and not (repo / ".git" / "index.lock").exists():
                extra = subprocess.run(
                    ["git", "-C", str(repo), "diff", "--no-index", "--", "/dev/null", "lib_moca/pixel_geometry.py"],
                    stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False,
                )
                if extra.returncode not in (0, 1):
                    fail(extra.stderr.decode(errors="replace"))
                content += extra.stdout
        path = destination / filename
        if content and (not path.exists() or path.read_bytes() != content):
            path.write_bytes(content)
        elif not content and path.exists():
            path.unlink()


def candidates() -> list[Path]:
    paths: list[Path] = []
    for filename in ROOT_FILES:
        path = ROOT / filename
        if path.is_file() and not path.is_symlink():
            paths.append(path)
    for dirname in SOURCE_ROOTS:
        directory = ROOT / dirname
        if not directory.is_dir():
            continue
        for current, dirs, files in os.walk(directory, followlinks=False):
            dirs[:] = sorted(d for d in dirs if d not in BLOCKED_PARTS and not d.startswith("."))
            for filename in sorted(files):
                path = Path(current) / filename
                if not path.is_symlink() and path.is_file() and allowed(path):
                    paths.append(path)
    for path in paths:
        if path.stat().st_size > MAX_FILE_BYTES:
            fail(f"Allowed source exceeds {MAX_FILE_BYTES} bytes: {path}")
        content = path.read_bytes()
        if b"\x00" in content or any(marker.search(content) for marker in SECRET_MARKERS):
            fail(f"Binary content or credential marker in allowed source: {path}")
    return sorted(set(paths))


def sync(paths: list[Path]) -> None:
    if not (ROOT / ".git").is_dir():
        fail("Git repository is not initialized in the project root")
    pre_staged = git("diff", "--cached", "--name-only", "-z")
    if pre_staged.returncode or pre_staged.stdout:
        fail("The Git index already has staged changes; refusing to combine them with auto-sync")
    selected = {p.relative_to(ROOT).as_posix() for p in paths}
    tracked_result = git("ls-files", "-z")
    if tracked_result.returncode:
        fail(tracked_result.stderr.decode(errors="replace"))
    tracked = {p.decode() for p in tracked_result.stdout.split(b"\0") if p}
    unselected = {p for p in tracked if not allowed(ROOT / p)}
    if unselected:
        fail(f"Tracked files outside upload allowlist: {sorted(unselected)[:10]}")
    pathspec = b"\0".join(os.fsencode(p) for p in sorted(selected)) + b"\0"
    staged = git("add", "--force", "--pathspec-from-file=-", "--pathspec-file-nul", input_data=pathspec)
    if staged.returncode:
        fail(staged.stderr.decode(errors="replace"))
    missing = sorted(tracked - selected)
    if missing:
        removed = git("rm", "--quiet", "--", *missing)
        if removed.returncode:
            fail(removed.stderr.decode(errors="replace"))
    changed = git("diff", "--cached", "--name-only", "-z")
    changed_paths = {p.decode() for p in changed.stdout.split(b"\0") if p}
    if changed.returncode or changed_paths - (selected | set(missing)):
        fail("Staged changes crossed upload allowlist")
    if changed_paths:
        commit = git("commit", "-m", "Sync project code and configuration")
        if commit.returncode:
            fail(commit.stderr.decode(errors="replace"))
        print(f"Committed {len(changed_paths)} files")
    ahead = git("rev-list", "--count", "origin/main..HEAD")
    if ahead.returncode:
        fail(ahead.stderr.decode(errors="replace"))
    if int(ahead.stdout.strip()) > 0:
        push = git("push", "origin", "main")
        if push.returncode:
            fail(push.stderr.decode(errors="replace"))
    print(f"Checked {len(paths)} allowed files; GitHub main is current")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--list", action="store_true", help="print upload allowlist only")
    args = parser.parse_args()
    if not args.list:
        generate_patches()
    paths = candidates()
    if args.list:
        for path in paths:
            print(path.relative_to(ROOT))
        print(f"TOTAL {len(paths)}", file=sys.stderr)
        return
    lock_path = ROOT / ".git" / "repo-sync.lock"
    if not lock_path.parent.is_dir():
        fail("Git repository is not initialized in the project root")
    with lock_path.open("w") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        sync(paths)


if __name__ == "__main__":
    main()
