"""Reproducible run-manifest construction."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import subprocess
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from biohub.config import Config


def _normal(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return list(value)
    return value


def config_sha256(cfg: Config) -> str:
    payload = {key: _normal(value) for key, value in asdict(cfg).items()}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def input_fingerprint(path: Path) -> str:
    """Hash paths, sizes, mtimes, and small metadata; avoids rereading huge volumes."""
    path = Path(path)
    digest = hashlib.sha256()
    files = [path] if path.is_file() else sorted(p for p in path.rglob("*") if p.is_file())
    for file in files:
        stat = file.stat()
        rel = file.name if path.is_file() else str(file.relative_to(path))
        digest.update(f"{rel}\0{stat.st_size}\0{stat.st_mtime_ns}\n".encode())
        if file.name in {"zarr.json", ".zarray", ".zattrs"} or stat.st_size <= 4096:
            digest.update(file.read_bytes())
    return digest.hexdigest()


def build_run_manifest(cfg: Config, inputs: Iterable[Path] = ()) -> dict:
    root = Path(__file__).resolve().parents[2]
    try:
        git_sha = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, check=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        git_sha = None
    packages = {}
    for name in ("numpy", "scipy", "pandas", "scikit-image", "zarr", "torch", "cellpose", "streamlit", "plotly"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "git_sha": git_sha,
        "source_sha256": {str(p.relative_to(root)).replace('\\','/'):hashlib.sha256(p.read_bytes()).hexdigest()
                          for folder in ('src','app') for p in sorted((root/folder).rglob('*.py'))} |
                         ({'app.py':hashlib.sha256((root/'app.py').read_bytes()).hexdigest()} if (root/'app.py').exists() else {}),
        "config_sha256": config_sha256(cfg),
        "config": {key: _normal(value) for key, value in asdict(cfg).items()},
        "inputs": [
            {"path": str(Path(path)), "fingerprint_sha256": input_fingerprint(Path(path))}
            for path in inputs
        ],
        "fingerprint_definition": "SHA-256 over relative paths, byte sizes, mtimes, and small metadata content; not a full byte checksum for large arrays",
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": packages,
    }


def write_run_manifest(path: Path, cfg: Config, inputs: Iterable[Path] = ()) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(build_run_manifest(cfg, inputs), indent=2))
    return path
