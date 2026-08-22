"""Run-directory and provenance helpers for agent-driven benchmark runs."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def update_status(run_dir: str | Path, role: str, state: str, **extra: Any) -> None:
    payload = {
        "role": role,
        "state": state,
        "pid": os.getpid(),
        "updated_at": time.time(),
        **extra,
    }
    write_json(Path(run_dir) / "roles" / role / "status.json", payload)


def _git(command: list[str]) -> str | None:
    try:
        return subprocess.check_output(
            command, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def init_run(output_root: str | Path, name: str) -> Path:
    timestamp = time.strftime("%Y%m%dT%H%M%S", time.gmtime())
    fingerprint = hashlib.sha256(
        f"{timestamp}:{name}:{os.getpid()}".encode()
    ).hexdigest()[:12]
    run_dir = Path(output_root).expanduser() / f"{timestamp}-{name}-{fingerprint}"
    run_dir.mkdir(parents=True, exist_ok=False)
    write_json(
        run_dir / "provenance" / "run_start.json",
        {
            "name": name,
            "created_at": time.time(),
            "argv": sys.argv,
            "python": sys.executable,
            "platform": platform.platform(),
            "git_commit": _git(["git", "rev-parse", "HEAD"]),
            "git_branch": _git(["git", "branch", "--show-current"]),
            "git_status": _git(["git", "status", "--short"]),
        },
    )
    return run_dir


def seal_run(run_dir: str | Path) -> None:
    run_path = Path(run_dir).expanduser()
    manifest: dict[str, Any] = {"sealed_at": time.time(), "run_dir": str(run_path)}
    for path in sorted(run_path.rglob("*.json")):
        if path.name in {"manifest.json", "SHA256SUMS.json"}:
            continue
        try:
            manifest[str(path.relative_to(run_path))] = json.loads(
                path.read_text(encoding="utf-8")
            )
        except (OSError, json.JSONDecodeError):
            continue
    write_json(run_path / "run_manifest.json", manifest)
    hashes = []
    for path in sorted(run_path.rglob("*")):
        if not path.is_file() or path.name == "SHA256SUMS":
            continue
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        hashes.append(f"{digest}  {path.relative_to(run_path)}")
    (run_path / "SHA256SUMS").write_text("\n".join(hashes) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    init_parser = subparsers.add_parser("init")
    init_parser.add_argument("--output-root", required=True)
    init_parser.add_argument("--name", required=True)
    seal_parser = subparsers.add_parser("seal")
    seal_parser.add_argument("--run-dir", required=True)
    args = parser.parse_args()
    if args.command == "init":
        print(init_run(args.output_root, args.name))
    else:
        seal_run(args.run_dir)


if __name__ == "__main__":
    main()
