#!/usr/bin/env python3
"""Generate a decoupled-verifier throughput profile from target-only startup."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import signal
import socket
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

from profile_common import (
    ProfileSpec,
    build_engine_kwargs,
    detect_gpu_name,
    load_spec,
    validate_profile,
)


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", dir=path.parent, prefix=path.name + ".", suffix=".tmp", delete=False
    ) as file:
        tmp_path = Path(file.name)
        json.dump(payload, file, indent=2, sort_keys=True)
        file.write("\n")
    os.replace(tmp_path, path)


def _git_provenance(repo: Path) -> dict[str, Any]:
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repo, text=True
    ).strip()
    status = subprocess.check_output(
        ["git", "status", "--porcelain"], cwd=repo, text=True
    )
    diff = subprocess.check_output(["git", "diff", "--binary", "HEAD"], cwd=repo)
    return {
        "repo": str(repo),
        "commit": commit,
        "dirty": bool(status.strip()),
        "status": status.splitlines(),
        "tracked_diff_sha256": hashlib.sha256(diff).hexdigest(),
    }


def _reserve_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _stop_process_group(process: subprocess.Popen[Any]) -> None:
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=60)
        return
    except subprocess.TimeoutExpired:
        pass
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        return
    process.wait(timeout=60)


def _worker(spec: ProfileSpec) -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(spec.gpus)
    repo_python = str(spec.repo / "python")
    if repo_python not in sys.path:
        sys.path.insert(0, repo_python)
    import sglang as sgl

    dist_init_addr = f"127.0.0.1:{_reserve_port()}"
    kwargs = build_engine_kwargs(spec, dist_init_addr)
    print("engine_kwargs=" + json.dumps(kwargs, sort_keys=True), flush=True)
    engine = None
    try:
        engine = sgl.Engine(**kwargs)
    finally:
        if engine is not None:
            engine.shutdown()


def _manifest_base(spec: ProfileSpec, config_bytes: bytes) -> dict[str, Any]:
    normalized = asdict(spec)
    for key in ("config_path", "repo", "output_dir"):
        normalized[key] = str(normalized[key])
    normalized["gpus"] = list(spec.gpus)
    normalized["batch_sizes"] = list(spec.batch_sizes)
    normalized["ctx_lens"] = list(spec.ctx_lens)
    normalized["steps"] = list(spec.steps)
    return {
        "status": "started",
        "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "config_sha256": hashlib.sha256(config_bytes).hexdigest(),
        "spec": normalized,
        "git": _git_provenance(spec.repo),
    }


def _run_parent(spec: ProfileSpec, force: bool) -> None:
    gpu_name = detect_gpu_name(spec.gpus)
    spec.output_dir.mkdir(parents=True, exist_ok=True)
    config_bytes = spec.config_path.read_bytes()
    lock_path = spec.output_dir / "profile-config.lock.toml"
    if lock_path.exists() and lock_path.read_bytes() != config_bytes:
        raise ValueError(f"config differs from immutable lock: {lock_path}")
    if not lock_path.exists():
        lock_path.write_bytes(config_bytes)

    if spec.profile_path.exists() and not force:
        try:
            validate_profile(spec, expected_gpu_name=gpu_name)
        except Exception as error:
            raise ValueError(
                f"existing profile is not valid for this grid: {error}; "
                "use --force only with explicit overwrite approval"
            ) from error

    manifest_path = spec.output_dir / "manifest.json"
    manifest = _manifest_base(spec, config_bytes)
    _atomic_write_json(manifest_path, manifest)
    if force and spec.profile_path.exists():
        backup = spec.profile_path.with_suffix(
            f".json.before-force-{time.strftime('%Y%m%d-%H%M%S')}"
        )
        spec.profile_path.replace(backup)
        manifest["profile_backup"] = str(backup)

    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--config",
        str(lock_path),
        "--worker",
    ]
    manifest["command"] = command
    _atomic_write_json(manifest_path, manifest)
    log_path = spec.output_dir / "profile.log"
    env = os.environ.copy()
    repo_python = str(spec.repo / "python")
    env["PYTHONPATH"] = os.pathsep.join(
        value for value in (repo_python, env.get("PYTHONPATH")) if value
    )
    with log_path.open("w") as log:
        log.write("# command: " + shlex.join(command) + "\n")
        log.flush()
        process = subprocess.Popen(
            command,
            cwd=spec.repo,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            text=True,
        )
        try:
            returncode = process.wait(timeout=spec.timeout_s)
        except subprocess.TimeoutExpired:
            _stop_process_group(process)
            manifest.update(status="timeout", finished_at=time.strftime("%F %T"))
            _atomic_write_json(manifest_path, manifest)
            raise
        except BaseException:
            _stop_process_group(process)
            manifest.update(status="interrupted", finished_at=time.strftime("%F %T"))
            _atomic_write_json(manifest_path, manifest)
            raise
    if returncode != 0:
        manifest.update(
            status="failed",
            returncode=returncode,
            finished_at=time.strftime("%F %T"),
        )
        _atomic_write_json(manifest_path, manifest)
        raise RuntimeError(
            f"profile worker failed with return code {returncode}: {log_path}"
        )
    try:
        validation = validate_profile(spec, expected_gpu_name=gpu_name)
    except Exception as error:
        manifest.update(
            status="invalid_profile",
            error=str(error),
            finished_at=time.strftime("%F %T"),
        )
        _atomic_write_json(manifest_path, manifest)
        raise
    profile_bytes = spec.profile_path.read_bytes()
    manifest.update(
        status="ok",
        returncode=0,
        finished_at=time.strftime("%F %T"),
        profile_sha256=hashlib.sha256(profile_bytes).hexdigest(),
        validation=validation,
    )
    _atomic_write_json(manifest_path, manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--check", action="store_true", help="Validate config without using GPUs."
    )
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    spec = load_spec(args.config)
    if args.check:
        print(
            json.dumps(
                {
                    "engine_kwargs": build_engine_kwargs(spec, "127.0.0.1:<auto>"),
                    "required_points": len(spec.required_points),
                },
                indent=2,
                sort_keys=True,
            )
        )
    elif args.worker:
        _worker(spec)
    else:
        _run_parent(spec, args.force)


if __name__ == "__main__":
    main()
