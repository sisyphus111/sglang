#!/usr/bin/env python3
"""Wait until decoupled-spec role status files and HTTP health checks are ready."""

from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None
    return value if isinstance(value, dict) else None


def _role_url(run_dir: Path, role: str, status: dict[str, Any]) -> str | None:
    host = status.get("host")
    port = status.get("port")
    if host is None or port is None:
        config = _read_json(run_dir / "server" / role / "resolved_config.json") or {}
        server_args = config.get("server_args", {})
        if isinstance(server_args, dict):
            host = server_args.get("host")
            port = server_args.get("port")
    if host is None or port is None:
        return None
    connect_host = "127.0.0.1" if str(host) in {"0.0.0.0", "::"} else str(host)
    return f"http://{connect_host}:{int(port)}"


def _http_ready(base_url: str, role: str, timeout_s: float) -> tuple[bool, str | None]:
    # The drafter deliberately rejects user-generation work, including the
    # generic generation-based health probe. /model_info proves that its HTTP
    # process is serving after the launcher has recorded http_ready.
    path = "/model_info" if role == "drafter" else "/health"
    try:
        with urllib.request.urlopen(  # noqa: S310 - URL comes from saved local config.
            base_url.rstrip("/") + path, timeout=timeout_s
        ) as response:
            status = int(response.status)
        return 200 <= status < 300, None if 200 <= status < 300 else f"HTTP {status}"
    except (OSError, urllib.error.URLError) as exc:
        return False, repr(exc)


def wait_for_roles(
    run_dir: Path,
    roles: list[str],
    timeout_s: float,
    poll_interval_s: float,
    check_http: bool,
    http_timeout_s: float,
) -> dict[str, Any]:
    if timeout_s <= 0 or poll_interval_s <= 0 or http_timeout_s <= 0:
        raise ValueError("timeouts and poll interval must be positive")
    started = time.monotonic()
    last: dict[str, Any] = {}
    while True:
        manifest_path = run_dir / "server" / "manifest.json"
        manifest = _read_json(manifest_path)
        if manifest_path.exists() and manifest is None:
            raise RuntimeError(f"cannot read unified server manifest: {manifest_path}")
        if manifest is not None:
            if manifest.get("schema_version") != 1:
                raise RuntimeError(
                    "unified server manifest schema_version must be 1, got "
                    f"{manifest.get('schema_version')!r}"
                )
            state = manifest.get("state")
            if state in {"failed", "stopping", "stopped"}:
                raise RuntimeError(
                    f"unified server fleet entered terminal state {state!r}: "
                    f"{manifest}"
                )
            engines = manifest.get("engines")
            if state == "ready" and isinstance(engines, list) and engines:
                ready = True
                targets = {}
                seen_roles = set()
                seen_role_ranks = set()
                physical_gpu_owners = set()
                for engine in engines:
                    if not isinstance(engine, dict):
                        raise RuntimeError(
                            f"invalid unified server engine entry: {engine!r}"
                        )
                    engine_id = str(engine.get("engine_id", ""))
                    role = str(engine.get("role", ""))
                    rank = engine.get("rank")
                    if role not in {"verifier", "drafter"}:
                        raise RuntimeError(
                            f"invalid unified server engine role: {engine!r}"
                        )
                    if type(rank) is not int or rank < 0:
                        raise RuntimeError(
                            f"invalid unified server engine rank: {engine!r}"
                        )
                    if engine_id in targets or (role, rank) in seen_role_ranks:
                        raise RuntimeError(
                            "duplicate unified server engine identity: "
                            f"engine_id={engine_id!r} role={role} rank={rank}"
                        )
                    seen_role_ranks.add((role, rank))
                    seen_roles.add(role)
                    tp_size = engine.get("tp_size")
                    node_actors = engine.get("node_actors")
                    rank_placements = engine.get("rank_placements")
                    if (
                        type(tp_size) is not int
                        or tp_size <= 0
                        or not isinstance(node_actors, list)
                        or not node_actors
                        or not isinstance(rank_placements, list)
                        or len(rank_placements) != tp_size
                    ):
                        raise RuntimeError(
                            f"invalid unified server Engine placement: {engine!r}"
                        )
                    for expected_node_rank, node_actor in enumerate(node_actors):
                        if (
                            not isinstance(node_actor, dict)
                            or node_actor.get("node_rank") != expected_node_rank
                            or not isinstance(node_actor.get("gpu_ids"), list)
                            or not node_actor["gpu_ids"]
                        ):
                            raise RuntimeError(
                                f"invalid Engine node actor: {node_actor!r}"
                            )
                        for gpu_id in node_actor["gpu_ids"]:
                            owner = (node_actor.get("node_id"), str(gpu_id))
                            if owner in physical_gpu_owners:
                                raise RuntimeError(f"GPU placement is reused: {owner}")
                            physical_gpu_owners.add(owner)
                    for tp_rank, placement in enumerate(rank_placements):
                        if (
                            not isinstance(placement, dict)
                            or placement.get("tp_rank") != tp_rank
                        ):
                            raise RuntimeError(
                                f"invalid TP{tp_rank} placement: {placement!r}"
                            )
                    base_url = engine.get("http_url")
                    record = {
                        "role": role,
                        "rank": rank,
                        "base_url": base_url,
                        "rank_placements": rank_placements,
                    }
                    if not engine_id or not isinstance(base_url, str):
                        ready = False
                        record["health_error"] = "manifest identity/http_url missing"
                    elif check_http:
                        healthy, error = _http_ready(base_url, role, http_timeout_s)
                        record["http_ready"] = healthy
                        record["health_error"] = error
                        ready = ready and healthy
                    targets[engine_id] = record
                topology = manifest.get("topology")
                if not isinstance(topology, dict):
                    raise RuntimeError("unified server manifest lacks topology")
                for role, count_field in (
                    ("verifier", "num_verifiers"),
                    ("drafter", "num_drafters"),
                ):
                    actual = sum(target["role"] == role for target in targets.values())
                    if topology.get(count_field) != actual:
                        raise RuntimeError(
                            f"unified server topology {count_field} mismatch: "
                            f"recorded={topology.get(count_field)!r}, actual={actual}"
                        )
                if not targets or not set(roles).issubset(seen_roles):
                    ready = False
                last = targets
                if ready:
                    return {
                        "ready": True,
                        "runtime_dir": str(run_dir),
                        "manifest_path": str(manifest_path),
                        "elapsed_s": time.monotonic() - started,
                        "engines": targets,
                    }
            else:
                last = {"server_manifest": {"state": state}}
            if time.monotonic() - started >= timeout_s:
                raise TimeoutError(
                    "unified server fleet did not become ready within "
                    f"{timeout_s}s; last={last}"
                )
            time.sleep(poll_interval_s)
            continue

        ready = True
        last = {}
        for role in roles:
            status_path = run_dir / "server" / role / "status.json"
            status = _read_json(status_path)
            state = status.get("state") if status else None
            record: dict[str, Any] = {
                "status_path": str(status_path),
                "state": state,
                "pid": status.get("pid") if status else None,
            }
            if state in {"failed", "exited"}:
                raise RuntimeError(
                    f"{role} entered terminal state {state!r} before readiness: {status}"
                )
            if state != "http_ready":
                ready = False
            url = _role_url(run_dir, role, status or {})
            record["base_url"] = url
            if state == "http_ready" and check_http:
                if url is None:
                    ready = False
                    record["health_error"] = "host/port missing from status and config"
                else:
                    healthy, error = _http_ready(url, role, http_timeout_s)
                    record["http_ready"] = healthy
                    record["health_error"] = error
                    ready = ready and healthy
            last[role] = record
        if ready:
            return {
                "ready": True,
                "runtime_dir": str(run_dir),
                "elapsed_s": time.monotonic() - started,
                "roles": last,
            }
        if time.monotonic() - started >= timeout_s:
            raise TimeoutError(
                f"roles did not become ready within {timeout_s}s; last={last}"
            )
        time.sleep(poll_interval_s)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-dir", required=True)
    parser.add_argument(
        "--role",
        action="append",
        choices=("verifier", "drafter"),
        dest="roles",
        help="Role to wait for; repeat to override the default pair.",
    )
    parser.add_argument("--timeout-s", type=float, default=600.0)
    parser.add_argument("--poll-interval-s", type=float, default=1.0)
    parser.add_argument("--http-timeout-s", type=float, default=1.0)
    parser.add_argument(
        "--check-http",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    args = parser.parse_args()
    report = wait_for_roles(
        Path(args.runtime_dir).expanduser().resolve(),
        args.roles or ["verifier", "drafter"],
        args.timeout_s,
        args.poll_interval_s,
        args.check_http,
        args.http_timeout_s,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
