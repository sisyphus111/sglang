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
        config = _read_json(run_dir / "roles" / role / "resolved_config.json") or {}
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
        ready = True
        last = {}
        for role in roles:
            status_path = run_dir / "roles" / role / "status.json"
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
                "run_dir": str(run_dir),
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
    parser.add_argument("--run-dir", required=True)
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
        Path(args.run_dir).expanduser().resolve(),
        args.roles or ["verifier", "drafter"],
        args.timeout_s,
        args.poll_interval_s,
        args.check_http,
        args.http_timeout_s,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
