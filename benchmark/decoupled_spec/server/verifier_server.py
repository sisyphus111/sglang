"""Launch only the verifier HTTP server for an agent-driven run."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

if Path(sys.path[0]).resolve() == Path(__file__).resolve().parent:
    sys.path.pop(0)
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from decoupled_spec.run_io import require_run_dir, update_status, write_json
from decoupled_spec.server.config import (
    add_role_cli_args,
    build_server_args,
    resolve_role_config,
    role_cli_overrides,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--runtime-dir", required=True)
    add_role_cli_args(parser)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--print-resolved-config", action="store_true")
    return parser


def main() -> None:
    cli = _parser().parse_args()
    resolved = resolve_role_config(cli.config, "verifier", role_cli_overrides(cli))
    runtime = resolved.get("runtime", {})
    for key, value in runtime.get("env", {}).items():
        os.environ[key] = str(value)
    visible = runtime.get("cuda_visible_devices")
    if visible:
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(visible)
    if cli.print_resolved_config or cli.check:
        print(json.dumps(resolved, ensure_ascii=False, indent=2, default=str))
    if cli.check:
        build_server_args(resolved)
        return
    run_dir = require_run_dir(cli.runtime_dir)
    role_dir = run_dir / "server" / "verifier"
    role_dir.mkdir(parents=True, exist_ok=True)
    write_json(role_dir / "resolved_config.json", resolved)
    update_status(
        run_dir, "server/verifier", "starting", config=str(Path(cli.config).resolve())
    )
    try:
        from sglang.srt.entrypoints.http_server import launch_server

        server_args = build_server_args(resolved)

        def on_ready() -> None:
            update_status(
                run_dir,
                "server/verifier",
                "http_ready",
                host=server_args.host,
                port=server_args.port,
                pid=os.getpid(),
            )

        launch_server(server_args, launch_callback=on_ready)
    except BaseException as exc:
        update_status(run_dir, "server/verifier", "failed", error=repr(exc))
        raise
    finally:
        update_status(run_dir, "server/verifier", "exited")


if __name__ == "__main__":
    main()
