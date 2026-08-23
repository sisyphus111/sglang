"""Launch all decoupled-spec verifier and drafter HTTP servers through Ray."""

from __future__ import annotations

import argparse
import importlib
import json
import signal
import sys
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.artifacts import update_status, write_json

_orchestrator = importlib.import_module("server-side.orchestrator")
RayServerOrchestrator = _orchestrator.RayServerOrchestrator
config_to_dict = _orchestrator.config_to_dict
load_config = _orchestrator.load_config
print_ready_manifest = _orchestrator.print_ready_manifest
validate_server_args_templates = _orchestrator.validate_server_args_templates


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Unified server fleet YAML.")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--ray-address", help="Override ray.address from YAML.")
    parser.add_argument("--ray-namespace", help="Override ray.namespace from YAML.")
    parser.add_argument(
        "--check",
        action="store_true",
        help="Validate and print the config without importing or connecting to Ray.",
    )
    parser.add_argument("--print-resolved-config", action="store_true")
    return parser


def main() -> None:
    args = _parser().parse_args()
    config = load_config(args.config)
    resolved_config = config_to_dict(config)
    if args.ray_address is not None:
        resolved_config["ray"]["address"] = args.ray_address
    if args.ray_namespace is not None:
        resolved_config["ray"]["namespace"] = args.ray_namespace
    if args.print_resolved_config or args.check:
        print(json.dumps(resolved_config, ensure_ascii=False, indent=2))
    if args.check:
        validate_server_args_templates(config)
        return

    run_dir = Path(args.run_dir).expanduser().resolve()
    write_json(run_dir / "server" / "resolved_config.json", resolved_config)
    update_status(
        run_dir,
        "server",
        "starting",
        config=str(Path(args.config).expanduser().resolve()),
    )
    orchestrator = RayServerOrchestrator(
        config,
        config_path=args.config,
        run_dir=run_dir,
        ray_address=args.ray_address,
        ray_namespace=args.ray_namespace,
    )
    stop = threading.Event()

    def request_stop(_signum: int, _frame: object) -> None:
        stop.set()

    previous_handlers = {
        sig: signal.signal(sig, request_stop) for sig in (signal.SIGINT, signal.SIGTERM)
    }
    final_state = "stopped"
    try:
        manifest = orchestrator.start(stop)
        print_ready_manifest(manifest, orchestrator.manifest_path)
        update_status(
            run_dir,
            "server",
            "ready",
            manifest=str(orchestrator.manifest_path),
            engine_count=len(manifest["engines"]),
        )
        orchestrator.wait(stop)
    except InterruptedError:
        final_state = "stopped"
    except BaseException:
        final_state = "failed"
        update_status(run_dir, "server", "failed")
        raise
    finally:
        orchestrator.shutdown(final_state=final_state)
        if final_state != "failed":
            update_status(run_dir, "server", "exited")
        for sig, handler in previous_handlers.items():
            signal.signal(sig, handler)


if __name__ == "__main__":
    main()
