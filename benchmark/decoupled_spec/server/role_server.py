"""Internal HTTP subprocess used by the unified Ray server launcher."""

from __future__ import annotations

import argparse
import json
import os
import socket
import sys
import time
from pathlib import Path
from typing import Any

if Path(sys.path[0]).resolve() == Path(__file__).resolve().parent:
    sys.path.pop(0)

from sglang.srt.server_args import ServerArgs


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _status_payload(config: dict[str, Any], state: str, **extra: Any) -> dict[str, Any]:
    args = config["server_args"]
    host = str(args["host"])
    port = int(args["port"])
    if ":" in host and not host.startswith("["):
        display_host = f"[{host}]"
    else:
        display_host = host
    return {
        "engine_id": config["engine_id"],
        "role": config["role"],
        "rank": config["rank"],
        "state": state,
        "pid": os.getpid(),
        "updated_at": time.time(),
        "http_url": f"http://{display_host}:{port}",
        **extra,
    }


def _take_http_socket(fd: int, expected_port: int) -> socket.socket:
    """Take ownership of one inherited listener without re-exporting its FD."""

    http_socket = socket.socket(fileno=fd)
    try:
        http_socket.set_inheritable(False)
        actual_port = int(http_socket.getsockname()[1])
        if actual_port != int(expected_port):
            raise RuntimeError(
                "Inherited HTTP socket port does not match resolved ServerArgs: "
                f"socket={actual_port} config={expected_port}"
            )
        return http_socket
    except BaseException:
        http_socket.close()
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resolved-config", required=True)
    parser.add_argument("--status-file", required=True)
    parser.add_argument("--http-socket-fd", type=int)
    args = parser.parse_args()
    config_path = Path(args.resolved_config)
    status_path = Path(args.status_file)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    runtime = config.get("runtime", {})
    for key, value in runtime.get("env", {}).items():
        os.environ[str(key)] = str(value)

    server_args = ServerArgs(**config["server_args"])
    http_socket = (
        None
        if args.http_socket_fd is None
        else _take_http_socket(args.http_socket_fd, server_args.port)
    )
    _write_json(status_path, _status_payload(config, "starting"))
    failed = False
    try:
        from sglang.srt.entrypoints.http_server import launch_server

        def on_ready() -> None:
            _write_json(status_path, _status_payload(config, "http_ready"))

        launch_server(
            server_args,
            launch_callback=on_ready,
            http_socket=http_socket,
        )
    except BaseException as exc:
        failed = True
        _write_json(status_path, _status_payload(config, "failed", error=repr(exc)))
        raise
    finally:
        if http_socket is not None:
            http_socket.close()
        if not failed:
            _write_json(status_path, _status_payload(config, "exited"))


if __name__ == "__main__":
    main()
