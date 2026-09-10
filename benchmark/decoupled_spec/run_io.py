"""Small shared IO helpers for component-owned benchmark directories."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any


def require_run_dir(run_dir: str | Path) -> Path:
    """Resolve a RUN_DIR that the user created before launching components."""

    path = Path(run_dir).expanduser().resolve()
    if not path.is_dir():
        raise FileNotFoundError(
            f"RUN_DIR does not exist; create it before launching components: {path}"
        )
    return path


def write_json(path: str | Path, value: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(target)


def update_run_config(
    run_dir: str | Path, section: str, config: dict[str, Any]
) -> dict[str, Any]:
    """Atomically merge the fixed server/client sections in RUN_DIR/config.json."""

    if section not in {"server", "client"}:
        raise ValueError(f"unsupported run config section: {section!r}")
    path = require_run_dir(run_dir) / "config.json"
    if path.exists():
        value = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(value, dict) or set(value) - {"server", "client"}:
            raise ValueError(f"invalid run config: {path}")
    else:
        value = {}
    existing = value.get(section)
    if existing is not None and existing != config:
        raise ValueError(
            f"RUN_DIR/config.json already has a different {section} config"
        )
    value[section] = config
    ordered = {key: value[key] for key in ("server", "client") if key in value}
    write_json(path, ordered)
    return ordered


def update_status(
    run_dir: str | Path, component_dir: str | Path, state: str, **extra: Any
) -> None:
    relative = Path(component_dir)
    if relative == Path(".") or relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"component_dir must stay inside RUN_DIR: {component_dir!r}")
    payload = {
        "component": relative.as_posix(),
        "state": state,
        "pid": os.getpid(),
        "updated_at": time.time(),
        **extra,
    }
    write_json(require_run_dir(run_dir) / relative / "status.json", payload)
