"""Role YAML loading, named CLI overrides, and strict pair validation."""

from __future__ import annotations

import argparse
import copy
import dataclasses
import json
from pathlib import Path
from typing import Any

import yaml

from sglang.srt.server_args import ServerArgs

SCHEMA_VERSION = 1
_ROLE_KEYS = {"schema_version", "role", "runtime", "server_args"}
_RUNTIME_KEYS = {"cuda_visible_devices", "env"}
_CPP_DATA_PLANE_ENV = "SGLANG_DECOUPLED_SPEC_USE_CPP_PYBIND"


def load_yaml(path: str | Path) -> dict[str, Any]:
    value = yaml.safe_load(Path(path).expanduser().read_text(encoding="utf-8"))
    if value is None:
        value = {}
    if not isinstance(value, dict):
        raise ValueError(f"config must be a YAML mapping: {path}")
    if int(value.get("schema_version", SCHEMA_VERSION)) != SCHEMA_VERSION:
        raise ValueError(
            f"unsupported schema_version in {path}; expected {SCHEMA_VERSION}"
        )
    return value


def add_role_cli_args(
    parser: argparse.ArgumentParser, prefix: str | None = None
) -> None:
    """Add role-local deployment overrides with argparse-visible types."""
    option_prefix = f"{prefix}-" if prefix else ""
    dest_prefix = f"{prefix}_" if prefix else ""
    parser.add_argument(
        f"--{option_prefix}model-path",
        dest=f"{dest_prefix}model_path",
        help="Override server_args.model_path from YAML.",
    )
    parser.add_argument(
        f"--{option_prefix}tp-size",
        dest=f"{dest_prefix}tp_size",
        type=int,
        help="Override server_args.tp_size from YAML.",
    )
    parser.add_argument(
        f"--{option_prefix}cuda-visible-devices",
        dest=f"{dest_prefix}cuda_visible_devices",
        nargs="+",
        help="Override runtime.cuda_visible_devices from YAML.",
    )
    parser.add_argument(
        f"--{option_prefix}host",
        dest=f"{dest_prefix}host",
        help="Override server_args.host from YAML.",
    )
    parser.add_argument(
        f"--{option_prefix}port",
        dest=f"{dest_prefix}port",
        type=int,
        help="Override server_args.port from YAML.",
    )


def role_cli_overrides(
    args: argparse.Namespace, prefix: str | None = None
) -> dict[str, dict[str, Any]]:
    """Return only explicitly supplied values so YAML remains the baseline."""
    dest_prefix = f"{prefix}_" if prefix else ""
    server_args = {
        field: value
        for field in ("model_path", "tp_size", "host", "port")
        if (value := getattr(args, f"{dest_prefix}{field}")) is not None
    }
    runtime = {}
    visible = getattr(args, f"{dest_prefix}cuda_visible_devices")
    if visible is not None:
        runtime["cuda_visible_devices"] = visible
    return {"runtime": runtime, "server_args": server_args}


def resolve_role_config(
    path: str | Path,
    role: str,
    cli_overrides: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    config = load_yaml(path)
    unknown = set(config) - _ROLE_KEYS
    if unknown:
        raise ValueError(f"unknown config keys in {path}: {sorted(unknown)}")
    if config.get("role") != role:
        raise ValueError(
            f"{path} declares role={config.get('role')!r}, expected {role!r}"
        )
    config.setdefault("runtime", {})
    config.setdefault("server_args", {})
    if not isinstance(config["runtime"], dict) or not isinstance(
        config["server_args"], dict
    ):
        raise ValueError("runtime and server_args must be mappings")
    unknown_runtime = set(config["runtime"]) - _RUNTIME_KEYS
    if unknown_runtime:
        raise ValueError(f"unknown runtime keys: {sorted(unknown_runtime)}")

    resolved = copy.deepcopy(config)
    resolved["schema_version"] = SCHEMA_VERSION
    for section, values in (cli_overrides or {}).items():
        if section not in {"runtime", "server_args"}:
            raise ValueError(f"unknown CLI override section: {section}")
        resolved[section].update(values)

    fields = {field.name for field in dataclasses.fields(ServerArgs)}
    unknown_args = set(resolved["server_args"]) - fields
    if unknown_args:
        raise ValueError(f"unknown ServerArgs fields: {sorted(unknown_args)}")

    args = resolved["server_args"]
    if not args.get("model_path"):
        raise ValueError(f"{path}: server_args.model_path is required")
    if args.get("decoupled_spec_role") != role:
        raise ValueError(f"{path}: server_args.decoupled_spec_role must be {role!r}")
    if args.get("decoupled_spec_rank") is None:
        raise ValueError(f"{path}: decoupled_spec_rank is required")
    if args.get("decoupled_spec_bind_endpoint") is None:
        raise ValueError(f"{path}: decoupled_spec_bind_endpoint is required")
    if args.get("decoupled_spec_connect_endpoints") is None:
        raise ValueError(f"{path}: decoupled_spec_connect_endpoints is required")
    visible = resolved["runtime"].get("cuda_visible_devices")
    if visible is not None:
        if not isinstance(visible, list) or not visible:
            raise ValueError("runtime.cuda_visible_devices must be a non-empty list")
        resolved["runtime"]["cuda_visible_devices"] = [str(item) for item in visible]
    env = resolved["runtime"].get("env", {})
    if not isinstance(env, dict) or any(not isinstance(k, str) for k in env):
        raise ValueError("runtime.env must be a string-keyed mapping")
    return resolved


def build_server_args(resolved: dict[str, Any]) -> ServerArgs:
    return ServerArgs(**resolved["server_args"])


def validate_role_pair(verifier: dict[str, Any], drafter: dict[str, Any]) -> None:
    verifier_args = verifier["server_args"]
    drafter_args = drafter["server_args"]

    verifier_algorithm = verifier_args.get("speculative_algorithm")
    if not isinstance(verifier_algorithm, str) or (
        verifier_algorithm.upper() != "DECOUPLED_VERIFY"
    ):
        raise ValueError("verifier speculative_algorithm must be DECOUPLED_VERIFY")
    if drafter_args.get("speculative_algorithm") is not None:
        raise ValueError("drafter speculative_algorithm must be unset")

    for field in (
        "speculative_num_steps",
        "speculative_eagle_topk",
        "speculative_num_draft_tokens",
    ):
        if verifier_args.get(field) != drafter_args.get(field):
            raise ValueError(
                f"role configs disagree on {field}: "
                f"verifier={verifier_args.get(field)!r}, "
                f"drafter={drafter_args.get(field)!r}"
            )

    num_steps = verifier_args.get("speculative_num_steps")
    if not isinstance(num_steps, int) or num_steps <= 0:
        raise ValueError("speculative_num_steps must be a positive integer")
    if verifier_args.get("speculative_eagle_topk") != 1:
        raise ValueError("phase-one decoupled speculation requires F=1 (topk=1)")
    if verifier_args.get("speculative_num_draft_tokens") != num_steps + 1:
        raise ValueError(
            "speculative_num_draft_tokens must equal speculative_num_steps + 1"
        )
    if not drafter_args.get("disable_overlap_schedule", False):
        raise ValueError("drafter must set disable_overlap_schedule=true")

    def cpp_data_plane_enabled(config: dict[str, Any]) -> bool:
        value = config.get("runtime", {}).get("env", {}).get(_CPP_DATA_PLANE_ENV, "0")
        normalized = str(value).strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"", "0", "false", "no", "off"}:
            return False
        raise ValueError(f"invalid {_CPP_DATA_PLANE_ENV} value: {value!r}")

    if cpp_data_plane_enabled(verifier) != cpp_data_plane_enabled(drafter):
        raise ValueError(
            "verifier and drafter must use the same Python/C++ data-plane backend"
        )

    verifier_bind = verifier_args["decoupled_spec_bind_endpoint"]
    drafter_bind = drafter_args["decoupled_spec_bind_endpoint"]
    if drafter_bind not in verifier_args["decoupled_spec_connect_endpoints"]:
        raise ValueError("verifier does not connect to drafter bind endpoint")
    if verifier_bind not in drafter_args["decoupled_spec_connect_endpoints"]:
        raise ValueError("drafter does not connect to verifier bind endpoint")
    verifier_gpus = set(verifier.get("runtime", {}).get("cuda_visible_devices", []))
    drafter_gpus = set(drafter.get("runtime", {}).get("cuda_visible_devices", []))
    if verifier_gpus and drafter_gpus and verifier_gpus & drafter_gpus:
        raise ValueError("verifier and drafter CUDA_VISIBLE_DEVICES overlap")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["validate"])
    parser.add_argument("--verifier-config", required=True)
    parser.add_argument("--drafter-config", required=True)
    add_role_cli_args(parser, "verifier")
    add_role_cli_args(parser, "drafter")
    return parser


def main() -> None:
    args = _parser().parse_args()
    verifier = resolve_role_config(
        args.verifier_config,
        "verifier",
        role_cli_overrides(args, "verifier"),
    )
    drafter = resolve_role_config(
        args.drafter_config,
        "drafter",
        role_cli_overrides(args, "drafter"),
    )
    validate_role_pair(verifier, drafter)
    print(
        json.dumps({"valid": True, "verifier": verifier, "drafter": drafter}, indent=2)
    )


if __name__ == "__main__":
    main()
