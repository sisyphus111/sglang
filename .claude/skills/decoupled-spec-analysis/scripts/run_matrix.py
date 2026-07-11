#!/usr/bin/env python3
"""Config-driven runner for decoupled-spec profile generation and case matrices."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import signal
import subprocess
import sys
import time
import tomllib
from pathlib import Path
from typing import Any

from analysis_common import ProfileTable


DEBUG_ENV_KEYS = (
    "SGLANG_TA_DEBUG",
    "SGLANG_LOG_FORWARD_ITERS",
    "SGLANG_RECORD_STEP_TIME",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="action", required=True)
    for action in ("profile", "run"):
        sub = subparsers.add_parser(action)
        sub.add_argument("--config", type=Path, required=True)
        sub.add_argument("--run-dir", type=Path, required=True)
        sub.add_argument("--force", action="store_true")
        sub.add_argument("--timeout-s", type=float, default=8 * 3600)
        if action == "profile":
            sub.add_argument("--poll-s", type=float, default=15.0)
    return parser.parse_args()


def load_config(path: Path) -> dict[str, Any]:
    with path.open("rb") as file:
        config = tomllib.load(file)
    required_sections = {"paths", "parallelism", "workload", "profile", "matrix"}
    missing = sorted(required_sections - set(config))
    if missing:
        raise ValueError(f"config missing sections: {', '.join(missing)}")
    workload = config["workload"]
    batch_size = int(workload["batch_size"])
    max_running = int(workload["max_running_requests"])
    if max_running < batch_size + 1:
        raise ValueError(
            f"max_running_requests={max_running} must be >= batch_size+1={batch_size + 1}"
        )
    if bool(workload.get("ignore_eos", False)):
        print("warning: ignore_eos=true changes request outputs", file=sys.stderr)
    if workload.get("sampling_seed") is not None and not isinstance(
        workload["sampling_seed"], int
    ):
        raise ValueError("workload.sampling_seed must be an integer")
    if "deterministic" in workload and not isinstance(
        workload["deterministic"], bool
    ):
        raise ValueError("workload.deterministic must be a boolean")
    capture_bs = [int(value) for value in config["profile"]["capture_bs"]]
    if batch_size not in capture_bs:
        raise ValueError(f"profile.capture_bs must contain workload batch_size={batch_size}")
    adaptive = config.get("adaptive", {})
    strategy = str(adaptive.get("strategy", "throughput_aware"))
    if strategy not in {"ema", "throughput_aware"}:
        raise ValueError(
            f"adaptive.strategy must be ema or throughput_aware, got {strategy!r}"
        )
    if strategy == "ema" and not adaptive.get("config"):
        raise ValueError("adaptive.config is required when adaptive.strategy=ema")
    return config


def write_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as file:
        file.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        file.flush()


def now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def stop_process_group(process: subprocess.Popen[Any]) -> None:
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


def build_env(config: dict[str, Any], allow_partial: bool) -> dict[str, str]:
    env = os.environ.copy()
    repo_python = str(Path(config["paths"]["repo"]) / "python")
    ambient_pythonpath = env.get("PYTHONPATH")
    env["PYTHONPATH"] = (
        os.pathsep.join((repo_python, ambient_pythonpath))
        if ambient_pythonpath
        else repo_python
    )
    env["PYTHONUNBUFFERED"] = "1"
    env["SGLANG_DECOUPLED_SPEC_ALLOW_PARTIAL"] = "1" if allow_partial else "0"
    configured = config.get("environment", {})
    for key in DEBUG_ENV_KEYS:
        value = str(configured.get(key, ""))
        if value:
            env[key] = value
        else:
            env.pop(key, None)
    return env


def base_command(
    config: dict[str, Any],
    output_dir: Path,
    steps: int,
    dynamic: bool,
    *,
    adaptive_strategy: str | None = None,
) -> list[str]:
    paths = config["paths"]
    parallelism = config["parallelism"]
    workload = config["workload"]
    profile = config["profile"]
    repo = Path(paths["repo"])
    entrypoint = Path(paths["entrypoint"])
    if not entrypoint.is_absolute():
        entrypoint = repo / entrypoint
    command = [
        sys.executable,
        str(entrypoint),
        "--dataset-path",
        str(paths["dataset"]),
        "--dataset-format",
        str(workload["dataset_format"]),
        "--batch-size",
        str(workload["batch_size"]),
        "--baseline",
        "none",
        "--target-model-path",
        str(paths["target_model"]),
        "--draft-model-path",
        str(paths["draft_model"]),
        "--target-tp-size",
        str(parallelism["target_tp_size"]),
        "--draft-tp-size",
        str(parallelism["draft_tp_size"]),
        "--n-gpu-per-node",
        str(parallelism["n_gpu_per_node"]),
        "--verify-ngpus",
        str(parallelism["verify_ngpus"]),
        "--draft-ngpus",
        str(parallelism["draft_ngpus"]),
        "--max-running-requests",
        str(workload["max_running_requests"]),
        "--cuda-graph-bs-decode",
        ",".join(str(value) for value in profile["capture_bs"]),
        "--max-new-tokens",
        str(workload["max_new_tokens"]),
        "--num-speculative-steps",
        str(steps),
        "--temperature",
        str(workload["temperature"]),
        "--decode-log-interval",
        str(workload.get("decode_log_interval", 1)),
        "--output-dir",
        str(output_dir),
    ]
    if workload.get("sampling_seed") is not None:
        command.extend(["--sampling-seed", str(workload["sampling_seed"])])
    if bool(workload.get("enable_thinking", False)):
        command.append("--enable-thinking")
    if bool(workload.get("deterministic", False)):
        command.append("--deterministic")
    if bool(workload.get("ignore_eos", False)):
        command.append("--ignore-eos")
    if dynamic:
        adaptive = config.get("adaptive", {})
        strategy = adaptive_strategy or str(
            adaptive.get("strategy", "throughput_aware")
        )
        command.extend(
            ["--speculative-adaptive", "--speculative-adaptive-strategy", strategy]
        )
        if strategy == "throughput_aware":
            command.extend(
                [
                    "--decoupled-verify-throughput-profile-path",
                    str(paths["profile"]),
                    "--decoupled-verify-throughput-profile-ctx-lens",
                    ",".join(str(value) for value in profile["ctx_lens"]),
                ]
            )
        elif strategy == "ema":
            adaptive_config = Path(str(adaptive["config"]))
            if not adaptive_config.is_absolute():
                adaptive_config = repo / adaptive_config
            command.extend(["--speculative-adaptive-config", str(adaptive_config)])
        else:
            raise ValueError(f"Unsupported adaptive strategy: {strategy!r}")
    for value in config.get("extra_args", {}).get("common", []):
        command.append(str(value))
    return command


def profile_complete(config: dict[str, Any]) -> tuple[bool, str]:
    path = Path(config["paths"]["profile"])
    if not path.exists():
        return False, "profile file does not exist"
    try:
        table = ProfileTable.load(path)
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as error:
        return False, f"invalid profile: {error}"
    profile = config["profile"]
    missing = table.missing_points(
        profile["capture_bs"], profile["steps"], profile["ctx_lens"]
    )
    if missing:
        return False, f"missing {len(missing)} points; first={missing[0]}"
    return True, f"complete with {len(table.data)} points"


def run_process(
    *,
    config: dict[str, Any],
    command: list[str],
    log_path: Path,
    allow_partial: bool,
    timeout_s: float,
) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    repo = Path(config["paths"]["repo"])
    env = build_env(config, allow_partial)
    with log_path.open("w") as log_file:
        log_file.write("# command: " + shlex.join(command) + "\n")
        log_file.write(
            "# env: "
            + json.dumps(
                {
                    "PYTHONPATH": env["PYTHONPATH"],
                    "SGLANG_DECOUPLED_SPEC_ALLOW_PARTIAL": env[
                        "SGLANG_DECOUPLED_SPEC_ALLOW_PARTIAL"
                    ],
                    **{key: env.get(key) for key in DEBUG_ENV_KEYS},
                },
                sort_keys=True,
            )
            + "\n"
        )
        log_file.flush()
        process = subprocess.Popen(
            command,
            cwd=repo,
            env=env,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            text=True,
        )
        try:
            return process.wait(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            stop_process_group(process)
            raise


def generate_profile(
    config: dict[str, Any], run_dir: Path, force: bool, timeout_s: float, poll_s: float
) -> None:
    status_path = run_dir / "status.jsonl"
    complete, reason = profile_complete(config)
    if complete and not force:
        write_jsonl(status_path, {"ts": now(), "status": "profile_skipped", "reason": reason})
        print(reason)
        return

    profile_path = Path(config["paths"]["profile"])
    if force and profile_path.exists():
        backup_path = profile_path.with_suffix(
            profile_path.suffix + f".before-force-{time.strftime('%Y%m%d-%H%M%S')}"
        )
        profile_path.replace(backup_path)
        write_jsonl(
            status_path,
            {
                "ts": now(),
                "status": "profile_backup_before_force",
                "profile": str(profile_path),
                "backup": str(backup_path),
            },
        )

    profile = config["profile"]
    max_steps = max(int(value) for value in profile["steps"])
    label = f"profile_probe_dl{max_steps}"
    output_dir = run_dir / "runs" / label
    log_path = run_dir / "logs" / f"{label}.log"
    output_dir.mkdir(parents=True, exist_ok=True)
    command = base_command(
        config,
        output_dir,
        max_steps,
        dynamic=True,
        adaptive_strategy="throughput_aware",
    )
    if "mem_fraction_static" in profile:
        command.extend(["--mem-fraction-static", str(profile["mem_fraction_static"])])
    if "prefill_budget" in profile:
        command.extend(
            [
                "--chunked-prefill-size",
                str(profile["prefill_budget"]),
                "--max-prefill-tokens",
                str(profile["prefill_budget"]),
            ]
        )
    env = build_env(config, allow_partial=False)
    repo = Path(config["paths"]["repo"])
    log_path.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(
        status_path,
        {"ts": now(), "status": "profile_started", "command": command, "log": str(log_path)},
    )
    start = time.time()
    with log_path.open("w") as log_file:
        log_file.write("# command: " + shlex.join(command) + "\n")
        log_file.flush()
        process = subprocess.Popen(
            command,
            cwd=repo,
            env=env,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            text=True,
        )
        try:
            while time.time() - start < timeout_s:
                complete, reason = profile_complete(config)
                if complete:
                    write_jsonl(
                        status_path,
                        {
                            "ts": now(),
                            "status": "profile_ok",
                            "elapsed_s": time.time() - start,
                            "reason": reason,
                        },
                    )
                    stop_process_group(process)
                    print(reason)
                    return
                if process.poll() is not None:
                    break
                time.sleep(poll_s)
        finally:
            stop_process_group(process)
    complete, reason = profile_complete(config)
    write_jsonl(
        status_path,
        {
            "ts": now(),
            "status": "profile_failed",
            "elapsed_s": time.time() - start,
            "reason": reason,
            "log": str(log_path),
        },
    )
    raise RuntimeError(f"profile generation failed: {reason}; log={log_path}")


def case_label(allow_partial: bool, dynamic: bool, step: int) -> str:
    return f"ap{int(allow_partial)}_{'dynamic' if dynamic else 'static'}_dl{step}"


def run_matrix(
    config: dict[str, Any], run_dir: Path, force: bool, timeout_s: float
) -> None:
    complete, reason = profile_complete(config)
    strategy = str(
        config.get("adaptive", {}).get("strategy", "throughput_aware")
    )
    if (
        strategy == "throughput_aware"
        and config["matrix"].get("dynamic_max_steps")
        and not complete
    ):
        raise RuntimeError(f"dynamic matrix requires a complete profile: {reason}")
    status_path = run_dir / "status.jsonl"
    cases: list[dict[str, Any]] = []
    for allow_partial in config["matrix"]["allow_partial"]:
        for dynamic, key in (
            (False, "static_steps"),
            (True, "dynamic_max_steps"),
        ):
            for raw_step in config["matrix"].get(key, []):
                step = int(raw_step)
                label = case_label(bool(allow_partial), dynamic, step)
                output_dir = run_dir / "runs" / label
                log_path = run_dir / "logs" / f"{label}.log"
                summary_path = output_dir / "summary.json"
                case = {
                    "label": label,
                    "allow_partial": bool(allow_partial),
                    "dynamic": dynamic,
                    "max_step": step,
                    "log": str(log_path),
                    "summary": str(summary_path),
                }
                cases.append(case)
                if summary_path.exists() and not force:
                    write_jsonl(
                        status_path, {"ts": now(), "status": "skipped_existing", **case}
                    )
                    continue
                output_dir.mkdir(parents=True, exist_ok=True)
                command = base_command(config, output_dir, step, dynamic)
                write_jsonl(
                    status_path,
                    {"ts": now(), "status": "started", "command": command, **case},
                )
                start = time.time()
                try:
                    returncode = run_process(
                        config=config,
                        command=command,
                        log_path=log_path,
                        allow_partial=bool(allow_partial),
                        timeout_s=timeout_s,
                    )
                    status = "ok" if returncode == 0 else "failed"
                except subprocess.TimeoutExpired:
                    returncode = None
                    status = "timeout"
                write_jsonl(
                    status_path,
                    {
                        "ts": now(),
                        "status": status,
                        "returncode": returncode,
                        "elapsed_s": time.time() - start,
                        **case,
                    },
                )
    manifest = {
        "config": str((run_dir / "config.toml").resolve()),
        "profile": str(Path(config["paths"]["profile"]).resolve()),
        "cases": cases,
    }
    (run_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    args.run_dir.mkdir(parents=True, exist_ok=True)
    config_copy = args.run_dir / "config.toml"
    if args.config.resolve() != config_copy.resolve():
        config_copy.write_bytes(args.config.read_bytes())
    if args.action == "profile":
        generate_profile(config, args.run_dir, args.force, args.timeout_s, args.poll_s)
    else:
        run_matrix(config, args.run_dir, args.force, args.timeout_s)


if __name__ == "__main__":
    main()
