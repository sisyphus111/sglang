#!/usr/bin/env python3
"""
Profile target-model decode throughput versus per-forward token batch size.

This tool intentionally does not enable decoupled speculation. For each probed
batch size, it creates a target-only engine with max_running_requests and decode
CUDA Graph capture size set to that batch size, then measures ordinary decode.
The measured peak/plateau gives a hardware/model/TP-specific budget candidate
for dynamic decoupled verifier target verify.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import socket
import subprocess
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean, median
from typing import Any

try:
    from . import common
except ImportError:
    import common

try:
    import sglang as sgl
except ImportError as exc:
    raise ImportError("sglang is required to run target token-batch profiling") from exc


LOCAL_HOST = "127.0.0.1"


@dataclass
class GpuSampleSummary:
    sample_count: int = 0
    avg_gpu_util_pct: float | None = None
    max_gpu_util_pct: float | None = None
    avg_mem_util_pct: float | None = None
    avg_power_w: float | None = None
    max_power_w: float | None = None
    avg_mem_used_mib: float | None = None
    max_mem_used_mib: float | None = None


@dataclass
class ProfileRow:
    token_batch_size: int
    batch_size: int
    repeat: int
    warmup: int
    max_new_tokens: int
    prompt_tokens_min: int
    prompt_tokens_max: int
    prompt_tokens_avg: float
    elapsed_s: float
    generated_tokens: int
    throughput_tok_s: float
    sec_per_token: float
    request_latency_p50_s: float | None
    request_latency_p90_s: float | None
    request_latency_max_s: float | None
    gpu_sample_count: int
    avg_gpu_util_pct: float | None
    max_gpu_util_pct: float | None
    avg_mem_util_pct: float | None
    avg_power_w: float | None
    max_power_w: float | None
    avg_mem_used_mib: float | None
    max_mem_used_mib: float | None


class NvidiaSmiSampler:
    def __init__(self, gpu_indices: list[int], interval_s: float) -> None:
        self.gpu_indices = list(gpu_indices)
        self.interval_s = float(interval_s)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._samples: list[dict[str, float]] = []

    def __enter__(self):
        if not self.gpu_indices or self.interval_s <= 0:
            return self
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc, tb):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(1.0, self.interval_s * 4))
        return False

    def _run(self) -> None:
        while not self._stop.is_set():
            self._sample_once()
            self._stop.wait(self.interval_s)

    def _sample_once(self) -> None:
        command = [
            "nvidia-smi",
            "-i",
            ",".join(str(index) for index in self.gpu_indices),
            "--query-gpu=utilization.gpu,utilization.memory,power.draw,memory.used",
            "--format=csv,noheader,nounits",
        ]
        try:
            output = subprocess.check_output(
                command,
                text=True,
                stderr=subprocess.DEVNULL,
                timeout=max(1.0, self.interval_s),
            )
        except Exception:
            return
        for line in output.strip().splitlines():
            parts = [part.strip() for part in line.split(",")]
            if len(parts) != 4:
                continue
            try:
                self._samples.append(
                    {
                        "gpu_util": float(parts[0]),
                        "mem_util": float(parts[1]),
                        "power_w": float(parts[2]),
                        "mem_used_mib": float(parts[3]),
                    }
                )
            except ValueError:
                continue

    def summarize(self) -> GpuSampleSummary:
        if not self._samples:
            return GpuSampleSummary()

        def values(name: str) -> list[float]:
            return [sample[name] for sample in self._samples]

        gpu_util = values("gpu_util")
        mem_util = values("mem_util")
        power_w = values("power_w")
        mem_used_mib = values("mem_used_mib")
        return GpuSampleSummary(
            sample_count=len(self._samples),
            avg_gpu_util_pct=mean(gpu_util),
            max_gpu_util_pct=max(gpu_util),
            avg_mem_util_pct=mean(mem_util),
            avg_power_w=mean(power_w),
            max_power_w=max(power_w),
            avg_mem_used_mib=mean(mem_used_mib),
            max_mem_used_mib=max(mem_used_mib),
        )


def parse_int_list(raw: str) -> list[int]:
    values: list[int] = []
    for item in raw.replace(",", " ").split():
        if "-" in item:
            pieces = item.split(":")
            if len(pieces) not in (2, 3):
                raise ValueError(f"invalid range item: {item!r}")
            bounds = pieces[0].split("-")
            if len(bounds) != 2:
                raise ValueError(f"invalid range item: {item!r}")
            start = int(bounds[0])
            stop = int(bounds[1])
            step = int(pieces[2] if len(pieces) == 3 else pieces[1])
            if step <= 0:
                raise ValueError(f"range step must be positive: {item!r}")
            values.extend(range(start, stop + 1, step))
        else:
            values.append(int(item))
    values = sorted(set(values))
    if not values or any(value <= 0 for value in values):
        raise ValueError(f"expected positive token batch sizes: {raw!r}")
    return values


def parse_gpu_indices(raw: str | None, *, base_gpu_id: int, tp_size: int) -> list[int]:
    if raw:
        return parse_int_list(raw)
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible:
        entries = [item.strip() for item in visible.split(",") if item.strip()]
        parsed: list[int] = []
        for entry in entries[:tp_size]:
            try:
                parsed.append(int(entry))
            except ValueError:
                return list(range(base_gpu_id, base_gpu_id + tp_size))
        return parsed
    return list(range(base_gpu_id, base_gpu_id + tp_size))


def reserve_local_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind((LOCAL_HOST, 0))
        return int(sock.getsockname()[1])


def percentile(values: list[float], pct: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * pct)))
    return ordered[index]


def output_token_count(output: dict[str, Any]) -> int:
    output_ids = output.get("output_ids", [])
    if isinstance(output_ids, list):
        return len(output_ids)
    return 0


def request_latency(output: dict[str, Any]) -> float | None:
    meta_info = output.get("meta_info", {}) or {}
    value = meta_info.get("e2e_latency")
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def run_generate(engine, input_ids: list[list[int]], sampling_params: dict[str, Any]):
    outputs = engine.generate(input_ids=input_ids, sampling_params=sampling_params)
    if not isinstance(outputs, list):
        outputs = [outputs]
    return outputs


def profile_one_size(
    *,
    engine,
    input_ids: list[list[int]],
    token_batch_size: int,
    max_new_tokens: int,
    temperature: float,
    repeat: int,
    warmup: int,
    gpu_indices: list[int],
    gpu_sample_interval_s: float,
) -> ProfileRow:
    batch_input_ids = input_ids[:token_batch_size]
    if len(batch_input_ids) != token_batch_size:
        raise RuntimeError(
            f"need {token_batch_size} prompts, only loaded {len(batch_input_ids)}"
        )
    sampling_params = {
        "temperature": temperature,
        "max_new_tokens": max_new_tokens,
        "ignore_eos": True,
    }
    for _ in range(warmup):
        run_generate(engine, batch_input_ids, sampling_params)

    all_latencies: list[float] = []
    generated_tokens = 0
    with NvidiaSmiSampler(gpu_indices, gpu_sample_interval_s) as sampler:
        tic = time.perf_counter()
        for _ in range(repeat):
            outputs = run_generate(engine, batch_input_ids, sampling_params)
            generated_tokens += sum(output_token_count(output) for output in outputs)
            all_latencies.extend(
                latency
                for latency in (request_latency(output) for output in outputs)
                if latency is not None
            )
        elapsed = time.perf_counter() - tic
    gpu_summary = sampler.summarize()

    prompt_lens = [len(ids) for ids in batch_input_ids]
    throughput = generated_tokens / elapsed if elapsed > 0 else 0.0
    return ProfileRow(
        token_batch_size=token_batch_size,
        batch_size=token_batch_size,
        repeat=repeat,
        warmup=warmup,
        max_new_tokens=max_new_tokens,
        prompt_tokens_min=min(prompt_lens),
        prompt_tokens_max=max(prompt_lens),
        prompt_tokens_avg=mean(prompt_lens),
        elapsed_s=elapsed,
        generated_tokens=generated_tokens,
        throughput_tok_s=throughput,
        sec_per_token=elapsed / generated_tokens if generated_tokens > 0 else 0.0,
        request_latency_p50_s=median(all_latencies) if all_latencies else None,
        request_latency_p90_s=percentile(all_latencies, 0.90),
        request_latency_max_s=max(all_latencies) if all_latencies else None,
        gpu_sample_count=gpu_summary.sample_count,
        avg_gpu_util_pct=gpu_summary.avg_gpu_util_pct,
        max_gpu_util_pct=gpu_summary.max_gpu_util_pct,
        avg_mem_util_pct=gpu_summary.avg_mem_util_pct,
        avg_power_w=gpu_summary.avg_power_w,
        max_power_w=gpu_summary.max_power_w,
        avg_mem_used_mib=gpu_summary.avg_mem_used_mib,
        max_mem_used_mib=gpu_summary.max_mem_used_mib,
    )


def build_engine_kwargs(
    args: argparse.Namespace,
    *,
    token_batch_size: int,
) -> dict[str, Any]:
    dist_init_addr = args.dist_init_addr
    if dist_init_addr is None and args.target_tp_size > 1:
        dist_init_addr = f"{LOCAL_HOST}:{reserve_local_port()}"

    engine_kwargs: dict[str, Any] = dict(
        model_path=args.target_model_path,
        tokenizer_path=args.tokenizer_path or args.target_model_path,
        tp_size=args.target_tp_size,
        base_gpu_id=args.base_gpu_id,
        dtype=args.dtype,
        trust_remote_code=args.trust_remote_code,
        disable_overlap_schedule=args.disable_overlap_schedule,
        disable_radix_cache=args.disable_radix_cache,
        enable_deterministic_inference=args.enable_deterministic,
        max_running_requests=token_batch_size,
        cuda_graph_backend_decode="full",
        cuda_graph_backend_prefill="disabled",
        cuda_graph_max_bs_decode=token_batch_size,
        cuda_graph_bs_decode=[token_batch_size],
        disable_cuda_graph_padding=not args.allow_cuda_graph_padding,
    )
    if dist_init_addr is not None:
        engine_kwargs["dist_init_addr"] = dist_init_addr
    if args.mem_fraction_static is not None:
        engine_kwargs["mem_fraction_static"] = args.mem_fraction_static
    if args.attention_backend is not None:
        engine_kwargs["attention_backend"] = args.attention_backend
    if args.sampling_backend is not None:
        engine_kwargs["sampling_backend"] = args.sampling_backend
    return engine_kwargs


def write_csv(path: Path, rows: list[ProfileRow]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(rows[0]).keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def choose_recommendations(
    rows: list[ProfileRow], *, plateau_ratio: float
) -> dict[str, Any]:
    if not rows:
        raise RuntimeError("no profile rows")
    peak = max(rows, key=lambda row: row.throughput_tok_s)
    threshold = peak.throughput_tok_s * plateau_ratio
    plateau = min(
        (row for row in rows if row.throughput_tok_s >= threshold),
        key=lambda row: row.token_batch_size,
    )
    return {
        "peak_token_batch_size": peak.token_batch_size,
        "peak_throughput_tok_s": peak.throughput_tok_s,
        "peak_budget": peak.token_batch_size,
        "plateau_ratio": plateau_ratio,
        "plateau_threshold_tok_s": threshold,
        "plateau_token_batch_size": plateau.token_batch_size,
        "plateau_throughput_tok_s": plateau.throughput_tok_s,
        "plateau_budget": plateau.token_batch_size,
    }


def verify_length_table(
    *,
    budget: int,
    batch_sizes: list[int],
    max_speculative_steps: int,
) -> list[dict[str, int]]:
    rows = []
    for bs in batch_sizes:
        budget_step_cap = (budget - 1) // bs - 1
        if budget_step_cap < 0:
            num_speculative_steps = -1
            verify_tokens_per_req = 0
        else:
            num_speculative_steps = min(max_speculative_steps, budget_step_cap)
            verify_tokens_per_req = num_speculative_steps + 1
        rows.append(
            {
                "batch_size": bs,
                "num_speculative_steps": num_speculative_steps,
                "verify_tokens_per_req": verify_tokens_per_req,
                "verify_token_batch_size": bs * verify_tokens_per_req,
                "budget": budget,
            }
        )
    return rows


def write_report(path: Path, summary: dict[str, Any], rows: list[ProfileRow]) -> None:
    lines = [
        "# Target Token-Batch Profile",
        "",
        "This profile uses target-only decode as a surrogate for target-verify "
        "token batch size. For decoupled verifier target verify, compare "
        "`bs * (num_speculative_steps + 1)` against the recommended budget.",
        "",
        "## Recommendation",
        "",
        f"- peak token_batch_size: {summary['recommendation']['peak_token_batch_size']}",
        f"- peak throughput: {summary['recommendation']['peak_throughput_tok_s']:.3f} tok/s",
        f"- peak budget: {summary['recommendation']['peak_budget']}",
        f"- plateau ratio: {summary['recommendation']['plateau_ratio']:.3f}",
        f"- plateau token_batch_size: {summary['recommendation']['plateau_token_batch_size']}",
        f"- plateau throughput: {summary['recommendation']['plateau_throughput_tok_s']:.3f} tok/s",
        f"- plateau budget: {summary['recommendation']['plateau_budget']}",
        "",
        "## Sweep",
        "",
        "| token_batch_size | throughput tok/s | sec/token | avg GPU util % | avg power W |",
        "|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| "
            f"{row.token_batch_size} | "
            f"{row.throughput_tok_s:.3f} | "
            f"{row.sec_per_token:.6f} | "
            f"{'' if row.avg_gpu_util_pct is None else f'{row.avg_gpu_util_pct:.1f}'} | "
            f"{'' if row.avg_power_w is None else f'{row.avg_power_w:.1f}'} |"
        )
    lines.extend(["", "## Verify-Length Table", ""])
    for name, table in summary["verify_length_tables"].items():
        lines.extend(
            [
                f"### {name}",
                "",
                "| batch_size | num_speculative_steps | verify_tokens_per_req | verify_token_batch_size |",
                "|---:|---:|---:|---:|",
            ]
        )
        for row in table:
            lines.append(
                "| "
                f"{row['batch_size']} | "
                f"{row['num_speculative_steps']} | "
                f"{row['verify_tokens_per_req']} | "
                f"{row['verify_token_batch_size']} |"
            )
        lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Profile target-only decode throughput versus token batch size and "
            "recommend decoupled verifier dynamic verify budgets."
        )
    )
    parser.add_argument("--target-model-path", required=True)
    parser.add_argument("--tokenizer-path", default=None)
    parser.add_argument("--target-tp-size", type=int, required=True)
    parser.add_argument("--base-gpu-id", type=int, default=0)
    parser.add_argument("--gpu-indices", default=None)
    parser.add_argument("--dist-init-addr", default=None)
    parser.add_argument("--mem-fraction-static", type=float, default=None)
    parser.add_argument("--attention-backend", default=None)
    parser.add_argument("--sampling-backend", default=None)
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--disable-radix-cache", action="store_true")
    parser.add_argument("--disable-overlap-schedule", action="store_true")
    parser.add_argument(
        "--allow-cuda-graph-padding",
        action="store_true",
        help=(
            "Allow decode CUDA Graph padding during profiling. By default each "
            "profile point captures exactly the probed batch size."
        ),
    )
    parser.add_argument("--enable-deterministic", action="store_true")

    parser.add_argument("--prompt", default=None)
    parser.add_argument("--dataset-path", default=None)
    parser.add_argument("--prompt-column", default=None)
    parser.add_argument(
        "--dataset-format",
        choices=["auto", "codeforces_raw", "dapo_math_17k"],
        default="auto",
    )
    parser.add_argument(
        "--code-language",
        choices=["python", "py", "cpp", "c++"],
        default="python",
    )
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--max-prompt-length", type=int, default=None)
    parser.add_argument("--disable-chat-template", action="store_true")
    parser.add_argument("--enable-thinking", action="store_true")

    parser.add_argument(
        "--token-batch-sizes",
        default="1 2 4 8 16 24 32 48 64 96 128",
        help=(
            "Space/comma separated sizes, with optional start-stop:step ranges, "
            "for example '1 2 4 8 16-128:16'."
        ),
    )
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--gpu-sample-interval-s", type=float, default=0.2)
    parser.add_argument("--plateau-ratio", type=float, default=0.95)
    parser.add_argument(
        "--verify-batch-sizes",
        default="1 2 4 8 16 32 64 128",
        help="Batch sizes to include in the recommended verify-length table.",
    )
    parser.add_argument("--max-speculative-steps", type=int, default=4)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def validate_args(args: argparse.Namespace, token_batch_sizes: list[int]) -> None:
    if args.target_tp_size <= 0:
        raise ValueError("--target-tp-size must be positive")
    if args.max_new_tokens <= 0:
        raise ValueError("--max-new-tokens must be positive")
    if args.repeat <= 0:
        raise ValueError("--repeat must be positive")
    if args.warmup < 0:
        raise ValueError("--warmup must be non-negative")
    if not (0 < args.plateau_ratio <= 1.0):
        raise ValueError("--plateau-ratio must be in (0, 1]")
    if args.max_speculative_steps < 0:
        raise ValueError("--max-speculative-steps must be non-negative")
    if max(token_batch_sizes) <= 0:
        raise ValueError("token batch sizes must be positive")


def main() -> None:
    args = parse_args()
    token_batch_sizes = parse_int_list(args.token_batch_sizes)
    verify_batch_sizes = parse_int_list(args.verify_batch_sizes)
    validate_args(args, token_batch_sizes)

    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    prompt_args = argparse.Namespace(**vars(args))
    prompt_args.batch_size = max(token_batch_sizes)
    prompt_args.context_length = args.max_new_tokens
    prompt_column, prompt_samples, total_rows = common.load_prompt_samples(prompt_args)
    input_ids = [list(sample.prompt_input_ids) for sample in prompt_samples]
    gpu_indices = parse_gpu_indices(
        args.gpu_indices, base_gpu_id=args.base_gpu_id, tp_size=args.target_tp_size
    )

    engine_kwargs: dict[str, Any] = dict(
        model_path=args.target_model_path,
        tokenizer_path=args.tokenizer_path or args.target_model_path,
        tp_size=args.target_tp_size,
        base_gpu_id=args.base_gpu_id,
        dtype=args.dtype,
        trust_remote_code=args.trust_remote_code,
        disable_overlap_schedule=args.disable_overlap_schedule,
        disable_radix_cache=args.disable_radix_cache,
        enable_deterministic_inference=args.enable_deterministic,
    )
    config = {
        "target_model_path": args.target_model_path,
        "tokenizer_path": args.tokenizer_path or args.target_model_path,
        "target_tp_size": args.target_tp_size,
        "base_gpu_id": args.base_gpu_id,
        "gpu_indices": gpu_indices,
        "dist_init_addr": args.dist_init_addr,
        "per_size_engine": True,
        "allow_cuda_graph_padding": args.allow_cuda_graph_padding,
        "cuda_graph_backend_decode": "full",
        "cuda_graph_backend_prefill": "disabled",
        "dataset_path": args.dataset_path,
        "dataset_format": args.dataset_format,
        "prompt_column": prompt_column,
        "total_rows": total_rows,
        "loaded_rows": [sample.row_index for sample in prompt_samples],
        "max_prompt_length": args.max_prompt_length,
        "token_batch_sizes": token_batch_sizes,
        "max_new_tokens": args.max_new_tokens,
        "repeat": args.repeat,
        "warmup": args.warmup,
        "temperature": args.temperature,
        "plateau_ratio": args.plateau_ratio,
        "max_speculative_steps": args.max_speculative_steps,
        "verify_batch_sizes": verify_batch_sizes,
    }
    (output_dir / "config.json").write_text(
        json.dumps(config, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    rows: list[ProfileRow] = []
    for token_batch_size in token_batch_sizes:
        print(
            f"profile_token_batch_size={token_batch_size} "
            f"max_running_requests={token_batch_size}",
            flush=True,
        )
        engine_kwargs = build_engine_kwargs(args, token_batch_size=token_batch_size)
        engine = sgl.Engine(**engine_kwargs)
        try:
            row = profile_one_size(
                engine=engine,
                input_ids=input_ids,
                token_batch_size=token_batch_size,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                repeat=args.repeat,
                warmup=args.warmup,
                gpu_indices=gpu_indices,
                gpu_sample_interval_s=args.gpu_sample_interval_s,
            )
            rows.append(row)
            print(
                f"  throughput={row.throughput_tok_s:.3f} tok/s "
                f"elapsed={row.elapsed_s:.3f}s generated={row.generated_tokens} "
                f"avg_gpu_util={row.avg_gpu_util_pct}",
                flush=True,
            )
            write_csv(output_dir / "profile.csv", rows)
        finally:
            engine.shutdown()

    recommendation = choose_recommendations(rows, plateau_ratio=args.plateau_ratio)
    summary = {
        "config": config,
        "recommendation": recommendation,
        "rows": [asdict(row) for row in rows],
        "verify_length_tables": {
            "plateau_budget": verify_length_table(
                budget=int(recommendation["plateau_budget"]),
                batch_sizes=verify_batch_sizes,
                max_speculative_steps=args.max_speculative_steps,
            ),
            "peak_budget": verify_length_table(
                budget=int(recommendation["peak_budget"]),
                batch_sizes=verify_batch_sizes,
                max_speculative_steps=args.max_speculative_steps,
            ),
        },
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    write_report(output_dir / "report.md", summary, rows)
    print(json.dumps(recommendation, indent=2), flush=True)
    print(f"profile_output_dir={output_dir}", flush=True)


if __name__ == "__main__":
    main()
