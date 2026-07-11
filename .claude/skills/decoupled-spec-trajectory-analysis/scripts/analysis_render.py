#!/usr/bin/env python3
"""Rendering helpers for decoupled-spec analysis figures and Markdown reports."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

matplotlib.rcParams["svg.fonttype"] = "none"


def plot_trajectories(
    raw_rows: list[dict[str, Any]],
    smooth_rows: list[dict[str, Any]],
    output_dir: Path,
    smooth_window: int,
) -> list[Path]:
    """Plot the observable runtime trajectory that feeds controller diagnosis."""
    paths: list[Path] = []
    for label in sorted({str(row["label"]) for row in raw_rows}):
        case_raw = [row for row in raw_rows if row["label"] == label]
        case_smooth = [row for row in smooth_rows if row["label"] == label]
        x = [float(row["elapsed_s"]) for row in case_raw]
        fig, axes = plt.subplots(
            3, 1, figsize=(12, 8), sharex=True, constrained_layout=True
        )

        axes[0].plot(
            x,
            [float(row["observed_throughput_tok_s"]) for row in case_raw],
            color="#6b7280",
            alpha=0.35,
            linewidth=0.8,
            label="raw",
        )
        axes[0].plot(
            x,
            [float(row["observed_throughput_tok_s_smooth"]) for row in case_smooth],
            color="#0072B2",
            linewidth=1.6,
            label=f"centered MA (window={smooth_window})",
        )
        axes[0].set_title("Observed throughput")
        axes[0].set_ylabel("token/s")
        axes[0].legend(loc="upper right", frameon=False, fontsize=8)

        accept_raw = [row for row in case_raw if row["accept_len"] != ""]
        accept_smooth = [
            row for row in case_smooth if row.get("accept_len_smooth", "") != ""
        ]
        if accept_raw:
            axes[1].plot(
                [float(row["elapsed_s"]) for row in accept_raw],
                [float(row["accept_len"]) for row in accept_raw],
                color="#6b7280",
                alpha=0.35,
                linewidth=0.8,
                label="raw",
            )
        if accept_smooth:
            axes[1].plot(
                [float(row["elapsed_s"]) for row in accept_smooth],
                [float(row["accept_len_smooth"]) for row in accept_smooth],
                color="#D55E00",
                linewidth=1.6,
                label=f"centered MA (window={smooth_window})",
            )
            axes[1].legend(loc="upper right", frameon=False, fontsize=8)
        elif not accept_raw:
            axes[1].text(
                0.5,
                0.5,
                "accept_len unavailable in scheduler log",
                transform=axes[1].transAxes,
                ha="center",
                va="center",
                color="#6b7280",
            )
        else:
            axes[1].legend(loc="upper right", frameon=False, fontsize=8)
        axes[1].set_title("Acceptance length")
        axes[1].set_ylabel("tokens/iteration")

        axes[2].step(
            x,
            [int(row["active_step"]) for row in case_raw],
            where="post",
            color="#009E73",
            linewidth=1.4,
        )
        axes[2].set_title("Active draft length")
        axes[2].set_ylabel("step")
        axes[2].set_xlabel("reconstructed decode time (s)")
        for axis in axes:
            axis.grid(True, linestyle=":", linewidth=0.5, alpha=0.5)
            axis.spines[["top", "right"]].set_visible(False)
        fig.suptitle(label, fontsize=11)
        path = output_dir / f"trajectory_{label}.svg"
        fig.savefig(path)
        fig.savefig(output_dir / f"trajectory_{label}.png", dpi=180)
        plt.close(fig)
        paths.append(path)
    return paths


def write_report(
    output_dir: Path,
    summaries: list[dict[str, Any]],
    e2e_rows: list[dict[str, Any]],
    speedups: list[dict[str, Any]],
    figures: list[Path],
) -> None:
    lines = [
        "# Decoupled Spec Analysis",
        "",
        "Structured source files: `decode_points.csv`, "
        "`decode_points_trajectory.csv`, `controller_switches.csv`, "
        "`case_summary.csv`, `e2e_summary.csv`, and `speedup_summary.csv`.",
        "",
        "## Runtime Summary",
        "",
        "| case | points | queue max | observed ITL ms | modeled ITL ms | observed/modeled thpt |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summaries:
        lines.append(
            f"| {row['label']} | {row['points']} | {row['queue_req_max']} | "
            f"{float(row['observed_itl_mean_ms']):.2f} | "
            f"{float(row['modeled_itl_mean_ms']):.2f} | "
            f"{float(row['observed_over_modeled_throughput_mean']):.3f} |"
        )
    lines.extend(
        [
            "",
            "## E2E Throughput",
            "",
            "| case | token/s | generation s | accept length |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    for row in e2e_rows:
        lines.append(
            f"| {row['label']} | {float(row['output_throughput_tok_per_s']):.2f} | "
            f"{float(row['generation_time_s']):.2f} | "
            f"{float(row['avg_spec_accept_length'] or 0):.3f} |"
        )
    lines.extend(
        [
            "",
            "## Dynamic Speedup",
            "",
            "| case | vs same step | vs best static | best static |",
            "| --- | ---: | ---: | --- |",
        ]
    )
    for row in speedups:
        same = row["speedup_vs_same_step"]
        same_text = f"{float(same):.3f}x" if same != "" else "N/A"
        lines.append(
            f"| {row['label']} | {same_text} | "
            f"{float(row['speedup_vs_best_static']):.3f}x | {row['best_static_label']} |"
        )
    lines.extend(["", "## Figures", ""])
    for path in figures:
        lines.append(f"![{path.stem}]({path.name})")
        lines.append("")
    (output_dir / "report.md").write_text("\n".join(lines).rstrip() + "\n")


def plot_static(
    rows: list[dict[str, Any]], output_dir: Path, smooth: bool
) -> list[Path]:
    paths: list[Path] = []
    suffix = "smooth" if smooth else "raw"
    observed_key = "observed_itl_ms_smooth" if smooth else "observed_itl_ms"
    modeled_key = "modeled_itl_ms_smooth" if smooth else "modeled_itl_ms"
    for allow_partial in (False, True):
        selected = [
            row
            for row in rows
            if not row["dynamic"] and bool(row["allow_partial"]) == allow_partial
        ]
        steps = sorted({int(row["max_step"]) for row in selected})
        if not steps:
            continue
        y_max = max(
            float(row[key])
            for row in selected
            for key in (observed_key, modeled_key)
        ) * 1.08
        fig, axes = plt.subplots(
            len(steps),
            1,
            figsize=(15, max(3.2 * len(steps), 4)),
            sharey=True,
            constrained_layout=True,
        )
        if len(steps) == 1:
            axes = [axes]
        fig.suptitle(
            f"Static decoupled speculation: runtime vs profile+CPU ({suffix}, "
            f"allow_partial={allow_partial})",
            fontsize=12,
        )
        for axis, step in zip(axes, steps):
            case_rows = [row for row in selected if int(row["max_step"]) == step]
            axis.plot(
                [float(row["elapsed_s"]) for row in case_rows],
                [float(row[observed_key]) for row in case_rows],
                linewidth=1.0,
                label=f"runtime step={step}",
            )
            axis.plot(
                [float(row["elapsed_s"]) for row in case_rows],
                [float(row[modeled_key]) for row in case_rows],
                linewidth=1.0,
                linestyle="--",
                label="profile lookup + CPU overhead",
            )
            axis.set_ylim(0, y_max)
            axis.set_ylabel("ITL (ms)")
            axis.grid(True, linestyle=":", linewidth=0.5, alpha=0.6)
            axis.legend(loc="upper right", fontsize=8)
        axes[-1].set_xlabel("reconstructed decode time (s)")
        path = output_dir / f"static_ap{int(allow_partial)}_latency_profile_gap_{suffix}.png"
        fig.savefig(path, dpi=180)
        plt.close(fig)
        paths.append(path)
    return paths


def plot_dynamic(
    rows: list[dict[str, Any]],
    switches: list[dict[str, Any]],
    output_dir: Path,
    smooth: bool,
) -> list[Path]:
    paths: list[Path] = []
    suffix = "smooth" if smooth else "raw"
    keys = {
        "observed_itl": "observed_itl_ms_smooth" if smooth else "observed_itl_ms",
        "modeled_itl": "modeled_itl_ms_smooth" if smooth else "modeled_itl_ms",
        "observed_thpt": (
            "observed_throughput_tok_s_smooth"
            if smooth
            else "observed_throughput_tok_s"
        ),
        "modeled_thpt": (
            "modeled_throughput_tok_s_smooth"
            if smooth
            else "modeled_throughput_tok_s"
        ),
    }
    for allow_partial in (False, True):
        selected = [
            row
            for row in rows
            if row["dynamic"] and bool(row["allow_partial"]) == allow_partial
        ]
        steps = sorted({int(row["max_step"]) for row in selected})
        if not steps:
            continue
        itl_max = max(
            float(row[keys[name]])
            for row in selected
            for name in ("observed_itl", "modeled_itl")
        ) * 1.08
        thpt_max = max(
            float(row[keys[name]])
            for row in selected
            for name in ("observed_thpt", "modeled_thpt")
        ) * 1.08
        fig, axes = plt.subplots(
            len(steps),
            2,
            figsize=(16, max(3.4 * len(steps), 4)),
            constrained_layout=True,
        )
        if len(steps) == 1:
            axes = [axes]
        fig.suptitle(
            f"Dynamic decoupled speculation: observed vs modeled ({suffix}, "
            f"allow_partial={allow_partial})",
            fontsize=12,
        )
        for row_index, step in enumerate(steps):
            case_rows = [row for row in selected if int(row["max_step"]) == step]
            x = [float(row["elapsed_s"]) for row in case_rows]
            axes[row_index][0].plot(
                x,
                [float(row[keys["observed_itl"]]) for row in case_rows],
                linewidth=1.0,
                label="observed ITL",
            )
            axes[row_index][0].plot(
                x,
                [float(row[keys["modeled_itl"]]) for row in case_rows],
                linewidth=1.0,
                linestyle="--",
                label="modeled ITL",
            )
            axes[row_index][1].plot(
                x,
                [float(row[keys["observed_thpt"]]) for row in case_rows],
                linewidth=1.0,
                label="observed throughput",
            )
            axes[row_index][1].plot(
                x,
                [float(row[keys["modeled_thpt"]]) for row in case_rows],
                linewidth=1.0,
                linestyle="--",
                label="modeled throughput",
            )
            for axis, y_max, ylabel in (
                (axes[row_index][0], itl_max, "ITL (ms)"),
                (axes[row_index][1], thpt_max, "throughput (token/s)"),
            ):
                axis.set_ylim(0, y_max)
                axis.set_ylabel(ylabel)
                axis.grid(True, linestyle=":", linewidth=0.5, alpha=0.6)
                axis.legend(loc="upper right", fontsize=8)
            axes[row_index][0].set_title(f"max step={step}")
            axes[row_index][1].set_title(f"max step={step}")
            for switch in [
                item for item in switches if item["label"] == case_rows[0]["label"]
            ]:
                candidates = [
                    item
                    for item in case_rows
                    if int(item["point_index"]) >= int(switch["batch_count"])
                ]
                if candidates:
                    switch_x = float(candidates[0]["elapsed_s"])
                    axes[row_index][0].axvline(
                        switch_x, color="#6b7280", alpha=0.35
                    )
                    axes[row_index][1].axvline(
                        switch_x, color="#6b7280", alpha=0.35
                    )
        axes[-1][0].set_xlabel("reconstructed decode time (s)")
        axes[-1][1].set_xlabel("reconstructed decode time (s)")
        path = (
            output_dir
            / f"dynamic_ap{int(allow_partial)}_observed_vs_modeled_{suffix}.png"
        )
        fig.savefig(path, dpi=180)
        plt.close(fig)
        paths.append(path)
    return paths
