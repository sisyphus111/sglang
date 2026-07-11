#!/usr/bin/env python3
"""Minimal throughput-profile coverage checks used by the experiment runner."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable


class ProfileTable:
    """Load exact (batch size, step, context) points for resume validation."""

    def __init__(
        self,
        rows: Iterable[dict[str, Any]],
        fingerprint: dict[str, Any] | None = None,
    ) -> None:
        self.data = {
            (int(row["batch_size"]), int(row["steps"]), int(row["ctx_len"])): float(
                row["cost_ms"]
            )
            for row in rows
        }
        if not self.data:
            raise ValueError("profile contains no costs")
        self.fingerprint = fingerprint or {}

    @classmethod
    def load(cls, path: Path) -> "ProfileTable":
        payload = json.loads(path.read_text())
        rows = payload["costs"] if isinstance(payload, dict) else payload
        if not isinstance(rows, list):
            raise ValueError(f"profile costs must be a list: {path}")
        fingerprint = payload.get("fingerprint") if isinstance(payload, dict) else None
        if not isinstance(fingerprint, dict):
            raise ValueError(f"profile fingerprint must be an object: {path}")
        return cls(rows, fingerprint)

    def missing_points(
        self,
        batch_sizes: Iterable[int],
        steps: Iterable[int],
        ctx_lens: Iterable[int],
    ) -> list[tuple[int, int, int]]:
        return [
            (int(bs), int(step), int(ctx))
            for step in steps
            for bs in batch_sizes
            for ctx in ctx_lens
            if (int(bs), int(step), int(ctx)) not in self.data
        ]
