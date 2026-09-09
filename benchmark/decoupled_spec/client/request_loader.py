"""Deterministic request preparation for decoupled-spec benchmarks."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import msgspec


class RequestSpec(msgspec.Struct, kw_only=True):
    request_id: str
    batch_row_index: int
    dataset_idx: int
    raw_prompt: str
    rendered_prompt: str
    input_ids: list[int] | None = None
    reference_response: str | None = None
    prompt_len: int = 0
    requested_output_len: int = 0
    source: dict[str, Any] = msgspec.field(default_factory=dict)


def _read_rows(path: Path, fmt: str) -> list[dict[str, Any]]:
    if fmt == "codeforces_raw":
        parquet_files = (
            [path]
            if path.is_file() and path.suffix == ".parquet"
            else sorted(path.rglob("*.parquet")) if path.is_dir() else []
        )
        if parquet_files:
            import pyarrow.parquet as pq

            table = pq.read_table(parquet_files[0])
            return [dict(row) for row in table.to_pylist()]
    if fmt in {"parquet", "generic_parquet", "gsm8k", "dapo_math_17k"}:
        import pyarrow.parquet as pq

        files = [path] if path.is_file() else sorted(path.rglob("*.parquet"))
        if not files:
            raise FileNotFoundError(f"no parquet files below {path}")
        if fmt == "gsm8k" and len(files) > 1:
            test_files = [item for item in files if "test" in item.name.lower()]
            files = test_files or files
        table = pq.read_table(files[0])
        return [dict(row) for row in table.to_pylist()]
    if fmt in {"jsonl", "generic_jsonl", "codeforces_raw", "sharegpt"}:
        files = [path] if path.is_file() else sorted(path.rglob("*.jsonl"))
        if not files:
            raise FileNotFoundError(f"no jsonl files below {path}")
        return [
            json.loads(line)
            for line in files[0].read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    raise ValueError(f"unsupported dataset format: {fmt}")


def _codeforces_messages(row: dict[str, Any], code_language: str) -> list[dict[str, str]]:
    required = (
        "description",
        "input_format",
        "output_format",
        "time_limit",
        "memory_limit",
    )
    missing = [field for field in required if not str(row.get(field) or "").strip()]
    if missing:
        raise ValueError(f"Codeforces row is missing required fields: {missing}")
    aliases = {
        "python": "Python",
        "python3": "Python",
        "cpp": "C++",
        "c++": "C++",
    }
    language = aliases.get(code_language.lower(), code_language)
    sections = [
        f"Write a correct and efficient {language} solution for the following "
        "competitive programming problem.",
        f"Time limit: {row['time_limit']} seconds\nMemory limit: {row['memory_limit']} MB",
    ]
    if row.get("title"):
        sections.append(f"Title: {row['title']}")
    sections.extend(
        [
            f"Problem description:\n{row['description']}",
            f"Input format:\n{row['input_format']}",
            f"Output format:\n{row['output_format']}",
        ]
    )
    if row.get("note"):
        sections.append(f"Notes:\n{row['note']}")
    examples = []
    for index, example in enumerate(row.get("examples") or [], start=1):
        if isinstance(example, dict):
            examples.append(
                f"Example {index}\nInput:\n{example.get('input', '')}\nOutput:\n{example.get('output', '')}"
            )
    if examples:
        sections.append("Examples:\n" + "\n\n".join(examples))
    sections.append(
        f"Return only the final {language} source code. Do not include explanations "
        "or Markdown fences."
    )
    return [
        {
            "role": "system",
            "content": (
                "You are an expert competitive programmer. Produce a correct, "
                "efficient solution that respects the stated constraints."
            ),
        },
        {"role": "user", "content": "\n\n".join(sections)},
    ]


def _nested_value(row: dict[str, Any], path: str) -> Any:
    value: Any = row
    for part in path.split("."):
        if not isinstance(value, dict):
            return None
        value = value.get(part)
    return value


def _render_messages(
    messages: list[dict[str, str]], chat_template: dict[str, Any], tokenizer: Any
) -> str:
    kwargs = {"tokenize": False, "add_generation_prompt": True}
    if "enable_thinking" in chat_template:
        kwargs["enable_thinking"] = bool(chat_template["enable_thinking"])
    try:
        return tokenizer.apply_chat_template(messages, **kwargs)
    except TypeError:
        kwargs.pop("enable_thinking", None)
        return tokenizer.apply_chat_template(messages, **kwargs)


def _render_prompt(
    raw_prompt: str, chat_template: dict[str, Any], tokenizer: Any
) -> str:
    mode = chat_template.get("mode", "none")
    if mode == "none":
        return raw_prompt
    if mode != "tokenizer":
        raise ValueError(f"unsupported chat_template.mode: {mode}")
    return _render_messages(
        [{"role": "user", "content": raw_prompt}], chat_template, tokenizer
    )


def load_requests(
    config: dict[str, Any], tokenizer: Any | None = None
) -> list[RequestSpec]:
    dataset = config.get("dataset", {})
    fmt = str(dataset.get("format", "synthetic_ids"))
    count = int(config.get("batch", {}).get("size", 1))
    if count <= 0:
        raise ValueError("batch.size must be positive")
    indexed_rows: list[tuple[int, dict[str, Any]]]
    if fmt == "synthetic_ids":
        indexed_rows = [
            (
                index,
                {"prompt": "", "input_len": int(dataset.get("prompt_len", 128))},
            )
            for index in range(count)
        ]
    else:
        path = dataset.get("path")
        if not path:
            raise ValueError("dataset.path is required for non-synthetic datasets")
        indexed_rows = list(enumerate(_read_rows(Path(path).expanduser(), fmt)))
        rng = random.Random(int(dataset.get("seed", 0)))
        if dataset.get("shuffle", False):
            rng.shuffle(indexed_rows)
        if len(indexed_rows) < count:
            raise ValueError(
                f"dataset contains {len(indexed_rows)} rows, fewer than batch.size={count}"
            )
        if fmt != "codeforces_raw":
            indexed_rows = indexed_rows[:count]
    chat_template = config.get("chat_template", {})
    generation = config.get("generation", {})
    output_len = int(generation.get("output_len", dataset.get("output_len", 128)))
    prompt_column = dataset.get(
        "prompt_column", "question" if fmt == "gsm8k" else "prompt"
    )
    reference_column = dataset.get(
        "reference_column",
        "reward_model.ground_truth" if fmt == "dapo_math_17k" else "answer",
    )
    requests: list[RequestSpec] = []
    for dataset_idx, row in indexed_rows:
        batch_row_index = len(requests)
        if fmt == "synthetic_ids":
            raw = ""
            prompt_len = int(row.get("input_len", dataset.get("prompt_len", 128)))
            ids = [int(dataset.get("token_id", 1))] * prompt_len
            rendered = raw
        else:
            if tokenizer is None:
                raise ValueError("tokenizer is required for text datasets")
            prompt_value = _nested_value(row, str(prompt_column))
            if prompt_value is None:
                prompt_value = row.get("text", "")
            if fmt == "codeforces_raw":
                try:
                    messages = _codeforces_messages(
                        row, str(dataset.get("code_language", "python"))
                    )
                except ValueError:
                    continue
                raw = messages[1]["content"]
                mode = chat_template.get("mode", "none")
                if mode == "tokenizer":
                    rendered = _render_messages(messages, chat_template, tokenizer)
                elif mode == "none":
                    rendered = "\n\n".join(
                        f"{message['role']}: {message['content']}"
                        for message in messages
                    )
                else:
                    raise ValueError(f"unsupported chat_template.mode: {mode}")
            elif fmt == "dapo_math_17k" and isinstance(prompt_value, list):
                messages = []
                for message in prompt_value:
                    if not isinstance(message, dict):
                        raise ValueError("DAPO prompt entries must be mappings")
                    role = message.get("role")
                    content = message.get("content")
                    if not isinstance(role, str) or not isinstance(content, str):
                        raise ValueError(
                            "DAPO prompt entries require string role and content"
                        )
                    messages.append({"role": role, "content": content})
                if not messages:
                    raise ValueError("DAPO prompt message list must not be empty")
                raw = (
                    messages[0]["content"]
                    if len(messages) == 1 and messages[0]["role"] == "user"
                    else json.dumps(messages, ensure_ascii=False)
                )
                mode = chat_template.get("mode", "none")
                if mode == "tokenizer":
                    rendered = _render_messages(messages, chat_template, tokenizer)
                elif mode == "none" and len(messages) == 1:
                    rendered = messages[0]["content"]
                else:
                    raise ValueError(
                        "multi-message DAPO prompts require chat_template.mode=tokenizer"
                    )
            else:
                raw = str(prompt_value)
                rendered = _render_prompt(raw, chat_template, tokenizer)
            ids = tokenizer.encode(rendered, add_special_tokens=False)
            prompt_len = len(ids)
        reference = _nested_value(row, str(reference_column))
        source = {"format": fmt, "dataset_idx": dataset_idx}
        if fmt == "dapo_math_17k":
            source["dataset_index"] = _nested_value(row, "extra_info.index")
        requests.append(
            RequestSpec(
                request_id=f"req-{batch_row_index:06d}",
                batch_row_index=batch_row_index,
                dataset_idx=dataset_idx,
                raw_prompt=raw,
                rendered_prompt=rendered,
                input_ids=ids,
                reference_response=None if reference is None else str(reference),
                prompt_len=prompt_len,
                requested_output_len=output_len,
                source=source,
            )
        )
        if len(requests) == count:
            break
    if len(requests) != count:
        raise ValueError(
            f"dataset contains only {len(requests)} valid requests, fewer than "
            f"batch.size={count}"
        )
    return requests
