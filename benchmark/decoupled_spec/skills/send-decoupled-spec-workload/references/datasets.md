# Dataset Contracts

The loader reads enough rows for one batch. The selected material is saved in
the fixed `client/content.json` contract.

| `dataset.format` | Source behavior | Prompt/reference behavior |
| --- | --- | --- |
| `gsm8k` | Parquet file or directory; when multiple files exist, a filename containing `test` is preferred, then the first sorted file is read | Defaults to `question` and `answer` |
| `parquet`, `generic_parquet` | Parquet file or directory; the first sorted parquet file is read | Use configured `prompt_column` and optional `reference_column` |
| `dapo_math_17k` | DAPO parquet file or directory; the first sorted parquet file is read | Reads the native `prompt` message list and defaults reference to `reward_model.ground_truth` |
| `jsonl`, `generic_jsonl`, `sharegpt` | JSONL file or directory; the first sorted JSONL file is read | Use configured `prompt_column` and optional `reference_column` |
| `codeforces_raw` | Codeforces parquet or JSONL; the first sorted matching file is read and invalid problem rows are skipped | Builds a competitive-programming system/user prompt from the limits, title, statement, I/O format, notes, and examples; `code_language` defaults to `python` |
| `synthetic_ids` | No dataset file; repeated token IDs are generated | Uses `prompt_len`, `token_id`, and generation output length |

`sharegpt` currently selects fields through the generic JSONL contract. Set
`prompt_column` to a field containing the exact text to benchmark.

DAPO `prompt` is a list of `{role, content}` messages. In `tokenizer` mode the
list is passed directly to the target tokenizer's chat template instead of
being converted to a Python string or wrapped in another user message. The
saved source metadata also retains `extra_info.index` as `dataset_index`.

## Selection

- `batch.size` must be positive and the source must contain at least that many
  rows.
- With `shuffle: false`, the first `batch.size` rows are used.
- With `shuffle: true`, rows are shuffled with a local RNG seeded by
  `dataset.seed`, then the first `batch.size` rows are used.
- `batch_row_index` is the index within the submitted batch after optional
  shuffle. `dataset_idx` is the zero-based physical row in the selected source
  file before shuffle and remains stable after selection.

## Synthetic inputs

`synthetic_ids` constructs every prompt as `[token_id] * prompt_len`. The client
still loads the configured target tokenizer so model/tokenizer provenance and
the client startup path remain consistent, but the generated token IDs do not
come from text encoding.
