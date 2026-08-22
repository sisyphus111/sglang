# Chat Template and Tokenization

Client-side prompt preparation uses the target tokenizer and saves the result
before traffic is sent.

## `chat_template.mode: tokenizer`

For each text prompt, construct one user message and call:

```python
tokenizer.apply_chat_template(
    [{"role": "user", "content": raw_prompt}],
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=<configured value>,
)
```

If the tokenizer rejects the `enable_thinking` keyword with `TypeError`, retry
without that keyword. Then encode the rendered string with
`add_special_tokens=False`.

Use this mode for instruct/chat checkpoints whose serving prompts require their
tokenizer template. Record `enable_thinking` explicitly for Qwen-family
experiments because it can materially change prompt tokens and generated
behavior.

## `chat_template.mode: none`

Encode the dataset prompt directly with `add_special_tokens=False`. Use this
only when the dataset already contains the exact model input text or when the
experiment intentionally measures raw completion prompts.

## Preflight checks

Compare the inspected prompt lengths with the intended workload. Do not replace
a dataset workload with synthetic IDs merely to reach a target length unless
the user requested a synthetic-length experiment. The saved
`rendered_prompt`, `input_ids`, and `prompt_len` are the authoritative inputs
for a completed run.
