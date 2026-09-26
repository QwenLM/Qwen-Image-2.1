# Prompt Enhancer (Qwen3.5-VL 9B) -- t2i + edit

Two prompt-enhancer models that share one codebase:

| `--task` | What it does | Input | Answer fields |
| -------- | ------------ | ----- | ------------- |
| `t2i` | **Text-to-image prompt expansion.** Turns a short, rough image request into a long English prompt that reads like a description of the finished picture. | text | `rewritten_prompt`, `wh_ratio` |
| `edit` | **Image-editing instruction rewrite.** Turns a vague edit instruction plus its source image(s) into a precise, actionable prompt a downstream editor can follow without guessing. | text + 1..N images | `rewritten_prompt`, `wh_ratio`, `ratio_follow` |

Both checkpoints are **fine-tuned Qwen3.5-VL 9B, not the official base model**:
stock architecture (`model_type: qwen3_5`, `Qwen3_5ForConditionalGeneration`,
hybrid linear/full attention, thinking on by default), post-trained weights.
Their `chat_template.jinja` and `tokenizer.json` are byte-identical, which is
what makes one codebase honest rather than merely convenient.

**Each task has its own checkpoint and its own system prompt.** They are not
interchangeable and there is no merged prompt: the answer contract is part of
what each model was trained on. Point `--ckpt` at one and give it that model's
prompt (via `--system-prompt`, or ship it as `system_prompt.txt` inside the
checkpoint directory and it is picked up automatically). A Hub id such as
`Qwen/Qwen-Image-2.1-PE-T2I` works the same way: its `system_prompt.txt` is
downloaded from the same repo as the weights.

Pointing `--ckpt` at the official open-source Qwen3.5-VL 9B release will load and
generate, but it was never trained against either system prompt, so it does not
reliably emit the answer JSON -- expect `parse_ok: false` on most rows.

## Files

| File | What it is |
| ---- | ---------- |
| `pe_core.py` | Task profiles, message construction, answer parsing, output records. Everything the two tasks genuinely share. |
| `run_vllm.py` | Offline batch through `LLM.chat()`. **Use this for real workloads.** |
| `run_transformers.py` | Plain HuggingFace, batch size 1. Clear and hackable; for sanity checks and for modifying. |
| `serve.sh` | Start an OpenAI-compatible vLLM server on one checkpoint. |
| `client.py` | Talk to that server: one prompt, or a JSONL batch. |

All four entry points emit the **same** output records, so online and offline
results are interchangeable and nothing downstream needs to know which produced
a file.

## Install

```bash
pip install -r requirements.txt
```

Tested with `transformers==5.5.4`, `vllm==0.19.1`, `torch==2.10.0+cu128` on CUDA
12.x. The checkpoint loads through `AutoModelForImageTextToText`, which
dispatches on `config.model_type` (`qwen3_5` here).

## Hardware

One GPU is enough for either task. Weights are ~20 GB in `bfloat16`; vLLM needs
headroom for the KV cache on top, so a 40 GB card is comfortable at the default
`--gpu-memory-utilization 0.85`, and a 24 GB card wants `--max-model-len 12000`
or a lower utilization. `--tp 2` splits the weights to ~12 GB per GPU but buys
nothing at this size. System RAM: 64 GB is plenty.

## Input format

One JSON object per line, for both tasks:

```json
{"id": "abc123", "prompt": "make the sky sunset", "input_images": ["images/abc123.png"], "task_type": "basic_edit"}
```

- `prompt` -- the user's raw request, in any language.
- `input_images` -- **`edit` only**, a list of 1..N paths resolved relative to
  the JSONL's own directory (absolute paths also work). Every image is sent in
  order, because the system prompt tells the model to address them as
  `<image1>`, `<image2>`, ... -- reorder them and every reference in the rewrite
  silently re-points. Passing images to `--task t2i` is an error, not a warning:
  dropping them quietly would look like a successful run of the wrong
  experiment.
- `task_type` -- optional, echoed to the output for your bookkeeping. Both
  models use one system prompt for every task type; there is no router.

`t2i` line:

```json
{"id": "t2i_1", "prompt": "一只在雨中弹吉他的柯基"}
```

Multi-image `edit` line:

```json
{"id": "multi_1", "prompt": "Place <image1>'s subject into <image2>'s scene, matching lighting.", "input_images": ["images/portrait.png", "images/scene.png"]}
```

## Output format

One JSON object per line, fields in a stable order:

```json
{"id": "abc123",
 "task": "edit",
 "raw_prompt": "make the sky sunset",
 "input_images": ["images/abc123.png"],
 "task_type": "basic_edit",
 "thinking": "The image shows ...",
 "positive_prompt": "Replace the daytime sky with a warm sunset ...",
 "negative_prompt": "",
 "wh_ratio": "",
 "ratio_follow": "<image1>",
 "parse_ok": true}
```

- `positive_prompt` -- the rewritten instruction; the prompt you render.
- **`wh_ratio` / `ratio_follow` decide the output canvas.** For `edit` they are
  mutually exclusive: exactly one carries a value.
  - `wh_ratio` (e.g. `"16:9"`) -- the model chose the shape, because the task
    generates a new composition rather than editing the existing frame. This is
    the only one of the two that `t2i` uses.
  - `ratio_follow` (e.g. `"<image2>"`) -- `edit` only: the output inherits that
    input image's aspect ratio, because that image is the canvas being edited.

  Pass both through to whatever renders the prompt. Ignoring them and rendering
  at the source image's ratio discards a real part of the rewrite: a prompt
  describing a wide two-subject composition, rendered onto a portrait canvas, is
  a different picture.
- `parse_ok` -- `false` when the answer did not parse as the expected JSON
  object. `positive_prompt` then holds the raw answer text (nothing is lost) and
  the ratio fields are empty. Every entry point prints a summary line; to audit
  a finished file:

  ```bash
  jq -s 'map(select(.parse_ok | not)) | length' out.jsonl
  ```
- `negative_prompt` -- always `""`; neither model emits one. The field exists
  because downstream editors expect the slot.
- `task` -- which profile produced the row, so mixed corpora stay sortable.

## Usage

Offline batch, the normal path:

```bash
# t2i
python run_vllm.py --task t2i \
    --ckpt Qwen/Qwen-Image-2.1-PE-T2I \
    --input data/t2i_example.jsonl --output out.jsonl

# edit
python run_vllm.py --task edit \
    --ckpt Qwen/Qwen-Image-2.1-PE-I2I \
    --input data/edit_example.jsonl --output out.jsonl
```

Single-sample sanity check without a serving stack:

```bash
python run_transformers.py --task edit \
    --ckpt Qwen/Qwen-Image-2.1-PE-I2I \
    --input data/edit_example.jsonl --output out.jsonl --limit 1
```

Online, when you want an endpoint:

```bash
CKPT=Qwen/Qwen-Image-2.1-PE-T2I PORT=8100 bash serve.sh
# in another shell, once `curl -sf localhost:8100/health` answers:
python client.py --task t2i --model Qwen/Qwen-Image-2.1-PE-T2I \
    --system-prompt prompts/system_prompt_t2i.txt \
    "a corgi playing guitar in the rain"
```

The server holds the weights but not the system prompt, so `client.py` always
needs `--system-prompt`. `prompts/` has a copy of each checkpoint's
`system_prompt.txt`: `system_prompt_t2i.txt` and `system_prompt_edit.txt`.

`client.py` also takes `--input/--output` for a batch over HTTP, and `--image`
(repeatable, in order) for a single `edit` request.

## Sampling defaults

Per task, matching each one's production inference settings. Unset flags fall
back to these; the run's first log line prints the **effective** values and
marks anything you overrode.

| | `t2i` | `edit` |
| --- | --- | --- |
| temperature | 1.0 | 1.0 |
| top_p | 0.95 | 0.95 |
| top_k | 20 | 20 |
| min_p | 0 | 0 |
| **presence_penalty** | **1.5** | **0** |
| max_new_tokens | 16256 | 24000 |
| thinking | on (required) | on (required) |

`presence_penalty` is the one that matters: the two values are not
interchangeable, and a wrong penalty does not fail loudly -- it quietly changes
the distribution you sample from. That is why there is no global default and why
the effective value is logged.

`transformers` has no native `presence_penalty` (`repetition_penalty` is
different math), so `run_transformers.py` implements the vLLM semantics as a
`LogitsProcessor`. Without it the t2i path would silently run at penalty 0.

Thinking is required: both models were trained with a `<think>` block and
degrade without it. The chat template opens one by default, and every entry
point also passes `enable_thinking=True` explicitly so the intent survives a
template change.

`--seed` defaults to 42, and what that buys you depends on the path:

- **`run_vllm.py` (offline batch) is reproducible.** Same input file, same
  sampling, same `--tp`, same graph/eager mode gives byte-identical output --
  verified across separate runs, including on a different GPU.
- **`client.py` against a server is not**, even with the same seed. The server
  batches concurrent requests, and a sampler seeded per request still sees a
  different batch composition each time. Two identical client runs differ. Use
  the offline path when you need reproducibility.
- Either way, `--enforce-eager` changes the numerics enough to diverge within a
  few hundred tokens, even under greedy decoding. Two batches rendered with
  different settings there are not comparable.

## Notes

**CUDA graphs are on by default.** On vLLM 0.19.1 this architecture's
linear-attention layers capture fine, and graphs are substantially faster, so
neither `serve.sh` nor `run_vllm.py` forces eager. `--enforce-eager` /
`EAGER=1` remain as escape hatches for other vLLM versions.

**Why one codebase.** The two tasks differ in exactly four places -- whether
images are accepted, whether `ratio_follow` exists, `presence_penalty`, and
`max_new_tokens` -- and all four live in one `Profile` dataclass in
`pe_core.py`. Everything else (chat template contract, thinking split, answer
parsing, output records, image resizing) is shared, so a fix to the parser
cannot land on one task and miss the other.

**Sockets.** Clusters often preset `GLOO_SOCKET_IFNAME` / `TP_SOCKET_IFNAME` to
interfaces that do not exist on the current host, which kills vLLM on init. Both
the runner and `serve.sh` unset any that name a missing interface.
