#!/usr/bin/env python3
"""Prompt enhancer -- client for a running `serve.sh` endpoint (both tasks).

One prompt from the command line, or a whole JSONL. Output records are identical
to the offline runners', so online and offline results are interchangeable.

    # one t2i prompt
    python client.py --task t2i \\
        --model Qwen/Qwen-Image-2.1-PE-T2I "a corgi playing guitar in the rain"

    # one edit instruction with its source image(s)
    python client.py --task edit \\
        --model Qwen/Qwen-Image-2.1-PE-I2I --image a.png --image b.png \\
        "put <image1>'s subject into <image2>'s scene"

    # a batch, same JSONL format as the offline runners
    python client.py --task edit \\
        --model Qwen/Qwen-Image-2.1-PE-I2I --input data/edit_example.jsonl --output out.jsonl
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from openai import OpenAI

import pe_core as core


def image_to_data_uri(path: Path, max_pixels: int) -> str:
    im = core.load_image(path, max_pixels)
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def _flatten(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Collapse a single text part to a plain string.

    The OpenAI chat schema accepts either; a bare string is what the server's own
    template examples use, and it keeps a text-only (t2i) request byte-identical
    to what `pe_rewrite`-style clients send.
    """
    out = []
    for m in messages:
        content = m["content"]
        if isinstance(content, list) and len(content) == 1 and content[0].get("type") == "text":
            out.append({"role": m["role"], "content": content[0]["text"]})
        else:
            out.append(m)
    return out


def rewrite_one(client: OpenAI, model: str, messages: list[dict[str, Any]],
                *, temperature: float, top_p: float, top_k: int, min_p: float,
                presence_penalty: float, max_tokens: int, seed: int,
                timeout: float) -> tuple[str, str]:
    """Return (thinking, answer) for one request.

    Streaming, so a 10k-token thinking block is not a silent multi-minute wait.
    `reasoning_content` is what `--reasoning-parser qwen3` gives us; if the server
    was started without it, the thinking arrives inline and `split_thinking`
    recovers it from the `</think>` tag instead -- so both server configurations
    produce the same record.
    """
    think: list[str] = []
    content: list[str] = []
    stream = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        top_p=top_p,
        presence_penalty=presence_penalty,
        max_tokens=max_tokens,
        seed=seed,
        stream=True,
        timeout=timeout,
        # top_k / min_p / enable_thinking are vLLM extensions, not OpenAI fields.
        extra_body={"top_k": top_k, "min_p": min_p,
                    "chat_template_kwargs": {"enable_thinking": True}},
    )
    for chunk in stream:
        if not chunk.choices:
            continue
        delta = chunk.choices[0].delta
        reasoning = getattr(delta, "reasoning_content", None) or getattr(delta, "reasoning", None)
        if reasoning:
            think.append(reasoning)
        if getattr(delta, "content", None):
            content.append(delta.content)
    answer = "".join(content)
    if think:
        return "".join(think).strip(), answer.strip()
    return core.split_thinking(answer)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("prompt", nargs="?", help="Single prompt (omit when using --input).")
    ap.add_argument("--task", required=True, choices=sorted(core.PROFILES))
    ap.add_argument("--model", required=True,
                    help="served-model-name given to serve.sh (its NAME).")
    ap.add_argument("--system-prompt", default=None,
                    help="System prompt file (default: system_prompt.txt from "
                         "--model, as a local directory or Hub id). Required "
                         "when --model is a custom served-model-name.")
    ap.add_argument("--url", default=os.environ.get("PE_URL"),
                    help="Base URL, e.g. http://localhost:8100/v1 "
                         "(default: built from --host/--port).")
    ap.add_argument("--host", default="localhost")
    ap.add_argument("--port", type=int, default=8100)
    ap.add_argument("--image", action="append", default=[], metavar="PATH",
                    help="Source image for a single edit request; repeat, in order.")
    ap.add_argument("--input", help="JSONL batch input (mutually exclusive with `prompt`).")
    ap.add_argument("--output", help="JSONL batch output (required with --input).")
    ap.add_argument("--workers", type=int, default=4,
                    help="Concurrent requests for a batch.")
    ap.add_argument("--timeout", type=float, default=900.0)
    # Unset sampling flags fall back to the task profile (its production setting).
    ap.add_argument("--temperature", type=float, default=None)
    ap.add_argument("--top-p", type=float, default=None)
    ap.add_argument("--top-k", type=int, default=None)
    ap.add_argument("--min-p", type=float, default=None)
    ap.add_argument("--presence-penalty", type=float, default=None)
    ap.add_argument("--max-new-tokens", type=int, default=None)
    ap.add_argument("--image-max-pixels", type=int, default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    if bool(args.input) == bool(args.prompt):
        ap.error("give exactly one of: a single `prompt` argument, or --input <jsonl>")
    if args.input and not args.output:
        ap.error("--input needs --output")

    profile = core.get_profile(args.task)
    pick = lambda cli, default: default if cli is None else cli  # noqa: E731
    sampling = dict(
        temperature=pick(args.temperature, profile.temperature),
        top_p=pick(args.top_p, profile.top_p),
        top_k=pick(args.top_k, profile.top_k),
        min_p=pick(args.min_p, profile.min_p),
        presence_penalty=pick(args.presence_penalty, profile.presence_penalty),
        max_tokens=pick(args.max_new_tokens, profile.max_new_tokens),
        seed=args.seed,
        timeout=args.timeout,
    )
    image_max_pixels = pick(args.image_max_pixels, profile.image_max_pixels)
    system_prompt = core.load_system_prompt(args.system_prompt, args.model)
    url = args.url or f"http://{args.host}:{args.port}/v1"
    client = OpenAI(base_url=url, api_key="dummy")

    if args.prompt:
        case = {"id": "cli", "prompt": args.prompt, "input_images": args.image,
                "task_type": ""}
        paths = core.resolve_image_paths(case, Path.cwd(), profile)
        uris = [image_to_data_uri(p, image_max_pixels) for p in paths]
        messages = _flatten(core.build_messages(system_prompt, args.prompt, uris))
        thinking, answer = rewrite_one(client, args.model, messages, **sampling)
        record = core.build_record(case, thinking, answer, profile)
        if not record["parse_ok"]:
            print("WARNING: answer did not parse as the expected JSON object; "
                  "positive_prompt holds the raw answer text.", file=sys.stderr)
        print(json.dumps(record, ensure_ascii=False, indent=2))
        return 0 if record["parse_ok"] else 1

    in_path = Path(args.input).resolve()
    base_dir = in_path.parent
    cases = core.load_cases(in_path, args.limit)
    image_paths = [core.resolve_image_paths(c, base_dir, profile) for c in cases]
    print(f"task={profile.name} cases={len(cases)} url={url} model={args.model}",
          flush=True)

    def run(item):
        case, paths = item
        uris = [image_to_data_uri(p, image_max_pixels) for p in paths]
        messages = _flatten(core.build_messages(system_prompt, case["prompt"], uris))
        thinking, answer = rewrite_one(client, args.model, messages, **sampling)
        return core.build_record(case, thinking, answer, profile)

    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        records = list(pool.map(run, zip(cases, image_paths)))
    out_path = Path(args.output).resolve()
    core.write_records(out_path, records)
    print(f"Wrote {len(records)} records to {out_path}")
    return 1 if core.report_parse_failures(records) else 0


if __name__ == "__main__":
    raise SystemExit(main())
