#!/usr/bin/env python3
"""Prompt enhancer -- vLLM offline batch, for both the t2i and edit tasks.

Recommended for anything beyond a few samples. One process loads the checkpoint
once and runs the whole JSONL through `LLM.chat()`.

    # text-to-image prompt expansion (no source images)
    python run_vllm.py --task t2i --ckpt Qwen/Qwen-Image-2.1-PE-T2I \\
        --input data/t2i_example.jsonl --output out.jsonl

    # image-editing instruction rewrite (1..N source images per case)
    python run_vllm.py --task edit --ckpt Qwen/Qwen-Image-2.1-PE-I2I \\
        --input data/edit_example.jsonl --output out.jsonl

The system prompt comes from the checkpoint's own `system_prompt.txt` (local
directory or Hugging Face Hub), or from `--system-prompt <file>`. The two tasks
have separate prompts; there is no unified one.
"""

from __future__ import annotations

import argparse
import base64
import io
import os
from pathlib import Path

# Some clusters preset GLOO/NCCL socket-interface envs to names that do not exist
# on the current host (e.g. bond1); vLLM's gloo backend then dies on init. Drop
# any that name a missing interface so vLLM auto-selects a live one.
for _v in ("GLOO_SOCKET_IFNAME", "TP_SOCKET_IFNAME", "NVSHMEM_BOOTSTRAP_UID_SOCK_IFNAME"):
    _iface = os.environ.get(_v)
    if _iface and not os.path.isdir(f"/sys/class/net/{_iface}"):
        os.environ.pop(_v, None)

from vllm import LLM, SamplingParams  # noqa: E402

import pe_core as core  # noqa: E402


def image_to_data_uri(path: Path, max_pixels: int) -> str:
    """Encode a (downscaled) source image as a PNG data URI for `LLM.chat`."""
    im = core.load_image(path, max_pixels)
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=sorted(core.PROFILES),
                    help="t2i = text only; edit = text + source image(s).")
    ap.add_argument("--ckpt", required=True, help="Local HF dir or Hub id.")
    ap.add_argument("--input", required=True,
                    help="JSONL: {id, prompt, input_images?, task_type?}.")
    ap.add_argument("--output", required=True, help="Output JSONL.")
    ap.add_argument("--system-prompt", default=None,
                    help="System prompt file (default: system_prompt.txt from "
                         "--ckpt, as a local directory or Hub id).")
    ap.add_argument("--tp", type=int, default=1, help="Tensor-parallel size.")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--max-model-len", type=int, default=24576)
    ap.add_argument("--limit-mm-per-prompt-image", type=int, default=10,
                    help="Max source images per case (edit only).")
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    ap.add_argument("--swap-space", type=int, default=8, help="CPU KV cache (GiB).")
    ap.add_argument("--max-num-seqs", type=int, default=32, help="Concurrent decode slots.")
    ap.add_argument("--enforce-eager", action="store_true",
                    help="Disable CUDA graph capture. Not needed on vLLM 0.19.1, "
                         "where this architecture's linear-attention layers run "
                         "fine under the default graph capture -- keep it as an "
                         "escape hatch for other vLLM versions.")
    # Sampling: unset means "use the task profile", which is the production
    # setting for that task. They are not interchangeable between tasks.
    ap.add_argument("--temperature", type=float, default=None)
    ap.add_argument("--top-p", type=float, default=None)
    ap.add_argument("--top-k", type=int, default=None)
    ap.add_argument("--min-p", type=float, default=None)
    ap.add_argument("--presence-penalty", type=float, default=None)
    ap.add_argument("--max-new-tokens", type=int, default=None)
    ap.add_argument("--image-max-pixels", type=int, default=None)
    ap.add_argument("--seed", type=int, default=42,
                    help="Engine and sampling seed. Reproducible for a fixed "
                         "engine config (same --tp, same graph/eager mode); "
                         "changing the engine changes the numerics.")
    ap.add_argument("--limit", type=int, default=0, help="Process only first N cases.")
    args = ap.parse_args()

    profile = core.get_profile(args.task)
    pick = lambda cli, default: default if cli is None else cli  # noqa: E731
    temperature = pick(args.temperature, profile.temperature)
    top_p = pick(args.top_p, profile.top_p)
    top_k = pick(args.top_k, profile.top_k)
    min_p = pick(args.min_p, profile.min_p)
    presence_penalty = pick(args.presence_penalty, profile.presence_penalty)
    max_new_tokens = pick(args.max_new_tokens, profile.max_new_tokens)
    image_max_pixels = pick(args.image_max_pixels, profile.image_max_pixels)

    system_prompt = core.load_system_prompt(args.system_prompt, args.ckpt)
    in_path = Path(args.input).resolve()
    base_dir = in_path.parent
    cases = core.load_cases(in_path, args.limit)

    # Resolve and encode every input before touching the GPU: a missing image or a
    # t2i case carrying images should fail in seconds, not after a model load.
    conversations = []
    for case in cases:
        paths = core.resolve_image_paths(case, base_dir, profile)
        uris = [image_to_data_uri(p, image_max_pixels) for p in paths]
        conversations.append(core.build_messages(system_prompt, case["prompt"], uris))

    print(f"task={profile.name} cases={len(cases)} sampling: "
          f"{profile.sampling_summary(temperature=temperature, top_p=top_p, top_k=top_k, min_p=min_p, presence_penalty=presence_penalty, max_new_tokens=max_new_tokens)}", flush=True)
    print(f"Loading vLLM engine from {args.ckpt} (tp={args.tp}) ...", flush=True)
    llm = LLM(
        model=args.ckpt,
        dtype=args.dtype,
        tensor_parallel_size=args.tp,
        max_model_len=args.max_model_len,
        limit_mm_per_prompt=({"image": args.limit_mm_per_prompt_image}
                             if profile.takes_images else None),
        gpu_memory_utilization=args.gpu_memory_utilization,
        swap_space=args.swap_space,
        max_num_seqs=args.max_num_seqs,
        enforce_eager=args.enforce_eager,
        enable_prefix_caching=False,
        seed=args.seed,
        trust_remote_code=True,
    )
    sampling = SamplingParams(
        temperature=temperature, top_p=top_p, top_k=top_k, min_p=min_p,
        presence_penalty=presence_penalty, max_tokens=max_new_tokens, seed=args.seed,
    )

    # enable_thinking is passed at the top level so vLLM forwards it to
    # apply_chat_template. The template defaults to thinking anyway; passing it
    # explicitly keeps the intent visible and survives a template change.
    outputs = llm.chat(conversations, sampling_params=sampling,
                       chat_template_kwargs={"enable_thinking": True})

    records = []
    for case, out in zip(cases, outputs):
        thinking, answer = core.split_thinking(out.outputs[0].text)
        records.append(core.build_record(case, thinking, answer, profile))
    out_path = Path(args.output).resolve()
    core.write_records(out_path, records)
    print(f"Wrote {len(records)} records to {out_path}")
    core.report_parse_failures(records)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
