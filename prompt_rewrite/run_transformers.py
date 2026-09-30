#!/usr/bin/env python3
"""Prompt enhancer -- HuggingFace transformers reference impl, batch size 1.

The clear, hackable path: no serving stack, one case at a time. Use `run_vllm.py`
for anything beyond a sanity check.

    python run_transformers.py --task t2i  --ckpt Qwen/Qwen-Image-2.1-PE-T2I  \\
        --input data/t2i_example.jsonl  --output out.jsonl
    python run_transformers.py --task edit --ckpt Qwen/Qwen-Image-2.1-PE-I2I \\
        --input data/edit_example.jsonl --output out.jsonl

Both tasks load through `AutoProcessor` / `AutoModelForImageTextToText`, even
t2i, which sends no image: the processor is a superset of the tokenizer for this
architecture, so one code path serves both and the prompt bytes are identical to
what the vLLM path builds.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm
from transformers import (AutoModelForImageTextToText, AutoProcessor,
                          LogitsProcessor, LogitsProcessorList)

import pe_core as core


class PresencePenalty(LogitsProcessor):
    """vLLM-style presence penalty: subtract a constant from every token already
    generated.

    transformers has no native `presence_penalty`, and `repetition_penalty` is
    different math (multiplicative, and it also penalises the prompt). The t2i
    profile runs at 1.5, so getting this wrong is not cosmetic.
    """

    def __init__(self, penalty: float, prompt_len: int):
        self.penalty = penalty
        self.prompt_len = prompt_len

    def __call__(self, input_ids, scores):
        for b in range(input_ids.shape[0]):
            generated = input_ids[b, self.prompt_len:]
            if generated.numel():
                scores[b, generated.unique()] -= self.penalty
        return scores


@torch.inference_mode()
def rewrite(model, processor, messages: list[dict[str, Any]], *,
            max_new_tokens: int, temperature: float, top_p: float, top_k: int,
            presence_penalty: float, seed: int) -> tuple[str, str]:
    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        enable_thinking=True,
    ).to(model.device)

    # This architecture marks image spans with mm_token_type_ids and needs them to
    # compute M-RoPE. Some processor versions populate it from
    # apply_chat_template, some don't -- create it when missing.
    if "mm_token_type_ids" not in inputs and hasattr(processor, "create_mm_token_type_ids"):
        inputs["mm_token_type_ids"] = processor.create_mm_token_type_ids(inputs["input_ids"])

    prompt_len = inputs["input_ids"].shape[1]
    processors = LogitsProcessorList()
    if presence_penalty:
        processors.append(PresencePenalty(presence_penalty, prompt_len))

    torch.manual_seed(seed)
    out = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=temperature > 0,
        temperature=temperature if temperature > 0 else None,
        top_p=top_p if temperature > 0 else None,
        top_k=top_k if temperature > 0 else None,
        logits_processor=processors,
        pad_token_id=processor.tokenizer.eos_token_id,
    )
    text = processor.tokenizer.decode(out[0, prompt_len:], skip_special_tokens=True)
    return core.split_thinking(text)


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
    ap.add_argument("--dtype", default="bfloat16",
                    choices=["bfloat16", "float16", "float32"])
    ap.add_argument("--device", default="cuda")
    # Unset sampling flags fall back to the task profile (its production setting).
    ap.add_argument("--temperature", type=float, default=None)
    ap.add_argument("--top-p", type=float, default=None)
    ap.add_argument("--top-k", type=int, default=None)
    ap.add_argument("--presence-penalty", type=float, default=None)
    ap.add_argument("--max-new-tokens", type=int, default=None)
    ap.add_argument("--image-max-pixels", type=int, default=None)
    ap.add_argument("--seed", type=int, default=42,
                    help="Reproducible for a fixed model/dtype/device; changing "
                         "any of those changes the numerics.")
    ap.add_argument("--limit", type=int, default=0, help="Process only first N cases.")
    args = ap.parse_args()

    profile = core.get_profile(args.task)
    pick = lambda cli, default: default if cli is None else cli  # noqa: E731
    temperature = pick(args.temperature, profile.temperature)
    top_p = pick(args.top_p, profile.top_p)
    top_k = pick(args.top_k, profile.top_k)
    presence_penalty = pick(args.presence_penalty, profile.presence_penalty)
    max_new_tokens = pick(args.max_new_tokens, profile.max_new_tokens)
    image_max_pixels = pick(args.image_max_pixels, profile.image_max_pixels)

    system_prompt = core.load_system_prompt(args.system_prompt, args.ckpt)
    in_path = Path(args.input).resolve()
    base_dir = in_path.parent
    cases = core.load_cases(in_path, args.limit)
    # Fail on bad inputs before the model load, not 40 GB of weights later.
    image_paths = [core.resolve_image_paths(c, base_dir, profile) for c in cases]

    print(f"task={profile.name} cases={len(cases)} sampling: "
          f"{profile.sampling_summary(temperature=temperature, top_p=top_p, top_k=top_k, presence_penalty=presence_penalty, max_new_tokens=max_new_tokens)}", flush=True)
    print(f"Loading model from {args.ckpt} ...", flush=True)
    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16,
             "float32": torch.float32}[args.dtype]
    processor = AutoProcessor.from_pretrained(args.ckpt)
    # low_cpu_mem_usage streams weights straight to the target dtype without a
    # full-precision CPU copy; .to(device) then moves the single copy to GPU.
    model = AutoModelForImageTextToText.from_pretrained(
        args.ckpt, dtype=dtype, low_cpu_mem_usage=True).to(args.device).eval()

    records = []
    out_path = Path(args.output).resolve()
    for case, paths in zip(tqdm(cases, desc=f"pe-{profile.name}"), image_paths):
        images = [core.load_image(p, image_max_pixels) for p in paths]
        messages = core.build_messages(system_prompt, case["prompt"], images)
        thinking, answer = rewrite(
            model, processor, messages,
            max_new_tokens=max_new_tokens, temperature=temperature, top_p=top_p,
            top_k=top_k, presence_penalty=presence_penalty, seed=args.seed)
        records.append(core.build_record(case, thinking, answer, profile))
        # Rewrite the file each step: a long run killed halfway still leaves a
        # complete, valid JSONL of everything finished so far.
        core.write_records(out_path, records)
    print(f"Wrote {len(records)} records to {out_path}")
    core.report_parse_failures(records)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
