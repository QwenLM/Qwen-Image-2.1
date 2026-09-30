#!/usr/bin/env python3
"""Shared core for the Qwen3.5-VL 9B prompt-enhancer family: task profiles,
chat-message construction, answer parsing, and output records.

Two tasks ship on the same base architecture, the same tokenizer and the same
chat template -- they differ only in what goes in and what comes back:

    t2i   text-to-image prompt expansion.  Text in, no source image.
    edit  image-editing instruction rewrite. Text + 1..N source images in.

Each task has its own checkpoint and its own system prompt file; nothing is
shared between them at the prompt level. What *is* shared is everything in this
module, so a caller only picks a task and gets the right sampling parameters,
the right message shape and the right output contract.

Answer contract (the model emits this JSON at the end of the answer section):

    t2i   {"rewritten_prompt": "...", "wh_ratio": "16:9"}
    edit  {"rewritten_prompt": "...", "wh_ratio": "", "ratio_follow": "<image1>"}

`wh_ratio` and `ratio_follow` decide the canvas the downstream image model
renders on; `ratio_follow` only exists for `edit`, where there is a source image
whose framing the output can inherit.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# json_repair fixes answers that are *nearly* valid JSON (a trailing comma, an
# unescaped quote). Optional: without it the balanced-brace scan below still
# handles every well-formed answer, it just gives up a little sooner.
try:
    import json_repair  # type: ignore
except ImportError:
    json_repair = None


# --------------------------------------------------------------------------- #
# Task profiles
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Profile:
    """Everything that differs between the two tasks, in one place.

    The sampling numbers are the production inference settings for each task --
    notably `presence_penalty`, which is 1.5 for t2i and 0 for edit. They are
    not interchangeable, which is why there is no global default: a wrong
    penalty does not fail, it quietly changes the distribution you sample from.
    """

    name: str
    takes_images: bool
    #: Answer fields to surface. `ratio_follow` is t2i-meaningless and stays "".
    has_ratio_follow: bool
    temperature: float = 1.0
    top_p: float = 0.95
    top_k: int = 20
    min_p: float = 0.0
    presence_penalty: float = 0.0
    max_new_tokens: int = 24000
    #: Cap on each source image's pixel count, matching training.
    image_max_pixels: int = 1024 * 1024

    def sampling_summary(self, **overrides: Any) -> str:
        """One-line sampling summary. Pass the *effective* values as overrides:
        logging profile defaults while CLI flags are in force is exactly the kind
        of stale-log trap that makes two runs look identical when they are not.
        """
        eff = {k: overrides.get(k, getattr(self, k)) for k in
               ("temperature", "top_p", "top_k", "min_p", "presence_penalty",
                "max_new_tokens")}
        changed = {k for k, v in eff.items() if v != getattr(self, k)}
        parts = [f"{k}={v}" + ("*" if k in changed else "") for k, v in eff.items()]
        tail = "   (* = overridden on the command line)" if changed else ""
        return " ".join(parts) + tail


PROFILES: dict[str, Profile] = {
    "t2i": Profile(
        name="t2i",
        takes_images=False,
        has_ratio_follow=False,
        presence_penalty=1.5,
        max_new_tokens=16256,
    ),
    "edit": Profile(
        name="edit",
        takes_images=True,
        has_ratio_follow=True,
        presence_penalty=0.0,
        max_new_tokens=24000,
    ),
}


def get_profile(task: str) -> Profile:
    try:
        return PROFILES[task]
    except KeyError:
        raise SystemExit(f"unknown --task {task!r}; choose one of {sorted(PROFILES)}")


# --------------------------------------------------------------------------- #
# System prompt
# --------------------------------------------------------------------------- #
def load_system_prompt(explicit: str | None, ckpt: str | None) -> str:
    """Resolve the system prompt: `--system-prompt` wins, else the checkpoint's
    own `system_prompt.txt`, from a local directory or the Hugging Face Hub.

    Preferring the file that ships *inside* the checkpoint is deliberate. The
    expert's answer contract is part of what the weights were trained on, so a
    prompt that travels with the weights can't drift out of sync with them --
    swapping checkpoints and forgetting to swap the prompt is the failure this
    avoids, and it fails silently (fluent output, wrong contract).
    """
    if explicit:
        path = Path(explicit)
        if not path.is_file():
            raise SystemExit(f"--system-prompt {explicit!r} is not a file")
        return path.read_text(encoding="utf-8").strip()
    if ckpt:
        checkpoint = Path(ckpt)
        path = checkpoint / "system_prompt.txt"
        if path.is_file():
            return path.read_text(encoding="utf-8").strip()
        # Existing local checkpoints must not fall through to a similarly named
        # Hub repository when their prompt is missing.
        is_local = (checkpoint.exists() or checkpoint.is_absolute()
                    or ckpt.startswith(("./", "../", "~")))
        if not is_local:
            # Lazy import: explicit files and local checkpoints need no Hub client.
            from huggingface_hub import hf_hub_download
            from huggingface_hub.errors import (HFValidationError, HfHubHTTPError,
                                                LocalEntryNotFoundError)

            try:
                path = Path(hf_hub_download(repo_id=ckpt, filename="system_prompt.txt"))
            except (HFValidationError, HfHubHTTPError, LocalEntryNotFoundError, OSError) as exc:
                raise SystemExit(
                    f"cannot load system_prompt.txt from {ckpt!r}: {exc}. "
                    "Pass --system-prompt <file> with the prompt for this checkpoint."
                ) from exc
            return path.read_text(encoding="utf-8").strip()
    raise SystemExit(
        "no system prompt: pass --system-prompt <file>, or put system_prompt.txt "
        "in the checkpoint directory. Each task has its own; they are not "
        "interchangeable."
    )


# --------------------------------------------------------------------------- #
# Input
# --------------------------------------------------------------------------- #
def load_cases(input_path: Path, limit: int = 0) -> list[dict[str, Any]]:
    """Read the input JSONL: ``{id, prompt, input_images?, task_type?}``."""
    with open(input_path, encoding="utf-8") as f:
        cases = [json.loads(line) for line in f if line.strip()]
    if limit:
        cases = cases[:limit]
    for i, case in enumerate(cases):
        if "prompt" not in case:
            raise SystemExit(f"line {i + 1} of {input_path} has no 'prompt' field")
        case.setdefault("id", str(i))
        case.setdefault("input_images", [])
        case.setdefault("task_type", "")
    return cases


def resolve_image_paths(case: dict[str, Any], base_dir: Path,
                        profile: Profile) -> list[Path]:
    """Resolve a case's source images against the JSONL's directory.

    Raises on a t2i case that carries images: silently dropping them would look
    like a successful run of the wrong experiment.
    """
    raw = case.get("input_images") or []
    if not profile.takes_images:
        if raw:
            raise SystemExit(
                f"case {case['id']!r}: task 't2i' takes no source images but the "
                f"record lists {len(raw)}. Use --task edit for image inputs."
            )
        return []
    if not raw:
        raise SystemExit(f"case {case['id']!r}: task 'edit' needs at least one input image")
    paths = [Path(p) if Path(p).is_absolute() else base_dir / p for p in raw]
    missing = [str(p) for p in paths if not p.is_file()]
    if missing:
        raise SystemExit(f"case {case['id']!r}: missing input image(s): {missing}")
    return paths


def load_image(path: Path, max_pixels: int):
    """Open an image and downscale it so W*H <= max_pixels, preserving aspect.

    Matches training's IMAGE_MAX_PIXELS. Imported lazily so the t2i path needs
    no Pillow.
    """
    from PIL import Image

    im = Image.open(path).convert("RGB")
    w, h = im.size
    if max_pixels and w * h > max_pixels:
        s = (max_pixels / float(w * h)) ** 0.5
        im = im.resize((max(1, int(w * s)), max(1, int(h * s))), Image.LANCZOS)
    return im


def build_messages(system_prompt: str, user_prompt: str,
                   images: list[Any] | None = None) -> list[dict[str, Any]]:
    """Build the two-turn conversation.

    Images come first and in order, because the system prompt tells the model to
    address them as ``<image1>``, ``<image2>``, ... -- reordering them silently
    re-points every reference in the rewrite.

    `images` entries may be PIL images (transformers path) or ``data:``/``http``
    URI strings (vLLM path); both are what the respective backend expects.
    """
    user_content: list[dict[str, Any]] = []
    for im in images or []:
        if isinstance(im, str):
            user_content.append({"type": "image_url", "image_url": {"url": im}})
        else:
            user_content.append({"type": "image", "image": im})
    user_content.append({"type": "text", "text": user_prompt})
    return [
        {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
        {"role": "user", "content": user_content},
    ]


# --------------------------------------------------------------------------- #
# Answer parsing
# --------------------------------------------------------------------------- #
def split_thinking(text: str) -> tuple[str, str]:
    """Split a decoded generation into (thinking, answer).

    The chat template pre-fills ``<think>\\n`` before generation, so the decoded
    text normally starts *inside* the thinking block and closes it with
    ``</think>``.
    """
    if "</think>" in text:
        think, _, answer = text.partition("</think>")
        if "<think>" in think:
            think = think.partition("<think>")[2]
        return think.strip(), answer.strip()
    if "<think>" in text:
        # Unterminated thinking block -- the generation hit the token budget.
        return text.partition("<think>")[2].strip(), ""
    return "", text.strip()


def _balanced_spans(answer: str) -> list[str]:
    """Return every balanced top-level ``{...}`` span in `answer`, in order.

    A single greedy ``\\{.*\\}`` is not enough: any brace in prose after the
    object stretches the match past its real end and the parse fails silently.
    Braces inside JSON string literals are skipped, so a rewrite containing
    ``{`` does not break the scan.
    """
    spans: list[str] = []
    depth = 0
    start = -1
    in_str = False
    escaped = False
    for i, ch in enumerate(answer):
        if in_str:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}" and depth > 0:
            depth -= 1
            if depth == 0 and start >= 0:
                spans.append(answer[start:i + 1])
    return spans


def _as_obj(candidate: str) -> dict[str, Any] | None:
    """Strict json first, then json_repair if it is installed."""
    try:
        obj = json.loads(candidate)
    except json.JSONDecodeError:
        if json_repair is None:
            return None
        obj = json_repair.repair_json(candidate, return_objects=True)
        if isinstance(obj, list):
            obj = obj[0] if obj else None
    return obj if isinstance(obj, dict) else None


def parse_answer(answer: str, profile: Profile) -> dict[str, Any]:
    """Parse the answer section into the task's declared fields.

    Returns ``{"positive_prompt", "wh_ratio", "ratio_follow", "parse_ok"}``.

    On failure `positive_prompt` falls back to the raw answer text so the model's
    output is never lost -- but `parse_ok` is then False, which is the only way
    to tell a fallback from a clean parse once the record is on disk. A batch
    that silently falls back on a few percent of its rows looks exactly like a
    clean batch; audit it:

        jq -s 'map(select(.parse_ok | not)) | length' out.jsonl
    """
    answer = (answer or "").strip()
    # The answer object is emitted at the end, so scan candidates last-first.
    candidates = list(reversed(_balanced_spans(answer)))
    for candidate in candidates:
        obj = _as_obj(candidate)
        if obj is None:
            continue
        # Some training runs mis-typed the key as `rewrited_prompt`; accept both.
        rewritten = obj.get("rewritten_prompt") or obj.get("rewrited_prompt")
        if not isinstance(rewritten, str) or not rewritten.strip():
            continue
        return {
            "positive_prompt": rewritten.strip(),
            "wh_ratio": str(obj.get("wh_ratio") or "").strip(),
            "ratio_follow": (str(obj.get("ratio_follow") or "").strip()
                             if profile.has_ratio_follow else ""),
            "parse_ok": True,
        }
    return {"positive_prompt": answer, "wh_ratio": "", "ratio_follow": "",
            "parse_ok": False}


# --------------------------------------------------------------------------- #
# Output
# --------------------------------------------------------------------------- #
#: Stable output field order. `parse_ok` stays last so later additions never
#: move it.
OUTPUT_FIELDS = (
    "id", "task", "raw_prompt", "input_images", "task_type", "thinking",
    "positive_prompt", "negative_prompt", "wh_ratio", "ratio_follow", "parse_ok",
)


def build_record(case: dict[str, Any], thinking: str, answer: str,
                 profile: Profile) -> dict[str, Any]:
    """Build one output record with fields in ``OUTPUT_FIELDS`` order."""
    parsed = parse_answer(answer, profile)
    record = {
        "id": case["id"],
        "task": profile.name,
        "raw_prompt": case["prompt"],
        "input_images": case.get("input_images", []),
        "task_type": case.get("task_type", ""),
        "thinking": thinking,
        "positive_prompt": parsed["positive_prompt"],
        "negative_prompt": "",
        "wh_ratio": parsed["wh_ratio"],
        "ratio_follow": parsed["ratio_follow"],
        "parse_ok": parsed["parse_ok"],
    }
    return {k: record[k] for k in OUTPUT_FIELDS}


def write_records(out_path: Path, records: list[dict[str, Any]]) -> None:
    """Write the JSONL atomically (.tmp + replace) so a reader never sees a
    half-written file and a crashed run leaves no truncated output."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(out_path.suffix + f".{os.getpid()}.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    os.replace(tmp, out_path)


def report_parse_failures(records: list[dict[str, Any]]) -> int:
    """Print a one-line parse summary; return the number of failures."""
    bad = [r for r in records if not r.get("parse_ok")]
    if bad:
        ids = ", ".join(str(r["id"]) for r in bad[:10])
        more = f", ... (+{len(bad) - 10})" if len(bad) > 10 else ""
        print(f"WARNING: {len(bad)}/{len(records)} answers did not parse as the "
              f"expected JSON object; positive_prompt fell back to the raw answer "
              f"text and wh_ratio/ratio_follow are empty for them. "
              f"ids: {ids}{more}", flush=True)
    else:
        print(f"Parsed {len(records)}/{len(records)} answers cleanly "
              f"(parse_ok=true).", flush=True)
    return len(bad)
