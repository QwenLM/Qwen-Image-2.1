#!/usr/bin/env python3
"""Qwen-Image-2.1 Studio.

A Gradio app for everything the model does: text to image, editing with up to 10
reference images, local edits marked by drawing on the image, transparent (RGBA)
output, and the official prompt enhancers (served by prompt_rewrite/serve.sh).

    python studio/app.py [--config FILE] [--host HOST] [--port PORT] [--share]

Jobs run in a background worker and the page polls their state, so switching
browser tabs, reloading or closing the page never interrupts a generation. Reopen
the page and it shows the job where it is, with the inputs that started it.
"""

from __future__ import annotations

import argparse
import atexit
import gc
import html
import json
import math
import os
import random
import signal
import subprocess
import sys
import threading
import time
import tomllib
import traceback
import urllib.request
from datetime import datetime
from functools import partial
from pathlib import Path

import gradio as gr
import torch
from PIL import Image, ImageOps

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
PE_DIR = REPO / "prompt_rewrite"
sys.path.insert(0, str(PE_DIR))
import client as pe_client  # noqa: E402  (message flattening, image encoding)
import pe_core as core  # noqa: E402  (task profiles, system prompts, answer parsing)


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
def _parse_args():
    ap = argparse.ArgumentParser(description="Qwen-Image-2.1 Studio")
    ap.add_argument("--config", default=str(HERE / "config.toml"),
                    help="Config file. A config.local.toml next to it overrides its values.")
    ap.add_argument("--host", help="Overrides server.host")
    ap.add_argument("--port", type=int, help="Overrides server.port")
    ap.add_argument("--share", action="store_true", help="Create a public gradio.live link")
    return ap.parse_known_args()[0]


def _merge(base: dict, over: dict) -> dict:
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _merge(base[k], v)
        else:
            base[k] = v
    return base


def load_config(path: Path) -> dict:
    with open(path, "rb") as f:
        cfg = tomllib.load(f)
    local = path.with_name(f"{path.stem}.local.toml")
    if local.is_file():
        with open(local, "rb") as f:
            _merge(cfg, tomllib.load(f))
    return cfg


ARGS = _parse_args()
CONFIG_PATH = Path(ARGS.config).resolve()
CFG = load_config(CONFIG_PATH)
PIPE_CFG, PE_CFG, DEF, OUT_CFG = CFG["pipeline"], CFG["enhancer"], CFG["defaults"], CFG["output"]


def _path(p: str) -> Path:
    """Expand ~ and make relative paths absolute from the repository root. Symlinks
    are kept as they are: resolving a venv's python would escape the venv."""
    q = Path(os.path.expanduser(str(p)))
    return Path(os.path.abspath(q if q.is_absolute() else REPO / q))


def _show(p) -> str:
    """A path for display: relative to the repository when it is inside it."""
    try:
        return str(Path(os.path.abspath(p)).relative_to(REPO))
    except ValueError:
        return str(p)


OUT_DIR = _path(OUT_CFG.get("dir", "studio/outputs"))
DEVICE = PIPE_CFG.get("device", "cuda")
GIB = 1024 ** 3

# --------------------------------------------------------------------------- #
# Sizes
# --------------------------------------------------------------------------- #
# The official 2K sizes from the README. Other resolutions keep the ratio at about
# resolution x resolution pixels, in multiples of 32, the way the pipeline does.
RATIOS_2K = {
    "1:1": (2048, 2048), "4:3": (2400, 1792), "3:4": (1792, 2400),
    "3:2": (2528, 1696), "2:3": (1696, 2528), "16:9": (2752, 1536),
    "9:16": (1536, 2752),
}
RATIO_CHOICES = list(RATIOS_2K) + ["21:9", "9:21", "2:1", "1:2", "4:5", "5:4"]
RESOLUTIONS = {"1K": 1024, "1.5K": 1536, "2K": 2048}


def _snap(v: float) -> int:
    return max(32, int(round(v / 32)) * 32)


def _parse_ratio(ratio: str) -> float | None:
    try:
        a, b = (float(x) for x in ratio.strip().split(":"))
        return a / b if a > 0 and b > 0 else None
    except (ValueError, AttributeError):
        return None


def _size_from_area(r: float, side: int) -> tuple[int, int]:
    w = math.sqrt(side * side * r)
    return _snap(w), _snap(w / r)


def size_for_ratio(ratio: str, res: str) -> tuple[int, int]:
    if res == "2K" and ratio in RATIOS_2K:
        return RATIOS_2K[ratio]
    return _size_from_area(_parse_ratio(ratio) or 1.0, RESOLUTIONS[res])


def size_like(img: Image.Image, res: str) -> tuple[int, int]:
    """Canvas with the same aspect as a reference image (the pipeline's default)."""
    return _size_from_area(img.width / img.height, RESOLUTIONS[res])


# --------------------------------------------------------------------------- #
# Image model
# --------------------------------------------------------------------------- #
class Stopped(Exception):
    pass


def _free_cuda():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _free_gib() -> tuple[float, float]:
    """Free VRAM across all processes, after returning this process's cached blocks."""
    _free_cuda()
    free, total = torch.cuda.mem_get_info()
    return free / GIB, total / GIB


class Engine:
    """Owns the diffusers pipeline and where its weights live."""

    def __init__(self):
        self.pipe = None
        self.stop = threading.Event()
        self.lock = threading.RLock()

    @property
    def offload(self) -> str:
        return PIPE_CFG.get("offload", "none")

    def load(self, notify=None):
        with self.lock:
            if self.pipe is None:
                from diffusers import QwenImage21Pipeline

                if notify:
                    notify("Loading Qwen-Image-2.1 (about 30 s the first time)")
                dtype = getattr(torch, PIPE_CFG.get("dtype", "bfloat16"))
                model = PIPE_CFG["model"]
                if _path(model).exists():
                    model = str(_path(model))
                pipe = QwenImage21Pipeline.from_pretrained(model, dtype=dtype)
                if self.offload == "model":
                    pipe.enable_model_cpu_offload(device=DEVICE)
                elif self.offload == "sequential":
                    pipe.enable_sequential_cpu_offload(device=DEVICE)
                pipe.set_progress_bar_config(disable=True)
                self.pipe = pipe
            if not self.on_gpu() and self.offload == "none":
                if notify:
                    notify("Moving the image model to the GPU")
                self.pipe.to(DEVICE)
            return self.pipe

    def on_gpu(self) -> bool:
        return (self.pipe is not None and self.offload == "none"
                and self.pipe.transformer.device.type == "cuda")

    def to_cpu(self):
        """Park the weights in CPU RAM (seconds to undo) to free VRAM."""
        if self.on_gpu():
            self.pipe.to("cpu")
            _free_cuda()

    def unload(self):
        with self.lock:
            self.pipe = None
            _free_cuda()

    def status(self) -> str:
        if self.pipe is None:
            return "not loaded"
        if self.offload != "none":
            return f"loaded (offload: {self.offload})"
        return "loaded on the GPU" if self.on_gpu() else "parked in CPU RAM"

    def weights_gib(self) -> float:
        if self.pipe is None:
            return 33.0
        n = 0
        for m in (self.pipe.transformer, self.pipe.text_encoder, self.pipe.vae):
            n += sum(p.numel() * p.element_size() for p in m.parameters())
        return n / GIB


ENGINE = Engine()


def _pipeline_need_gib(w: int, h: int, n: int, n_refs: int, res: str) -> float:
    """Rough VRAM the image model needs beyond what is already resident.
    Measured: 2048x2048, one image, no references peaks at about 57 GB with about
    33 GB of weights."""
    act = 4.0 + 20.0 * (w * h * n) / (2048 * 2048)
    act += n_refs * 3.0 * (RESOLUTIONS[res] ** 2) / (1024 * 1024)
    if ENGINE.offload == "none":
        return act + (0 if ENGINE.on_gpu() else ENGINE.weights_gib())
    return act + 18.0


# --------------------------------------------------------------------------- #
# Prompt enhancer (vLLM server started from prompt_rewrite/serve.sh)
# --------------------------------------------------------------------------- #
class Enhancer:
    TASKS = ("t2i", "edit")

    def __init__(self):
        self.procs: dict[str, subprocess.Popen] = {}
        self.names: dict[str, str] = {}

    def model(self, task):
        return PE_CFG["t2i_model"] if task == "t2i" else PE_CFG["edit_model"]

    def port(self, task):
        return int(PE_CFG[f"{task}_port"])

    def util(self, task):
        return float(PE_CFG[f"{task}_gpu_memory_utilization"])

    def python(self) -> str:
        return str(_path(PE_CFG["python"])) if PE_CFG.get("python") else sys.executable

    def healthy(self, task) -> bool:
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{self.port(task)}/health", timeout=2) as r:
                return r.status == 200
        except Exception:
            return False

    def running(self, task) -> bool:
        p = self.procs.get(task)
        return p is not None and p.poll() is None

    def served_name(self, task) -> str:
        if task not in self.names:
            with urllib.request.urlopen(f"http://127.0.0.1:{self.port(task)}/v1/models", timeout=5) as r:
                self.names[task] = json.load(r)["data"][0]["id"]
        return self.names[task]

    def stop(self, task):
        p = self.procs.pop(task, None)
        self.names.pop(task, None)
        if p is None or p.poll() is not None:
            return
        try:
            os.killpg(p.pid, signal.SIGTERM)
            p.wait(timeout=60)
        except subprocess.TimeoutExpired:
            os.killpg(p.pid, signal.SIGKILL)
            p.wait()
        except ProcessLookupError:
            pass
        time.sleep(2)  # let the driver release the memory

    def stop_all(self):
        for t in list(self.procs):
            self.stop(t)

    def status(self) -> str:
        parts = []
        for t in self.TASKS:
            if self.healthy(t):
                parts.append(f"{t}: ready" + ("" if self.running(t) else " (started elsewhere)"))
            elif self.running(t):
                parts.append(f"{t}: starting")
            else:
                parts.append(f"{t}: stopped")
        return ", ".join(parts)

    def ensure(self, task, notify):
        """Start the task's server unless it is already up."""
        if self.healthy(task):
            return
        if PE_CFG.get("one_at_a_time", True):
            for other in self.TASKS:
                if other != task:
                    self.stop(other)
        free, total = _free_gib()
        need = self.util(task) * total + 1.0
        if free < need and ENGINE.on_gpu():
            notify("Parking the image model in CPU RAM to make room for the enhancer")
            ENGINE.to_cpu()
            free, total = _free_gib()
        if free < need:
            raise gr.Error(
                f"Not enough free GPU memory for the {task} enhancer: it reserves "
                f"{need - 1:.0f} GiB but only {free:.0f} GiB is free. Lower "
                f"{task}_gpu_memory_utilization in the config or free some VRAM.")
        log_dir = OUT_DIR / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / f"enhancer_{task}.log"
        env = dict(os.environ,
                   CKPT=self.model(task), PORT=str(self.port(task)), PY=self.python(),
                   MEM_UTIL=str(self.util(task)), MAX_LEN=str(PE_CFG[f"{task}_max_model_len"]),
                   MAX_IMGS="10")
        with open(log_path, "w") as log:
            self.procs[task] = subprocess.Popen(
                ["bash", str(PE_DIR / "serve.sh")], cwd=PE_DIR, env=env,
                stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                start_new_session=True)
        notify(f"Starting the {task} prompt enhancer (1 to 2 minutes the first time)")
        t0 = time.time()
        timeout = float(PE_CFG.get("startup_timeout", 600))
        while not self.healthy(task):
            if not self.running(task):
                self.procs.pop(task, None)
                tail = log_path.read_text(errors="replace").splitlines()[-12:]
                raise gr.Error(f"The {task} enhancer exited during startup. "
                               f"Log {_show(log_path)}:\n" + "\n".join(tail))
            if time.time() - t0 > timeout:
                self.stop(task)
                raise gr.Error(f"The {task} enhancer was not ready after {timeout:.0f} s. "
                               f"See {_show(log_path)}")
            if ENGINE.stop.is_set():
                self.stop(task)
                raise Stopped()
            time.sleep(2)

    def enhance(self, task, prompt, image_paths, seed, notify, on_stream) -> dict:
        """Rewrite `prompt` with the task's enhancer and return the output record."""
        self.ensure(task, notify)
        from openai import OpenAI

        profile = core.get_profile(task)
        try:
            system_prompt = core.load_system_prompt(None, self.model(task))
        except SystemExit:
            system_prompt = (PE_DIR / "prompts" / f"system_prompt_{task}.txt").read_text(encoding="utf-8").strip()
        uris = [pe_client.image_to_data_uri(Path(p), profile.image_max_pixels) for p in image_paths]
        messages = pe_client._flatten(core.build_messages(system_prompt, prompt, uris))
        client = OpenAI(base_url=f"http://127.0.0.1:{self.port(task)}/v1", api_key="unused")
        notify("Enhancing the prompt")
        stream = client.chat.completions.create(
            model=self.served_name(task), messages=messages,
            temperature=profile.temperature, top_p=profile.top_p,
            presence_penalty=profile.presence_penalty,
            max_tokens=profile.max_new_tokens, seed=seed, stream=True, timeout=900,
            extra_body={"top_k": profile.top_k, "min_p": profile.min_p,
                        "chat_template_kwargs": {"enable_thinking": True}})
        think, content = [], []
        last = 0.0
        try:
            for chunk in stream:
                if ENGINE.stop.is_set():
                    raise Stopped()
                if not chunk.choices:
                    continue
                delta = chunk.choices[0].delta
                reasoning = getattr(delta, "reasoning_content", None) or getattr(delta, "reasoning", None)
                if reasoning:
                    think.append(reasoning)
                if getattr(delta, "content", None):
                    content.append(delta.content)
                if time.time() - last > 0.3:
                    last = time.time()
                    on_stream("".join(think), "".join(content))
        finally:
            stream.close()
        answer = "".join(content)
        if think:
            thinking, answer = "".join(think).strip(), answer.strip()
        else:
            thinking, answer = core.split_thinking(answer)
        case = {"id": "studio", "prompt": prompt, "input_images": list(image_paths), "task_type": ""}
        record = core.build_record(case, thinking, answer, profile)
        if not PE_CFG.get("keep_running", True):
            self.stop(task)
        return record


ENHANCER = Enhancer()
atexit.register(ENHANCER.stop_all)
signal.signal(signal.SIGTERM, lambda *a: sys.exit(0))


# --------------------------------------------------------------------------- #
# Jobs: one GPU job at a time, run in a background thread; the page polls state
# --------------------------------------------------------------------------- #
TABS = ("t2i", "edit", "draw")
TAB_TITLES = {"t2i": "Text to Image", "edit": "Image Edit", "draw": "Local Edit"}


class TabState:
    """Everything a tab shows. Survives page reloads because it lives on the server."""

    def __init__(self):
        self.lock = threading.Lock()
        self.stage = "idle"
        self.message = "Results will appear here"
        self.progress: tuple[int, int] | None = None
        self.started: float | None = None
        self.elapsed: float | None = None
        self.images: list = []
        self.info = ""
        self.final = ""
        self.thinking = ""
        self.inputs: list | None = None
        self.v_status = self.v_images = self.v_info = self.v_text = 0


STATES = {t: TabState() for t in TABS}
BUSY = {"tab": None}
BUSY_LOCK = threading.Lock()


class Job:
    """Handle a running job uses to report into its tab."""

    def __init__(self, tab):
        self.st = STATES[tab]

    def status(self, message, progress=None):
        with self.st.lock:
            self.st.message, self.st.progress = message, progress
            self.st.v_status += 1

    def text(self, final=None, thinking=None):
        with self.st.lock:
            if final is not None:
                self.st.final = final
            if thinking is not None:
                self.st.thinking = thinking
            self.st.v_text += 1

    def result(self, images=None, info=None):
        with self.st.lock:
            if images is not None:
                self.st.images = images
                self.st.v_images += 1
            if info is not None:
                self.st.info = info
                self.st.v_info += 1

    def finish(self, stage, message):
        with self.st.lock:
            self.st.stage, self.st.message, self.st.progress = stage, message, None
            self.st.elapsed = time.time() - (self.st.started or time.time())
            self.st.v_status += 1


def start_job(tab, inputs, target, *args):
    """Start `target(job, *args)` in the background, or refuse if the GPU is busy."""
    with BUSY_LOCK:
        if BUSY["tab"] is not None:
            raise gr.Error(f"A {TAB_TITLES[BUSY['tab']]} job is still running. "
                           "Wait for it to finish or press Stop.")
        BUSY["tab"] = tab
    st = STATES[tab]
    with st.lock:
        st.stage, st.message, st.progress = "running", "Starting", None
        st.started, st.elapsed, st.inputs = time.time(), None, inputs
        st.v_status += 1
    ENGINE.stop.clear()

    def run():
        job = Job(tab)
        try:
            message = target(job, *args)
            if ENGINE.stop.is_set():
                job.finish("stopped", "Stopped early")
            else:
                job.finish("done", message or "Done")
        except Stopped:
            job.finish("stopped", "Stopped")
        except gr.Error as exc:
            job.finish("error", getattr(exc, "message", str(exc)))
        except Exception as exc:
            traceback.print_exc()
            job.finish("error", f"{type(exc).__name__}: {exc}")
        finally:
            with BUSY_LOCK:
                BUSY["tab"] = None

    threading.Thread(target=run, name=f"studio-{tab}", daemon=True).start()
    return render_status(tab)


def request_stop(tab):
    busy = BUSY["tab"]
    if busy is None:
        gr.Info("Nothing is running.")
    else:
        ENGINE.stop.set()
        Job(busy).status("Stopping after the current step")
    return render_status(tab)


# --------------------------------------------------------------------------- #
# Generation
# --------------------------------------------------------------------------- #
RGBA_PREFIX = "This is an RGBA image with transparency."
RGBA_SUFFIX = "The image has alpha channel and the background is transparent."


def rgba_wrap(prompt: str) -> str:
    """The prompt format the README recommends for transparent output."""
    p = prompt.strip()
    if p.startswith(RGBA_PREFIX):
        return p
    return f"{RGBA_PREFIX} {p.rstrip('. ')}. {RGBA_SUFFIX}"


def _gallery_paths(items) -> list[str]:
    paths = []
    for it in items or []:
        if isinstance(it, (list, tuple)):
            it = it[0]
        if isinstance(it, dict):
            it = it.get("path") or it.get("image") or it.get("name")
            if isinstance(it, dict):
                it = it.get("path")
        if isinstance(it, str) and it:
            paths.append(it)
    return paths


def _open(path) -> Image.Image:
    with Image.open(path) as im:
        im.load()
        return im.copy()


def _normalize_inputs(images: list[Image.Image], stamp: str) -> tuple[list[Image.Image], list[str]]:
    """Apply EXIF rotation, keep alpha, and save copies, so the enhancer and the image
    model see exactly the same pixels and the page can restore them after a reload."""
    d = OUT_DIR / "inputs"
    d.mkdir(parents=True, exist_ok=True)
    out, paths = [], []
    for i, im in enumerate(images, 1):
        im = ImageOps.exif_transpose(im)
        im = im.convert("RGBA") if im.mode in ("RGBA", "LA", "P", "PA") else im.convert("RGB")
        p = d / f"{stamp}_image{i}.png"
        im.save(p)
        out.append(im)
        paths.append(str(p))
    return out, paths


def _finalize(img: Image.Image) -> Image.Image:
    """Output is always RGBA; drop the alpha channel when it is fully opaque."""
    if img.mode == "RGBA" and img.getchannel("A").getextrema() == (255, 255):
        return img.convert("RGB")
    return img


def _stamp() -> str:
    return datetime.now().strftime("%H%M%S_%f")[:-3]


def _seed(seed, randomize) -> int:
    return random.randint(0, 2 ** 31 - 1) if randomize else int(seed or 0)


def _save(images, meta: dict, stamp: str, mode: str) -> list[str]:
    if not OUT_CFG.get("save", True):
        return []
    d = OUT_DIR / datetime.now().strftime("%Y-%m-%d")
    d.mkdir(parents=True, exist_ok=True)
    paths = []
    for i, im in enumerate(images):
        base = d / f"{stamp}_{mode}_seed{meta['seeds'][i]}"
        im.save(f"{base}.png")
        with open(f"{base}.json", "w", encoding="utf-8") as f:
            json.dump({**meta, "seed": meta["seeds"][i], "file": _show(f"{base}.png")}, f,
                      ensure_ascii=False, indent=2)
        paths.append(f"{base}.png")
    return paths


def _run_pipeline(job, prompt, images, w, h, res, steps, seed, n, neg, cfg, kv):
    steps = int(steps)
    if ENGINE.offload == "none" or ENGINE.pipe is None:
        need = _pipeline_need_gib(w, h, n, len(images or []), res)
        free, _ = _free_gib()
        if free < need and any(ENHANCER.running(t) for t in ENHANCER.TASKS):
            job.status("Stopping the prompt enhancer to free VRAM for the image model")
            ENHANCER.stop_all()
    pipe = ENGINE.load(job.status)
    if ENGINE.stop.is_set():
        raise Stopped()
    gens = [torch.Generator(DEVICE).manual_seed(seed + i) for i in range(n)]
    job.status(f"Generating {w}×{h}", (0, steps))

    def on_step(p, i, t, kwargs):
        job.status(f"Generating {w}×{h}", (i + 1, steps))
        if ENGINE.stop.is_set():
            p._interrupt = True
        return kwargs

    kwargs = dict(prompt=prompt, width=w, height=h, num_inference_steps=steps,
                  num_images_per_prompt=n, generator=gens if n > 1 else gens[0],
                  output_resolution=RESOLUTIONS[res], use_kv_cache=bool(kv),
                  callback_on_step_end=on_step)
    if images:
        kwargs["image"] = images
    if cfg > 1:
        kwargs["true_cfg_scale"] = float(cfg)
        kwargs["negative_prompt"] = neg or ""
    try:
        with torch.inference_mode():
            out = pipe(**kwargs).images
    except torch.OutOfMemoryError:
        _free_cuda()
        raise gr.Error("Out of GPU memory. Lower the resolution or the number of images, "
                       "set offload = \"model\" in the config, or free some VRAM.")
    job.status("Decoding and saving")
    return [_finalize(im) for im in out]


def _enhance(job, task, prompt, paths, seed) -> dict:
    def on_stream(think, answer):
        job.text(final=answer or None, thinking=think)

    rec = ENHANCER.enhance(task, prompt, paths, seed, job.status, on_stream)
    job.text(final=rec["positive_prompt"], thinking=rec["thinking"])
    return rec


def _deliver(job, imgs, meta, stamp, mode, notes, t0) -> str:
    if ENGINE.stop.is_set():
        job.result(imgs, "Stopped early. The image is only partly denoised and was not saved.")
        return "Stopped early"
    saved = _save(imgs, meta, stamp, mode)
    seeds = meta["seeds"]
    lines = [f"**{meta['width']}×{meta['height']}** · {meta['steps']} steps · "
             f"seed{'s' if len(seeds) > 1 else ''} {', '.join(map(str, seeds))} · "
             f"{time.time() - t0:.1f} s"]
    lines += notes
    if saved:
        lines.append("Saved to " + ", ".join(f"`{_show(p)}`" for p in saved))
    job.result(imgs, "\n\n".join(lines))
    return f"Done in {time.time() - t0:.1f} s"


def t2i_job(job, v, seed):
    t0, stamp, n, res = time.time(), _stamp(), int(v["n"]), v["res"]
    final, ratio, notes = v["prompt"].strip(), None, []
    job.text(final=final, thinking="")
    if v["enhance"]:
        rec = _enhance(job, "t2i", final, [], seed)
        final, ratio = rec["positive_prompt"], rec["wh_ratio"]
        if not rec["parse_ok"]:
            notes.append("The enhancer answer did not parse, so its raw text was used.")
    if v["transparent"]:
        final = rgba_wrap(final)
    if v["aspect"] == "Custom":
        w, h = _snap(v["width"]), _snap(v["height"])
    elif v["enhance"] and v["pe_ratio"] and _parse_ratio(ratio or ""):
        w, h = size_for_ratio(ratio, res)
        notes.append(f"Aspect ratio {ratio} chosen by the enhancer.")
    else:
        w, h = size_for_ratio(v["aspect"], res)
    job.text(final=final)
    imgs = _run_pipeline(job, final, None, w, h, res, v["steps"], seed, n, v["neg"], v["cfg"], v["kv"])
    meta = dict(mode="t2i", prompt=v["prompt"], final_prompt=final, transparent=v["transparent"],
                enhanced=bool(v["enhance"]), width=w, height=h, steps=int(v["steps"]), resolution=res,
                seeds=[seed + i for i in range(n)], true_cfg_scale=v["cfg"],
                negative_prompt=v["neg"] if v["cfg"] > 1 else "", use_kv_cache=bool(v["kv"]))
    return _deliver(job, imgs, meta, stamp, "t2i", notes, t0)


def edit_job(job, v, images, paths, seed, mode):
    t0, stamp, n, res = time.time(), _stamp(), int(v["n"]), v["res"]
    final, notes, rec = v["prompt"].strip(), [], None
    job.text(final=final, thinking="")
    if v["enhance"]:
        rec = _enhance(job, "edit", final, paths, seed)
        final = rec["positive_prompt"]
        if not rec["parse_ok"]:
            notes.append("The enhancer answer did not parse, so its raw text was used.")
    if v["transparent"]:
        final = rgba_wrap(final)
    follow = rec["ratio_follow"] if rec else ""
    pe_ratio = rec["wh_ratio"] if rec else ""
    k_follow = int(follow[6:-1]) if follow.startswith("<image") and follow[6:-1].isdigit() else 0
    if v["enhance"] and v["pe_ratio"] and 1 <= k_follow <= len(images):
        w, h = size_like(images[k_follow - 1], res)
        notes.append(f"Canvas follows reference image {k_follow}, as the enhancer chose.")
    elif v["enhance"] and v["pe_ratio"] and _parse_ratio(pe_ratio):
        w, h = size_for_ratio(pe_ratio, res)
        notes.append(f"Aspect ratio {pe_ratio} chosen by the enhancer.")
    elif v["size_mode"] == "Aspect ratio preset":
        w, h = size_for_ratio(v["aspect"], res)
    elif v["size_mode"] == "Custom":
        w, h = _snap(v["width"]), _snap(v["height"])
    else:
        k = min(max(int(v["ref_index"] or 1), 1), len(images))
        w, h = size_like(images[k - 1], res)
    job.text(final=final)
    imgs = _run_pipeline(job, final, images, w, h, res, v["steps"], seed, n, v["neg"], v["cfg"], v["kv"])
    meta = dict(mode=mode, prompt=v["prompt"], final_prompt=final, transparent=v["transparent"],
                enhanced=bool(v["enhance"]), input_images=[_show(p) for p in paths], width=w, height=h,
                steps=int(v["steps"]), resolution=res, seeds=[seed + i for i in range(n)],
                true_cfg_scale=v["cfg"], negative_prompt=v["neg"] if v["cfg"] > 1 else "",
                use_kv_cache=bool(v["kv"]))
    return _deliver(job, imgs, meta, stamp, mode, notes, t0)


def enhance_job(job, task, prompt, paths, seed):
    job.text(final="", thinking="")
    rec = _enhance(job, task, prompt, paths, seed)
    if rec["wh_ratio"]:
        extra = f", aspect ratio {rec['wh_ratio']}"
    elif rec["ratio_follow"]:
        extra = f", canvas follows {rec['ratio_follow']}"
    else:
        extra = ""
    ok = "" if rec["parse_ok"] else " (the answer did not parse, raw text shown)"
    return f"Prompt enhanced{extra}{ok}"


# --------------------------------------------------------------------------- #
# Submit handlers (validate fast, then hand the work to the background worker)
# --------------------------------------------------------------------------- #
T2I_KEYS = ["prompt", "transparent", "enhance", "pe_ratio", "aspect", "res", "width", "height",
            "steps", "seed", "randomize", "n", "neg", "cfg", "kv"]
EDIT_KEYS = ["prompt", "transparent", "enhance", "pe_ratio", "size_mode", "ref_index", "aspect",
             "res", "width", "height", "steps", "seed", "randomize", "n", "neg", "cfg", "kv"]


def _need_prompt(prompt):
    if not prompt or not prompt.strip():
        raise gr.Error("Write a prompt first.")


def _check_refs(images):
    if not images:
        raise gr.Error("Add at least one reference image.")
    if len(images) > 10:
        raise gr.Error(f"Qwen-Image-2.1 takes up to 10 reference images, got {len(images)}.")


def _editor_value(path):
    return {"background": path, "layers": [], "composite": path}


def submit_t2i(*values):
    v = dict(zip(T2I_KEYS, values))
    _need_prompt(v["prompt"])
    return start_job("t2i", list(values), t2i_job, v, _seed(v["seed"], v["randomize"]))


def submit_t2i_enhance(*values):
    v = dict(zip(T2I_KEYS, values))
    _need_prompt(v["prompt"])
    return start_job("t2i", list(values), enhance_job, "t2i", v["prompt"].strip(), [],
                     _seed(v["seed"], v["randomize"]))


def _edit_inputs(files):
    images = [_open(p) for p in _gallery_paths(files)]
    _check_refs(images)
    return _normalize_inputs(images, _stamp())


def _draw_inputs(editor):
    comp = editor.get("composite") if isinstance(editor, dict) else None
    if comp is None:
        raise gr.Error("Upload an image and draw on it first.")
    return _normalize_inputs([comp], _stamp())


def submit_edit(files, *values):
    v = dict(zip(EDIT_KEYS, values))
    _need_prompt(v["prompt"])
    images, paths = _edit_inputs(files)
    return start_job("edit", [paths] + list(values), edit_job, v, images, paths,
                     _seed(v["seed"], v["randomize"]), "edit")


def submit_edit_enhance(files, *values):
    v = dict(zip(EDIT_KEYS, values))
    _need_prompt(v["prompt"])
    _, paths = _edit_inputs(files)
    return start_job("edit", [paths] + list(values), enhance_job, "edit", v["prompt"].strip(),
                     paths, _seed(v["seed"], v["randomize"]))


def submit_draw(editor, *values):
    v = dict(zip(EDIT_KEYS, values))
    _need_prompt(v["prompt"])
    images, paths = _draw_inputs(editor)
    return start_job("draw", [_editor_value(paths[0])] + list(values), edit_job, v, images, paths,
                     _seed(v["seed"], v["randomize"]), "local_edit")


def submit_draw_enhance(editor, *values):
    v = dict(zip(EDIT_KEYS, values))
    _need_prompt(v["prompt"])
    _, paths = _draw_inputs(editor)
    return start_job("draw", [_editor_value(paths[0])] + list(values), enhance_job, "edit",
                     v["prompt"].strip(), paths, _seed(v["seed"], v["randomize"]))


# --------------------------------------------------------------------------- #
# Polling and restoring
# --------------------------------------------------------------------------- #
STAGE_LABELS = {"idle": "Ready", "running": "Running", "done": "Done", "error": "Error",
                "stopped": "Stopped"}


def _fmt_secs(s: float) -> str:
    s = int(s)
    return f"{s // 60}:{s % 60:02d}"


def render_status(tab) -> str:
    st = STATES[tab]
    with st.lock:
        stage, message, progress = st.stage, st.message, st.progress
        started, elapsed = st.started, st.elapsed
    busy = BUSY["tab"]
    if stage == "running":
        # The clock keeps counting in the browser (see HEAD) without a server round trip.
        timer = f'<span class="qs-time" data-qs-elapsed="{time.time() - started:.1f}"></span>'
    elif elapsed is not None:
        timer = f'<span class="qs-time">{_fmt_secs(elapsed)}</span>'
    else:
        timer = ""
    note = ""
    if stage != "running" and busy and busy != tab:
        note = f'<div class="qs-note">The GPU is busy with a {TAB_TITLES[busy]} job.</div>'
    bar = ""
    if stage == "running":
        if progress and progress[1]:
            pct = 100.0 * progress[0] / progress[1]
            bar = (f'<div class="qs-bar"><div class="qs-fill" style="width:{pct:.1f}%"></div></div>'
                   f'<div class="qs-steps">step {progress[0]} of {progress[1]}</div>')
        else:
            bar = '<div class="qs-bar qs-indeterminate"><div class="qs-fill"></div></div>'
    return (f'<div class="qs-status qs-{stage}"><div class="qs-row">'
            f'<span class="qs-pill">{STAGE_LABELS[stage]}</span>'
            f'<span class="qs-msg">{html.escape(message)}</span>{timer}</div>{bar}{note}</div>')


def poll(seen_json):
    """Send each tab only what changed since this page last saw it.

    What the page has seen is kept in the browser (a hidden textbox that every poll
    updates), not in server-side session state: if a response is lost, for example
    when the connection drops, the page still holds the old versions and the next
    poll sends the missing updates again."""
    try:
        seen = json.loads(seen_json or "{}")
    except json.JSONDecodeError:
        seen = {}
    out = []
    busy = BUSY["tab"]
    # Re-send the status cards every ~10 s even when unchanged, as a safety net.
    tick = seen.get("_tick", 0) + 1
    seen["_tick"] = tick
    refresh = tick % 12 == 0
    for tab in TABS:
        st = STATES[tab]
        with st.lock:
            ks = [st.v_status, busy]
            ki, kinfo, kt = st.v_images, st.v_info, st.v_text
            images, info, final, thinking = list(st.images), st.info, st.final, st.thinking
        out.append(render_status(tab) if refresh or seen.get(f"{tab}.s") != ks else gr.skip())
        out.append(images if seen.get(f"{tab}.i") != ki else gr.skip())
        out.append(info if seen.get(f"{tab}.info") != kinfo else gr.skip())
        out.append(final if seen.get(f"{tab}.t") != kt else gr.skip())
        out.append(thinking if seen.get(f"{tab}.t") != kt else gr.skip())
        seen.update({f"{tab}.s": ks, f"{tab}.i": ki, f"{tab}.info": kinfo, f"{tab}.t": kt})
    return [json.dumps(seen)] + out


def restore():
    """Runs once the page has loaded: put back each tab's last inputs, then start
    polling from scratch so the page receives the full current state."""
    out = []
    for tab, n in (("t2i", len(T2I_KEYS)), ("edit", len(EDIT_KEYS) + 1), ("draw", len(EDIT_KEYS) + 1)):
        with STATES[tab].lock:
            inputs = STATES[tab].inputs
        out += list(inputs) if inputs else [gr.skip()] * n
    return out + ["{}", gr.Timer(active=True)]


def job_status(tab: str) -> dict:
    """API helper: the state of a tab's latest job, for scripts."""
    if tab not in STATES:
        raise gr.Error(f"tab must be one of {', '.join(TABS)}")
    st = STATES[tab]
    with st.lock:
        return {"stage": st.stage, "message": st.message, "progress": st.progress,
                "info": st.info, "final_prompt": st.final, "thinking_chars": len(st.thinking),
                "images": len(st.images)}


# --------------------------------------------------------------------------- #
# System tab
# --------------------------------------------------------------------------- #
def system_status():
    lines = [f"**Image model:** {ENGINE.status()}", f"**Prompt enhancers:** {ENHANCER.status()}"]
    busy = BUSY["tab"]
    lines.append(f"**Current job:** {TAB_TITLES[busy] if busy else 'none'}")
    if torch.cuda.is_available():
        free, total = _free_gib()
        lines.append(f"**GPU:** {torch.cuda.get_device_name(0)}, {total - free:.1f} of "
                     f"{total:.1f} GiB in use (all processes)")
    else:
        lines.append("**GPU:** CUDA is not available")
    return "\n\n".join(lines)


def load_model():
    if BUSY["tab"]:
        raise gr.Error("Wait for the running job to finish.")
    ENGINE.load()
    return system_status()


def unload_model():
    if BUSY["tab"]:
        raise gr.Error("Wait for the running job to finish.")
    ENGINE.unload()
    return system_status()


def stop_enhancers():
    if BUSY["tab"]:
        raise gr.Error("Wait for the running job to finish.")
    ENHANCER.stop_all()
    return system_status()


# --------------------------------------------------------------------------- #
# UI
# --------------------------------------------------------------------------- #
T2I_EXAMPLES = [
    'A neon shop sign that reads "QWEN IMAGE 2.1", rainy night, reflections on wet pavement',
    "A panoramic mountain landscape at sunrise, low clouds drifting through the valleys",
    'A minimalist poster for a jazz festival, the title "BLUE NOTES" in bold serif type',
    "A cute cartoon dragon sticker",
    "Studio portrait of an elderly fisherman, soft window light, shallow depth of field",
]
EDIT_EXAMPLES = [
    "Change the background to a sunset beach",
    "Remove the background, and output a PNG image",
    "These characters are sitting around a campfire in a forest",
    "Put the outfit from <image2> on the person in <image1>",
    "Translate the sign text into Hindi",
    "Turn this photo into a 360 degree panorama of the same place",
]
DRAW_EXAMPLES = [
    "Remove the object circled in red",
    "Change the hair inside the red circle to blonde",
    "Replace the clothing marked in red with a black leather jacket",
    "Remove the red marks and everything they circle",
]

THEME = gr.themes.Soft(
    primary_hue=gr.themes.colors.violet,
    secondary_hue=gr.themes.colors.indigo,
    neutral_hue=gr.themes.colors.slate,
    font=[gr.themes.GoogleFont("Inter"), "ui-sans-serif", "system-ui", "sans-serif"],
    font_mono=[gr.themes.GoogleFont("JetBrains Mono"), "ui-monospace", "monospace"],
    radius_size=gr.themes.sizes.radius_lg,
).set(
    body_background_fill="#f6f5fb",
    body_background_fill_dark="#0b0b14",
    block_background_fill_dark="#14141f",
    block_border_width="1px",
    block_shadow="0 1px 2px rgba(15, 12, 40, 0.04)",
    button_primary_background_fill="linear-gradient(135deg, #7c3aed 0%, #4f46e5 100%)",
    button_primary_background_fill_hover="linear-gradient(135deg, #8b5cf6 0%, #6366f1 100%)",
    button_primary_background_fill_dark="linear-gradient(135deg, #7c3aed 0%, #4f46e5 100%)",
    button_primary_background_fill_hover_dark="linear-gradient(135deg, #8b5cf6 0%, #6366f1 100%)",
    button_primary_text_color="#ffffff",
    button_primary_text_color_dark="#ffffff",
)

CSS = """
.gradio-container {max-width: 1480px !important; margin: 0 auto !important;}
#qs-hero {
  position: relative; overflow: hidden; border-radius: 22px; padding: 28px 32px 24px;
  color: #fff; margin-bottom: 8px;
  background:
    radial-gradient(900px 320px at 8% -20%, rgba(196, 181, 253, 0.45), transparent 60%),
    radial-gradient(700px 300px at 100% 120%, rgba(34, 211, 238, 0.35), transparent 60%),
    linear-gradient(125deg, #1e1b4b 0%, #5b21b6 48%, #0e7490 100%);
  box-shadow: 0 18px 40px -18px rgba(76, 29, 149, 0.55);
}
#qs-hero h1 {margin: 0; font-size: 32px; line-height: 1.15; font-weight: 750; letter-spacing: -0.02em; color: #fff;}
#qs-hero h1 span {background: linear-gradient(90deg, #e9d5ff, #a5f3fc); -webkit-background-clip: text; background-clip: text; color: transparent;}
#qs-hero p {margin: 8px 0 0; max-width: 820px; font-size: 15px; line-height: 1.55; color: rgba(255, 255, 255, 0.86);}
.qs-chips {display: flex; flex-wrap: wrap; gap: 8px; margin-top: 16px;}
.qs-chip {font-size: 12.5px; font-weight: 550; padding: 5px 12px; border-radius: 999px; color: #fff;
  background: rgba(255, 255, 255, 0.13); border: 1px solid rgba(255, 255, 255, 0.24); backdrop-filter: blur(4px);}

.qs-actions {gap: 10px !important; margin-top: 2px;}
.qs-actions button {min-height: 48px; font-size: 15px; font-weight: 650;}
.qs-go {box-shadow: 0 10px 24px -10px rgba(124, 58, 237, 0.75);}
.qs-stop {background: transparent !important; color: #dc2626 !important; border: 1px solid rgba(220, 38, 38, 0.45) !important;}
.qs-stop:hover {background: rgba(220, 38, 38, 0.08) !important;}
.dark .qs-stop {color: #f87171 !important; border-color: rgba(248, 113, 113, 0.45) !important;}
.qs-go:hover {transform: translateY(-1px);}
.qs-go, .qs-go:hover {transition: transform 0.15s ease, filter 0.15s ease;}

.qs-status {border: 1px solid var(--border-color-primary); background: var(--block-background-fill);
  border-radius: 16px; padding: 14px 16px;}
.qs-row {display: flex; align-items: center; gap: 12px;}
.qs-pill {flex: none; font-size: 12px; font-weight: 700; letter-spacing: 0.02em; padding: 4px 11px;
  border-radius: 999px; background: rgba(100, 116, 139, 0.14); color: #475569;}
.dark .qs-pill {color: #cbd5e1;}
.qs-running .qs-pill {background: rgba(124, 58, 237, 0.14); color: #6d28d9;}
.dark .qs-running .qs-pill {color: #c4b5fd;}
.qs-done .qs-pill {background: rgba(16, 185, 129, 0.15); color: #047857;}
.dark .qs-done .qs-pill {color: #6ee7b7;}
.qs-error .qs-pill {background: rgba(239, 68, 68, 0.14); color: #b91c1c;}
.dark .qs-error .qs-pill {color: #fca5a5;}
.qs-stopped .qs-pill {background: rgba(245, 158, 11, 0.16); color: #b45309;}
.dark .qs-stopped .qs-pill {color: #fcd34d;}
.qs-msg {flex: 1; min-width: 0; font-size: 14.5px; color: var(--body-text-color); white-space: pre-wrap; word-break: break-word;}
.qs-time {flex: none; font-family: var(--font-mono); font-size: 13px; color: var(--body-text-color-subdued);}
.qs-bar {position: relative; height: 8px; margin-top: 12px; border-radius: 999px; overflow: hidden; background: rgba(100, 116, 139, 0.18);}
.qs-fill {height: 100%; border-radius: 999px; background: linear-gradient(90deg, #7c3aed, #6366f1, #06b6d4); transition: width 0.5s ease;}
.qs-indeterminate .qs-fill {width: 38%; animation: qs-slide 1.4s ease-in-out infinite;}
@keyframes qs-slide {0% {transform: translateX(-110%);} 100% {transform: translateX(280%);}}
.qs-steps {margin-top: 6px; font-size: 12px; color: var(--body-text-color-subdued);}
.qs-note {margin-top: 8px; font-size: 12.5px; color: var(--body-text-color-subdued);}

.qs-section {overflow: visible !important; margin: 10px 2px 0 !important; padding: 0 !important; min-height: 0 !important;}
.qs-section, .qs-section * {font-size: 12px !important; font-weight: 700 !important; letter-spacing: 0.06em;
  text-transform: uppercase; color: var(--body-text-color-subdued) !important; margin-bottom: 0 !important;}
.qs-help {font-size: 13px; color: var(--body-text-color-subdued);}
footer {display: none !important;}
"""

# Keeps the elapsed clock of a running job ticking between server updates.
HEAD = """
<script>
(() => {
  const fmt = s => `${Math.floor(s / 60)}:${String(Math.floor(s % 60)).padStart(2, "0")}`;
  setInterval(() => {
    document.querySelectorAll(".qs-time[data-qs-elapsed]").forEach(el => {
      if (!el.dataset.qsT0) el.dataset.qsT0 = String(performance.now() / 1000 - parseFloat(el.dataset.qsElapsed));
      el.textContent = fmt(performance.now() / 1000 - parseFloat(el.dataset.qsT0));
    });
  }, 250);
})();
</script>
"""

HERO = """
<div id="qs-hero">
  <h1><span>Qwen-Image-2.1</span> Studio</h1>
  <p>Create images from text, edit with up to ten reference images, mark local edits by drawing
  on the picture, and export transparent PNGs. Jobs keep running in the background, so you can
  switch tabs or reload the page at any time.</p>
  <div class="qs-chips">
    <span class="qs-chip">Native 2K</span><span class="qs-chip">Up to 10 references</span>
    <span class="qs-chip">Circle and paint edits</span><span class="qs-chip">Transparent RGBA</span>
    <span class="qs-chip">Official prompt enhancers</span>
  </div>
</div>
"""


def _section(title):
    gr.Markdown(title, elem_classes="qs-section")


def _action_bar():
    with gr.Row(elem_classes="qs-actions"):
        go = gr.Button("Generate", variant="primary", size="lg", scale=3, min_width=130, elem_classes="qs-go")
        enh = gr.Button("Enhance only", variant="secondary", size="lg", scale=2, min_width=120)
        stop = gr.Button("Stop", variant="stop", size="lg", scale=1, min_width=90, elem_classes="qs-stop")
    return go, enh, stop


def _options(enhance_label):
    with gr.Row():
        transparent = gr.Checkbox(label="Transparent background (RGBA PNG)")
        enhance = gr.Checkbox(label=enhance_label)
        pe_ratio = gr.Checkbox(value=True, label="Let the enhancer pick the canvas")
    return transparent, enhance, pe_ratio


def _settings(default_res, is_edit):
    """Size, sampling and advanced controls shared by the three tabs."""
    _section("Size")
    with gr.Group():
        if is_edit:
            size_mode = gr.Radio(["Match reference image", "Aspect ratio preset", "Custom"],
                                 value="Match reference image", label="Output canvas")
            ref_index = gr.Number(value=1, precision=0, minimum=1, maximum=10,
                                  label="Reference image to match")
        else:
            size_mode = ref_index = None
        with gr.Row():
            aspect = gr.Dropdown(RATIO_CHOICES + ([] if is_edit else ["Custom"]),
                                 value=DEF.get("aspect_ratio", "1:1"), label="Aspect ratio",
                                 visible=not is_edit)
            res = gr.Radio(list(RESOLUTIONS), value=default_res, label="Resolution",
                           info="2K is native quality, 1K is about 4× faster")
        with gr.Row(visible=False) as custom_row:
            width = gr.Slider(256, 3072, value=1024, step=32, label="Width")
            height = gr.Slider(256, 3072, value=1024, step=32, label="Height")
    _section("Sampling")
    with gr.Group():
        with gr.Row():
            steps = gr.Slider(1, 100, value=int(DEF.get("steps", 40)), step=1, label="Steps",
                              info="40 is the official default, 25 is a good fast setting")
            n = gr.Slider(1, 4, value=int(DEF.get("num_images", 1)), step=1, label="Images")
        with gr.Row():
            seed = gr.Number(value=int(DEF.get("seed", 42)), precision=0, label="Seed")
            randomize = gr.Checkbox(value=bool(DEF.get("random_seed", True)), label="Random seed")
    with gr.Accordion("Advanced", open=False):
        cfg = gr.Slider(1.0, 10.0, value=1.0, step=0.5, label="True CFG scale",
                        info="1 is off, which is how the model is meant to be sampled. Above 1 "
                             "uses the negative prompt and takes twice as long.")
        neg = gr.Textbox(label="Negative prompt", lines=2, placeholder="Used only when True CFG scale is above 1")
        kv = gr.Checkbox(value=True, label="Prefix KV cache",
                         info="Reuses the text and reference context across steps. Faster; "
                              "turning it off gives a different but equally valid sample.")

    if is_edit:
        def _mode(m):
            return (gr.update(visible=m == "Match reference image"),
                    gr.update(visible=m == "Aspect ratio preset"),
                    gr.update(visible=m == "Custom"))
        size_mode.change(_mode, size_mode, [ref_index, aspect, custom_row], api_visibility="private")
    else:
        aspect.change(lambda a: gr.update(visible=a == "Custom"), aspect, custom_row,
                      api_visibility="private")
    return dict(size_mode=size_mode, ref_index=ref_index, aspect=aspect, res=res, width=width,
                height=height, steps=steps, n=n, seed=seed, randomize=randomize, cfg=cfg, neg=neg, kv=kv)


def _results(tab):
    status = gr.HTML(render_status(tab))
    gallery = gr.Gallery(label="Result", format="png", columns=2, height=640, preview=True,
                         object_fit="contain", interactive=False)
    info = gr.Markdown()
    with gr.Accordion("Prompt used and enhancer thinking", open=False):
        final = gr.Textbox(label="Final prompt", lines=6, buttons=["copy"])
        thinking = gr.Textbox(label="Enhancer thinking", lines=8, max_lines=24, buttons=["copy"])
        use = gr.Button("Use the final prompt as my prompt", size="sm")
    return status, gallery, info, final, thinking, use


def build_ui():
    with gr.Blocks(title="Qwen-Image-2.1 Studio") as demo:
        gr.HTML(HERO)
        seen = gr.Textbox("{}", visible=False)
        panels = {}

        with gr.Tabs():
            with gr.Tab("Text to Image", id="t2i"):
                with gr.Row(equal_height=False):
                    with gr.Column(scale=5, min_width=380):
                        t_prompt = gr.Textbox(label="Prompt", lines=4, autofocus=True,
                                              placeholder="Describe the image. Put any text you want rendered in double quotes.")
                        t_go, t_enh, t_stop = _action_bar()
                        t_opts = _options("Enhance prompt (PE-T2I)")
                        t = _settings(DEF.get("t2i_resolution", "2K"), is_edit=False)
                        gr.Examples(T2I_EXAMPLES, t_prompt, label="Try one of these")
                    with gr.Column(scale=6, min_width=420):
                        panels["t2i"] = _results("t2i")
                t_inputs = [t_prompt, *t_opts, t["aspect"], t["res"], t["width"], t["height"],
                            t["steps"], t["seed"], t["randomize"], t["n"], t["neg"], t["cfg"], t["kv"]]

            with gr.Tab("Image Edit", id="edit"):
                with gr.Row(equal_height=False):
                    with gr.Column(scale=5, min_width=380):
                        e_prompt = gr.Textbox(label="Edit instruction", lines=3,
                                              placeholder="Change the background to a sunset beach. With several images, call them <image1>, <image2>, …")
                        e_go, e_enh, e_stop = _action_bar()
                        e_files = gr.Gallery(label="Reference images, in order <image1>, <image2>, … (up to 10)",
                                             type="filepath", interactive=True, file_types=["image"],
                                             columns=5, height=240, object_fit="contain")
                        e_opts = _options("Enhance instruction (PE-I2I)")
                        e = _settings(DEF.get("edit_resolution", "1K"), is_edit=True)
                        gr.Examples(EDIT_EXAMPLES, e_prompt, label="Try one of these")
                    with gr.Column(scale=6, min_width=420):
                        panels["edit"] = _results("edit")
                e_inputs = [e_files, e_prompt, *e_opts, e["size_mode"], e["ref_index"], e["aspect"],
                            e["res"], e["width"], e["height"], e["steps"], e["seed"], e["randomize"],
                            e["n"], e["neg"], e["cfg"], e["kv"]]

            with gr.Tab("Local Edit", id="draw"):
                with gr.Row(equal_height=False):
                    with gr.Column(scale=5, min_width=380):
                        d_prompt = gr.Textbox(label="Edit instruction", lines=3,
                                              placeholder="Remove the object circled in red")
                        d_go, d_enh, d_stop = _action_bar()
                        gr.Markdown("Upload an image, then circle or paint over what should change. "
                                    "The marked image is the reference, which is how the model reads "
                                    "circles and painted annotations.", elem_classes="qs-help")
                        d_editor = gr.ImageEditor(
                            label="Image with your marks", type="pil", image_mode="RGBA", height=520,
                            brush=gr.Brush(colors=["#FF0000", "#0066FF", "#00C040", "#FFD400", "#FFFFFF", "#000000"],
                                           default_color="#FF0000", default_size=8, color_mode="fixed"))
                        d_opts = _options("Enhance instruction (PE-I2I)")
                        d = _settings(DEF.get("edit_resolution", "1K"), is_edit=True)
                        gr.Examples(DRAW_EXAMPLES, d_prompt, label="Try one of these")
                    with gr.Column(scale=6, min_width=420):
                        panels["draw"] = _results("draw")
                d_inputs = [d_editor, d_prompt, *d_opts, d["size_mode"], d["ref_index"], d["aspect"],
                            d["res"], d["width"], d["height"], d["steps"], d["seed"], d["randomize"],
                            d["n"], d["neg"], d["cfg"], d["kv"]]

            with gr.Tab("System", id="system"):
                s_status = gr.Markdown(system_status())
                with gr.Row():
                    s_refresh = gr.Button("Refresh")
                    s_load = gr.Button("Load image model")
                    s_unload = gr.Button("Unload image model")
                    s_stop = gr.Button("Stop prompt enhancers")
                gr.Markdown(f"Config: `{_show(CONFIG_PATH)}`, plus `config.local.toml` next to it when "
                            f"present. Restart the app after editing. Outputs and enhancer logs: "
                            f"`{_show(OUT_DIR)}`.", elem_classes="qs-help")

        # Wiring
        wiring = (
            ("t2i", t_go, t_enh, t_stop, t_inputs, submit_t2i, submit_t2i_enhance, t_prompt),
            ("edit", e_go, e_enh, e_stop, e_inputs, submit_edit, submit_edit_enhance, e_prompt),
            ("draw", d_go, d_enh, d_stop, d_inputs, submit_draw, submit_draw_enhance, d_prompt),
        )
        for tab, go, enh, stop, inputs, fn_go, fn_enh, prompt in wiring:
            status, _, _, final, _, use = panels[tab]
            go.click(fn_go, inputs, status, api_name=f"{tab}_generate", show_progress="hidden")
            enh.click(fn_enh, inputs, status, api_name=f"{tab}_enhance", show_progress="hidden")
            stop.click(partial(request_stop, tab), None, status, api_name=f"{tab}_stop",
                       show_progress="hidden", concurrency_limit=None)
            use.click(lambda f: f, final, prompt, show_progress="hidden", api_visibility="private")

        poll_outputs = [seen]
        for tab in TABS:
            status, gallery, info, final, thinking, _ = panels[tab]
            poll_outputs += [status, gallery, info, final, thinking]
        # Polls bypass the queue and never wait on each other, so a request lost to a
        # dropped connection cannot stall the ones after it.
        timer = gr.Timer(0.8, active=False)
        timer.tick(poll, seen, poll_outputs, show_progress="hidden", queue=False,
                   trigger_mode="multiple", api_visibility="private")
        demo.load(restore, None, t_inputs + e_inputs + d_inputs + [seen, timer],
                  show_progress="hidden", api_visibility="private")

        s_refresh.click(system_status, None, s_status, concurrency_limit=None, api_visibility="private")
        s_load.click(load_model, None, s_status, api_name="load_model")
        s_unload.click(unload_model, None, s_status, api_name="unload_model")
        s_stop.click(stop_enhancers, None, s_status, api_name="stop_enhancers")

        # Scriptable status for API users.
        api_tab = gr.Textbox(visible=False)
        api_out = gr.JSON(visible=False)
        api_tab.submit(job_status, api_tab, api_out, api_name="job_status")
    return demo


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if PIPE_CFG.get("preload", False):
        print("Preloading the image model", flush=True)
        ENGINE.load()
    srv = CFG["server"]
    demo = build_ui()
    demo.queue(default_concurrency_limit=1)
    demo.launch(server_name=ARGS.host or srv.get("host", "127.0.0.1"),
                server_port=ARGS.port or int(srv.get("port", 7860)),
                share=ARGS.share or bool(srv.get("share", False)),
                allowed_paths=[str(OUT_DIR)], show_error=True,
                theme=THEME, css=CSS, head=HEAD, footer_links=[])


if __name__ == "__main__":
    main()
