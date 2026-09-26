# Qwen-Image-2.1 Studio

A Gradio app for trying everything Qwen-Image-2.1 does, on your own GPU.

| Tab | What it does |
| --- | --- |
| Text to Image | Prompt to image at 1K, 1.5K or native 2K, with the recommended aspect ratios or a custom size. |
| Image Edit | Edit with 1 to 10 reference images. Refer to them as `<image1>`, `<image2>`, … in the instruction. |
| Local Edit | Draw circles or paint on an image to mark what should change, then describe the change. |
| System | Model and enhancer status, GPU memory, load or unload the model, stop the enhancers. |

Every generation tab also has:

* **Transparent background**, which wraps the prompt in the recommended RGBA format and saves a real alpha channel.
* **Enhance**, which rewrites the prompt with the official prompt enhancer ([`prompt_rewrite/`](../prompt_rewrite/)) before generating. The enhancer can also choose the canvas. **Enhance only** shows the rewrite without generating.
* Steps, seed, number of images, true CFG with a negative prompt, and the prefix KV cache switch.

Jobs run in a background worker and the page polls their state. Switching browser tabs, reloading or closing the page does not interrupt a generation. Reopen the page and it shows the job where it is, with the inputs that started it. One job runs at a time.

## Install

```bash
pip install -r studio/requirements.txt
```

The prompt enhancers are optional. They run as vLLM servers through `prompt_rewrite/serve.sh`, which needs its own environment because vLLM pins an older transformers:

```bash
python -m venv .venv-pe
.venv-pe/bin/pip install -r prompt_rewrite/requirements.txt
```

Then set `python = ".venv-pe/bin/python"` under `[enhancer]` in `studio/config.local.toml`.

## Run

From the repository root:

```bash
bash studio/run.sh                 # foreground, Ctrl+C to quit
bash studio/run.sh start           # background
bash studio/run.sh status
bash studio/run.sh logs
bash studio/run.sh stop            # also stops any enhancer the app started
```

Open http://127.0.0.1:7860. On a remote GPU, forward the port over SSH (`ssh -L 7860:127.0.0.1:7860 <host>`). `run.sh` passes extra arguments to the app, for example `bash studio/run.sh start --port 7861`.

## Configure

`studio/config.toml` documents every option: port, offload mode, enhancer memory and ports, default steps and resolution, and the output folder. Put your changes in `studio/config.local.toml` (git-ignored) using the same sections and keys, then restart.

## Memory

With `offload = "none"` the image model keeps about 33 GB of weights on the GPU and peaks at about 57 GB at 2048×2048. `offload = "model"` needs much less VRAM at some cost in speed.

An enhancer server reserves a fixed fraction of the GPU (0.35 for text to image, 0.45 for editing). When both do not fit, the app parks the image model in CPU RAM while the enhancer runs, and stops the enhancer before the next image when the image model needs the room. The first enhancer start takes 1 to 2 minutes.

## Outputs

Each result is saved to `studio/outputs/<date>/` as a PNG and a JSON file with the prompt, the final prompt, the seed and every setting. Fully opaque results are saved as RGB and transparent ones as RGBA. Enhancer logs are in `studio/outputs/logs/`.

## API

Every button is also a Gradio API endpoint (`/t2i_generate`, `/edit_generate`, `/draw_generate`, the matching `_enhance` and `_stop` endpoints, and `/job_status`), so the app can be scripted with `gradio_client`. The generate endpoints return as soon as the job starts; poll `/job_status` with the tab name (`t2i`, `edit` or `draw`) to follow it.
