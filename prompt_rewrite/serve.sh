#!/bin/bash
# Serve a prompt-enhancer checkpoint as an OpenAI-compatible vLLM endpoint.
# Same server for both tasks -- the task only decides which checkpoint and which
# system prompt the *client* sends. Use this for interactive / online use;
# run_vllm.py is faster for batch.
#
#   CKPT=Qwen/Qwen-Image-2.1-PE-T2I bash serve.sh
#   CKPT=Qwen/Qwen-Image-2.1-PE-I2I GPUS=0,1 PORT=8100 bash serve.sh
#
# Environment (all optional):
#   CKPT       checkpoint dir or Hub id       (required, no default on purpose)
#   NAME       served model name              default: the Hub id, or basename of a dir
#   PORT       listen port                    default 8100
#   GPUS       CUDA_VISIBLE_DEVICES           default 0..TP-1
#   TP         tensor-parallel size           default = #visible GPUs, in {1,2,4,8}
#   MAX_LEN    max sequence length            default 24576
#   MEM_UTIL   GPU memory fraction            default 0.90
#   MAX_IMGS   max images per request         default 10 (edit only; harmless for t2i)
#   QUANT      "" | fp8                       default "" (bf16); only if VRAM-bound
#   EAGER      1 = add --enforce-eager        default 0 (see note below)
#   PY         python with vLLM               default: `python`
#
# No default CKPT: the two tasks use different weights, and a server silently
# started on the wrong ones answers fluently and wrongly.
#
# EAGER: CUDA-graph capture works on vLLM 0.19.1 for this architecture and is
# substantially faster, so graphs stay ON by default. Set EAGER=1 if you hit a
# capture failure on another vLLM version.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

CKPT="${CKPT:?CKPT must be a checkpoint directory or a Hub id}"
# A Hub id is served under its full id, so clients pass the same string as --model.
if [ -d "$CKPT" ]; then NAME="${NAME:-$(basename "$CKPT")}"
elif [[ "$CKPT" =~ ^[A-Za-z0-9._-]+/[A-Za-z0-9._-]+$ ]]; then NAME="${NAME:-$CKPT}"
else echo "CKPT $CKPT is neither a directory nor a Hub id" >&2; exit 2; fi
PORT="${PORT:-8100}"
PY="${PY:-python}"
MAX_LEN="${MAX_LEN:-24576}"
MEM_UTIL="${MEM_UTIL:-0.90}"
MAX_IMGS="${MAX_IMGS:-10}"
QUANT="${QUANT:-}"
EAGER="${EAGER:-0}"

if [ -n "${GPUS:-}" ]; then NGPU=$(awk -F, '{print NF}' <<<"$GPUS")
else NGPU=$(nvidia-smi -L 2>/dev/null | wc -l); [ "${NGPU:-0}" -ge 1 ] || NGPU=1; fi
_tp(){ for t in 8 4 2 1; do [ "$t" -le "$1" ] && { echo "$t"; return; }; done; echo 1; }
TP="${TP:-$(_tp "$NGPU")}"
GPUS="${GPUS:-$(seq -s, 0 $((TP - 1)))}"

EXTRA=()
[ -n "$QUANT" ] && EXTRA+=(--quantization "$QUANT")
[ "$EAGER" = "1" ] && EXTRA+=(--enforce-eager)

# Drop socket-interface envs naming an interface that doesn't exist here; vLLM's
# gloo backend dies on init otherwise (clusters often preset these globally).
for v in GLOO_SOCKET_IFNAME TP_SOCKET_IFNAME NVSHMEM_BOOTSTRAP_UID_SOCK_IFNAME; do
    iface="${!v:-}"
    if [ -n "$iface" ] && [ ! -d "/sys/class/net/$iface" ]; then unset "$v"; fi
done

echo "[serve] ckpt=$CKPT  served-model-name=$NAME"
echo "[serve] GPUS=$GPUS TP=$TP dtype=${QUANT:-bf16} port=$PORT max_len=$MAX_LEN eager=$EAGER"
echo "[serve] ready check:  curl -sf localhost:$PORT/health && echo ok"
echo "[serve] then:         python client.py --task <t2i|edit> --port $PORT --model $NAME ..."

# --reasoning-parser qwen3 makes the server split the <think> block into
# `reasoning_content`, so the client gets thinking and answer already separated.
exec env CUDA_VISIBLE_DEVICES="$GPUS" "$PY" -m vllm.entrypoints.openai.api_server \
    --model "$CKPT" --served-model-name "$NAME" --port "$PORT" \
    --dtype bfloat16 "${EXTRA[@]}" \
    --tensor-parallel-size "$TP" --max-model-len "$MAX_LEN" \
    --gpu-memory-utilization "$MEM_UTIL" \
    --limit-mm-per-prompt "{\"image\": $MAX_IMGS}" \
    --reasoning-parser qwen3
