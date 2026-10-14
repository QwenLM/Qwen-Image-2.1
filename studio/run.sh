#!/bin/bash
# Qwen-Image-2.1 Studio launcher.
#
#   bash studio/run.sh [app args]          run in the foreground (Ctrl+C to quit)
#   bash studio/run.sh start [app args]    run in the background (survives closing the terminal)
#   bash studio/run.sh stop                stop the background app and any enhancer it started
#   bash studio/run.sh status              is it running, and where
#   bash studio/run.sh logs                follow the background log
#
# App args are passed to app.py, e.g. --port 7861 or --share.
# Python: $PYTHON if set, else .venv/bin/python in the repo root if it exists, else python3.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(dirname "$HERE")"
if [ -n "${PYTHON:-}" ]; then PY="$PYTHON"
elif [ -x "$REPO/.venv/bin/python" ]; then PY="$REPO/.venv/bin/python"
else PY=python3; fi
RUN_DIR="$HERE/.run"
PID_FILE="$RUN_DIR/studio.pid"
LOG_FILE="$RUN_DIR/studio.log"
mkdir -p "$RUN_DIR"

is_running() { [ -f "$PID_FILE" ] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; }
url() { grep -o 'Running on local URL: *[^ ]*' "$LOG_FILE" 2>/dev/null | tail -1 | awk '{print $NF}'; }

cmd="${1:-fg}"
case "$cmd" in
  start|stop|status|logs|fg) shift || true ;;
  *) cmd=fg ;;
esac

case "$cmd" in
  fg)
    cd "$REPO"
    exec "$PY" "$HERE/app.py" "$@"
    ;;
  start)
    if is_running; then echo "Already running (pid $(cat "$PID_FILE")) at $(url)"; exit 0; fi
    cd "$REPO"
    PYTHONUNBUFFERED=1 nohup "$PY" "$HERE/app.py" "$@" > "$LOG_FILE" 2>&1 < /dev/null &
    echo $! > "$PID_FILE"
    for _ in $(seq 1 120); do
      if [ -n "$(url)" ]; then echo "Studio is running at $(url) (pid $(cat "$PID_FILE"))"; exit 0; fi
      is_running || { echo "Studio exited during startup. Last log lines:"; tail -20 "$LOG_FILE"; rm -f "$PID_FILE"; exit 1; }
      sleep 1
    done
    echo "Still starting. Follow it with: bash studio/run.sh logs"
    ;;
  stop)
    if is_running; then
      kill -TERM "$(cat "$PID_FILE")"
      for _ in $(seq 1 90); do is_running || break; sleep 1; done
      if is_running; then kill -KILL "$(cat "$PID_FILE")"; fi
      echo "Stopped"
    else
      echo "Not running"
    fi
    rm -f "$PID_FILE"
    ;;
  status)
    if is_running; then echo "Running (pid $(cat "$PID_FILE")) at $(url)"; else echo "Not running"; fi
    ;;
  logs)
    tail -f "$LOG_FILE"
    ;;
esac
