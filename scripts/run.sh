#!/bin/bash
# Unattended benchmark orchestrator (run under nohup on the box).
# Minimizes SSH: one launch runs everything; poll /root/bench/progress.log.
set -u
BENCH=/root/bench
mkdir -p "$BENCH/results"
cd "$BENCH"
log(){ echo "[$(date -u +%H:%M:%S)] $*" | tee -a "$BENCH/progress.log"; }

MAXD="${1:-131072}"     # max context depth to bench this pass

log "======== RUN START (MAXD=$MAXD) ========"
log "probe -sm options (looking for undocumented 'internal' 2-card mode):"
/root/llama.cpp/build/bin/llama-server --help 2>&1 | grep -iE "split-mode|-sm," | sed 's/^/    /' | tee -a "$BENCH/progress.log"

log "ensure tokenizers in /venv/main"
/venv/main/bin/python -c "import tokenizers" 2>/dev/null || /venv/main/bin/pip install -q tokenizers 2>&1 | tail -2

log "build fixtures (coding corpus = llama.cpp source, Qwen tokenizer)"
if [ ! -f "$BENCH/fixtures.json" ]; then
  /venv/main/bin/python "$BENCH/make_prompts.py" \
    --tokenizer /root/models/fp8/tokenizer.json \
    --corpus /root/llama.cpp \
    --depths 1024,4096,16384,32768,65536,131072,262144 \
    --out "$BENCH/fixtures.json" 2>&1 | tail -14 | sed 's/^/    /' | tee -a "$BENCH/progress.log"
fi

log "llama.cpp tensor sweep (2gpu + 4gpu, MTP on/off)"
bash "$BENCH/bench_llama.sh" "$BENCH/fixtures.json" "$BENCH/results" "$MAXD" 2 256 2>&1 | tee -a "$BENCH/progress.log"

log "======== LLAMA SWEEP COMPLETE ========"
log "results files:"; ls -1 "$BENCH/results"/*.json 2>/dev/null | sed 's/^/    /' | tee -a "$BENCH/progress.log"
echo "RUN_ALL_DONE" | tee -a "$BENCH/progress.log"
