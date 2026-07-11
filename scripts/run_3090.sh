#!/bin/bash
# 2x RTX 3090 spike: llama.cpp Q8_0, 2-GPU tensor, bs=1, depths 1k + ~255k.
#   Run A: MTP off, f16 KV        Run B: MTP on, q8_0 KV
# Self-contained: builds fixtures then sweeps. Poll /root/bench/progress.log.
set -u
BENCH=/root/bench; OUT=$BENCH/results
BIN=/root/llama.cpp/build/bin/llama-server
MODEL=/root/models/q8-mtp/Qwen3.6-27B-Q8_0.gguf
PORT=18080; REP="${1:-2}"; GEN=256
mkdir -p "$OUT"; cd "$BENCH"
log(){ echo "[$(date -u +%H:%M:%S)] $*" | tee -a "$BENCH/progress.log"; }

log "======== 3090 RUN START ========"
/venv/main/bin/python -c "import tokenizers" 2>/dev/null || /venv/main/bin/pip install -q tokenizers
if [ ! -f "$BENCH/fixtures.json" ]; then
  log "build fixtures (1k + ~255k)"
  /venv/main/bin/python "$BENCH/make_prompts.py" --tokenizer /root/models/tok/tokenizer.json \
    --corpus /root/llama.cpp --depths 1024,261120 --out "$BENCH/fixtures.json" 2>&1 | tail -4 | tee -a "$BENCH/progress.log"
fi

# config: label | spec(nomtp|mtp) | kv(f16|q8_0)
CONFIGS=( "3090-nomtp-f16|nomtp|f16" "3090-mtp-q8|mtp|q8_0" )
ALL_CTX=(262144 131072 65536 32768 8192); ACHIEVED=0

start_server(){ # spec kv logbase
  local spec="$1" kv="$2" logbase="$3" specargs="" kvargs=""
  [ "$spec" = mtp ] && specargs="--spec-type draft-mtp --spec-draft-n-max 4"
  [ "$kv" = q8_0 ] && kvargs="--cache-type-k q8_0 --cache-type-v q8_0"
  for ctx in "${ALL_CTX[@]}"; do
    local nctx=$(( ctx + GEN + 512 )); [ "$nctx" -gt 262144 ] && nctx=262144
    pkill -f llama-server 2>/dev/null; sleep 3
    CUDA_VISIBLE_DEVICES=0,1 nohup "$BIN" -m "$MODEL" -ngl 99 -sm tensor -fa on -c "$nctx" \
      --alias qwen $specargs $kvargs --host 127.0.0.1 --port $PORT --no-webui \
      > "${logbase}_ctx${ctx}.log" 2>&1 &
    local pid=$!
    for i in $(seq 1 160); do
      curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && { ACHIEVED="$ctx"; echo "    healthy at ctx=$ctx"; return 0; }
      kill -0 "$pid" 2>/dev/null || { echo "    died at ctx=$ctx (OOM?), falling back"; break; }
      sleep 2
    done
  done; return 1
}

for cfg in "${CONFIGS[@]}"; do
  IFS='|' read -r label spec kv <<< "$cfg"
  log ">> $label (spec=$spec kv=$kv)"
  if ! start_server "$spec" "$kv" "$OUT/server_${label}"; then log "  SKIP $label (never healthy)"; continue; fi
  nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader > "$OUT/vram_${label}.txt"
  log "  achieved ctx=$ACHIEVED  VRAM: $(paste -sd' | ' "$OUT/vram_${label}.txt")"
  maxd=$(( ACHIEVED - GEN - 512 ))
  /usr/bin/python3 "$BENCH/bench_client.py" --base-url "http://127.0.0.1:$PORT" --model qwen \
    --fixtures "$BENCH/fixtures.json" --out "$OUT/${label}.json" --label "$label" \
    --gen "$GEN" --repeats "$REP" --max-depth "$maxd" 2>&1 | tee -a "$BENCH/progress.log"
done
pkill -f llama-server 2>/dev/null
log "======== 3090 RUN DONE ========"; echo "RUN_3090_DONE" | tee -a "$BENCH/progress.log"
