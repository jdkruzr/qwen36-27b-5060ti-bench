#!/bin/bash
# Extras parts 2-4 only (telemetry already captured). No bare `wait` (that bug
# hung the first attempt on the never-exiting llama-server).
set -u
BIN=/root/llama.cpp/build/bin/llama-server
MODEL=/root/models/q8-mtp/Qwen3.6-27B-Q8_0.gguf
BENCH=/root/bench; OUT=$BENCH/results; PORT=18080; FIX=$BENCH/fixtures.json
LOGF=$BENCH/progress_extras2.log
log(){ echo "[$(date -u +%H:%M:%S)] $*" | tee -a "$LOGF"; }

start(){ # devices kv spec allreduce ctx logbase
  local devs="$1" kv="$2" spec="$3" ar="$4" ctx="$5" lb="$6"
  local specargs="" kvargs="" arenv=""
  [ "$spec" = mtp ] && specargs="--spec-type draft-mtp --spec-draft-n-max 4"
  [ "$kv" = q8_0 ] && kvargs="--cache-type-k q8_0 --cache-type-v q8_0"
  [ -n "$ar" ] && arenv="GGML_CUDA_ALLREDUCE=$ar"
  pkill -f llama-server 2>/dev/null; sleep 3
  CUDA_VISIBLE_DEVICES="$devs" env $arenv nohup "$BIN" -m "$MODEL" -ngl 99 -sm tensor -fa on -c "$ctx" \
    --alias qwen $specargs $kvargs --host 127.0.0.1 --port $PORT --no-webui > "${lb}.log" 2>&1 &
  local pid=$!
  for i in $(seq 1 170); do curl -sf localhost:$PORT/health >/dev/null 2>&1 && return 0; kill -0 $pid 2>/dev/null || { echo "  DIED"; tail -4 "${lb}.log"; return 1; }; sleep 2; done; return 1
}
bench(){ python3 "$BENCH/bench_client.py" --base-url http://127.0.0.1:$PORT --model qwen \
  --fixtures "$FIX" --out "$OUT/$1.json" --label "$1" --gen 256 --repeats 1 --max-depth "$2" 2>&1 | tee -a "$LOGF"; }
arline(){ grep -iE "AllReduce|butterfly|internal|falling back|NCCL" "$1" | tail -2 | sed 's/^/     /' | tee -a "$LOGF"; }

log "======== EXTRAS2 START ========"

log ">> [2] q8-KV isolation: 2x3090 nomtp q8_0 @255k"
start "0,1" q8_0 nomtp internal 262144 "$OUT/extra_nomtp_q8" && bench "3090-nomtp-q8" 261120

log ">> [3] butterfly AR: 2x3090 nomtp f16 @255k (GGML_CUDA_ALLREDUCE=none)"
if start "0,1" f16 nomtp none 262144 "$OUT/extra_butterfly"; then arline "$OUT/extra_butterfly.log"; bench "3090-nomtp-butterfly" 261120; fi

log ">> [4a] 4x3090 nomtp f16 @1k+255k"
if start "0,1,2,3" f16 nomtp "" 262144 "$OUT/extra_4gpu_nomtp"; then
  arline "$OUT/extra_4gpu_nomtp.log"
  nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | sed 's/^/     /' | tee -a "$LOGF"
  bench "4x3090-nomtp-f16" 261120
fi
log ">> [4b] 4x3090 mtp f16 @1k+255k"
start "0,1,2,3" f16 mtp "" 262144 "$OUT/extra_4gpu_mtp" && bench "4x3090-mtp-f16" 261120

pkill -f llama-server 2>/dev/null
log "======== EXTRAS2 DONE ========"; echo "EXTRAS2_DONE" | tee -a "$LOGF"
