#!/bin/bash
# llama.cpp bs=1 coding-latency sweep: PP + TG vs context, tensor-parallel
# topologies, MTP on/off, Q8_0 weights, f16 (unquantized) KV.
#
# Usage: bench_llama.sh <fixtures.json> <outdir> [max_depth] [repeats] [gen]
set -u
BIN=/root/llama.cpp/build/bin/llama-server
MODEL=/root/models/q8-mtp/Qwen3.6-27B-Q8_0.gguf
FIX="${1:?fixtures}"; OUT="${2:?outdir}"; MAXD="${3:-262144}"; REP="${4:-3}"; GEN="${5:-256}"
PORT=18080
PY=python3
mkdir -p "$OUT"

# config: name | CUDA_VISIBLE_DEVICES | split-mode  (tensor-parallel only)
CONFIGS=(
  "4gpu-tensor|0,1,2,3|tensor"
)

# context ladder (descending); we try the largest <= MAXD first, fall back on OOM
ALL_CTX=(262144 131072 65536 32768 16384 8192)
ACHIEVED_CTX=0

start_server() {  # devs sm spec logbase
  local devs="$1" sm="$2" spec="$3" logbase="$4"
  local specargs=""
  [ "$spec" = "mtp" ] && specargs="--spec-type draft-mtp --spec-draft-n-max 4"
  for ctx in "${ALL_CTX[@]}"; do
    [ "$ctx" -gt "$MAXD" ] && continue
    local nctx=$(( ctx + GEN + 512 ))
    pkill -f llama-server 2>/dev/null; sleep 3
    CUDA_VISIBLE_DEVICES="$devs" nohup "$BIN" -m "$MODEL" \
      -ngl 99 -sm "$sm" -fa on -c "$nctx" --alias qwen \
      $specargs --host 127.0.0.1 --port $PORT --no-webui \
      > "${logbase}_ctx${ctx}.log" 2>&1 &
    local pid=$!
    local ok=1
    for i in $(seq 1 150); do
      if curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then ok=0; break; fi
      kill -0 "$pid" 2>/dev/null || { echo "    server died at ctx=$ctx (likely OOM), falling back"; ok=1; break; }
      sleep 2
    done
    if [ "$ok" -eq 0 ]; then ACHIEVED_CTX="$ctx"; echo "    healthy at ctx=$ctx"; return 0; fi
  done
  return 1
}

for cfg in "${CONFIGS[@]}"; do
  IFS='|' read -r name devs sm <<< "$cfg"
  for spec in nomtp mtp; do
    label="llamacpp-${name}-${spec}"
    echo "=========================================================="
    echo ">> $label (devices=$devs sm=$sm)"
    if ! start_server "$devs" "$sm" "$spec" "$OUT/server_${label}"; then
      echo "  SKIP $label (never healthy on any ctx)"; continue
    fi
    nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader > "$OUT/vram_${label}.txt"
    echo "  achieved ctx=$ACHIEVED_CTX  VRAM:"; cat "$OUT/vram_${label}.txt" | sed 's/^/    /'
    local_maxd=$(( ACHIEVED_CTX - GEN - 512 ))
    [ "$local_maxd" -gt "$MAXD" ] && local_maxd="$MAXD"
    $PY "$OUT/../bench_client.py" \
      --base-url "http://127.0.0.1:$PORT" --model qwen \
      --fixtures "$FIX" --out "$OUT/${label}.json" --label "$label" \
      --gen "$GEN" --repeats "$REP" --max-depth "$local_maxd"
    echo "  wrote $OUT/${label}.json (max_depth=$local_maxd)"
  done
done
pkill -f llama-server 2>/dev/null
echo "ALL_LLAMA_BENCH_DONE"
