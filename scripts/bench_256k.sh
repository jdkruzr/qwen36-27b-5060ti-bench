#!/bin/bash
# Targeted 256k backfill (the main sweep's max-depth filter skipped depth 262144).
# Same labels as the main run so aggregate.py merges by (label, depth).
set -u
BIN=/root/llama.cpp/build/bin/llama-server
MODEL=/root/models/q8-mtp/Qwen3.6-27B-Q8_0.gguf
BENCH=/root/bench; OUT=$BENCH/results; PORT=18080
NCTX=263424    # 262144 prompt + 256 gen + ~1k headroom

python3 -c "import json; d=json.load(open('$BENCH/fixtures.json')); json.dump([x for x in d if x['depth']==262144], open('$BENCH/fixtures_256k.json','w'))"

start(){ local spec="$1" sa=""; [ "$spec" = mtp ] && sa="--spec-type draft-mtp --spec-draft-n-max 4"
  pkill -f llama-server 2>/dev/null; sleep 3
  CUDA_VISIBLE_DEVICES=0,1,2,3 nohup "$BIN" -m "$MODEL" -ngl 99 -sm tensor -fa on -c $NCTX --alias qwen \
    $sa --host 127.0.0.1 --port $PORT --no-webui > "$OUT/server_256k_${spec}.log" 2>&1 &
  local pid=$!
  for i in $(seq 1 200); do
    curl -sf localhost:$PORT/health >/dev/null 2>&1 && return 0
    kill -0 $pid 2>/dev/null || { echo "  server died ($spec)"; tail -4 "$OUT/server_256k_${spec}.log"; return 1; }
    sleep 2
  done; return 1
}

for spec in nomtp mtp; do
  label="llamacpp-4gpu-tensor-${spec}"
  echo ">> $label @256k $(date -u +%T)"
  if start "$spec"; then
    nvidia-smi --query-gpu=index,memory.used --format=csv,noheader > "$OUT/vram_${label}_256k.txt"
    python3 "$BENCH/bench_client.py" --base-url http://127.0.0.1:$PORT --model qwen \
      --fixtures "$BENCH/fixtures_256k.json" --out "$OUT/${label}-256k.json" \
      --label "$label" --gen 256 --repeats 2 --max-depth 262144
  else echo "  SKIP $label @256k"; fi
done
pkill -f llama-server 2>/dev/null
echo "BENCH_256K_DONE"
