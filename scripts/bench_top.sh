#!/bin/bash
# Clean near-max-context point: prompt ~255k leaves room for 256 decode tokens
# inside the model's 262144 ceiling (the 262144 fixture left only ~11 tokens).
set -u
BIN=/root/llama.cpp/build/bin/llama-server
MODEL=/root/models/q8-mtp/Qwen3.6-27B-Q8_0.gguf
BENCH=/root/bench; OUT=$BENCH/results; PORT=18080
DEPTH=261120     # ~255k; +256 gen = 261376 < 262144 max

pkill -f llama-server 2>/dev/null; sleep 3
/venv/main/bin/python "$BENCH/make_prompts.py" --tokenizer /root/models/fp8/tokenizer.json \
  --corpus /root/llama.cpp --depths $DEPTH --out "$BENCH/fixtures_top.json" 2>&1 | tail -3

start(){ local spec="$1" sa=""; [ "$spec" = mtp ] && sa="--spec-type draft-mtp --spec-draft-n-max 4"
  pkill -f llama-server 2>/dev/null; sleep 3
  CUDA_VISIBLE_DEVICES=0,1,2,3 nohup "$BIN" -m "$MODEL" -ngl 99 -sm tensor -fa on -c 262144 --alias qwen \
    $sa --host 127.0.0.1 --port $PORT --no-webui > "$OUT/server_top_${spec}.log" 2>&1 &
  local pid=$!
  for i in $(seq 1 220); do curl -sf localhost:$PORT/health >/dev/null 2>&1 && return 0; kill -0 $pid 2>/dev/null || return 1; sleep 2; done; return 1
}
for spec in nomtp mtp; do
  label="llamacpp-4gpu-tensor-${spec}"
  echo ">> $label top(depth=$DEPTH) $(date -u +%T)"
  if start "$spec"; then
    python3 "$BENCH/bench_client.py" --base-url http://127.0.0.1:$PORT --model qwen \
      --fixtures "$BENCH/fixtures_top.json" --out "$OUT/${label}-top.json" \
      --label "$label" --gen 256 --repeats 1 --max-depth 300000
  else echo "  SKIP $label top"; fi
done
pkill -f llama-server 2>/dev/null
echo "BENCH_TOP_DONE"
