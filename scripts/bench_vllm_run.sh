#!/bin/bash
# vLLM FP8 TP=4 single-stream sweep, 1k..255k, matching the llama.cpp ladder.
# --enforce-eager: torch.compile startup is impractically slow on 5060 Ti (few SMs).
set -u
export VLLM_USE_DEEP_GEMM=0   # DeepGemm FP8 kernels assert "Unknown recipe" on Blackwell; use CUTLASS
VLLM=/venv/main/bin/vllm
MODEL=/root/models/fp8
BENCH=/root/bench; OUT=$BENCH/results; PORT=18080
MAXLEN=262144; REP=1; GEN=256

# combined fixtures: clean depths 1k..128k from fixtures.json + 255k from fixtures_top.json
python3 -c "
import json
d=[x for x in json.load(open('$BENCH/fixtures.json')) if x['depth']<=131072]
d+=json.load(open('$BENCH/fixtures_top.json'))
json.dump(d, open('$BENCH/fixtures_vllm.json','w'))
print('depths:', [x['depth'] for x in d])
"

start(){ # spec(nomtp|mtp)
  local spec="$1" extra=""
  [ "$spec" = mtp ] && extra='--speculative-config {"method":"qwen3_next_mtp","num_speculative_tokens":2}'
  # kill any prior vllm by pattern that does NOT match this script's name
  pkill -f "vllm serve" 2>/dev/null; sleep 6
  CUDA_VISIBLE_DEVICES=0,1,2,3 nohup "$VLLM" serve "$MODEL" \
    --tensor-parallel-size 4 --max-num-seqs 1 --max-model-len $MAXLEN \
    --served-model-name qwen --no-enable-prefix-caching --enforce-eager \
    --disable-custom-all-reduce --port $PORT $extra \
    > "$OUT/vllm_${spec}.log" 2>&1 &
  local pid=$!
  for i in $(seq 1 210); do
    curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && return 0
    kill -0 $pid 2>/dev/null || { echo "  vllm $spec died"; tail -6 "$OUT/vllm_${spec}.log"; return 1; }
    sleep 3
  done; return 1
}

for spec in nomtp; do
  label="vllm-tp4-${spec}"
  echo ">> $label $(date -u +%T)"
  if start "$spec"; then
    nvidia-smi --query-gpu=index,memory.used --format=csv,noheader > "$OUT/vram_${label}.txt"
    python3 "$BENCH/bench_client.py" --base-url "http://127.0.0.1:$PORT" --model qwen \
      --fixtures "$BENCH/fixtures_vllm.json" --out "$OUT/${label}.json" \
      --label "$label" --gen $GEN --repeats $REP --max-depth 261120
    echo "  wrote $OUT/${label}.json"
  else echo "  SKIP $label"; fi
done
pkill -f "vllm serve" 2>/dev/null
echo "BENCH_VLLM_DONE"
