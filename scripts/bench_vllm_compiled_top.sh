#!/bin/bash
# ONE compiled-mode vLLM point at ~255k: default torch.compile + CUDA graphs
# (no --enforce-eager) for a FAIR bs=1 decode number. DeepGemm off (CUTLASS).
# Bounded ~14min compile wait; if it never converges, that's the finding.
set -u
export VLLM_USE_DEEP_GEMM=0
VLLM=/venv/main/bin/vllm; MODEL=/root/models/fp8
BENCH=/root/bench; OUT=$BENCH/results; PORT=18080

pkill -f "vllm serve" 2>/dev/null; sleep 6
echo ">> compiled vLLM @255k, loading+compiling $(date -u +%T)"
CUDA_VISIBLE_DEVICES=0,1,2,3 nohup "$VLLM" serve "$MODEL" \
  --tensor-parallel-size 4 --max-num-seqs 1 --max-model-len 262144 \
  --served-model-name qwen --no-enable-prefix-caching \
  --disable-custom-all-reduce --port $PORT \
  > "$OUT/vllm_compiled_top.log" 2>&1 &
pid=$!
ok=1
for i in $(seq 1 450); do
  curl -sf localhost:$PORT/health >/dev/null 2>&1 && { ok=0; echo "  healthy after $((i*2))s"; break; }
  kill -0 $pid 2>/dev/null || { echo "  server died"; tail -8 "$OUT/vllm_compiled_top.log"; break; }
  sleep 2
done
if [ $ok -ne 0 ]; then echo "COMPILED_TOP_FAILED (no health in ~15min)"; exit 1; fi
nvidia-smi --query-gpu=index,memory.used,power.draw --format=csv,noheader > "$OUT/vram_vllm-compiled-top.txt"
python3 "$BENCH/bench_client.py" --base-url http://127.0.0.1:$PORT --model qwen \
  --fixtures "$BENCH/fixtures_top.json" --out "$OUT/vllm-tp4-compiled-top.json" \
  --label "vllm-tp4-compiled" --gen 256 --repeats 1 --max-depth 261120
pkill -f "vllm serve" 2>/dev/null
echo "BENCH_VLLM_COMPILED_TOP_DONE"
