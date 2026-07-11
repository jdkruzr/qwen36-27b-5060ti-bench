#!/bin/bash
# vLLM MTP @255k, compiled (fair vs the 28.4 compiled baseline). Native method
# qwen3_5_mtp, 1 MTP layer -> num_speculative_tokens=1. DeepGemm off.
set -u
export VLLM_USE_DEEP_GEMM=0
VLLM=/venv/main/bin/vllm; MODEL=/root/models/fp8
BENCH=/root/bench; OUT=$BENCH/results; PORT=18080
SPEC='{"method": "qwen3_5_mtp", "num_speculative_tokens": 1}'

pkill -f "vllm serve" 2>/dev/null; sleep 6
echo ">> vLLM MTP @255k (compiled) $(date -u +%T)"
CUDA_VISIBLE_DEVICES=0,1,2,3 nohup "$VLLM" serve "$MODEL" \
  --tensor-parallel-size 4 --max-num-seqs 1 --max-model-len 262144 \
  --served-model-name qwen --no-enable-prefix-caching --disable-custom-all-reduce \
  --speculative-config "$SPEC" --port $PORT \
  > "$OUT/vllm_mtp_top.log" 2>&1 &
pid=$!
ok=1
for i in $(seq 1 480); do
  curl -sf localhost:$PORT/health >/dev/null 2>&1 && { ok=0; echo "  healthy after $((i*2))s"; break; }
  kill -0 $pid 2>/dev/null || { echo "  died"; tail -14 "$OUT/vllm_mtp_top.log"; break; }
  sleep 2
done
if [ $ok -ne 0 ]; then echo "VLLM_MTP_FAILED"; grep -iE "error|not support|assert|valueerror|speculat|recipe" "$OUT/vllm_mtp_top.log" | tail -10; exit 1; fi
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader > "$OUT/vram_vllm-mtp-top.txt"
python3 "$BENCH/bench_client.py" --base-url http://127.0.0.1:$PORT --model qwen \
  --fixtures "$BENCH/fixtures_top.json" --out "$OUT/vllm-tp4-mtp-top.json" \
  --label "vllm-tp4-mtp" --gen 256 --repeats 1 --max-depth 261120
echo "=== spec acceptance (from server log) ==="
grep -iE "accept|draft|spec_token|speculat" "$OUT/vllm_mtp_top.log" | tail -6
pkill -f "vllm serve" 2>/dev/null
echo "BENCH_VLLM_MTP_DONE"
