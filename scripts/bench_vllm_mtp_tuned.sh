#!/bin/bash
# vLLM MTP @255k, compiled, TUNED: raise --max-num-batched-tokens (vLLM warned the
# spec-decode auto-cap of 2048 may be suboptimal). Tests whether the scheduler
# throttle explains the 3.7 tok/s decode collapse.
set -u
export VLLM_USE_DEEP_GEMM=0
VLLM=/venv/main/bin/vllm; MODEL=/root/models/fp8
BENCH=/root/bench; OUT=$BENCH/results; PORT=18080
SPEC='{"method": "qwen3_5_mtp", "num_speculative_tokens": 1}'

pkill -f "vllm serve" 2>/dev/null; sleep 6
echo ">> vLLM MTP @255k (compiled, max-num-batched-tokens=16384) $(date -u +%T)"
CUDA_VISIBLE_DEVICES=0,1,2,3 nohup "$VLLM" serve "$MODEL" \
  --tensor-parallel-size 4 --max-num-seqs 1 --max-model-len 262144 \
  --max-num-batched-tokens 16384 \
  --served-model-name qwen --no-enable-prefix-caching --disable-custom-all-reduce \
  --speculative-config "$SPEC" --port $PORT \
  > "$OUT/vllm_mtp_tuned.log" 2>&1 &
pid=$!
ok=1
for i in $(seq 1 480); do
  curl -sf localhost:$PORT/health >/dev/null 2>&1 && { ok=0; echo "  healthy after $((i*2))s"; break; }
  kill -0 $pid 2>/dev/null || { echo "  died"; tail -14 "$OUT/vllm_mtp_tuned.log"; break; }
  sleep 2
done
if [ $ok -ne 0 ]; then echo "VLLM_MTP_TUNED_FAILED"; grep -iE "error|assert|valueerror|batched" "$OUT/vllm_mtp_tuned.log" | tail -10; exit 1; fi
python3 "$BENCH/bench_client.py" --base-url http://127.0.0.1:$PORT --model qwen \
  --fixtures "$BENCH/fixtures_top.json" --out "$OUT/vllm-tp4-mtp-tuned-top.json" \
  --label "vllm-tp4-mtp-tuned" --gen 256 --repeats 1 --max-depth 261120
echo "=== spec acceptance ==="; grep -iE "acceptance length|acceptance rate|Accepted throughput" "$OUT/vllm_mtp_tuned.log" | tail -4
pkill -f "vllm serve" 2>/dev/null
echo "BENCH_VLLM_MTP_TUNED_DONE"
