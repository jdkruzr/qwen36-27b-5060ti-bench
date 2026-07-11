#!/bin/bash
# 2x3090: compare decode speed of -sm tensor vs -sm row (the "TP if supported" mode)
# at low context, to pick the faster mode for the main runs.
BIN=/root/llama.cpp/build/bin/llama-server
MODEL=/root/models/q8-mtp/Qwen3.6-27B-Q8_0.gguf
PORT=18080

test_mode(){ local sm="$1"
  pkill -f llama-server 2>/dev/null; sleep 3
  CUDA_VISIBLE_DEVICES=0,1 nohup "$BIN" -m "$MODEL" -ngl 99 -sm "$sm" -fa on -c 8192 \
    --alias qwen --host 127.0.0.1 --port $PORT --no-webui > /root/bench/smoke_$sm.log 2>&1 &
  local pid=$!; local ok=1
  for i in $(seq 1 90); do
    curl -sf localhost:$PORT/health >/dev/null 2>&1 && { ok=0; break; }
    kill -0 $pid 2>/dev/null || { echo ">> $sm: SERVER DIED"; grep -iE "error|not impl|assert|abort" /root/bench/smoke_$sm.log | tail -3; return; }
    sleep 2
  done
  [ $ok -ne 0 ] && { echo ">> $sm: no health in 180s"; return; }
  curl -s localhost:$PORT/completion -d '{"prompt":"def quicksort(arr):","n_predict":8,"cache_prompt":false}' >/dev/null 2>&1   # warmup
  R=$(curl -s localhost:$PORT/completion -d '{"prompt":"def quicksort(arr):","n_predict":160,"cache_prompt":false}')
  echo ">> $sm: $(echo "$R" | python3 -c "import sys,json;t=json.load(sys.stdin)['timings'];print('TG',round(t['predicted_per_second'],1),'tok/s')" 2>/dev/null || echo 'gen parse failed')"
  nvidia-smi --query-gpu=index,memory.used,power.draw --format=csv,noheader | head -2 | sed 's/^/     /'
}
for sm in tensor row; do test_mode "$sm"; done
pkill -f llama-server 2>/dev/null
echo SMOKE_SPLIT_DONE
