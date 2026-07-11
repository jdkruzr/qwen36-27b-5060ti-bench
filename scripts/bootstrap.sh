#!/bin/bash
# One-shot bring-up for a fresh Vast box: builds llama.cpp (sm_120), installs
# vLLM, and downloads Qwen3.6-27B weights (Q8_0 MTP GGUF + FP8) — all in
# parallel, each logging to /root/logs. Then poll the logs for *_DONE markers.
#
# Usage: bootstrap.sh <HF_TOKEN>
set -u
HF_TOKEN_IN="${1:?pass HF token as arg1}"
mkdir -p /root/logs /root/bench/results
umask 077; printf "%s" "$HF_TOKEN_IN" > /root/.hf_token; umask 022

########## 1. llama.cpp build (sm_120) ##########
cat > /root/_build_llama.sh <<'EOS'
#!/bin/bash
set -e
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq && apt-get install -y -qq libcurl4-openssl-dev ccache >/dev/null 2>&1 || true
cd /root
[ -d llama.cpp ] || git clone --depth 1 https://github.com/ggml-org/llama.cpp
cd llama.cpp
git log -1 --oneline
cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=120 -DCMAKE_BUILD_TYPE=Release -DLLAMA_CURL=ON -DGGML_CUDA_FORCE_CUBLAS=ON
cmake --build build --config Release -j "$(nproc)"
ls build/bin/ | grep -E "llama-(server|bench|batched-bench)"
echo "LLAMA_BUILD_DONE"
EOS

########## 2. vLLM install ##########
cat > /root/_install_vllm.sh <<'EOS'
#!/bin/bash
set -e
source /venv/main/bin/activate
command -v uv >/dev/null 2>&1 || pip install -q uv
uv pip install --upgrade vllm
python - <<'PY'
import vllm, torch
print("vllm", vllm.__version__, "| torch", torch.__version__, "| cuda", torch.version.cuda, "| devices", torch.cuda.device_count())
PY
echo "VLLM_INSTALL_DONE"
EOS

########## 3. model downloads ##########
cat > /root/_download.sh <<'EOS'
#!/bin/bash
set -e
export HF_TOKEN=$(cat /root/.hf_token)
command -v uv >/dev/null 2>&1 || source /venv/main/bin/activate
uv venv /root/hfenv >/dev/null 2>&1 || true
uv pip install --python /root/hfenv/bin/python -q huggingface_hub hf_transfer
HF=/root/hfenv/bin/hf
mkdir -p /root/models
$HF download unsloth/Qwen3.6-27B-MTP-GGUF Qwen3.6-27B-Q8_0.gguf --local-dir /root/models/q8-mtp
echo "Q8_DONE"
$HF download Qwen/Qwen3.6-27B-FP8 --local-dir /root/models/fp8
echo "FP8_DONE"
du -sh /root/models/* 2>/dev/null
echo "ALL_DOWNLOADS_DONE"
EOS

chmod +x /root/_build_llama.sh /root/_install_vllm.sh /root/_download.sh
nohup /root/_build_llama.sh   > /root/logs/llama_build.log   2>&1 & echo "llama-build   PID $!"
nohup /root/_install_vllm.sh  > /root/logs/vllm_install.log  2>&1 & echo "vllm-install  PID $!"
nohup /root/_download.sh      > /root/logs/download.log      2>&1 & echo "downloads     PID $!"
echo "BOOTSTRAP_LAUNCHED — poll /root/logs/*.log for *_DONE markers"
