#!/bin/bash
# Lighter bring-up for the 2x RTX 3090 (Ampere sm_86) spike: llama.cpp only.
# Builds llama.cpp for sm_86 and downloads just the Q8_0 MTP GGUF + tokenizer.
# Usage: bootstrap_3090.sh <HF_TOKEN>
set -u
HF_TOKEN_IN="${1:?pass HF token as arg1}"
mkdir -p /root/logs /root/bench/results
umask 077; printf "%s" "$HF_TOKEN_IN" > /root/.hf_token; umask 022

########## llama.cpp build (sm_86 for Ampere/3090) ##########
cat > /root/_build_llama.sh <<'EOS'
#!/bin/bash
set -e
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq && apt-get install -y -qq libcurl4-openssl-dev ccache >/dev/null 2>&1 || true
cd /root
[ -d llama.cpp ] || git clone --depth 1 https://github.com/ggml-org/llama.cpp
cd llama.cpp
git log -1 --oneline
cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=86 -DCMAKE_BUILD_TYPE=Release -DLLAMA_CURL=ON
cmake --build build --config Release -j "$(nproc)"
ls build/bin/ | grep -E "llama-(server|bench)"
echo "LLAMA_BUILD_DONE"
EOS

########## downloads: Q8_0 MTP GGUF (27GB) + tokenizer.json only ##########
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
# tokenizer for make_prompts (tiny)
$HF download Qwen/Qwen3.6-27B-FP8 tokenizer.json --local-dir /root/models/tok
echo "Q8_DONE"; du -sh /root/models/* 2>/dev/null
echo "ALL_DOWNLOADS_DONE"
EOS

chmod +x /root/_build_llama.sh /root/_download.sh
nohup /root/_build_llama.sh > /root/logs/llama_build.log 2>&1 & echo "llama-build PID $!"
nohup /root/_download.sh    > /root/logs/download.log    2>&1 & echo "downloads   PID $!"
echo "BOOTSTRAP_3090_LAUNCHED — poll /root/logs/*.log for *_DONE"
