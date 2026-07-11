# Qwen3.6-27B on 4× RTX 5060 Ti — single-stream coding benchmark

Prefill (PP) and decode (TG) throughput for **Qwen3.6-27B** across context length, at
**batch = 1** — the regime a coding harness actually runs in — on four consumer Blackwell
cards (~$2K of hardware). Q8_0 weights, **unquantized f16 KV**, tensor-parallel, with and
without MTP speculative decoding. Includes a second-engine comparison (llama.cpp Q8_0 vs
vLLM FP8) and an interconnect analysis.

**📊 Live report:** https://jdkruzr.github.io/qwen36-27b-5060ti-bench/
&nbsp;·&nbsp; [full version incl. vLLM](https://jdkruzr.github.io/qwen36-27b-5060ti-bench/full-with-vllm.html)

## TL;DR

- **262,144 context fits, unquantized.** Q8_0 weights + f16 KV sit at **~11.6 GB of 16 GB per
  card** at full native context — the hybrid linear/full attention keeps KV tiny. No KV
  quantization required.
- **Decode is remarkably flat; prefill is the long-context cost.** Baseline TG drifts 39 → 32
  tok/s from 1k → 128k, then to ~27 at 255k. Prefill sits on a flat ~800 tok/s ceiling, so
  time-to-first-token grows linearly (minutes at max context).
- **MTP ~doubles decode in llama.cpp** (1.7–2.3×) for a ~6% prefill tax — a clear win for coding.
- **The limiter is collective *latency*, not the interconnect.** No P2P on these cards, but PCIe
  runs only ~15–20% utilized; GPUs stall on per-layer all-reduce sync (SM% dips, ~65% power at
  100% "util"). A 2-card P2P (Geohot/reBAR) A/B on a sister rig changed nothing — P2P is a
  bandwidth fix, and this isn't bandwidth-bound.
- **Same feature, opposite result across engines.** At 255k with ~identical ~90% MTP draft
  acceptance, MTP is a **2× win in llama.cpp (26.7 → 52.2)** and an **8× *loss* in vLLM
  (28.4 → 3.7)** — the spec-decode implementation, not the algorithm, decides.

## Results (bs=1, f16 KV, cold prefill)

### llama.cpp Q8_0, 4-GPU tensor-parallel

| ctx | 1k | 4k | 16k | 32k | 64k | 128k | 255k |
|----|----|----|-----|-----|-----|------|------|
| **TG** baseline (tok/s) | 39.4 | 38.9 | 37.8 | 36.7 | 35.3 | 31.9 | 26.7 |
| **TG** MTP (tok/s) | 69.0 | 67.2 | 75.2 | 65.0 | 81.0 | 66.7 | 52.2 |
| MTP speedup | 1.75× | 1.73× | 1.99× | 1.77× | 2.29× | 2.09× | 1.96× |
| **PP** baseline (tok/s) | 795 | 810 | 831 | 807 | 760 | 696 | 608 |

### Engine comparison @ 255k

| | decode baseline | decode MTP | prefill | draft accept |
|---|---|---|---|---|
| **llama.cpp** Q8_0 | 26.7 | **52.2** | 608 | high |
| **vLLM** FP8 (compiled) | 28.4 | **3.7** | 848 | ~90% |
| vLLM FP8 (eager) | 10.8 | — | 774 | — |

vLLM's CUTLASS FP8 **prefill** leads llama.cpp by ~25–30% everywhere. Its decode needs CUDA
graphs (eager is ~4× slower, launch-bound) and a ~4-min `torch.compile`; its MTP has excellent
acceptance but collapses throughput at long context (unchanged by `--max-num-batched-tokens`
tuning). llama.cpp is lean at bs=1 out of the box, and MTP is one flag.

## Second platform: 2× RTX 3090 (Ampere)

Same harness on a pair of RTX 3090s (no NVLink; on exactly 2 GPUs llama.cpp uses its **internal**
2-GPU all-reduce, selected via `GGML_CUDA_ALLREDUCE`), plus a 4×3090 bonus, vs the 4×5060 Ti:

| config (bs=1, Q8_0) | 1k TG | 255k TG | 1k PP | 255k PP |
|---|---|---|---|---|
| 2×3090 nomtp · f16 KV | 44.1 | 27.4 | 1148 | 839 |
| 2×3090 MTP · q8 KV | 69.9 | 45.2 | 982 | 744 |
| 4×3090 nomtp · f16 KV | 35.3 | 29.3 | 435 | 436 |
| 4×3090 MTP · f16 KV | 62.9 | 50.6 | 405 | 401 |
| 4×5060 Ti nomtp · f16 KV | 39.4 | 26.7 | 795 | 608 |
| 4×5060 Ti MTP · f16 KV | 69.0 | 52.2 | — | 562 |

- **Memory-bound vs latency-bound (the crux).** Live `dmon` on 2×3090 decode: SM 97%, mem-controller
  86%, PCIe ~0.1 GB/s, ~325 W — the pair hits the *right* bs=1 ceiling (VRAM bandwidth). The 5060 Ti
  quad stalls on collective latency (~100 W, SM dipping to 4–90%).
- **internal vs butterfly all-reduce:** internal (2-GPU only) beats the generic butterfly by +21%
  decode / +64% prefill at 1k.
- **q8-KV tax:** ~15% slower decode at 255k (23.4 vs 27.4). Like-for-like MTP speedup ≈ 1.93× on 3090
  ≈ 1.96× on 5060 Ti — MTP works equally on both.
- **More cards hurt at bs=1:** 4×3090 (butterfly) is slower than 2×3090 (internal) except a marginal
  max-context decode gain — adding GPUs loses the internal fast path.
- **Max-context MTP is a tie:** 4×3090 f16+MTP (50.6) ≈ 5060 Ti quad (52.2), within noise. At 4 cards
  both fall to butterfly and go latency-bound, so the 3090's bandwidth edge (real at 2-card/low-context)
  isn't on the critical path.
- **Value:** 2×3090 ≈ $2,400 vs 4×5060 Ti ≈ $2,000. The cheaper quad wins VRAM (f16 KV + MTP at 255k),
  Blackwell FP8/FP4, and power (~400 W vs ~650 W); the 3090 pair wins prefill/TTFT (~1.4×). Bandwidth is
  a non-issue (measured ~1–3 GB/s; even PCIe 4.0 ×4 has ~3× headroom) — the real variable is collective
  *latency* / topology (favor CPU-direct lanes).

## Hardware & software

- **GPUs:** 4× NVIDIA RTX 5060 Ti (16 GB, Blackwell **sm_120**), PCIe Gen4 ×8, **no NVLink, no
  P2P** (chipset-disabled). Vast.ai container, driver 580.95, CUDA 13.0.
- **Model:** Qwen3.6-27B — [`unsloth/Qwen3.6-27B-MTP-GGUF`](https://huggingface.co/unsloth/Qwen3.6-27B-MTP-GGUF)
  `Q8_0` (MTP heads embedded) for llama.cpp; [`Qwen/Qwen3.6-27B-FP8`](https://huggingface.co/Qwen/Qwen3.6-27B-FP8)
  for vLLM.
- **Engines:** current `llama.cpp` (`--split-mode tensor`, `--spec-type draft-mtp`);
  vLLM 0.24.0 (`--tensor-parallel-size 4`, `qwen3_5_mtp` speculative method, `VLLM_USE_DEEP_GEMM=0`).

## Method

Identical fixed-length coding prompts (real source corpus + an implement-this-function task),
streamed over the OpenAI-compatible API. **PP** = prompt tokens ÷ time-to-first-token (cold
prefill, prompt cache off). **TG** = decode tok/s over 256 generated tokens at bs=1. The 255k
point uses a ~255k prompt (not the literal 262,144 max) so 256 decode tokens fit inside the
model's trained context ceiling.

## Repo layout

```
index.html            # the report (llama.cpp only) — served by GitHub Pages
full-with-vllm.html   # full report incl. the vLLM comparison
scripts/              # the benchmark harness used to produce these numbers
```

## Reproduce

The scripts assume a box laid out like the Vast instance used here:
`/root/llama.cpp` (built), `/root/models/{q8-mtp,fp8}` (weights), `/root/bench` (this harness),
`/venv/main` (Python env with vLLM). Adjust paths for your setup.

```bash
# 1. one-shot bring-up: build llama.cpp (sm_120), install vLLM, download weights
bash scripts/bootstrap.sh <HF_TOKEN>

# 2. llama.cpp sweep (1k..256k, baseline + MTP) -> /root/bench/results/*.json
bash scripts/run.sh 262144

# 3. vLLM legs (FP8 TP=4)
bash scripts/bench_vllm_run.sh            # eager sweep
bash scripts/bench_vllm_compiled_top.sh   # compiled 255k baseline
bash scripts/bench_vllm_mtp_top.sh        # MTP @255k

# 4. collect into tables
python3 scripts/aggregate.py /root/bench/results
```

`make_prompts.py` builds the fixed-length fixtures; `bench_client.py` is the engine-agnostic
bs=1 streaming client (works against any OpenAI-compatible server).

## Caveats

Single box, single stream — throughput/serving is a different regime where the trade-offs flip.
Prefill measured cold to isolate the scaling curve; a real harness with prompt caching sees much
lower effective TTFT. llama.cpp numbers are medians of repeated runs; vLLM points are single
runs. Engines differ in quantization (Q8_0 vs FP8) — this is an end-to-end stack comparison, not
a like-for-like kernel test.
