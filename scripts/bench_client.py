#!/usr/bin/env python3
"""Engine-agnostic single-stream (bs=1) latency benchmark.

Streams OpenAI /v1/completions against any server (llama-server, vLLM, SGLang),
records TTFT and inter-token timing, derives:
  - PP (prefill) tok/s  = prompt_tokens / TTFT
  - TG (decode)  tok/s  = (gen_tokens - 1) / (t_last - t_first)
Repeats each point and reports the median. Forces full-length generation
(ignore_eos) so TG is measured over a stable token count.
"""
import argparse, json, time, statistics, sys, urllib.request

def stream_once(base_url, model, prompt, gen, extra):
    body = {
        "model": model,
        "prompt": prompt,
        "max_tokens": gen,
        "temperature": 0.0,
        "stream": True,
        "stream_options": {"include_usage": True},
        "ignore_eos": True,
        "min_tokens": gen,
        "cache_prompt": False,   # llama.cpp: force cold prefill each rep (clean PP);
                                 # vLLM ignores unknown field (prefix caching off at server)
    }
    body.update(extra or {})
    data = json.dumps(body).encode()
    req = urllib.request.Request(base_url.rstrip("/") + "/v1/completions",
                                 data=data, headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    t_first = None
    t_last = None
    ntok = 0
    usage = {}
    with urllib.request.urlopen(req, timeout=1800) as resp:
        for raw in resp:
            line = raw.decode("utf-8", "ignore").strip()
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                break
            try:
                obj = json.loads(payload)
            except Exception:
                continue
            if obj.get("usage"):
                usage = obj["usage"]
            choices = obj.get("choices") or []
            if choices and choices[0].get("text"):
                now = time.perf_counter()
                if t_first is None:
                    t_first = now
                t_last = now
                ntok += 1
    return {"t0": t0, "t_first": t_first, "t_last": t_last, "ntok": ntok, "usage": usage}

def bench(base_url, model, fixtures, gen, repeats, label, extra):
    rows = []
    for fx in fixtures:
        depth = fx["depth"]
        pp_list, tg_list, ttft_list, prompt_tok = [], [], [], None
        for r in range(repeats):
            try:
                m = stream_once(base_url, model, fx["prompt"], gen, extra)
            except Exception as e:
                print(f"[{label}] depth {depth} rep {r} ERROR: {e}", file=sys.stderr)
                continue
            if m["t_first"] is None or m["ntok"] < 2:
                print(f"[{label}] depth {depth} rep {r}: no/short output", file=sys.stderr)
                continue
            ttft = m["t_first"] - m["t0"]
            ptok = (m["usage"] or {}).get("prompt_tokens") or fx["actual_tokens"]
            prompt_tok = ptok
            pp = ptok / ttft if ttft > 0 else 0
            tg = (m["ntok"] - 1) / (m["t_last"] - m["t_first"]) if m["t_last"] > m["t_first"] else 0
            ttft_list.append(ttft); pp_list.append(pp); tg_list.append(tg)
            print(f"[{label}] depth={depth} rep={r} ttft={ttft:.3f}s PP={pp:.0f} TG={tg:.1f} tok/s "
                  f"(gen={m['ntok']})", file=sys.stderr)
        if pp_list:
            rows.append({
                "label": label, "depth": depth, "prompt_tokens": prompt_tok, "gen_tokens": gen,
                "ttft_s": round(statistics.median(ttft_list), 4),
                "pp_tok_s": round(statistics.median(pp_list), 1),
                "tg_tok_s": round(statistics.median(tg_list), 2),
                "repeats": len(pp_list),
            })
    return rows

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--fixtures", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--gen", type=int, default=256)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--extra", default="{}", help="JSON merged into request body")
    ap.add_argument("--max-depth", type=int, default=10**9)
    args = ap.parse_args()

    with open(args.fixtures) as f:
        fixtures = [fx for fx in json.load(f) if fx["depth"] <= args.max_depth]
    extra = json.loads(args.extra)
    rows = bench(args.base_url, args.model, fixtures, args.gen, args.repeats, args.label, extra)
    with open(args.out, "w") as f:
        json.dump(rows, f, indent=2)
    print(f"\nwrote {len(rows)} rows -> {args.out}", file=sys.stderr)
    for r in rows:
        print(json.dumps(r))

if __name__ == "__main__":
    main()
