#!/usr/bin/env python3
"""Build fixed-token-length coding prompts from a real code corpus.

Emits identical prompts reused across every engine so comparisons are fair.
Each prompt = a large code context (mirrors a coding harness resending repo
files each turn) + a concrete implementation task at the end so generation
is code, not prose.
"""
import argparse, json, os, glob, sys

def load_corpus(root, exts):
    texts = []
    for ext in exts:
        for p in glob.glob(os.path.join(root, "**", f"*.{ext}"), recursive=True):
            try:
                with open(p, "r", errors="ignore") as f:
                    texts.append(f"// ===== FILE: {os.path.relpath(p, root)} =====\n" + f.read())
            except Exception:
                pass
    return "\n\n".join(texts)

TASK = (
    "\n\n// ===== TASK =====\n"
    "// Given the code above, implement a new function `estimate_kv_bytes` that\n"
    "// returns the KV-cache size in bytes for a given context length, number of\n"
    "// layers, and head dim. Return only the function implementation.\n"
    "// IMPLEMENTATION:\n"
)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer", required=True, help="dir or file with tokenizer.json")
    ap.add_argument("--corpus", required=True, help="root dir of source files")
    ap.add_argument("--exts", default="cpp,h,hpp,c,cu,py,ts,js,go,rs")
    ap.add_argument("--depths", default="1024,4096,16384,32768,65536,131072")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    from tokenizers import Tokenizer
    tk_path = args.tokenizer
    if os.path.isdir(tk_path):
        tk_path = os.path.join(tk_path, "tokenizer.json")
    tk = Tokenizer.from_file(tk_path)

    corpus = load_corpus(args.corpus, args.exts.split(","))
    task_ids = tk.encode(TASK).ids
    corpus_ids = tk.encode(corpus).ids
    print(f"corpus tokens available: {len(corpus_ids)}", file=sys.stderr)

    depths = [int(d) for d in args.depths.split(",")]
    fixtures = []
    for d in depths:
        budget = d - len(task_ids)
        if budget <= 0:
            continue
        if budget > len(corpus_ids):
            print(f"WARN depth {d}: corpus too small ({len(corpus_ids)} tok), repeating", file=sys.stderr)
            reps = (budget // len(corpus_ids)) + 1
            ids = (corpus_ids * reps)[:budget]
        else:
            ids = corpus_ids[:budget]
        prompt = tk.decode(ids) + TASK
        actual = len(tk.encode(prompt).ids)
        fixtures.append({"depth": d, "target_tokens": d, "actual_tokens": actual, "prompt": prompt})
        print(f"depth {d}: actual {actual} tok", file=sys.stderr)

    with open(args.out, "w") as f:
        json.dump(fixtures, f)
    print(f"wrote {len(fixtures)} fixtures -> {args.out}", file=sys.stderr)

if __name__ == "__main__":
    main()
