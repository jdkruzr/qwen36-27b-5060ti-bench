#!/usr/bin/env python3
"""Combine per-config bench JSONs into one table + compute MTP speedup.
Run on the box: python3 aggregate.py /root/bench/results > summary.md
Also writes <dir>/combined.json for downstream charting.
"""
import sys, os, json, glob

def load(d):
    rows = []
    for f in glob.glob(os.path.join(d, "*.json")):
        if os.path.basename(f) == "combined.json":
            continue
        try:
            for r in json.load(open(f)):
                rows.append(r)
        except Exception as e:
            print(f"<!-- skip {f}: {e} -->", file=sys.stderr)
    return rows

def main():
    d = sys.argv[1] if len(sys.argv) > 1 else "."
    rows = load(d)
    if not rows:
        print("no results yet"); return
    # index by (label, depth)
    depths = sorted({r["depth"] for r in rows})
    labels = sorted({r["label"] for r in rows})
    by = {(r["label"], r["depth"]): r for r in rows}

    def fmt_depth(x): return f"{x//1024}k" if x >= 1024 else str(x)

    print("## PP (prefill) tok/s — bs=1, cold prefill\n")
    print("| config | " + " | ".join(fmt_depth(x) for x in depths) + " |")
    print("|" + "---|"*(len(depths)+1))
    for lab in labels:
        cells = [f"{by[(lab,x)]['pp_tok_s']:.0f}" if (lab,x) in by else "·" for x in depths]
        print(f"| {lab} | " + " | ".join(cells) + " |")

    print("\n## TG (decode) tok/s — bs=1\n")
    print("| config | " + " | ".join(fmt_depth(x) for x in depths) + " |")
    print("|" + "---|"*(len(depths)+1))
    for lab in labels:
        cells = [f"{by[(lab,x)]['tg_tok_s']:.1f}" if (lab,x) in by else "·" for x in depths]
        print(f"| {lab} | " + " | ".join(cells) + " |")

    # MTP speedup: pair labels differing only by nomtp/mtp
    print("\n## MTP TG speedup (mtp / nomtp)\n")
    print("| engine | " + " | ".join(fmt_depth(x) for x in depths) + " |")
    print("|" + "---|"*(len(depths)+1))
    bases = sorted({lab.replace("-mtp","").replace("-nomtp","") for lab in labels})
    for base in bases:
        n, m = f"{base}-nomtp", f"{base}-mtp"
        cells = []
        for x in depths:
            if (n,x) in by and (m,x) in by and by[(n,x)]['tg_tok_s']>0:
                cells.append(f"{by[(m,x)]['tg_tok_s']/by[(n,x)]['tg_tok_s']:.2f}x")
            else:
                cells.append("·")
        print(f"| {base} | " + " | ".join(cells) + " |")

    json.dump({"depths": depths, "labels": labels, "rows": rows},
              open(os.path.join(d, "combined.json"), "w"), indent=2)
    print(f"\n<!-- wrote {os.path.join(d,'combined.json')} ({len(rows)} rows) -->")

if __name__ == "__main__":
    main()
