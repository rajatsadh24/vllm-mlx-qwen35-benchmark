#!/usr/bin/env python3
"""4-bit KV cache quality A/B vs fp16 KV.

Greedy (temperature=0) completions over a fixed, diverse prompt set — so any
output difference is attributable to KV-cache quantization, not sampling. Two
modes:

  collect  POST the prompts to a running server, save completions to JSON.
  compare  Load two JSON collections, compute similarity metrics -> CSV + MD.

Usage:
  python3 quality_ab.py collect --url http://localhost:8000 --model m --out qual_fp16.json
  python3 quality_ab.py compare qual_fp16.json qual_quant.json --csv out.csv --md out.md
"""
import sys, json, csv, argparse, difflib, urllib.request

_LONG_CODE = '''Review this Python service for bugs, security, and performance:

import sqlite3, hashlib
def get_user(uid):
    conn = sqlite3.connect("app.db")
    q = "SELECT * FROM users WHERE id = " + str(uid)
    cur = conn.execute(q)
    return cur.fetchone()
def check_pw(user, pw):
    return user[2] == hashlib.md5(pw.encode()).hexdigest()
def list_orders(uid):
    conn = sqlite3.connect("app.db")
    rows = conn.execute("SELECT * FROM orders").fetchall()
    return [r for r in rows if r[1] == uid]

Give a detailed review with concrete fixes.'''

_LONG_LOG = '''You are an SRE. Analyze these production logs and write a short post-mortem
(timeline, root cause, remediation):

03:42:01 api  WARN  db pool exhausted (50/50), queue depth 120
03:42:03 api  ERROR upstream timeout calling payments (2000ms)
03:42:05 db   WARN  replication lag 8s and rising
03:42:09 api  ERROR 503 returned to 412 clients in last 5s
03:43:00 db   ERROR primary failover initiated
03:44:10 api  INFO  pool recovering (12/50)
03:48:00 api  INFO  error rate back to baseline
Keep it concise.'''

PROMPTS = [
    "What are the three laws of thermodynamics? Explain each one briefly.",
    "Write a Python function that checks if a string is a palindrome. Include type hints.",
    "Explain the difference between a process and a thread, with one example each.",
    "A farmer has 17 sheep. All but 9 run away. How many are left? Explain your reasoning.",
    "Summarize the main causes of World War I in five bullet points.",
    "Implement binary search in Python and explain its time complexity.",
    "Explain how HTTPS establishes a secure connection, step by step.",
    "Write a professional email declining a meeting invite, then explain your tone choices.",
    _LONG_CODE,
    _LONG_LOG,
]

def collect(url, model, out, max_tokens):
    results = []
    for i, p in enumerate(PROMPTS):
        body = json.dumps({
            "model": model, "messages": [{"role": "user", "content": p}],
            "temperature": 0, "max_tokens": max_tokens,
            "chat_template_kwargs": {"enable_thinking": False},
        }).encode()
        req = urllib.request.Request(url + "/v1/chat/completions", data=body,
                                     headers={"Content-Type": "application/json"})
        r = json.load(urllib.request.urlopen(req, timeout=600))
        c = r["choices"][0]
        results.append({"idx": i, "prompt": p[:70].replace("\n", " "),
                        "text": c["message"]["content"],
                        "finish": c["finish_reason"],
                        "tokens": r.get("usage", {}).get("completion_tokens")})
        print(f"  [{i+1}/{len(PROMPTS)}] {c['finish_reason']:>6} "
              f"{r.get('usage', {}).get('completion_tokens')} tok", flush=True)
    json.dump(results, open(out, "w"), indent=1)
    print("wrote", out)

def common_prefix(a, b):
    n = 0
    for ca, cb in zip(a, b):
        if ca == cb: n += 1
        else: break
    return n

def compare(fa, fb, outcsv, outmd):
    A = json.load(open(fa)); B = json.load(open(fb))
    rows = []
    for x, y in zip(A, B):
        ta, tb = x["text"], y["text"]
        pl = common_prefix(ta, tb)
        sim = difflib.SequenceMatcher(None, ta, tb).ratio()
        rows.append(dict(idx=x["idx"], prompt=x["prompt"],
                         identical=(ta == tb), prefix_chars=pl,
                         prefix_frac=round(pl / max(1, min(len(ta), len(tb))), 3),
                         similarity=round(sim, 3),
                         tok_fp16=x["tokens"], tok_quant=y["tokens"]))
    with open(outcsv, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    n = len(rows)
    ident = sum(r["identical"] for r in rows)
    msim = sum(r["similarity"] for r in rows) / n
    mpref = sum(r["prefix_frac"] for r in rows) / n
    low = [r for r in rows if r["similarity"] < 0.85]
    # markdown side-by-side for the most-divergent prompts
    md = ["# 4-bit KV cache — quality A/B (greedy decode)\n",
          f"- prompts: **{n}** · identical outputs: **{ident}/{n}** "
          f"· mean similarity: **{msim:.3f}** · mean prefix-agreement: **{mpref:.3f}**",
          f"- outputs with similarity < 0.85: **{len(low)}/{n}**\n",
          "| # | prompt | identical | similarity | prefix | tok fp16/4bit |",
          "|---|--------|-----------|-----------|--------|----------------|"]
    for r in rows:
        md.append(f"| {r['idx']} | {r['prompt']} | {'yes' if r['identical'] else 'no'} "
                  f"| {r['similarity']:.3f} | {r['prefix_frac']:.2f} | {r['tok_fp16']}/{r['tok_quant']} |")
    if low:
        md.append("\n## Most-divergent outputs (manual review)\n")
        for r in sorted(low, key=lambda z: z["similarity"])[:3]:
            a = next(z for z in A if z["idx"] == r["idx"])["text"]
            b = next(z for z in B if z["idx"] == r["idx"])["text"]
            md += [f"### Prompt {r['idx']} — {r['prompt']} (sim {r['similarity']:.3f})",
                   "**fp16 KV:**\n```\n" + a[:700] + "\n```",
                   "**4-bit KV:**\n```\n" + b[:700] + "\n```\n"]
    open(outmd, "w").write("\n".join(md))
    print(f"\n=== 4-bit KV quality vs fp16 ===")
    print(f"identical: {ident}/{n} | mean similarity: {msim:.3f} | "
          f"mean prefix-agreement: {mpref:.3f} | sim<0.85: {len(low)}/{n}")
    print(f"wrote {outcsv} and {outmd}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("collect"); c.add_argument("--url", required=True)
    c.add_argument("--model", required=True); c.add_argument("--out", required=True)
    c.add_argument("--max-tokens", type=int, default=768)
    k = sub.add_parser("compare"); k.add_argument("a"); k.add_argument("b")
    k.add_argument("--csv", default="data/final/quality_ab.csv")
    k.add_argument("--md", default="assets/quality_ab.md")
    args = ap.parse_args()
    if args.cmd == "collect":
        collect(args.url, args.model, args.out, args.max_tokens)
    else:
        compare(args.a, args.b, args.csv, args.md)
