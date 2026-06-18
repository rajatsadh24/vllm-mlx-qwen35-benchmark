#!/usr/bin/env python3
"""Authoritative consolidation of the vllm-mlx benchmark run into final CSVs.

Reads the raw per-point CSVs and powermetrics logs from RAW_DIR and writes three
clean summary CSVs to FINAL_DIR:
  - prefill_summary.csv     (TTFT & prompt-processing by prompt size x concurrency)
  - decode_comparison.csv   (baseline fp16 KV vs 4-bit KV, with power & tokens/joule)
  - power_stats.csv         (per-point GPU/CPU power: mean/median/p90/idle%)

Power is summarised with the MEAN (idle fraction is <2.4% everywhere, and the
mean is robust to the bimodal compute/idle duty-cycle that the median exaggerates
at low concurrency). Median and p90 are emitted alongside for transparency.
"""
import csv, glob, re, os, statistics, sys

RAW_DIR   = sys.argv[1] if len(sys.argv) > 1 else "."
FINAL_DIR = sys.argv[2] if len(sys.argv) > 2 else "."
CONCS = [1, 2, 4, 8, 16]

def rows(path):
    if not os.path.exists(path): return []
    with open(path) as fh: return list(csv.DictReader(fh))

def fmean(xs):  return statistics.fmean(xs) if xs else 0.0
def fmed(xs):   return statistics.median(xs) if xs else 0.0
def p90(xs):
    if not xs: return 0.0
    s = sorted(xs); return s[min(len(s)-1, int(round(0.9*(len(s)-1))))]

def parse_power(path):
    """Return (gpu_W list, cpu_W list) from a powermetrics dump (mW -> W)."""
    gpu, cpu = [], []
    pat = re.compile(r'^(GPU|CPU) Power:\s*([0-9.]+)\s*mW', re.I)
    if not os.path.exists(path): return gpu, cpu
    with open(path, errors="ignore") as fh:
        for line in fh:
            m = pat.match(line.strip())
            if m:
                (gpu if m.group(1).upper()=="GPU" else cpu).append(float(m.group(2))/1000.0)
    return gpu, cpu

# ---------------------------------------------------------------- prefill
pf = rows(os.path.join(RAW_DIR, "prefill.csv"))
groups = {}
for r in pf:
    groups.setdefault((r["prompt_set"], int(r["concurrency"])), []).append(r)
order = {"short":0, "medium":1, "long":2}
prefill_out = []
for (ps, c) in sorted(groups, key=lambda k: (order.get(k[0],9), k[1])):
    rs = groups[(ps,c)]
    g = lambda col: fmean([float(x[col]) for x in rs if x.get(col)])
    prefill_out.append(dict(
        prompt_set=ps, concurrency=c, n_reps=len(rs),
        prompt_tokens=round(g("prompt_tokens")),
        ttft_ms=round(g("ttft_ms"),1),
        prompt_tps=round(g("prompt_tps"),1),
        note=("raw prefill" if c <= 4 else "queue-dominated (max-num-seqs cap)"),
    ))

# ---------------------------------------------------------------- decode + power
def arm_point(prefix, c):
    csv_rows = rows(os.path.join(RAW_DIR, f"{prefix}_c{c}.csv"))
    if not csv_rows: return None
    g = lambda col: fmean([float(x[col]) for x in csv_rows if x.get(col)])
    gpu, cpu = parse_power(os.path.join(RAW_DIR, f"{prefix}_c{c}.txt"))
    gpu_mean, cpu_mean = fmean(gpu), fmean(cpu)
    comb = gpu_mean + cpu_mean
    thr = g("throughput_tps")
    gen = g("gen_tps")
    return dict(
        gen_tps=round(gen,1), agg_tps=round(thr,1), ttft_ms=round(g("ttft_ms"),0),
        gpu_W_mean=round(gpu_mean,1), gpu_W_med=round(fmed(gpu),1), gpu_W_p90=round(p90(gpu),1),
        cpu_W_mean=round(cpu_mean,1), comb_W=round(comb,1),
        idle_pct=round(100.0*sum(1 for v in gpu if v<0.1)/len(gpu),1) if gpu else 0.0,
        n_pwr=len(gpu),
        tok_per_J=round(thr/comb,2) if comb>0 else 0.0,
        failed=(gen < 1.0),
    )

decode_out, power_out = [], []
for c in CONCS:
    b = arm_point("dpow_base", c)
    q = arm_point("dpow_quant", c)
    row = dict(concurrency=c)
    for tag, d in (("base", b), ("quant", q)):
        if d:
            row[f"{tag}_gen_tps"]   = d["gen_tps"]
            row[f"{tag}_agg_tps"]   = d["agg_tps"]
            row[f"{tag}_ttft_ms"]   = d["ttft_ms"]
            row[f"{tag}_power_W"]   = d["comb_W"]
            row[f"{tag}_tok_per_J"] = d["tok_per_J"]
            row[f"{tag}_status"]    = "FAIL" if d["failed"] else "ok"
            power_out.append(dict(
                arm=("baseline_fp16" if tag=="base" else "kvquant_4bit"),
                concurrency=c, gpu_W_mean=d["gpu_W_mean"], gpu_W_median=d["gpu_W_med"],
                gpu_W_p90=d["gpu_W_p90"], cpu_W_mean=d["cpu_W_mean"],
                combined_W_mean=d["comb_W"], idle_pct=d["idle_pct"], n_samples=d["n_pwr"],
            ))
    decode_out.append(row)

# ---------------------------------------------------------------- write
def write_csv(path, dicts):
    if not dicts: return
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(dicts[0].keys())); w.writeheader(); w.writerows(dicts)
    print(f"wrote {path} ({len(dicts)} rows)")

write_csv(os.path.join(FINAL_DIR, "prefill_summary.csv"), prefill_out)
write_csv(os.path.join(FINAL_DIR, "decode_comparison.csv"), decode_out)
write_csv(os.path.join(FINAL_DIR, "power_stats.csv"), power_out)

# echo decode comparison for the report
print("\n=== DECODE COMPARISON (mean-based power) ===")
print(f"{'c':>3} | {'base agg':>8} {'base W':>6} {'base tJ':>7} | {'quant agg':>9} {'quant W':>7} {'quant tJ':>8}")
for r in decode_out:
    print(f"{r['concurrency']:>3} | {r.get('base_agg_tps','-'):>8} {r.get('base_power_W','-'):>6} {r.get('base_tok_per_J','-'):>7} | "
          f"{r.get('quant_agg_tps','-'):>9} {r.get('quant_power_W','-'):>7} {r.get('quant_tok_per_J','-'):>8}")
