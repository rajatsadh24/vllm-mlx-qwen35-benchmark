#!/usr/bin/env python3
"""Join decode CSVs with powermetrics logs -> tokens/joule, baseline vs KV-quant."""
import csv, glob, re, os

def _median(xs):
    if not xs:
        return 0.0
    s = sorted(xs)
    n = len(s)
    return s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2

def avg_power_w(path):
    """Median (GPU+CPU) power in watts from a powermetrics text dump.

    Median, not mean, so the handful of startup/idle samples can't skew it.
    """
    if not os.path.exists(path):
        return None
    gpu, cpu = [], []
    pat = re.compile(r'^(GPU|CPU) Power:\s*([0-9.]+)\s*mW', re.I)
    with open(path, errors="ignore") as fh:
        for line in fh:
            m = pat.match(line.strip())
            if m:
                (gpu if m.group(1).upper() == "GPU" else cpu).append(float(m.group(2)))
    g = _median(gpu)
    c = _median(cpu)
    return (g + c) / 1000.0, g / 1000.0, c / 1000.0, len(gpu)

def csv_metrics(path):
    if not os.path.exists(path):
        return None
    with open(path) as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        return None
    def mean(col):
        v = [float(r[col]) for r in rows if r.get(col)]
        return sum(v) / len(v) if v else 0.0
    return mean("gen_tps"), mean("throughput_tps"), mean("metal_peak_gb")

def collect(prefix):
    out = {}
    for f in glob.glob(f"{prefix}_c*.csv"):
        c = int(re.search(r"_c(\d+)\.csv$", f).group(1))
        m = csv_metrics(f)
        p = avg_power_w(f.replace(".csv", ".txt"))
        if m:
            out[c] = (m, p)
    return out

base = collect("dpow_base")
quant = collect("dpow_quant")
concs = sorted(set(base) | set(quant))

def fmt(d, c):
    if c not in d:
        return None
    (gen, thr, peak), pwr = d[c]
    if gen < 1.0:                      # near-zero gen => OOM/errored point
        return ("FAIL", None, None, None, peak)
    if not pwr:
        return ("ok", gen, thr, None, peak)
    comb, g, cw, n = pwr
    tpj = thr / comb if comb else 0.0
    return ("ok", gen, thr, (comb, tpj, n), peak)

print(f"\n{'conc':>4} | {'BASELINE (fp16 KV)':^38} | {'KV-QUANT (4-bit)':^38}")
print(f"{'':>4} | {'gen_tps':>7} {'aggTPS':>7} {'pwr_W':>6} {'tok/J':>6} | {'gen_tps':>7} {'aggTPS':>7} {'pwr_W':>6} {'tok/J':>6}")
print("-" * 90)
for c in concs:
    cells = []
    for d in (base, quant):
        r = fmt(d, c)
        if r is None:
            cells.append(f"{'-':>7} {'-':>7} {'-':>6} {'-':>6}")
        elif r[0] == "FAIL":
            cells.append(f"{'OOM/FAIL':>30}")
        else:
            _, gen, thr, pw, peak = r
            if pw:
                comb, tpj, _ = pw
                cells.append(f"{gen:>7.1f} {thr:>7.1f} {comb:>6.1f} {tpj:>6.2f}")
            else:
                cells.append(f"{gen:>7.1f} {thr:>7.1f} {'n/a':>6} {'n/a':>6}")
    print(f"{c:>4} | {cells[0]:^38} | {cells[1]:^38}")

print("\nNotes:")
print(" - tok/J = aggregate throughput_tps / (avg GPU+CPU watts). Higher = more efficient.")
print(" - power window includes the bench warmup round, so it is a slight under-estimate of")
print("   steady-state efficiency (a little idle/warmup power is averaged in).")
print(" - 'OOM/FAIL' = gen_tps collapsed (<1 tok/s); that point crashed.")
