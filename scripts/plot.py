#!/usr/bin/env python3
"""Generate benchmark figures from data/final/*.csv into assets/.

Charts (PNG @150dpi + SVG):
  fig1_batching_tradeoff   aggregate up / per-user down (the core serving truth)
  fig2_latency_cliff       TTFT log-scale, baseline vs quant (the capacity money shot)
  fig3_efficiency          throughput up + power up -> tok/J flat (3 small multiples)
  fig4_prefill             prefill TTFT & tok/s by prompt size
  fig5_pareto              per-user vs aggregate, operating point circled
  fig6_memory              why it OOM'd: cache_limit 32GB vs 2GB against 16GB
  infographic              composite for social (panels 1-3 + headline + footer)

Usage: python3 scripts/plot.py [final_dir=data/final] [out_dir=assets]
"""
import csv, os, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

FINAL = sys.argv[1] if len(sys.argv) > 1 else "data/final"
OUT   = sys.argv[2] if len(sys.argv) > 2 else "assets"
os.makedirs(OUT, exist_ok=True)

# ---- colorblind-safe palette (Okabe-Ito-ish) ----
BASE  = "#7F7F86"   # neutral slate = baseline (fp16 KV, 4-wide)
QUANT = "#0072B2"   # blue = optimized (4-bit KV, 8-wide)
ACC   = "#D55E00"   # vermillion = annotations/highlights
INK   = "#1A1A1A"
GRID  = "#D9D9D9"

plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 150, "font.size": 11,
    "axes.edgecolor": "#666", "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": INK, "ytick.color": INK, "axes.titleweight": "bold",
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.axisbelow": True, "font.family": "sans-serif",
})

def load(name):
    with open(os.path.join(FINAL, name)) as fh:
        return list(csv.DictReader(fh))

dc = load("decode_comparison.csv")
C  = [int(r["concurrency"]) for r in dc]
def col(k): return [float(r[k]) for r in dc]
b_agg, q_agg = col("base_agg_tps"), col("quant_agg_tps")
b_gen, q_gen = col("base_gen_tps"), col("quant_gen_tps")
b_ttft, q_ttft = col("base_ttft_ms"), col("quant_ttft_ms")
b_pwr, q_pwr = col("base_power_W"), col("quant_power_W")
b_tpj, q_tpj = col("base_tok_per_J"), col("quant_tok_per_J")
X = list(range(len(C)))
XT = [str(c) for c in C]

def save(fig, name):
    for ext in ("png", "svg"):
        fig.savefig(os.path.join(OUT, f"{name}.{ext}"), bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("wrote", name)

# ---- fig1: batching trade-off (stacked panels) ----
def fig1():
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 7), sharex=True)
    ax1.plot(X, q_agg, "-o", color=QUANT, lw=2.5, ms=7, label="4-bit KV (8-wide)")
    ax1.plot(X, b_agg, "-o", color=BASE, lw=2.5, ms=7, label="fp16 KV (4-wide)")
    ax1.set_ylabel("Aggregate throughput (tok/s)")
    ax1.set_title("Batching trades per-user speed for total throughput")
    ax1.legend(frameon=False, loc="lower right")
    ax1.annotate("+28%", xy=(3, q_agg[3]), xytext=(3, q_agg[3]+10),
                 color=ACC, fontweight="bold", ha="center",
                 arrowprops=dict(arrowstyle="->", color=ACC))
    ax2.plot(X, q_gen, "-o", color=QUANT, lw=2.5, ms=7)
    ax2.plot(X, b_gen, "-o", color=BASE, lw=2.5, ms=7)
    ax2.set_ylabel("Per-request speed (tok/s)")
    ax2.set_xlabel("Offered concurrency")
    ax2.set_ylim(0, 40)
    ax2.text(0.02, 0.06, "Aggregate climbs while each user slows — the continuous-batching reality.",
             transform=ax2.transAxes, fontsize=9, style="italic", color="#555")
    for ax in (ax1, ax2):
        ax.set_xticks(X); ax.set_xticklabels(XT)
    save(fig, "fig1_batching_tradeoff")

# ---- fig2: latency cliff (log bars) ----
def fig2():
    fig, ax = plt.subplots(figsize=(8, 5))
    w = 0.38
    xb = [x - w/2 for x in X]; xq = [x + w/2 for x in X]
    ax.bar(xb, b_ttft, w, color=BASE, label="fp16 KV (4-wide)")
    ax.bar(xq, q_ttft, w, color=QUANT, label="4-bit KV (8-wide)")
    ax.set_yscale("log")
    ax.set_ylabel("Time to first token (ms, log scale)")
    ax.set_xlabel("Offered concurrency")
    ax.set_title("The latency cliff: a wider engine kills the queue")
    ax.set_xticks(X); ax.set_xticklabels(XT)
    ax.legend(frameon=False, loc="upper left")
    ax.axvspan(2.5, 4.5, color=ACC, alpha=0.06)
    ax.annotate("16.5 s vs 0.58 s\nat conc 8",
                xy=(3.19, q_ttft[3]), xytext=(2.0, 3000),
                color=ACC, fontweight="bold", fontsize=10,
                arrowprops=dict(arrowstyle="->", color=ACC))
    ax.text(0.99, 0.02, "shaded = queue-saturated (offered load > engine width)",
            transform=ax.transAxes, fontsize=8, color="#777", ha="right", style="italic")
    save(fig, "fig2_latency_cliff")

# ---- fig3: efficiency causal chain (3 small multiples) ----
def fig3():
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    series = [("Aggregate throughput\n(tok/s)", b_agg, q_agg, None),
              ("Combined power\n(W, GPU+CPU)", b_pwr, q_pwr, None),
              ("Energy efficiency\n(tokens / joule)", b_tpj, q_tpj, (0, 6))]
    for ax, (title, bvals, qvals, ylim) in zip(axes, series):
        ax.plot(X, qvals, "-o", color=QUANT, lw=2.3, ms=6, label="4-bit KV")
        ax.plot(X, bvals, "-o", color=BASE, lw=2.3, ms=6, label="fp16 KV")
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("Concurrency")
        ax.set_xticks(X); ax.set_xticklabels(XT)
        if ylim: ax.set_ylim(*ylim)
    axes[0].legend(frameon=False, loc="lower right", fontsize=9)
    axes[2].axhspan(3.4, 5.1, color=ACC, alpha=0.07)
    axes[2].text(0.5, 0.06, "flat", transform=axes[2].transAxes, color=ACC,
                 fontweight="bold", ha="center")
    fig.suptitle("Throughput ↑ and power ↑ rise together → efficiency is flat. "
                 "Batch for throughput, not energy.", fontweight="bold", y=1.02)
    save(fig, "fig3_efficiency")

# ---- fig4: prefill by prompt size ----
def fig4():
    pf = load("prefill_summary.csv")
    rows = [r for r in pf if int(r["concurrency"]) == 1]
    sets = [r["prompt_set"] for r in rows]
    toks = [int(r["prompt_tokens"]) for r in rows]
    ttft = [float(r["ttft_ms"]) for r in rows]
    tps  = [float(r["prompt_tps"]) for r in rows]
    labels = [f"{s}\n({t} tok)" for s, t in zip(sets, toks)]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 4.2))
    a1.bar(labels, ttft, color=QUANT)
    a1.set_ylabel("TTFT (ms)"); a1.set_title("Prefill latency by prompt size (single stream)")
    for i, v in enumerate(ttft):
        a1.text(i, v, f"{v:.0f} ms", ha="center", va="bottom", fontsize=9, fontweight="bold")
    a2.bar(labels, tps, color=BASE)
    a2.set_ylabel("Prompt processing (tok/s)"); a2.set_title("Prefill throughput (amortizes with length)")
    for i, v in enumerate(tps):
        a2.text(i, v, f"{v:.0f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
    save(fig, "fig4_prefill")

# ---- fig5: operating-point Pareto ----
def fig5():
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    ax.plot(b_gen, b_agg, "-o", color=BASE, lw=1.5, ms=8, label="fp16 KV (4-wide)")
    ax.plot(q_gen, q_agg, "-o", color=QUANT, lw=1.5, ms=8, label="4-bit KV (8-wide)")
    for x, y, c in zip(b_gen, b_agg, C): ax.annotate(f"c{c}", (x, y), fontsize=8, color=BASE,
                                                     xytext=(4, -10), textcoords="offset points")
    for x, y, c in zip(q_gen, q_agg, C): ax.annotate(f"c{c}", (x, y), fontsize=8, color=QUANT,
                                                     xytext=(4, 6), textcoords="offset points")
    # operating point = quant conc 8
    ix = C.index(8)
    ax.scatter([q_gen[ix]], [q_agg[ix]], s=520, facecolors="none", edgecolors=ACC, lw=2.5, zorder=5)
    ax.annotate("operating point\nconc 8 · ~14 tok/s/user · 86 tok/s",
                (q_gen[ix], q_agg[ix]), xytext=(q_gen[ix]+3, q_agg[ix]-22),
                color=ACC, fontweight="bold", fontsize=9,
                arrowprops=dict(arrowstyle="->", color=ACC))
    ax.axvline(10, color="#aaa", ls="--", lw=1)
    ax.text(10.3, ax.get_ylim()[0]+3, "min interactive\n~10 tok/s/user", fontsize=8, color="#777")
    ax.set_xlabel("Per-user speed (tok/s)"); ax.set_ylabel("Aggregate throughput (tok/s)")
    ax.set_title("Choosing the operating point")
    ax.legend(frameon=False, loc="upper center")
    save(fig, "fig5_pareto")

# ---- fig6: memory budget (why it OOM'd) ----
def fig6():
    fig, ax = plt.subplots(figsize=(7.5, 5))
    # illustrative budget (GB): OS, model weights, KV+activations, MLX retained cache
    comps = ["OS", "Model (4-bit)", "KV + activations", "MLX buffer cache"]
    colors = ["#B0B0B0", "#7F7F86", "#56A0C0", "#0072B2"]
    before = [5.5, 4.0, 3.0, 19.5]   # cache_limit 32GB -> cache balloons
    after  = [5.5, 4.0, 3.0, 2.0]    # cache_limit 2GB
    bot_b = bot_a = 0
    for c, cb, ca, col_ in zip(comps, before, after, colors):
        ax.bar("32 GB cache\n(crashes)", cb, bottom=bot_b, color=col_, label=c)
        ax.bar("2 GB cache\n(patched)", ca, bottom=bot_a, color=col_)
        bot_b += cb; bot_a += ca
    ax.axhline(16, color=ACC, ls="--", lw=2)
    ax.text(1.45, 16.4, "16 GB physical", color=ACC, fontweight="bold", ha="right", fontsize=10)
    ax.annotate("OOM → abort", xy=(0, 24), color=ACC, fontweight="bold", ha="center", fontsize=11)
    ax.set_ylabel("Unified memory (GB, illustrative)")
    ax.set_title("Why it OOM'd: a hardcoded 32 GB MLX cache limit")
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    ax.set_ylim(0, 34)
    save(fig, "fig6_memory")

# ---- composite infographic ----
def infographic():
    fig = plt.figure(figsize=(13, 9))
    gs = fig.add_gridspec(3, 2, height_ratios=[0.16, 1, 0.14], hspace=0.42, wspace=0.22)
    # headline
    axh = fig.add_subplot(gs[0, :]); axh.axis("off")
    axh.text(0.5, 0.7, "Qwen3.5-4B (4-bit) on a 16 GB Apple Silicon Mac",
             ha="center", fontsize=19, fontweight="bold")
    axh.text(0.5, 0.12, "4-bit KV cache → 4→8 concurrent users · +28% throughput · flat tokens/joule",
             ha="center", fontsize=12.5, color=QUANT)
    # panel A: aggregate
    axA = fig.add_subplot(gs[1, 0])
    axA.plot(X, q_agg, "-o", color=QUANT, lw=2.5, ms=7, label="4-bit KV (8-wide)")
    axA.plot(X, b_agg, "-o", color=BASE, lw=2.5, ms=7, label="fp16 KV (4-wide)")
    axA.set_title("Aggregate throughput"); axA.set_xlabel("Concurrency"); axA.set_ylabel("tok/s")
    axA.set_xticks(X); axA.set_xticklabels(XT); axA.legend(frameon=False, fontsize=9, loc="lower right")
    axA.annotate("+28%\n@ c8", xy=(3, q_agg[3]), xytext=(2.0, 80), color=ACC, fontweight="bold",
                 fontsize=10, arrowprops=dict(arrowstyle="->", color=ACC))
    # panel B: latency cliff
    axB = fig.add_subplot(gs[1, 1])
    w = 0.38; xb = [x - w/2 for x in X]; xq = [x + w/2 for x in X]
    axB.bar(xb, b_ttft, w, color=BASE); axB.bar(xq, q_ttft, w, color=QUANT)
    axB.set_yscale("log"); axB.set_title("Time to first token (queueing)")
    axB.set_xlabel("Concurrency"); axB.set_ylabel("ms (log)")
    axB.set_xticks(X); axB.set_xticklabels(XT)
    axB.annotate("16.5 s → 0.58 s", xy=(3.19, q_ttft[3]), xytext=(1.4, 4000), color=ACC,
                 fontweight="bold", fontsize=10, arrowprops=dict(arrowstyle="->", color=ACC))
    # footer stats
    axf = fig.add_subplot(gs[2, :]); axf.axis("off")
    stats = "35 tok/s single-stream   ·   86 tok/s @ 8 users   ·   ~14 tok/s per user   ·   ~18 W   ·   ~4–5 tokens/joule"
    axf.text(0.5, 0.6, stats, ha="center", fontsize=12, fontweight="bold")
    axf.text(0.5, 0.05, "vllm-mlx · continuous batching · fresh-server-per-point · powermetrics  |  github.com/rajatsadh24/vllm-mlx-qwen35-benchmark",
             ha="center", fontsize=8.5, color="#777")
    for ext in ("png", "svg"):
        fig.savefig(os.path.join(OUT, f"infographic.{ext}"), bbox_inches="tight", facecolor="white")
    plt.close(fig); print("wrote infographic")

for f in (fig1, fig2, fig3, fig4, fig5, fig6, infographic):
    f()
print("done ->", OUT)
