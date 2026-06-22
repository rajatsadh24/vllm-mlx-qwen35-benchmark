#!/usr/bin/env python3
"""5-slide LinkedIn carousel (square PDF) from data/final/*.csv.

LinkedIn "document" posts (PDF carousels) get high reach. Produces
assets/linkedin_carousel.pdf at 1080x1080 per slide.

Usage: python3 scripts/carousel.py [final_dir=data/final] [out=assets/linkedin_carousel.pdf]
"""
import csv, os, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import FancyBboxPatch

FINAL = sys.argv[1] if len(sys.argv) > 1 else "data/final"
OUT   = sys.argv[2] if len(sys.argv) > 2 else "assets/linkedin_carousel.pdf"
os.makedirs(os.path.dirname(OUT) or ".", exist_ok=True)

BASE, QUANT, ACC, NAVY, INK, MUT = "#9AA0A6", "#0B6FB8", "#E8590C", "#0b1f33", "#1b1d21", "#5b6470"
REPO = "github.com/rajatsadh24/vllm-mlx-qwen35-benchmark"
plt.rcParams.update({"font.family": "sans-serif", "font.size": 13})

def load(name):
    with open(os.path.join(FINAL, name)) as fh:
        return list(csv.DictReader(fh))
dc = load("decode_comparison.csv")
C  = [int(r["concurrency"]) for r in dc]; CX = [str(c) for c in C]; X = list(range(len(C)))
def col(k): return [float(r[k]) for r in dc]
b_agg, q_agg = col("base_agg_tps"), col("quant_agg_tps")
b_ttft, q_ttft = col("base_ttft_ms"), col("quant_ttft_ms")
b_tpj, q_tpj = col("base_tok_per_J"), col("quant_tok_per_J")

def new_slide():
    fig = plt.figure(figsize=(10, 10)); fig.patch.set_facecolor("white")
    return fig

def header(fig, num):
    fig.text(0.07, 0.945, "vLLM on MLX  ·  Apple M5  ·  16 GB", color=QUANT, fontsize=13.5,
             fontweight="bold", va="top")
    fig.text(0.93, 0.945, f"{num} / 5", color=MUT, fontsize=12.5, ha="right", va="top")
    fig.add_artist(plt.Line2D([0.07, 0.93], [0.915, 0.915], color="#e6e8ec", lw=1.2))

def footer(fig):
    fig.text(0.07, 0.045, REPO, color=MUT, fontsize=11)

def title(fig, t):
    fig.text(0.07, 0.875, t, color=INK, fontsize=27, fontweight="bold", va="top")

def takeaway(fig, t):
    fig.text(0.07, 0.165, t, color=INK, fontsize=15, va="top", ha="left", wrap=True,
             bbox=dict(boxstyle="round,pad=0.9", fc="#f2f6fa", ec="#dfe7ee", lw=1))

def chart_axes(fig):
    return fig.add_axes([0.10, 0.27, 0.82, 0.45])

def style_ax(ax, ylab, xlab="Concurrency", ylog=False):
    ax.set_ylabel(ylab, fontsize=13); ax.set_xlabel(xlab, fontsize=13)
    ax.grid(True, color="#ECEDF0", lw=0.9); ax.set_axisbelow(True)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    if ylog: ax.set_yscale("log")

# ---------- Slide 1: cover ----------
def slide_cover(pdf):
    fig = new_slide(); fig.patch.set_facecolor(NAVY)
    fig.text(0.08, 0.88, "LOCAL AGENTIC WORKLOADS  ·  BENCHMARK", color="#7fb3d5",
             fontsize=14, fontweight="bold", va="top")
    fig.text(0.08, 0.80, "AI agents fan out into\nmany parallel calls —\ncan a 16 GB M5 serve them?",
             color="white", fontsize=33, fontweight="bold", va="top", linespacing=1.15)
    fig.text(0.08, 0.50, "vLLM (MLX backend)  ·  continuous batching  ·  4-bit KV cache",
             color="#cfe0f0", fontsize=15, va="top")
    ax = fig.add_axes([0, 0, 1, 1]); ax.axis("off"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    tiles = [("4 → 8", "parallel requests"), ("+28%", "aggregate throughput"),
             ("16.5s → 0.58s", "first-token latency"), ("lossless", "10/10 identical outputs")]
    pos = [(0.08, 0.30), (0.52, 0.30), (0.08, 0.12), (0.52, 0.12)]
    for (val, lab), (x, y) in zip(tiles, pos):
        ax.add_patch(FancyBboxPatch((x, y), 0.40, 0.155, boxstyle="round,pad=0.012",
                     fc="#15324c", ec="#26496a", lw=1.2, transform=ax.transAxes))
        ax.text(x + 0.025, y + 0.105, val, color="white", fontsize=25, fontweight="bold",
                transform=ax.transAxes, va="center")
        ax.text(x + 0.025, y + 0.038, lab, color="#aac6de", fontsize=12.5, transform=ax.transAxes)
    fig.text(0.08, 0.045, REPO, color="#7fb3d5", fontsize=11.5)
    pdf.savefig(fig, facecolor=NAVY); plt.close(fig)

# ---------- Slide 2: throughput ----------
def slide_throughput(pdf):
    fig = new_slide(); header(fig, 2)
    title(fig, "4-bit KV cache doubled\nparallel capacity")
    ax = chart_axes(fig)
    ax.plot(X, q_agg, "-o", color=QUANT, lw=3, ms=9, label="4-bit KV · 8-wide")
    ax.plot(X, b_agg, "-o", color=BASE, lw=3, ms=9, label="fp16 KV · 4-wide")
    ax.annotate("+28%", xy=(3, q_agg[3]), xytext=(3, q_agg[3] + 12), ha="center",
                color=ACC, fontweight="bold", fontsize=16,
                arrowprops=dict(arrowstyle="->", color=ACC, lw=2))
    ax.set_xticks(X); ax.set_xticklabels(CX); style_ax(ax, "Aggregate throughput (tok/s)")
    ax.legend(frameon=False, fontsize=12, loc="lower right")
    takeaway(fig, "Switching the KV cache to 4-bit freed enough memory to run 8 parallel\n"
                  "requests instead of 4 — +28% aggregate, with no single-request penalty.")
    footer(fig); pdf.savefig(fig); plt.close(fig)

# ---------- Slide 3: latency cliff ----------
def slide_latency(pdf):
    fig = new_slide(); header(fig, 3)
    title(fig, "A wider engine stops\nyour agent stalling")
    ax = chart_axes(fig); w = 0.38
    ax.bar([x - w/2 for x in X], b_ttft, w, color=BASE, label="fp16 KV · 4-wide")
    ax.bar([x + w/2 for x in X], q_ttft, w, color=QUANT, label="4-bit KV · 8-wide")
    ax.set_xticks(X); ax.set_xticklabels(CX); style_ax(ax, "Time to first token (ms, log)", ylog=True)
    ax.annotate("16.5 s → 0.58 s\nat 8 users", xy=(3.19, q_ttft[3]), xytext=(1.6, 4000),
                color=ACC, fontweight="bold", fontsize=14,
                arrowprops=dict(arrowstyle="->", color=ACC, lw=2))
    ax.legend(frameon=False, fontsize=12, loc="upper left")
    takeaway(fig, "With a 4-wide engine, 4 of every 8 agent calls wait in a queue. An 8-wide\n"
                  "engine serves them all at once — first-token latency drops 16.5 s → 0.58 s.")
    footer(fig); pdf.savefig(fig); plt.close(fig)

# ---------- Slide 4: efficiency ----------
def slide_efficiency(pdf):
    fig = new_slide(); header(fig, 4)
    title(fig, "Throughput scales —\nefficiency stays flat")
    ax = chart_axes(fig)
    ax.axhspan(3.4, 5.1, color=ACC, alpha=0.08)
    ax.plot(X, q_tpj, "-o", color=QUANT, lw=3, ms=9, label="4-bit KV")
    ax.plot(X, b_tpj, "-o", color=BASE, lw=3, ms=9, label="fp16 KV")
    ax.set_ylim(0, 6); ax.set_xticks(X); ax.set_xticklabels(CX)
    style_ax(ax, "Energy efficiency (tokens / joule)")
    ax.legend(frameon=False, fontsize=12, loc="lower right")
    takeaway(fig, "GPU power rises with concurrency in step with throughput, so tokens/joule\n"
                  "stays flat (~4–5) across every load. Batch for throughput, not energy.")
    footer(fig); pdf.savefig(fig); plt.close(fig)

# ---------- Slide 5: the bug + CTA ----------
def slide_bug(pdf):
    fig = new_slide(); header(fig, 5)
    title(fig, "Bonus: a 32 GB cache limit\nthat OOM'd 16 GB Macs")
    ax = chart_axes(fig)
    comps = ["OS", "Model (4-bit)", "KV + activations", "MLX buffer cache"]
    colors = ["#B7BCC4", "#9AA0A6", "#5FA8D0", QUANT]
    before = [5.5, 4.0, 3.0, 19.5]; after = [5.5, 4.0, 3.0, 2.0]
    bb = ba = 0
    for cmp, cb, ca, cl in zip(comps, before, after, colors):
        ax.bar("32 GB cache\n(crashes)", cb, bottom=bb, color=cl, label=cmp)
        ax.bar("2 GB cache\n(patched)", ca, bottom=ba, color=cl); bb += cb; ba += ca
    ax.axhline(16, color=ACC, ls="--", lw=2)
    ax.text(1.45, 16.6, "16 GB physical", color=ACC, fontweight="bold", ha="right", fontsize=12)
    ax.set_ylim(0, 34); style_ax(ax, "Unified memory (GB)", xlab="")
    ax.legend(frameon=False, fontsize=10.5, loc="upper right")
    takeaway(fig, "A hardcoded 32 GB MLX buffer-cache limit lets memory grow until a 16 GB Mac\n"
                  "OOMs. Traced it, patched to 2 GB, and reported upstream.   ▶  Full benchmark,\n"
                  "scripts & live dashboard at " + REPO)
    footer(fig); pdf.savefig(fig); plt.close(fig)

with PdfPages(OUT) as pdf:
    slide_cover(pdf); slide_throughput(pdf); slide_latency(pdf)
    slide_efficiency(pdf); slide_bug(pdf)
print("wrote", OUT, "(5 slides, 1080x1080)")
