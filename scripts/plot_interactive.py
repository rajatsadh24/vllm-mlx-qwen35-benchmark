#!/usr/bin/env python3
"""Interactive Plotly dashboard from data/final/*.csv -> assets/dashboard.html.

A single self-contained page (Plotly via CDN) with 6 linked panels. Hover shows
exact values, so no static label crowding — the reason to use Plotly here.
Static PNG/SVG for README/social still come from scripts/plot.py (matplotlib).

Usage: python3 scripts/plot_interactive.py [final_dir=data/final] [out=assets/dashboard.html]
"""
import csv, os, sys
import plotly.graph_objects as go
from plotly.subplots import make_subplots

FINAL = sys.argv[1] if len(sys.argv) > 1 else "data/final"
OUT   = sys.argv[2] if len(sys.argv) > 2 else "assets/dashboard.html"
os.makedirs(os.path.dirname(OUT) or ".", exist_ok=True)

BASE, QUANT = "#7F7F86", "#0072B2"

def load(name):
    with open(os.path.join(FINAL, name)) as fh:
        return list(csv.DictReader(fh))

dc = load("decode_comparison.csv")
C  = [int(r["concurrency"]) for r in dc]
CX = [str(c) for c in C]                      # categorical x = even spacing
def col(k): return [float(r[k]) for r in dc]
b_agg, q_agg = col("base_agg_tps"), col("quant_agg_tps")
b_gen, q_gen = col("base_gen_tps"), col("quant_gen_tps")
b_ttft, q_ttft = col("base_ttft_ms"), col("quant_ttft_ms")
b_pwr, q_pwr = col("base_power_W"), col("quant_power_W")
b_tpj, q_tpj = col("base_tok_per_J"), col("quant_tok_per_J")

fig = make_subplots(
    rows=3, cols=2,
    subplot_titles=("Aggregate throughput (tok/s)", "Per-request speed (tok/s)",
                    "Time to first token (ms, log)", "Energy efficiency (tokens/joule)",
                    "Combined power (W, GPU+CPU)", "Operating-point Pareto"),
    vertical_spacing=0.10, horizontal_spacing=0.09,
)

def line(x, y, name, color, group, show, row, col_, dash=None, hover="%{y}"):
    fig.add_trace(go.Scatter(
        x=x, y=y, name=name, mode="lines+markers",
        line=dict(color=color, width=2.5, dash=dash), marker=dict(size=8),
        legendgroup=group, showlegend=show, hovertemplate=hover + "<extra>" + name + "</extra>",
    ), row=row, col=col_)

# panels 1-5 over concurrency
panels = [
    (1, 1, b_agg, q_agg, "conc %{x} → %{y:.1f} tok/s"),
    (1, 2, b_gen, q_gen, "conc %{x} → %{y:.1f} tok/s/req"),
    (2, 1, b_ttft, q_ttft, "conc %{x} → %{y:.0f} ms"),
    (2, 2, b_tpj, q_tpj, "conc %{x} → %{y:.2f} tok/J"),
    (3, 1, b_pwr, q_pwr, "conc %{x} → %{y:.1f} W"),
]
for i, (r, c, bv, qv, hov) in enumerate(panels):
    first = (i == 0)
    line(CX, qv, "4-bit KV (8-wide)", QUANT, "q", first, r, c, hover=hov)
    line(CX, bv, "fp16 KV (4-wide)", BASE, "b", first, r, c, hover=hov)
fig.update_yaxes(type="log", row=2, col=1)

# panel 6: Pareto (per-user vs aggregate), hover shows concurrency
def pareto(gx, gy, name, color, group):
    fig.add_trace(go.Scatter(
        x=gx, y=gy, name=name, mode="lines+markers",
        line=dict(color=color, width=2), marker=dict(size=10),
        legendgroup=group, showlegend=False, customdata=C,
        hovertemplate="conc %{customdata}<br>%{x:.1f} tok/s/user<br>%{y:.1f} tok/s agg<extra>" + name + "</extra>",
    ), row=3, col=2)
pareto(q_gen, q_agg, "4-bit KV (8-wide)", QUANT, "q")
pareto(b_gen, b_agg, "fp16 KV (4-wide)", BASE, "b")
# highlight operating point (quant conc 8)
ix = C.index(8)
fig.add_trace(go.Scatter(
    x=[q_gen[ix]], y=[q_agg[ix]], mode="markers", name="operating point",
    marker=dict(size=22, color="rgba(0,0,0,0)", line=dict(color="#D55E00", width=3)),
    showlegend=False, hovertemplate="operating point<br>conc 8 · ~14 tok/s/user · 86 tok/s<extra></extra>",
), row=3, col=2)

fig.update_xaxes(title_text="Per-user speed (tok/s)", row=3, col=2)
fig.update_yaxes(title_text="Aggregate (tok/s)", row=3, col=2)
for r, c in [(1,1),(1,2),(2,1),(2,2),(3,1)]:
    fig.update_xaxes(title_text="Concurrency", row=r, col=c)

fig.update_layout(
    title=dict(text="<b>Qwen3.5-4B (4-bit) on a 16 GB Mac — vllm-mlx</b><br>"
                    "<sup>fp16 KV (4-wide) vs 4-bit KV (8-wide) · hover for exact values</sup>",
               x=0.5, xanchor="center"),
    template="plotly_white", height=1100, width=1100,
    legend=dict(orientation="h", y=1.06, x=0.5, xanchor="center"),
    font=dict(size=12), margin=dict(t=120),
)

fig.write_html(OUT, include_plotlyjs="cdn", full_html=True,
               config={"displaylogo": False})
print("wrote", OUT)
