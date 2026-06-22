#!/usr/bin/env python3
"""Designed interactive dashboard from data/final/*.csv -> assets/dashboard.html.

A styled single-page report (hero + KPI cards + sectioned narrative) with polished
Plotly charts embedded as responsive divs. Plotly loaded once via CDN.
Static PNG/SVG for README/social still come from scripts/plot.py (matplotlib).

Usage: python3 scripts/plot_interactive.py [final_dir=data/final] [out=assets/dashboard.html]
"""
import csv, os, sys
import plotly.graph_objects as go

FINAL = sys.argv[1] if len(sys.argv) > 1 else "data/final"
OUT   = sys.argv[2] if len(sys.argv) > 2 else "assets/dashboard.html"
os.makedirs(os.path.dirname(OUT) or ".", exist_ok=True)

BASE, QUANT, ACC, INK, MUT = "#9AA0A6", "#0B6FB8", "#E8590C", "#222", "#667"
FONT = "Inter, -apple-system, BlinkMacSystemFont, 'Segoe UI', system-ui, sans-serif"

def load(name):
    with open(os.path.join(FINAL, name)) as fh:
        return list(csv.DictReader(fh))

dc = load("decode_comparison.csv")
C  = [int(r["concurrency"]) for r in dc]
CX = [str(c) for c in C]
def col(k): return [float(r[k]) for r in dc]
b_agg, q_agg = col("base_agg_tps"), col("quant_agg_tps")
b_gen, q_gen = col("base_gen_tps"), col("quant_gen_tps")
b_ttft, q_ttft = col("base_ttft_ms"), col("quant_ttft_ms")
b_pwr, q_pwr = col("base_power_W"), col("quant_power_W")
b_tpj, q_tpj = col("base_tok_per_J"), col("quant_tok_per_J")

def theme(fig, height=350, ylog=False):
    fig.update_layout(
        height=height, autosize=True, font=dict(family=FONT, size=12.5, color=INK),
        margin=dict(l=58, r=22, t=18, b=42),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", y=1.13, x=0, font=dict(size=11.5),
                    bgcolor="rgba(0,0,0,0)"),
        hovermode="x unified",
        hoverlabel=dict(font_size=12.5, font_family=FONT, bgcolor="white"),
    )
    fig.update_xaxes(gridcolor="#EDEDF0", zeroline=False, linecolor="#DADADD")
    fig.update_yaxes(gridcolor="#EDEDF0", zeroline=False, linecolor="#DADADD",
                     type="log" if ylog else "linear")
    return fig

def trace(fig, x, y, name, color, hover):
    fig.add_trace(go.Scatter(x=x, y=y, name=name, mode="lines+markers",
        line=dict(color=color, width=3), marker=dict(size=9),
        hovertemplate=hover + "<extra>" + name + "</extra>"))

def div(fig, did):
    return fig.to_html(full_html=False, include_plotlyjs=False, div_id=did,
                       config={"responsive": True, "displayModeBar": False})

# ---- charts ----
def c_throughput():
    f = go.Figure()
    trace(f, CX, q_agg, "4-bit KV · 8-wide", QUANT, "%{y:.1f} tok/s")
    trace(f, CX, b_agg, "fp16 KV · 4-wide", BASE, "%{y:.1f} tok/s")
    f.add_annotation(x="8", y=q_agg[3], text="<b>+28%</b>", showarrow=True, arrowcolor=ACC,
                     arrowhead=2, ax=-2, ay=-34, font=dict(color=ACC, size=13))
    theme(f); f.update_xaxes(title_text="Concurrency"); f.update_yaxes(title_text="tok/s")
    return div(f, "c_thr")

def c_latency():
    f = go.Figure()
    f.add_bar(x=CX, y=b_ttft, name="fp16 KV · 4-wide", marker_color=BASE,
              hovertemplate="%{y:.0f} ms<extra>fp16</extra>")
    f.add_bar(x=CX, y=q_ttft, name="4-bit KV · 8-wide", marker_color=QUANT,
              hovertemplate="%{y:.0f} ms<extra>4-bit</extra>")
    f.update_layout(barmode="group")
    f.add_annotation(xref="paper", yref="paper", x=0.5, y=0.97, showarrow=False,
                     text="<b>16.5 s → 0.58 s</b> at 8 users", font=dict(color=ACC, size=13))
    theme(f, ylog=True); f.update_xaxes(title_text="Concurrency")
    f.update_yaxes(title_text="TTFT (ms, log)")
    f.update_layout(hovermode="x")
    return div(f, "c_lat")

def c_power():
    f = go.Figure()
    trace(f, CX, q_pwr, "4-bit KV", QUANT, "%{y:.1f} W")
    trace(f, CX, b_pwr, "fp16 KV", BASE, "%{y:.1f} W")
    theme(f); f.update_xaxes(title_text="Concurrency"); f.update_yaxes(title_text="Watts (GPU+CPU)")
    return div(f, "c_pwr")

def c_tpj():
    f = go.Figure()
    trace(f, CX, q_tpj, "4-bit KV", QUANT, "%{y:.2f} tok/J")
    trace(f, CX, b_tpj, "fp16 KV", BASE, "%{y:.2f} tok/J")
    f.add_hrect(y0=3.4, y1=5.1, fillcolor=ACC, opacity=0.07, line_width=0)
    f.add_annotation(xref="paper", x=0.5, y=4.9, showarrow=False, text="<b>flat ~4–5</b>",
                     font=dict(color=ACC, size=12.5))
    theme(f); f.update_xaxes(title_text="Concurrency")
    f.update_yaxes(title_text="tokens / joule", range=[0, 6])
    return div(f, "c_tpj")

def c_pareto():
    f = go.Figure()
    for gx, gy, nm, cl in ((q_gen, q_agg, "4-bit KV · 8-wide", QUANT),
                           (b_gen, b_agg, "fp16 KV · 4-wide", BASE)):
        f.add_trace(go.Scatter(x=gx, y=gy, name=nm, mode="lines+markers",
            line=dict(color=cl, width=2.5), marker=dict(size=11), customdata=C,
            hovertemplate="conc %{customdata}<br>%{x:.1f} tok/s/user<br>%{y:.1f} tok/s agg<extra>" + nm + "</extra>"))
    ix = C.index(8)
    f.add_trace(go.Scatter(x=[q_gen[ix]], y=[q_agg[ix]], mode="markers", showlegend=False,
        marker=dict(size=26, color="rgba(0,0,0,0)", line=dict(color=ACC, width=3)),
        hovertemplate="<b>operating point</b><br>conc 8 · ~14 tok/s/user · 86 tok/s<extra></extra>"))
    f.add_annotation(x=q_gen[ix], y=q_agg[ix], text="<b>operating point</b>", showarrow=True,
                     arrowcolor=ACC, ax=90, ay=-8, font=dict(color=ACC, size=12.5))
    f.add_vline(x=10, line=dict(color="#C7C7CC", dash="dash", width=1))
    theme(f, height=430); f.update_layout(hovermode="closest")
    f.update_xaxes(title_text="Per-user speed (tok/s)")
    f.update_yaxes(title_text="Aggregate throughput (tok/s)")
    return div(f, "c_par")

def c_prefill():
    pf = load("prefill_summary.csv")
    rows = [r for r in pf if int(r["concurrency"]) == 1]
    labs = [f"{r['prompt_set']}<br>{r['prompt_tokens']} tok" for r in rows]
    ttft = [float(r["ttft_ms"]) for r in rows]
    tps  = [float(r["prompt_tps"]) for r in rows]
    f1 = go.Figure([go.Bar(x=labs, y=ttft, marker_color=QUANT, text=[f"{v:.0f} ms" for v in ttft],
                           textposition="outside", hovertemplate="%{y:.0f} ms<extra></extra>")])
    theme(f1, height=320); f1.update_yaxes(title_text="TTFT (ms)", range=[0, max(ttft)*1.18])
    f2 = go.Figure([go.Bar(x=labs, y=tps, marker_color=BASE, text=[f"{v:.0f}" for v in tps],
                           textposition="outside", hovertemplate="%{y:.0f} tok/s<extra></extra>")])
    theme(f2, height=320); f2.update_yaxes(title_text="Prompt tok/s", range=[0, max(tps)*1.18])
    return div(f1, "c_pf1"), div(f2, "c_pf2")

thr, lat, pwr, tpj, par = c_throughput(), c_latency(), c_power(), c_tpj(), c_pareto()
pf1, pf2 = c_prefill()

# ---- KPI cards ----
KPIS = [
    ("35", "tok/s", "single-stream decode"),
    ("86", "tok/s", "peak aggregate · 8 users"),
    ("+28%", "", "throughput @ conc 8 (4-bit KV)"),
    ("4 → 8", "", "concurrent users (engine width)"),
    ("~14", "tok/s", "per active user"),
    ("~4–5", "tok/J", "energy efficiency (flat)"),
]
kpi_html = "".join(
    f'<div class="kpi"><div class="kpi-val">{v}<span>{u}</span></div>'
    f'<div class="kpi-lab">{l}</div></div>' for v, u, l in KPIS)

CSS = """
:root{
  --bg:#f6f7f9; --surface:#ffffff; --ink:#1b1d21; --muted:#6b7280;
  --base:#9AA0A6; --quant:#0B6FB8; --accent:#E8590C; --border:#e8eaed;
  --radius:16px; --shadow:0 1px 2px rgba(16,24,40,.05),0 10px 30px rgba(16,24,40,.05);
  --maxw:1180px;
}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);
  font-family:Inter,-apple-system,BlinkMacSystemFont,'Segoe UI',system-ui,sans-serif;
  line-height:1.5;-webkit-font-smoothing:antialiased}
a{color:var(--quant);text-decoration:none}a:hover{text-decoration:underline}
.wrap{max-width:var(--maxw);margin:0 auto;padding:0 22px}
header.hero{background:linear-gradient(135deg,#0b1f33 0%,#0B6FB8 100%);color:#fff;
  padding:54px 0 110px}
.hero h1{font-size:clamp(1.6rem,1.1rem+2.2vw,2.6rem);font-weight:800;margin:0 0 8px;letter-spacing:-.02em}
.hero p{margin:0;font-size:1.05rem;opacity:.92;max-width:60ch}
.badges{margin-top:18px;display:flex;flex-wrap:wrap;gap:8px}
.badge{background:rgba(255,255,255,.14);border:1px solid rgba(255,255,255,.22);
  padding:5px 11px;border-radius:999px;font-size:.82rem;font-weight:500}
.badge a{color:#fff;text-decoration:underline}
.kpis{display:grid;grid-template-columns:repeat(6,1fr);gap:14px;margin-top:-72px;margin-bottom:34px}
.kpi{background:var(--surface);border:1px solid var(--border);border-radius:var(--radius);
  padding:18px 16px;box-shadow:var(--shadow);transition:transform .15s ease}
.kpi:hover{transform:translateY(-3px)}
.kpi-val{font-size:1.7rem;font-weight:800;letter-spacing:-.02em;line-height:1.1}
.kpi-val span{font-size:.78rem;font-weight:600;color:var(--muted);margin-left:4px}
.kpi-lab{margin-top:6px;font-size:.78rem;color:var(--muted)}
section.block{margin:0 0 30px}
.block > h2{font-size:1.28rem;font-weight:750;letter-spacing:-.01em;margin:30px 0 4px;
  display:flex;align-items:center;gap:10px}
.block > h2::before{content:"";width:8px;height:22px;border-radius:3px;background:var(--quant)}
.takeaway{color:var(--muted);font-size:.96rem;margin:0 0 16px;max-width:80ch}
.takeaway b{color:var(--accent)}
.grid2{display:grid;grid-template-columns:1fr 1fr;gap:16px}
.card{background:var(--surface);border:1px solid var(--border);border-radius:var(--radius);
  box-shadow:var(--shadow);padding:14px 12px 6px}
.card h3{margin:2px 6px 0;font-size:.95rem;font-weight:650}
footer{background:#0b1f33;color:#cfd8e3;padding:30px 0;margin-top:20px;font-size:.9rem}
footer a{color:#9fd0f5}
.legend-key{display:flex;gap:18px;align-items:center;font-size:.85rem;color:var(--muted);margin:2px 6px 10px}
.dot{display:inline-block;width:11px;height:11px;border-radius:3px;margin-right:6px;vertical-align:-1px}
@media(max-width:860px){.kpis{grid-template-columns:repeat(3,1fr)}.grid2{grid-template-columns:1fr}}
@media(max-width:520px){.kpis{grid-template-columns:repeat(2,1fr)}}
"""

REPO = "https://github.com/rajatsadh24/vllm-mlx-qwen35-benchmark"
HTML = f"""<!doctype html><html lang="en"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>vllm-mlx · Qwen3.5-4B (4-bit) benchmark · 16 GB Mac</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap" rel="stylesheet">
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js" charset="utf-8"></script>
<style>{CSS}</style></head><body>
<header class="hero"><div class="wrap">
  <h1>Concurrent LLM serving on a 16&nbsp;GB Apple&nbsp;M5</h1>
  <p>A serving benchmark of <b>vLLM's MLX backend</b> — continuous batching, a 4-bit KV-cache
  optimization that doubles concurrent users (4 → 8), and energy in tokens/joule.</p>
  <div class="badges">
    <span class="badge">model: Qwen3.5-4B-OptiQ-4bit</span>
    <span class="badge">Apple M5 · 16 GB · MLX</span>
    <span class="badge"><a href="{REPO}">repo ↗</a></span>
    <span class="badge"><a href="{REPO}/blob/main/REPORT.md">full report ↗</a></span>
  </div>
</div></header>
<main class="wrap">
  <div class="kpis">{kpi_html}</div>
  <div class="legend-key"><span><span class="dot" style="background:{QUANT}"></span>4-bit KV · 8-wide (optimized)</span>
    <span><span class="dot" style="background:{BASE}"></span>fp16 KV · 4-wide (baseline)</span></div>

  <section class="block"><h2>The optimization</h2>
    <p class="takeaway">4-bit KV cache lets the engine decode <b>8 sequences in parallel instead of 4</b>:
    peak aggregate throughput rises and, crucially, the wider engine stops queueing —
    first-token latency at 8 users drops from <b>16.5 s to 0.58 s</b>.</p>
    <div class="grid2">
      <div class="card"><h3>Aggregate throughput</h3>{thr}</div>
      <div class="card"><h3>Time to first token</h3>{lat}</div>
    </div>
  </section>

  <section class="block"><h2>Energy efficiency</h2>
    <p class="takeaway">GPU power rises with concurrency in step with throughput, so
    <b>tokens/joule stays flat (~4–5)</b> across every config. Batch for throughput, not energy.</p>
    <div class="grid2">
      <div class="card"><h3>Combined power (GPU + CPU)</h3>{pwr}</div>
      <div class="card"><h3>Energy efficiency</h3>{tpj}</div>
    </div>
  </section>

  <section class="block"><h2>Choosing the operating point</h2>
    <p class="takeaway">Each curve runs concurrency 1 → 16 (aggregate ↑, per-user ↓). The sweet spot is
    <b>conc 8 on 4-bit KV</b>: ~86 tok/s aggregate with ~14 tok/s per user — above interactive reading speed.
    Hover any point for exact values.</p>
    <div class="card"><h3>Per-user speed vs aggregate throughput</h3>{par}</div>
  </section>

  <section class="block"><h2>Prefill cost by prompt size</h2>
    <p class="takeaway">Single-stream prefill runs ~1300 tok/s; a 2511-token prompt costs
    <b>~1.9 s to first token</b>. Throughput amortizes with prompt length.</p>
    <div class="grid2">
      <div class="card"><h3>Prefill latency (TTFT)</h3>{pf1}</div>
      <div class="card"><h3>Prefill throughput</h3>{pf2}</div>
    </div>
  </section>
</main>
<footer><div class="wrap">
  Method: continuous batching · fresh server per sweep point (OOM-resilient) · powermetrics for energy ·
  3 reps · thinking disabled. Power summarized by mean (GPU power is bimodal).
  <br>Generated from <code>data/final/*.csv</code> · <a href="{REPO}">github.com/rajatsadh24/vllm-mlx-qwen35-benchmark</a>
</div></footer>
</body></html>"""

with open(OUT, "w") as fh:
    fh.write(HTML)
print("wrote", OUT, f"({len(HTML)//1024} KB)")
