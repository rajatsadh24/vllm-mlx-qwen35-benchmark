# vLLM-on-MLX Serving Benchmark — Qwen3.5-4B-OptiQ-4bit on a 16 GB Apple M5

**Date:** 2026-06-18
**Model:** `mlx-community/Qwen3.5-4B-OptiQ-4bit` (served as `qwen35-optiq`) — a 4-bit, reasoning ("thinking") LLM
**Server:** `vllm-mlx serve` (continuous batching + paged KV cache), local patch applied (see §3.2 / Appendix B)
**Hardware:** Apple **M5** (10-core GPU, Metal 4), **16 GB** unified memory, macOS 26.5.1 (arm64) — bench fingerprint logged the chip as `Unknown` (tooling predates M5)
**Harness:** `vllm-mlx bench-serve`, fresh server per sweep point, `powermetrics` sidecar for energy

---

## 1. Executive Summary

We benchmarked a 4-bit 4B reasoning model under `vllm-mlx` on a memory-constrained 16 GB Mac, separating the two cost centers of LLM inference (**prefill** and **decode**), and compared a baseline against a **4-bit KV-cache** optimization while measuring **energy (tokens/joule)**.

| Question | Answer |
|----------|--------|
| Single-stream decode speed | **~35 tok/s** (clean, reproducible) |
| Aggregate decode ceiling, baseline (fp16 KV, 4-wide) | **~78 tok/s**, saturating ~conc 16 |
| Aggregate decode ceiling, optimized (4-bit KV, 8-wide) | **~86 tok/s @ conc 8** (+28% vs baseline at conc 8) |
| Real concurrent-decode capacity | **4 → 8 streams** after optimization, no OOM through conc 16 |
| Prefill rate (single stream, long prompt) | **~1300 tok/s**; TTFT 1.9 s for a 2511-token prompt |
| Energy efficiency | **~3.5–5.0 tok/J across all configs — essentially flat.** Batching buys throughput, not efficiency, because GPU power scales with concurrency. |
| The binding constraint | **Memory, not compute.** Compute sat idle while the box OOM'd. |

**Headline:** On this 16 GB Mac the model serves **~8 concurrent users at ~14 tok/s each (≈86 tok/s aggregate)** once 4-bit KV cache lifts the engine from 4-wide to 8-wide. The optimization is a clear throughput/capacity win at no single-stream cost and roughly neutral energy efficiency. Two non-obvious bugs had to be fixed first to get *any* clean data: a reasoning-model/validation interaction, and a hardcoded 32 GB MLX cache limit that OOM'd the box.

---

## 2. Environment

| Item | Value |
|------|-------|
| Chip / memory | Apple M5 (10-core GPU, Metal 4), 16 GB unified |
| OS | macOS 26.5.1 (Darwin arm64) |
| Model | `mlx-community/Qwen3.5-4B-OptiQ-4bit`, 4-bit weights, reasoning model |
| Model quirks | Detected as a VLM (`vision_config` + `text_config`) → loaded `strict=False`; ships MTP (speculative-decode) weights that were stripped/disabled — no effect on correctness |
| Server | `vllm-mlx serve`, `--continuous-batching --use-paged-cache`, metrics on |
| Power | `powermetrics --samplers cpu_power,gpu_power -i 500` (GPU+CPU on-die sensors) |

Test discipline: **AC power, Low Power Mode OFF, `caffeinate` to prevent sleep, run from a low-memory state (~4–5 GB used) with other apps closed.** Starting memory state matters a lot on 16 GB (see §3.3).

---

## 3. The Investigation (what had to be fixed before the data was trustworthy)

### 3.1 Why every benchmark first reported `FAIL`

The initial sweeps showed healthy TTFT/TPS yet `FAIL` on every row. Root cause was **not** performance:

- `bench-serve` validates each response; `validate_response()` marks `finish_reason == "length"` as **"Truncated" → FAIL** (independent of timing). A config is `PASS` only if every response validated.
- Every request hit the `--max-tokens` cap, so every `finish_reason` was `length`.
- **Why it never stopped naturally:** the model is a *reasoning* model. EOS works fine (a trivial prompt stops at 121 tokens), but with thinking enabled it emits long `<think>` chains and can enter **overthinking loops** — e.g. "explain the three laws of thermodynamics briefly" spent 2048 tokens arguing with itself about whether there are three or four laws and never produced an answer.
- **Fix:** disable thinking (`--default-chat-template-kwargs '{"enable_thinking": false}'`) → the same prompt finishes cleanly with `finish_reason=stop` at ~436 tokens. For raw throughput sweeps we also pass `--validate false` (truncation at the token cap is intentional in those regimes). A reasoning parser (`--reasoning-parser`) or a thinking-token budget (`--default-thinking-token-budget`) are the production-grade alternatives.

**Insight:** for reasoning models, "FAIL" in a serving benchmark usually means *the model didn't finish within max_tokens*, not that it's slow — and an unconfigured reasoning model can burn the entire token budget thinking.

### 3.2 The Metal OOM — root cause and fix

Once running, the server repeatedly **aborted** (`Abort trap: 6`) with:
```
[METAL] Command buffer execution failed: Insufficient Memory (kIOGPUCommandBufferCallbackErrorOutOfMemory)
```
The crash landed at a roughly **constant cumulative request count**, regardless of batch size — i.e. memory was *climbing per request*, not spiking on one big allocation. The cause:

- `vllm-mlx` hardcodes **`mx.set_cache_limit(32 GB)`**. MLX retains freed buffers in a cache up to that limit; on unified memory 32 GB ≫ 16 GB physical, so MLX never releases — resident memory grows until the OS is exhausted.
- `--gpu-memory-utilization` only sets `mx.set_memory_limit()`, a **soft** limit (MLX exceeds it rather than failing). It does **not** bound the buffer cache. So lowering it did not fix the OOM.
- The original `--max-cache-blocks 1000 × 64 = 64k tokens` KV pool was an independent ~9 GB bomb on top of ~4 GB of weights.

**Fix (Appendix B / `scripts/PATCH_cache_limit.md`):** patch the cache limit to **2 GB** at both call sites in `engine/batched.py`. Combined with bounding the KV pool (`--max-cache-blocks`) and the engine width (`--max-num-seqs`), this made the run survive.

### 3.3 Memory tuning for 16 GB

The levers, in order of impact on physical memory:

| Lever | Effect | Setting used |
|-------|--------|--------------|
| `set_cache_limit` (patched) | **hard** cap on MLX buffer retention — the real OOM fix | 2 GB |
| `--max-cache-blocks` | **hard** KV-pool size (`blocks × 64` tokens) | 256 (baseline) / 512 (quant) |
| `--max-num-seqs` | active batch width → peak activation memory | 4 (baseline) / 8 (quant) |
| `--gpu-memory-utilization` | soft alloc limit + emergency-cache-clear threshold | 0.70 |
| `--chunked-prefill-tokens` | bounds prefill memory spikes on long prompts | 512 |
| `--disable-prefix-cache` | removes an accumulating allocation | on |

Also critical and easy to miss: **system memory state.** Runs started at 6.6 GB used (apps open) OOM'd where runs from 4.1 GB used survived. A 2.5 GB swing in OS footprint is the difference between success and crash on 16 GB. Reproducible runs require a clean low-memory start.

---

## 4. Methodology

**Two regimes** isolate the two cost centers (`bench-serve` records both, but a blended sweep muddies them):
- **Prefill-bound:** long prompts, `--max-tokens 1`. TTFT and `prompt_tps` ≈ prompt-processing cost.
- **Decode-bound:** short prompt, `--max-tokens 1024`, thinking off. `gen_tps`/`tpot` ≈ pure generation; `throughput_tps` = aggregate.

**Two arms** (decode):
- **Baseline** — fp16 KV, `--max-num-seqs 4`.
- **Optimized** — 4-bit KV (`--kv-cache-quantization --kv-cache-quantization-bits 4`), `--max-num-seqs 8`. 4-bit KV is ~4× smaller than fp16, which is what frees the memory to double the engine width.

**Resilience:** a **fresh server per sweep point**, so a Metal OOM at the top of the ladder costs only that point, not the whole run. Per-point CSVs + power logs, merged afterward.

**Concurrency ladder:** 1, 2, 4, 8, 16. We deliberately **dropped 32/64** — on a 16 GB Mac a 4B model cannot fit that many parallel sequences; those points only measure admission queueing and crashes (proven: baseline OOM'd before 64).

**Energy:** `tok/J = throughput_tps / avg_power_W` (units cancel — tokens/s ÷ joules/s = tokens/joule), where `avg_power_W` is the **mean** of GPU+CPU on-die power over the bench window (mean chosen over median; see §7).

**Reps:** 3 per point (sufficient — see §7 reproducibility).

---

## 5. Results

### 5.1 Prefill — TTFT & prompt processing
(single-stream is the clean measurement; conc > 4 is queue-dominated by the 4-wide cap)

| Prompt | Tokens | TTFT @ conc 1 | prompt_tps @ conc 1 |
|--------|-------:|--------------:|--------------------:|
| short  | 26     | 135 ms        | 192 |
| medium | 599    | 508 ms        | 1179 |
| long   | 2511   | 1916 ms       | 1311 |

- Prefill amortizes fixed overhead, so longer prompts hit a higher tok/s (**~1300 tok/s** on the 2511-token prompt).
- A 2511-token prompt costs **~1.9 s to first token** single-stream — the relevant number for long-context latency SLAs.
- **Caveat:** above conc 4, TTFT is dominated by queue wait (e.g. `long conc 64` shows 83 s — almost entirely waiting behind a 4-wide engine), *not* prefill compute. Read the prefill curve as offered-load latency, not prefill scaling. Full table: `data/final/prefill_summary.csv`.

### 5.2 & 5.3 Decode — baseline vs 4-bit KV optimization
(mean-based power; full data `data/final/decode_comparison.csv`)

| conc | base gen_tps | base aggTPS | base W | base tok/J | quant gen_tps | quant aggTPS | quant W | quant tok/J |
|-----:|-------------:|------------:|-------:|-----------:|--------------:|-------------:|--------:|------------:|
| 1  | 34.6 | 34.2 | 7.3  | 4.70 | 35.3 | 34.8 | 7.0  | 5.01 |
| 2  | 29.9 | 44.7 | 12.7 | 3.52 | 30.1 | 41.4 | 11.6 | 3.55 |
| 4  | 21.6 | 65.1 | 17.5 | 3.71 | 24.6 | 71.4 | 19.6 | 3.64 |
| 8  | 21.6 | 66.9 | 14.9 | 4.49 | **14.5** | **85.7** | 19.3 | 4.45 |
| 16 | 20.7 | 78.5 | 16.3 | 4.81 | 12.9 | 81.2 | 16.5 | 4.92 |

**The optimization works:**
- **Real parallel capacity doubled, 4 → 8 streams**, sustained through conc 16 with **zero OOM**.
- **Peak aggregate throughput +9%** (78.5 → 85.7) and reached *sooner* (conc 8 vs conc 16).
- **+28% aggregate at conc 8** (66.9 → 85.7) — the regime where baseline is queueing 4 extra requests and quant is running all 8.
- **No single-stream penalty** (34.6 → 35.3 tok/s at conc 1).
- The lower quant `gen_tps` at conc 8/16 (14.5 vs 21.6) is **not** a regression — it is the batching trade-off: quant runs 8 truly-parallel streams (each sharing bandwidth), while baseline runs only 4 and *queues* the rest. More total work, lower per-stream rate. **Compare aggregate across arms; compare per-request only within an arm.**

### 5.4 Power & efficiency

- **GPU power scales with concurrency:** ~5 W single-stream (decode is memory-bandwidth-bound, GPU barely working) → ~16–20 W batched. CPU power is small (~1–3 W) and falls as the GPU dominates.
- **tok/J is essentially flat at ~3.5–5.0 across every config and both arms.** Because power rises roughly in step with throughput, **batching buys throughput, not energy efficiency.** Single-stream is among the most efficient per token (5 tok/J at 7 W); the slightly lower mid-ladder efficiency reflects the quant arm drawing more power to run 8 streams.
- Idle/baseline GPU draw is ~5 W even at rest (display/WindowServer), so the tok/J denominator has a real floor — the efficiency numbers are honest total-system figures, not compute-only.

Full per-point power statistics (mean/median/p90/idle%): `data/final/power_stats.csv`.

---

## 6. Capacity & Recommended Operating Point

**Defensible capacity claim:** *"On a 16 GB Apple Silicon Mac, this model serves **8 concurrent users at ~14 tok/s each (≈86 tok/s aggregate)** using 4-bit KV cache and an 8-wide engine, within ~16–20 W."*

- This is **measured**, with the engine actually running 8 sequences in parallel (not queued).
- ~14 tok/s/user is comfortably above reading speed (~5–7 tok/s), so the interactive experience holds.
- **Do not** claim "supports 4 users" from the baseline — the 4 was a memory-safety cap we chose, not a hardware limit. The honest framing is engine-width × per-user rate at a stated SLA.

**Recommended operating point:** **4-bit KV, `--max-num-seqs 8`, operate around conc 8** — peak aggregate throughput, full utilization, healthy per-user rate, comfortable memory headroom (16.5 W, no OOM at conc 16).

---

## 7. Measurement Quality & Threats to Validity

*(statistical pass over the raw per-rep data and power logs)*

**Reproducibility — strong.** Across 3 reps, **28 of 30 config cells have CoV < 5%**:
- Noisy: `quant c16 ttft_ms` (CoV 15.3%, ±4.3 s on a 28 s mean) and `base c8 throughput_tps` (CoV 11.5%). Both are in the queue-saturated regime; widen intervals or add reps there.
- Everything at conc 1/2/4/16 (both arms) reproduces tightly.

**Power logs — well-gated.** Idle fraction (GPU < 100 mW) is ≤ 2.4% everywhere, so idle/startup dilution is **not** a problem; mean ≈ median within ±10% for 8 of 10 points.

**The conc=2 power "anomaly" — explained.** GPU power is **bimodal**: it alternates between a ~7–8 W inter-batch phase and a ~13–14 W compute phase (sustained blocks, not transient spikes). At conc 2 the two arms' *medians* land in different clusters, which falsely implied a ~5 W gap (13.1 vs 7.5 W). The **means** (10.9 vs 9.9 W) show the true gap is ~1 W. **This is why the report uses mean-based power**, and why the conc=2 tok/J values are ~equal (3.52 vs 3.55) rather than divergent.

**Honest limitations:**
- `--max-num-seqs` caps real concurrency; high-concurrency prefill/TTFT rows reflect **admission queueing**, not model latency.
- `gen_tps` is not apples-to-apples across arms above conc 4 (4 active+queue vs 8 active).
- **4-bit KV quality — validated (greedy):** a fixed 10-prompt A/B (greedy decode, thinking off, incl. long-context code review + log analysis) produced **10/10 byte-identical outputs** vs fp16 KV (mean similarity 1.000; same token counts/finish reasons). For this model the 4-bit affine KV quant is effectively lossless for deterministic decoding — so the capacity win is free. Scope: one model, greedy only; `data/final/quality_ab.csv`, `assets/quality_ab.md`, `scripts/quality_ab.py`.
- Single machine, 3 reps, thinking disabled, short decode prompt. Power windows include the warmup round (slight efficiency under-estimate).
- TTFT in saturated regimes conflates queue-wait with first-token compute; a future run should separate them.

**To tighten a future run:** 5 reps on saturated cells; gate power capture to the steady-state decode window; fixed-duration power windows so low-concurrency points get comparable sample mass; always report power as mean+median+p90.

---

## 8. Recommendations & Next Steps

1. **Ship the 4-bit KV + 8-wide config** as the serving default on 16 GB; operate near conc 8.
2. **Stretch test `--max-num-seqs 12`** with 4-bit KV — conc 16 finished with headroom (16.5 W), so the true ceiling is likely > 8.
3. ~~Quality A/B (fp16 vs 4-bit KV)~~ **Done** — 10/10 byte-identical greedy outputs (§7). Optional follow-up: repeat at temperature > 0 with an LLM-judge for sampled-decoding parity.
4. **Reasoning in production:** if thinking is wanted, set `--reasoning-parser` and/or `--default-thinking-token-budget` rather than leaving it unbounded (avoids the overthinking-loop truncations of §3.1).
5. **Upstream the cache-limit fix** as a CLI flag scaled to `--gpu-memory-utilization`; the 32 GB hardcode is a latent OOM on any ≤32 GB Mac.
6. **More RAM removes both ceilings** (the OOM and the `max-num-seqs` cap) — a 32/64 GB Mac would let you measure true concurrency scaling past 16-way.

---

## Appendix A — Configuration & Reproducibility Reference (source-backed)

*Verified against the installed `vllm-mlx` source.*

### Serve flags used (with upstream defaults)

| Flag | Default | What it does |
|------|---------|--------------|
| `--gpu-memory-utilization` | 0.90 | Fraction of `max_recommended_working_set_size` → `mx.set_memory_limit()` (soft) + emergency-cache-clear threshold. |
| `--max-cache-blocks` | 1000 | Max paged KV blocks (× block size = pooled tokens). **Hard** memory bound. |
| `--paged-cache-block-size` | 64 | Tokens per KV block. |
| `--max-num-seqs` | 256 | Max concurrent sequences (engine batch width); excess offered load queues. |
| `--chunked-prefill-tokens` | 0 (off) | Max prefill tokens per scheduler step; bounds long-prefill memory/starvation. |
| `--disable-prefix-cache` | off (prefix cache on by default) | Disables prompt-prefix reuse. |
| `--kv-cache-quantization` | off | Master switch for KV-cache quantization. |
| `--kv-cache-quantization-bits` | 8 (choices 4/8) | KV quant bit width; 4-bit ≈ ¼ the fp16 KV memory. |
| `--kv-cache-min-quantize-tokens` | 256 | First 256 tokens of a sequence stay unquantized; quant applies beyond. |
| `--default-chat-template-kwargs` | none | JSON kwargs applied when a request omits them (e.g. `{"enable_thinking": false}`). |
| `--continuous-batching` | off | Selects the batched engine (multi-user). |
| `--use-paged-cache` | off | Enables the paged KV cache. |
| `--max-tokens` / `--max-request-tokens` | 32768 / 32768 | Default generation cap / max client-requested cap. |
| `--enable-metrics` | off | Prometheus `/metrics`. |

### Metal memory model
`soft_limit = max_recommended_working_set_size × gpu_memory_utilization`, applied via `mx.set_memory_limit(soft_limit)` (a **soft** limit — MLX frees cache to stay under it, but will exceed rather than fail). Separately, `mx.set_cache_limit(...)` bounds MLX's retained buffer cache — **patched from 32 GB → 2 GB** (Appendix B).

### bench-serve validation & metrics
- `validate_response()`: `status_code≥400` → FAIL; `finish_reason is None` → FAIL; **`finish_reason=="length"` → FAIL "Truncated"**; empty content+no tool calls → FAIL. `--validate false` skips this (results default `validated=True`).
- `ttft_ms = (t_first_token − t_start)·1000`. `tpot_ms` = mean inter-token gap.
- `gen_tps = completion_tokens / (t_last_token − t_first_token)` (excludes HTTP teardown).
- `prompt_tps = prompt_tokens / (t_first_token − t_start)` (i.e. over the TTFT window — why it looks "slow" under queueing).
- `throughput_tps = Σ completion_tokens / max(e2e_latency)` (slowest request defines the window); `requests_per_s = conc / max(e2e_latency)`.

### Exact commands
See `scripts/run_bench.sh` (prefill + decode envelope) and `scripts/run_power_kvquant.sh` (the two-arm decode + power sweep). Re-derive final CSVs with `scripts/consolidate.py <raw_dir> <final_dir>`.

---

## Appendix B — The cache-limit patch
See `scripts/PATCH_cache_limit.md`. Summary: `engine/batched.py`, two sites, `mx.set_cache_limit(32 GB → 2 GB)`. Lost on package upgrade — re-apply.

---

## Appendix C — Data index

- `data/final/prefill_summary.csv` — TTFT & prompt_tps by prompt_set × concurrency (mean of reps).
- `data/final/decode_comparison.csv` — baseline vs 4-bit KV: gen_tps, aggTPS, power_W, tok/J, status, per concurrency.
- `data/final/power_stats.csv` — per-point GPU/CPU power (mean/median/p90), idle%, sample count.
- `data/raw/` — all per-point CSVs (`prefill_c*`, `decode_c*`, `dpow_base_c*`, `dpow_quant_c*`), `powermetrics` logs (`dpow_*.txt`), merged `prefill.csv`/`decode.csv`, and `server.log`.
- `data/raw/archive/` — `sweep*.csv`, the original pre-fix runs (mostly `FAIL`) kept for provenance.

**Authoritative datasets:** prefill → `prefill.csv` (reps 5, conc 1–64; conc>4 queue-dominated). Decode → the `dpow_*` power-sweep files (reps 3, conc 1–16, with energy). The first-sweep `decode.csv` is superseded (its conc 64 row is an OOM artifact).
