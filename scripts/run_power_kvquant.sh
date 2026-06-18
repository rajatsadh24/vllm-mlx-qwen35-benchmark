#!/usr/bin/env bash
# Decode sweep x2 (baseline fp16 KV vs 4-bit KV), with a powermetrics sidecar
# per point so we can report tokens/joule. Fresh server per point (OOM-safe).
# Run as a normal user (NOT sudo); powermetrics is invoked with sudo internally.
set -uo pipefail

MODEL_DIR="./models/qwen35-optiq"
SERVED="qwen35-optiq"
URL="http://localhost:8000"
PORT=8000
REPS=3
# Realistic ladder for a 16 GB Mac. 32/64 are unfittable for a 4B model here —
# they only measure queueing/OOM, so we cap at 16 (= the optimized batch width).
CONCS=(1 2 4 8 16)

# Per-arm engine width. 4-bit KV is ~4x smaller than fp16, so the quant arm can
# decode twice as many sequences in parallel within the same 16 GB.
MAXSEQS=4          # set per arm below
CACHEBLOCKS=256    # set per arm below

echo "==> priming sudo for powermetrics (one prompt)"; sudo -v || { echo "sudo required"; exit 1; }

start_server() {  # extra serve args passed through; uses $MAXSEQS / $CACHEBLOCKS
  caffeinate -dimsu vllm-mlx serve "$MODEL_DIR" --served-model-name "$SERVED" \
    --default-chat-template-kwargs '{"enable_thinking": false}' \
    --continuous-batching --use-paged-cache \
    --gpu-memory-utilization 0.70 --max-cache-blocks "$CACHEBLOCKS" --paged-cache-block-size 64 \
    --max-num-seqs "$MAXSEQS" --chunked-prefill-tokens 512 --disable-prefix-cache \
    --max-tokens 2048 --max-request-tokens 2048 --enable-metrics --port "$PORT" \
    "$@" > server.log 2>&1 &
  SERVER_PID=$!
  for i in $(seq 1 120); do
    kill -0 "$SERVER_PID" 2>/dev/null || { echo "  !! server died"; tail -5 server.log; return 1; }
    curl -sf "$URL/health" >/dev/null 2>&1 && return 0
    sleep 1
  done
  return 1
}
stop_server() { kill "$SERVER_PID" 2>/dev/null || true; wait "$SERVER_PID" 2>/dev/null || true; sleep 2; }
trap 'stop_server; sudo pkill -x powermetrics 2>/dev/null || true' EXIT

# $1=csv out  $2=power out  $3=concurrency ; rest = extra serve args
run_point() {
  local csv="$1" pwr="$2" conc="$3"; shift 3
  echo "  conc=$conc -> $csv"
  start_server "$@" || { echo "    server failed; skipping"; return; }
  sudo powermetrics --samplers cpu_power,gpu_power -i 500 > "$pwr" 2>/dev/null &
  sleep 1
  vllm-mlx bench-serve --url "$URL" --model "$SERVED" \
    --prompts short --concurrency "$conc" --max-tokens 1024 \
    --enable-thinking false --validate false \
    --repetitions "$REPS" --format csv --output "$csv" || true
  sudo pkill -INT -x powermetrics 2>/dev/null || true; sleep 1
  stop_server
}

echo "==> BASELINE decode sweep (fp16 KV, max-num-seqs=4)"
MAXSEQS=4; CACHEBLOCKS=256
for C in "${CONCS[@]}"; do run_point "dpow_base_c${C}.csv" "dpow_base_c${C}.txt" "$C"; done

echo "==> KV-QUANT decode sweep (4-bit KV, max-num-seqs=8 — the optimization)"
MAXSEQS=8; CACHEBLOCKS=512
for C in "${CONCS[@]}"; do
  run_point "dpow_quant_c${C}.csv" "dpow_quant_c${C}.txt" "$C" \
    --kv-cache-quantization --kv-cache-quantization-bits 4
done

echo "==> computing tokens/joule"
python3 power_summary.py
