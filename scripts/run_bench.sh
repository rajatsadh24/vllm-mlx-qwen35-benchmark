#!/usr/bin/env bash
# End-to-end vLLM-MLX benchmark for a 16 GB Apple Silicon Mac.
# Resilient design: a FRESH server per sweep point, so a Metal OOM at high
# concurrency only loses that point, not the whole run. Run on AC, no sudo,
# Low Power Mode OFF, with as little else open as possible.
set -uo pipefail

MODEL_DIR="./models/qwen35-optiq"
SERVED="qwen35-optiq"
URL="http://localhost:8000"
PORT=8000
CONCS=(1 2 4 8 16 32 64)
REPS=3

mem_used_gb() {
  vm_stat | awk '/page size/{ps=$8} /Pages active/{a=$3} /Pages wired/{w=$4} \
    /Pages occupied by compressor/{c=$5} END{printf "%.1f",(a+w+c)*ps/1073741824}'
}

start_server() {
  caffeinate -dimsu vllm-mlx serve "$MODEL_DIR" --served-model-name "$SERVED" \
    --default-chat-template-kwargs '{"enable_thinking": false}' \
    --continuous-batching --use-paged-cache \
    --gpu-memory-utilization 0.70 \
    --max-cache-blocks 256 --paged-cache-block-size 64 \
    --max-num-seqs 4 --chunked-prefill-tokens 512 \
    --disable-prefix-cache \
    --max-tokens 2048 --max-request-tokens 2048 \
    --enable-metrics --port "$PORT" > server.log 2>&1 &
  SERVER_PID=$!
  for i in $(seq 1 120); do
    kill -0 "$SERVER_PID" 2>/dev/null || { echo "  !! server died on startup"; tail -5 server.log; return 1; }
    curl -sf "$URL/health" >/dev/null 2>&1 && { echo "  server up (${i}s)"; return 0; }
    sleep 1
  done
  echo "  !! server not healthy in 120s"; return 1
}

stop_server() { kill "$SERVER_PID" 2>/dev/null || true; wait "$SERVER_PID" 2>/dev/null || true; sleep 2; }
trap 'stop_server' EXIT

# one bench-serve call against a fresh server; $1=outfile, rest=bench args
run_point() {
  local out="$1"; shift
  echo "  -> $out (mem $(mem_used_gb) GB used)"
  start_server || { echo "  skipped $out (server failed)"; return; }
  vllm-mlx bench-serve --url "$URL" --model "$SERVED" \
    --validate false --repetitions "$REPS" --format csv --output "$out" "$@" || true
  stop_server
}

# Pre-flight: 16 GB is tight; bail if the box is loaded.
U=$(mem_used_gb)
echo "==> system memory in use: ${U} GB"
if awk "BEGIN{exit !($U > 5.5)}"; then
  echo "!! ${U} GB used is too high for a clean 16 GB run. Close apps / reboot, then retry."
  exit 1
fi

echo "==> PREFILL sweep (max-tokens 1)"
for C in "${CONCS[@]}"; do
  echo "[prefill conc=$C]"
  run_point "prefill_c${C}.csv" --prompts short,medium,long --concurrency "$C" --max-tokens 1
done

echo "==> DECODE sweep (max-tokens 1024)"
for C in "${CONCS[@]}"; do
  echo "[decode conc=$C]"
  run_point "decode_c${C}.csv" --prompts short --concurrency "$C" --max-tokens 1024 --enable-thinking false
done

# Merge per-point CSVs (header from the first that exists)
merge() {  # $1=glob prefix, $2=output
  local first=1
  : > "$2"
  for f in "$1"*.csv; do
    [ -f "$f" ] || continue
    if [ "$first" = 1 ]; then cat "$f" >> "$2"; first=0; else tail -n +2 "$f" >> "$2"; fi
  done
}
merge "prefill_c" prefill.csv
merge "decode_c"  decode.csv
echo "==> DONE. Merged: prefill.csv, decode.csv (per-point: prefill_c*.csv, decode_c*.csv)"
