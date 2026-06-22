#!/usr/bin/env bash
# 4-bit KV cache quality A/B: same prompts, greedy decode, fp16 KV vs 4-bit KV.
# No sudo needed. Run from repo root.
set -uo pipefail
MODEL_DIR="/Users/user/Desktop/vllm/models/qwen35-optiq"; SERVED="qwen35-optiq"
URL="http://localhost:8000"; PORT=8000
PY=/Users/user/.venv-vllm-metal/bin/python
VLLM=/Users/user/.venv-vllm-metal/bin/vllm-mlx

start() {  # extra serve args
  caffeinate -dimsu "$VLLM" serve "$MODEL_DIR" --served-model-name "$SERVED" \
    --default-chat-template-kwargs '{"enable_thinking": false}' \
    --continuous-batching --use-paged-cache --gpu-memory-utilization 0.70 \
    --max-cache-blocks 256 --max-num-seqs 2 --disable-prefix-cache \
    --max-tokens 1024 --max-request-tokens 1024 --port "$PORT" "$@" > qual_server.log 2>&1 &
  SPID=$!
  for i in $(seq 1 120); do
    kill -0 "$SPID" 2>/dev/null || { echo "server died"; tail -5 qual_server.log; return 1; }
    curl -sf "$URL/health" >/dev/null 2>&1 && { echo "  server up (${i}s)"; return 0; }
    sleep 1
  done; return 1
}
stop() { kill "$SPID" 2>/dev/null || true; wait "$SPID" 2>/dev/null || true; sleep 2; }
trap 'stop' EXIT

echo "==> fp16 KV (baseline) collect"
start || exit 1
"$PY" scripts/quality_ab.py collect --url "$URL" --model "$SERVED" --out qual_fp16.json
stop

echo "==> 4-bit KV collect"
start --kv-cache-quantization --kv-cache-quantization-bits 4 || exit 1
"$PY" scripts/quality_ab.py collect --url "$URL" --model "$SERVED" --out qual_quant.json
stop

echo "==> compare"
"$PY" scripts/quality_ab.py compare qual_fp16.json qual_quant.json \
  --csv data/final/quality_ab.csv --md assets/quality_ab.md
echo "==> DONE"
