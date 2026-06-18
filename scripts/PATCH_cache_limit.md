# Required local patch: MLX buffer-cache limit (32 GB → 2 GB)

**File:** `<venv>/lib/python3.12/site-packages/vllm_mlx/engine/batched.py`
**Sites:** two identical blocks — the MLLM path (~line 312) and the LLM path (~line 512).

`vllm-mlx` hardcodes the MLX buffer-cache limit to **32 GB**. On Apple Silicon the GPU
shares unified memory with the OS, so MLX retaining up to 32 GB of *freed-but-cached*
buffers means resident memory climbs monotonically across requests until a 16 GB machine
exhausts physical RAM and the process aborts with a hard Metal OOM
(`[METAL] Command buffer execution failed: Insufficient Memory ... kIOGPUCommandBufferCallbackErrorOutOfMemory`).
`mx.set_memory_limit()` (the `--gpu-memory-utilization` soft limit) does **not** prevent
this — the cache limit is a separate knob.

## Change (apply at BOTH sites)

```diff
-                    mx.set_cache_limit(32 * 1024 * 1024 * 1024)  # 32GB
+                    mx.set_cache_limit(2 * 1024 * 1024 * 1024)  # 2GB small buffer cache (was hardcoded 32GB; OOMs small-RAM Macs)
```

And the corresponding log string:

```diff
-                        f"cache_limit=32GB"
+                        f"cache_limit=2.0GB"
```

## Verify

```bash
grep -n "set_cache_limit\|cache_limit=" \
  "$(python3 -c 'import vllm_mlx, os; print(os.path.dirname(vllm_mlx.__file__))')/engine/batched.py"
# expect: 2 GB at both sites, no remaining 32 GB literal
```

> This patch lives in site-packages and will be lost on `pip install --upgrade vllm-mlx`.
> Re-apply after any reinstall. A cleaner upstream fix would expose the cache limit as a
> CLI flag (e.g. `--metal-cache-limit-gb`) scaled to `--gpu-memory-utilization`.
