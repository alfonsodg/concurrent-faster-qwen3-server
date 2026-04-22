# qwen3-tts-server — Development Status

High-performance Rust TTS server wrapping `qwen3-tts-rs` (Qwen3-TTS-12Hz-0.6B-Base)
with batched inference, streaming, and voice cloning for call center workloads.

**Current version:** v0.7.4

## Architecture

```
                    ┌──────────────────────────────────────────────┐
                    │              axum HTTP server                │
                    │  POST /v1/audio/speech  GET /health /metrics │
                    │  POST /v1/embeddings/preload                 │
                    └──────┬──────────────────┬────────────────────┘
                           │                  │
                    ┌──────▼──────┐    ┌──────▼──────────────┐
                    │ BatchEngine │    │  Streaming Worker    │
                    │  (thread)   │    │  (thread)           │
                    │ collect N   │    │  batched streaming   │
                    │ reqs → batch│    │  + forward threads   │
                    │ forward     │    │  + AtomicBool stop   │
                    └──────┬──────┘    └──────┬──────────────┘
                           │                  │
                    ┌──────▼──────────────────▼───────┐
                    │   Arc<Qwen3TTS> shared model    │
                    │   (single load, ~2.7GB VRAM)    │
                    └─────────────────────────────────┘
```

### Crate layout

| Path | Description |
|------|-------------|
| `server/src/main.rs` | Axum server, routes, streaming worker with AtomicBool stop signals |
| `server/src/batch.rs` | `BatchEngine` — batched forward pass, adaptive max_length |
| `server/src/bin/` | Benchmarks: `bench_batch`, `bench_stream`, `bench_n`, `profile` |
| `vendor/qwen3-tts-rs/` | Vendored fork with batching, streaming, speculative patches |

### Key modifications to vendor

1. `synthesize_batch_streaming()` — frame-by-frame generation with chunked vocoder decode
2. Fixed-window vocoder context (4 frames) with cross-fade (48 samples, ~2ms)
3. Token repetition early stop (threshold=3)
4. External stop signal via `&[&AtomicBool]` — forward thread signals generation loop
5. Send-failure detection — stops generation when receiver drops
6. `KVCache::truncate()` for speculative decoding rollback
7. Adaptive `max_length` for streaming: `(words*4)+30`, clamped 40-120

## Performance — Current State (v0.7.4)

### Batch throughput (L4, 0.6B)

| Batch | Audio | Wall time | Throughput | Latency/req | VRAM |
|-------|-------|-----------|-----------|-------------|------|
| 1 | 5.0s | 2.4s | 2.12x RT | 2.4s | 2.7GB |
| 4 | 19.5s | 2.8s | 6.99x RT | 0.7s | ~3.5GB |
| 8 | 41.4s | 3.6s | 11.49x RT | 0.4s | ~4GB |
| 16 | 84.6s | 5.1s | 16.59x RT | 0.3s | ~5GB |

### Streaming with voice clone (L4, 0.6B)

| Phrase | Words | TTFA | Total | Audio | RTF |
|--------|-------|------|-------|-------|-----|
| Short | 6 | 331ms | 4.88s | 4.06s | 1.20x |
| Medium | 17 | 326ms | 6.25s | 4.96s | 1.26x |
| Long | 27 | 321ms | 7.03s | 5.63s | 1.25x |

Stream close delay: <0.5s. Warmup: ~600ms.

## v0.7.x Optimization History

| Version | Change | Impact |
|---------|--------|--------|
| v0.7.0 | Voice_id preload endpoint | Had fake streaming bug (vocoder O(n²)) |
| v0.7.1 | Fixed-window vocoder O(1) + cross-fade + adaptive max_length | 10.67s → 4.56s for short phrase |
| v0.7.2 | Early stop on token repetition (rep=5) | Long phrases 28% faster |
| v0.7.3 | AtomicBool signal + rep=3 + vocoder context 8→4 | Clone matches no-clone (~1.25x RT) |
| v0.7.4 | Stop generation on send failure | Stream close delay 4.3s → 0.4s |

### Experimental (not merged)

| Branch | Attempt | Result |
|--------|---------|--------|
| `feature/speculative-decoding` | Draft 1 token ahead, verify with extra forward | +9% on medium phrases — not worth complexity |

### Earlier optimizations (v0.1–v0.5)

| Optimization | Impact |
|-------------|--------|
| Batched transformer forward pass | Baseline batching |
| Batched code predictor | 4.46x RT @ batch=8 |
| PreAlloc KV cache + batched EOS | Eliminated per-frame allocations |
| Arc shared model weights | VRAM 5174→2734 MB idle |
| Batched vocoder decoding | Batch=16: 13.43x → 16.59x RT (+24%) |
| Adaptive max_length + OOM batch splitting | KV cache 2048→122 for call center text |
| Batched streaming worker (up to 8 streams) | TTFA 761ms → 490ms |
| Speaker embedding cache + warmup | Cold TTFA 6.3s → 785ms |
| Voice_id preload endpoint | Zero encoder TTFA |

## Profiling Breakdown (single-stream, L4)

```
Prefill:      12ms   (0.5%)
Generation: 2,385ms  (92.5%) — 65 frames, 36.7ms/frame
Decode:      184ms   (7.1%)
Total:     2,580ms
```

Generation loop (transformer + code predictor) is 92.5% of wall time.

## Build

```bash
# Compile on Modal H100 targeting L4 (sm_89)
modal run modal_compile.py

# Download binary
modal volume get tts-compiled qwen3-tts-server ./qwen3-tts-server

# Run
MODEL_DIR=models/0.6b-base PORT=8090 ./qwen3-tts-server
```

Binary size: ~215 MB (statically linked CUDA + flash-attn).

## Dependencies

| Crate | Version | Purpose |
|-------|---------|---------|
| `qwen3-tts` | vendored | TTS inference engine |
| `axum` | 0.8 | HTTP server |
| `tokio` | 1.x | Async runtime |
| `hound` | 3.5 | WAV encoding |
| `base64` | 0.22 | Voice clone audio decoding |
| `tokio-stream` | 0.1 | Streaming response body |
