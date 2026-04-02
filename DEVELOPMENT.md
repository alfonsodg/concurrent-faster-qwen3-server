# qwen3-tts-server — Development Status

High-performance Rust TTS server wrapping `qwen3-tts-rs` (Qwen3-TTS-12Hz-0.6B-Base)
with batched inference, streaming, and voice cloning for call center workloads.

## Architecture

```
                    ┌──────────────────────────────────────────────┐
                    │              axum HTTP server                │
                    │  POST /v1/audio/speech  GET /health          │
                    └──────┬──────────────────┬────────────────────┘
                           │                  │
                    ┌──────▼──────┐    ┌──────▼──────┐
                    │ BatchEngine │    │  Streaming   │
                    │  (thread)   │    │   Worker     │
                    │             │    │  (thread)    │
                    │ collect N   │    │  single-seq  │
                    │ reqs → batch│    │  frame-by-   │
                    │ forward     │    │  frame       │
                    └──────┬──────┘    └──────┬───────┘
                           │                  │
                    ┌──────▼──────────────────▼───────┐
                    │   Arc<Qwen3TTS> shared model    │
                    │   (single load, ~2.7GB VRAM)    │
                    └─────────────────────────────────┘
```

### Crate layout

| Path | Description |
|------|-------------|
| `server/src/main.rs` | Axum server, routes, WAV encoding, streaming response |
| `server/src/batch.rs` | `BatchEngine` — collects requests, runs batched forward pass |
| `server/src/streaming.rs` | Batched streaming stub (currently non-incremental) |
| `server/src/bin/bench_batch.rs` | Batch throughput benchmark (1/2/4/8 + sequential baseline + TTFA) |
| `server/src/bin/bench_stream.rs` | Streaming latency benchmark |
| `server/src/bin/bench_n.rs` | Concurrent HTTP benchmark |
| `server/src/bin/profile.rs` | VRAM profiling |
| `vendor/qwen3-tts-rs/` | Vendored fork of `TrevorS/qwen3-tts-rs` with batching patches |

### Key modifications to vendor (`vendor/qwen3-tts-rs/src/lib.rs`)

347 lines added/modified vs upstream. Changes:

1. `synthesize_batch()` — batched autoregressive loop: single transformer forward for N sequences per frame
2. `synthesize_batch_with_voices()` — batched inference with per-request voice clone prompts (mixed batches supported)
3. Pre-allocated KV caches per sequence
4. Batched greedy sampling (argmax across batch in one op)
5. Batched EOS detection and frame transfer
6. Pre-allocated zero tensor for done sequences (avoids per-frame allocation)
7. `unsafe impl Send/Sync for Qwen3TTS` — enables `Arc<Qwen3TTS>` sharing

Also patched: `src/models/code_predictor.rs` (+77 lines), `src/models/talker.rs` (+18/-18).

## Performance — Current State

### Modal L4 (23GB VRAM) — 2026-04-01

Build: flash-attn enabled, `CUDA_COMPUTE_CAP=89`, compiled on H100.

| Batch | Audio total | Wall time | Throughput (RTx) | Latency/req |
|-------|------------|-----------|------------------|-------------|
| 1 | 5.0s | 2.4s | 2.12x | 2.4s |
| 2 | 9.8s | 2.5s | 3.84x | 1.3s |
| 4 | 19.5s | 2.8s | 6.99x | 0.7s |
| 8 | 41.4s | 3.6s | 11.49x | 0.4s |
| 16 | 84.6s | 5.1s | 16.59x | 0.3s |

Streaming TTFA (Time To First Audio): ~450ms (single sequence).

Adaptive `max_length`: ~6 frames/word capped at 512 (vs default 2048). Reduces KV cache pre-allocation by ~75%.

### Upstream `qwen3-tts-rs` baseline (same L4)

| Metric | Value |
|--------|-------|
| Single stream | 0.91x RT (5.76s audio / 6.35s wall) |
| Concurrent 2 (process pool) | 1.00x throughput |
| Concurrent 4 (process pool) | 1.19x throughput |

### VRAM usage

| State | VRAM |
|-------|------|
| Idle (model loaded, Arc shared) | 2,734 MB |
| During inference (batch=8) | ~3,984 MB |
| OOM threshold | batch=16 on L4 (23GB) |

### vs vLLM-Omni (production L40S)

| Metric | vLLM-Omni | qwen3-tts-server |
|--------|-----------|------------------|
| Single RTx (denise voice) | 1.9x | 2.12x |
| Batch throughput | ~5.5x at 8 CCU (estimated) | 16.59x at batch=16 |
| VRAM idle | ~8GB+ | 2.7GB |
| Voice cloning | Yes (via API) | Yes (ref_audio base64, batched) |
| Streaming | WebSocket | HTTP chunked transfer |

## Optimization History

| Commit | Optimization | Impact |
|--------|-------------|--------|
| `51d256f` | Working batched forward pass | Baseline batching |
| `79ff201` | Batched code predictor | 4.46x RT @ batch=8 |
| `d5300d7` | Deferred GPU frame transfer | Reduced per-frame overhead |
| `4cd7ac0` | PreAlloc KV cache + batched EOS | Eliminated per-frame allocations |
| `4d447fb` | Batched greedy sampling | 6.24x RT @ batch=8 |
| `f8214ef` | Reverted embedding (stable) | 8.97x RT @ batch=16 (H100) |
| `7469300` | Arc shared model weights | VRAM 5174→2734 MB idle |
| `49ccef4` | Pre-allocate zero tensor | Minor allocation reduction |
| `18ce1e9` | Voice cloning in batch mode + busy-wait fix (10ms) | Batch=8: 9.56x → 9.98x RT, voice clone no longer falls back to sequential |
| `31acfab` | Batched streaming worker (up to 8 concurrent streams) | Streaming uses `synthesize_batch_streaming`, decodes every 10 frames |
| `1c3fc94` | Batched code predictor in streaming path | TTFA 761ms → 490ms, streaming now uses `generate_acoustic_codes_batched()` |
| `059ac10` | Adaptive max_length + OOM batch splitting | Batch=16 on L4: 13.43x RT (0.4s/req), KV cache 2048→122 for call center text |
| `23ba457` | /metrics endpoint + streaming format header | Prometheus-compatible metrics, atomic counters, x-audio-format header |
| `4838106` | Batched vocoder decoding | Batch=16: 13.43x → 16.59x RT (+24%), TTFA 482ms → 450ms |

### Failed experiments

| Commit | Attempt | Result |
|--------|---------|--------|
| `b8c9667` | Batched initial sampling + batched embedding + parallel decoder | Regression — reverted in `ba84ceb` |

## API

### `POST /v1/audio/speech`

```json
{
  "text": "Buenos días, ¿en qué puedo ayudarle?",
  "language": "spanish",
  "ref_audio": "<base64 WAV>",
  "ref_text": "Reference transcript",
  "temperature": 0.7,
  "stream": false
}
```

Response: `audio/wav` with `x-rtf` header.

With `"stream": true`: chunked `audio/wav` (WAV header + PCM16 chunks).

### `GET /health`

```json
{"status": "ok", "queue_depth": 0, "max_batch": 8}
```

### Environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_DIR` | `models/0.6b-base` | Path to Qwen3-TTS model |
| `MAX_BATCH` | `8` | Maximum batch size |
| `MAX_WAIT_MS` | `200` | Max wait to fill batch before processing |
| `PORT` | `8090` | HTTP listen port |

## Build

Requires Modal H100 for flash-attn compilation targeting L4 (sm_89):

```bash
# Compile on Modal H100, binary saved to volume
modal run modal_compile.py

# Download binary
modal run modal_compile.py download

# Benchmark on Modal L4
modal run modal_flash_batch.py
```

Binary size: ~233 MB (statically linked CUDA + flash-attn + ort).

## Pending Work — Performance

### High Priority

| Issue | Title | Expected Impact |
|-------|-------|-----------------|
| [#1](https://scovil.labtau.com/ccvass/ai-audio/qwen3-tts-server/-/issues/1) | Speculative decoding for single-stream | Batch=1: 2.6s → ~1.5-1.8s (35-40% reduction) |
| [#2](https://scovil.labtau.com/ccvass/ai-audio/qwen3-tts-server/-/issues/2) | ~~Batched vocoder decoding (Phase 4)~~ ✅ `4838106` | Batch=16: 13.43x → 16.59x RT (+24%) |

### Medium Priority

| Issue | Title | Expected Impact |
|-------|-------|-----------------|
| [#3](https://scovil.labtau.com/ccvass/ai-audio/qwen3-tts-server/-/issues/3) | ~~Batched prefill embedding (Phase 1)~~ ✅ `0ead9c5` | Code cleanup, shared tensors pre-computed |

### Profiling Breakdown (single-stream, L4)

```
Prefill:      12ms   (0.5%)
Generation: 2,385ms  (92.5%) — 65 frames, 36.7ms/frame
Decode:      184ms   (7.1%)
Total:     2,580ms
```

Generation loop (transformer + code predictor) is 92.5% of wall time. Further optimization requires profiling the per-frame breakdown (transformer forward vs code predictor's 15 sequential acoustic steps).

## Pending Work — Features

### P0 — Critical for production

1. ~~**Incremental batched streaming**~~ ✅ `31acfab` — Streaming worker now uses `synthesize_batch_streaming()`. Collects up to 8 concurrent stream requests within 50ms window, runs batched generation, decodes and sends PCM chunks every 10 frames (~800ms). Single request has no added latency (batch=1).

2. ~~**Voice cloning in batch mode**~~ ✅ `18ce1e9` — `synthesize_batch_with_voices()` accepts per-request `VoiceClonePrompt`. Mixed batches (some clone, some Serena) supported. `BatchEngine` creates prompts from ref_audio and passes to batch API.

3. ~~**Batch=16 OOM on L4**~~ ✅ `059ac10` — Adaptive `max_length` (~6 frames/word, cap 512) reduces KV cache from 2048→~122 for call center text. OOM recovery splits batch and falls back to sequential. Batch=16: 13.43x RT, 0.4s/req.

### P1 — Important

4. ~~**BlockingRecvTimeout busy-wait**~~ ✅ `18ce1e9` — Replaced 1ms busy-wait poll with 10ms OS sleep.

5. ~~**Streaming worker is single-threaded**~~ ✅ `31acfab` — Now batches up to 8 concurrent streams with forwarding threads per request.

6. ~~**WAV header data length**~~ ✅ `23ba457` — Added `x-audio-format: pcm-s16le-24000-mono` header. WAV `0xFFFFFFFF` is standard for streaming.

7. ~~**Error propagation in streaming**~~ ✅ Already handled — `synthesize_batch_streaming` errors sent to all clients, forwarding threads terminate on sender drop.

### P2 — Nice to have

8. **GGUF quantized models** — The `qts` project (also in workspace) supports GGUF quantized Qwen3-TTS. Could reduce VRAM significantly but needs vocoder ONNX integration.

9. ~~**Metrics endpoint**~~ ✅ `23ba457` — `GET /metrics` with Prometheus text format: requests_total, streaming, errors, audio_seconds, gen_seconds, avg_rtf, queue_depth. Atomic counters, no external deps.

10. **WebSocket streaming** — Current streaming uses HTTP chunked transfer. WebSocket would allow bidirectional control (cancel, pause).

11. **Multi-language batch** — Current batch assumes homogeneous language. The batch loop supports mixed languages but hasn't been tested.

12. **Proper README** — Replace GitLab template with actual project documentation.

## Dependencies

| Crate | Version | Purpose |
|-------|---------|---------|
| `qwen3-tts` | vendored | TTS inference engine |
| `axum` | 0.8 | HTTP server |
| `tokio` | 1.x | Async runtime |
| `hound` | 3.5 | WAV encoding |
| `tower-http` | 0.6 | CORS, tracing middleware |
| `base64` | 0.22 | Voice clone audio decoding |

## Infrastructure

- GitLab: `scovil.labtau.com/ccvass/ai-audio/qwen3-tts-server`
- Branch: `feature/perf-optimizations`
- Build: Modal H100 → binary → deploy to L4/L40S
- Model: `Qwen/Qwen3-TTS-12Hz-0.6B-Base` from HuggingFace (1.2GB safetensors + speech_tokenizer)
