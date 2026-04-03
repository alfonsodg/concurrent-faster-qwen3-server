# concurrent-faster-qwen3-server

High-performance concurrent Rust TTS server for [Qwen3-TTS-12Hz-0.6B-Base](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-0.6B-Base). Batched inference with voice cloning, streaming, and flash-attention on NVIDIA GPUs.

Takes the upstream [qwen3-tts-rs](https://github.com/TrevorS/qwen3-tts-rs) single-stream engine (0.91x real-time) and optimizes it to **16.59x real-time** at batch=16 on a single NVIDIA L4.

Uses a [modified fork of qwen3-tts-rs](https://github.com/alfonsodg/qwen3-tts-rs) with batched inference, ICL voice clone fixes, and concurrent server optimizations. See the [fork's CHANGELOG](https://github.com/alfonsodg/qwen3-tts-rs/blob/main/CHANGELOG.md) for all modifications.

## Performance (NVIDIA L4, 23GB)

| Batch | Throughput | Latency/req | Concurrent calls (real-time) | VRAM |
|-------|-----------|-------------|------------------------------|------|
| 1 | 2.12x RT | 2.4s | 2 | 2.7GB |
| 4 | 6.99x RT | 0.7s | 6 | ~3.5GB |
| 8 | 11.49x RT | 0.4s | 11 | ~4GB |
| 16 | 16.59x RT | 0.3s | 16 | ~5GB |

Streaming TTFA: 450ms. In a real call center scenario (~10% TTS duty cycle), a single L4 handles ~60-80 simultaneous calls.

### vs other TTS models (L4)

| Model | Batching | Best Throughput | Voice Clone | VRAM |
|-------|----------|----------------|-------------|------|
| **This server** | ✅ Batch=16 | **16.59x RT** | ✅ ICL + x_vector | 2.7GB idle |
| OmniVoice 0.6B | ❌ Sequential | 6.8x RT | ✅ (slow: 0.25x RT) | 1.9GB |
| Kokoro 82M | ❌ Single | 15x RT | ✅ (via RVC) | 0.3GB |
| Higgs Audio V2 3B | ✅ vLLM | 8.0x RT @8 CCU | ✅ | 38GB (L40S only) |

Full comparison: [docs/TTS_STT_EVALUATION.md](docs/TTS_STT_EVALUATION.md) | Complete evaluation of 29+ TTS and STT models: [docs/FULL_TTS_STT_EVALUATION.md](docs/FULL_TTS_STT_EVALUATION.md)

## Features

- Batched inference: up to 16 concurrent requests in a single GPU forward pass
- Voice cloning: clone any voice from a short reference audio (ICL + x_vector modes)
- Streaming: chunked WAV output with 450ms time-to-first-audio
- Adaptive batching: automatic max_length tuning for call center text
- OOM recovery: automatic batch splitting on GPU memory exhaustion
- Prometheus metrics: `/metrics` endpoint for monitoring
- Low VRAM: 2.7GB idle, ~4GB during inference

## Optimization Journey

Starting from [qwen3-tts-rs](https://github.com/TrevorS/qwen3-tts-rs) (single-stream, 0.91x RT), we applied the following optimizations:

### Step 1: Shared model weights via `Arc<Qwen3TTS>`

Wrapped the model in `Arc` with `unsafe impl Send+Sync` to share weights across batch and streaming workers. Reduced VRAM from 5.1GB to 2.7GB idle (single model load instead of per-worker copies).

### Step 2: Batched autoregressive generation

Modified `synthesize_batch()` to run N sequences through a single transformer forward pass per frame. Each frame: batched code predictor → batched step input construction → single transformer forward → batched greedy sampling → batched EOS detection.

Result: batch=8 went from 0.91x to ~4.5x RT.

### Step 3: Pre-allocated KV cache with batched writes

Replaced concat-based KV cache with pre-allocated fixed-size buffers using CUDA `InplaceOp2` + `copy2d` for zero-allocation writes during generation.

### Step 4: Batched greedy sampling + EOS detection

Combined per-sequence argmax into a single batched operation. Stacked all new tokens for a single GPU→CPU transfer instead of N separate transfers.

Result: batch=8 reached ~6.2x RT.

### Step 5: Batched code predictor in streaming path

The streaming path was using sequential `generate_acoustic_codes()` per request. Switched to `generate_acoustic_codes_batched()` matching the batch path.

Result: TTFA improved from 761ms to 450ms.

### Step 6: Batched vocoder decoding

Phase 4 (vocoder) was decoding N sequences sequentially. Stacked all code tensors `[N, 16, T_max]` and ran a single vocoder forward pass, then split and trimmed output waveforms.

Result: batch=16 went from 13.4x to **16.59x RT** (+24%).

### Step 7: Adaptive max_length

KV cache pre-allocation used `max_length=2048` (model default). For call center text (~12 words), this wastes VRAM. Changed to `~6 frames/word + 50`, capped at 512. This reduced KV cache from 2048 to ~122 positions, unlocking batch=16 on L4 (previously OOM).

### Step 8: Batched streaming worker

Replaced single-sequence streaming worker with a batched version that collects up to 8 concurrent stream requests within a 50ms window, then runs `synthesize_batch_streaming()`.

### Step 9: Voice cloning in batch mode

Extended `synthesize_batch()` to accept per-request `VoiceClonePrompt`. Mixed batches (some with voice clone, some with default Serena voice) are supported. Failed voice clone requests return explicit errors instead of silent fallback.

### Step 10: ICL voice clone warm-up fix

The Rust ICL implementation generates warm-up frames for `ref_text` before target text. Fixed by skipping `ref_text_tokens - 3` frames from generated codes, then prepending original ref_codes for vocoder context with proportional cut. The `-3` margin preserves the onset of the first phoneme.

### What didn't work

| Attempt | Result |
|---------|--------|
| KV cache INT8 quantization | Dequantize overhead in candle (no fused kernels) caused regression |
| GGUF/ggml backend (qts) | 10x slower than candle — ggml CUDA is immature |
| Audio resampling for ICL speed correction | Destroyed pitch (chipmunk audio) |
| Codec frame dropping for ICL speed correction | Same pitch destruction — vocoder needs consecutive frames |
| Batched embedding + parallel decoder | Regression — reverted |

## Requirements

- Linux x86_64
- NVIDIA GPU with CUDA 12.x and compute capability >= 8.9 (L4, L40S, A100, H100)
- Minimum 6GB VRAM (8GB+ recommended for batch > 4)
- Rust toolchain (1.94+)
- CMake, clang, pkg-config, libssl-dev, libasound2-dev

## Quick Start

### 1. Build

```bash
git clone https://github.com/alfonsodg/concurrent-faster-qwen3-server.git
cd concurrent-faster-qwen3-server

# With flash-attn (recommended, set CUDA_COMPUTE_CAP for your GPU)
CUDA_COMPUTE_CAP=89 cargo build --release --features cuda,flash-attn  # L4, L40S
CUDA_COMPUTE_CAP=80 cargo build --release --features cuda,flash-attn  # A100
CUDA_COMPUTE_CAP=90 cargo build --release --features cuda,flash-attn  # H100

# Without flash-attn (simpler, slightly slower)
cargo build --release --features cuda
```

### 2. Download the model

```bash
pip install huggingface-hub[cli] hf-xet
mkdir -p models/0.6b-base
huggingface-cli download Qwen/Qwen3-TTS-12Hz-0.6B-Base \
  --local-dir models/0.6b-base \
  --include "model.safetensors" "config.json" "generation_config.json" \
  "preprocessor_config.json" "tokenizer_config.json" "vocab.json" "merges.txt" \
  "speech_tokenizer/model.safetensors" "speech_tokenizer/config.json"
```

### 3. Run

```bash
MODEL_DIR=models/0.6b-base PORT=8090 MAX_BATCH=8 ./target/release/qwen3-tts-server
```

### 4. Test

```bash
# Basic synthesis
curl -X POST http://localhost:8090/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello, how can I help you?", "language": "english"}' \
  --output test.wav

# Voice cloning
REF_B64=$(base64 -w0 reference.wav)
curl -X POST http://localhost:8090/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d "{\"text\": \"Hello\", \"language\": \"english\", \"ref_audio\": \"$REF_B64\", \"ref_text\": \"Transcript of reference audio.\"}" \
  --output cloned.wav
```

## API Reference

### `POST /v1/audio/speech`

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `text` | string | yes | — | Text to synthesize |
| `language` | string | no | `"spanish"` | `spanish`/`es`, `english`/`en`, `french`/`fr` |
| `temperature` | float | no | `0.7` | Sampling temperature (0.0-1.0) |
| `stream` | bool | no | `false` | Enable chunked streaming response |
| `ref_audio` | string | no | — | Base64-encoded WAV for voice cloning |
| `ref_text` | string | no | — | Transcript of ref_audio (enables ICL mode) |

Response: `audio/wav` (24kHz, 16-bit PCM, mono). Header `x-rtf` contains real-time factor.

### `GET /health`

```json
{"status": "ok", "queue_depth": 0, "max_batch": 8}
```

### `GET /metrics`

Prometheus text format: `tts_requests_total`, `tts_avg_rtf`, `tts_queue_depth`, etc.

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_DIR` | `models/0.6b-base` | Path to Qwen3-TTS model |
| `MAX_BATCH` | `8` | Maximum batch size (16 fits on L4) |
| `MAX_WAIT_MS` | `200` | Max wait to fill batch (ms) |
| `PORT` | `8090` | HTTP listen port |
| `RUST_LOG` | `info` | Log level |

## Benchmarking

```bash
# Throughput + concurrent latency
python3 scripts/bench_server.py --url http://localhost:8090

# Voice cloning
python3 scripts/bench_voice_clone.py --ref reference.wav --ref-text "Transcript."

# Streaming TTFA
python3 scripts/bench_streaming.py --url http://localhost:8090
```

## Architecture

- Axum HTTP server with dedicated batch engine thread
- `Arc<Qwen3TTS>` shared model weights across batch + streaming workers
- Batched transformer forward pass (N sequences per GPU call)
- Batched vocoder decoding (single pass for all sequences)
- Batched streaming worker (up to 8 concurrent streams)
- Adaptive `max_length` based on text word count
- OOM recovery with automatic batch splitting

See [DEVELOPMENT.md](DEVELOPMENT.md) for profiling data and full technical details.

## License

Apache-2.0. See [LICENSE](LICENSE).

## Acknowledgments

- [qwen3-tts-rs](https://github.com/TrevorS/qwen3-tts-rs) by TrevorS — Rust inference engine for Qwen3-TTS (Apache-2.0). This project uses a [modified fork](https://github.com/alfonsodg/qwen3-tts-rs) with batched inference, ICL voice clone fixes, and concurrent server optimizations.
- [Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS) by Alibaba Cloud Qwen Team — the underlying TTS model (Apache-2.0).
- [candle](https://github.com/huggingface/candle) by Hugging Face — Rust ML framework.
