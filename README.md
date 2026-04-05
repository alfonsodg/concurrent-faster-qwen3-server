# qwen3-tts-server

High-performance Rust TTS server for Qwen3-TTS-12Hz-0.6B-Base. Batched inference with voice cloning, streaming, and flash-attention on NVIDIA GPUs.

## Features

- Batched inference: up to 16 concurrent requests in a single GPU forward pass
- Voice cloning: clone any voice from a short reference audio (with or without transcript)
- Streaming: chunked WAV output with 450ms time-to-first-audio
- Adaptive batching: automatic max_length tuning for call center text
- OOM recovery: automatic batch splitting on GPU memory exhaustion
- Prometheus metrics: `/metrics` endpoint for monitoring
- Low VRAM: 2.7GB idle, ~4GB during inference

## Performance (NVIDIA L4, 23GB)

| Batch | Throughput | Latency/req | Concurrent calls (real-time) | VRAM |
|-------|-----------|-------------|------------------------------|------|
| 1 | 2.12x RT | 2.4s | 2 | 2.7GB |
| 4 | 6.99x RT | 0.7s | 6 | ~3.5GB |
| 8 | 11.49x RT | 0.4s | 11 | ~4GB |
| 16 | 16.59x RT | 0.3s | 16 | ~5GB |

Streaming TTFA: 700ms (with voice cloning, cached). In a real call center scenario (conversational duty cycle ~10%), a single L4 can handle ~60-80 simultaneous calls.

### vs other TTS models (L4)

| Model | Batching | Best Throughput | Voice Clone | VRAM |
|-------|----------|----------------|-------------|------|
| **qwen3-tts-server** (ours) | ✅ Batch=16 | **16.59x RT** | ✅ ICL + x_vector | 2.7GB idle |
| OmniVoice 0.6B | ❌ Sequential | 6.8x RT | ✅ (slow: 0.25x RT) | 1.9GB |
| Kokoro 82M | ❌ Single | 15x RT | ✅ (via RVC) | 0.3GB |
| Supertonic 2 66M | ONNX threads | 68x RT | ❌ (10 fixed) | 0 (CPU) |
| Higgs Audio V2 3B | ✅ vLLM | 8.0x RT @8 CCU | ✅ | 38GB (L40S only) |

Full comparison with 29+ models: [docs/TTS_STT_EVALUATION.md](docs/TTS_STT_EVALUATION.md)

## Requirements

- Linux x86_64
- NVIDIA GPU with CUDA 12.x and compute capability >= 8.9 (L4, L40S, A100, H100)
- Minimum 6GB VRAM (8GB+ recommended for batch > 4)
- Model: [Qwen/Qwen3-TTS-12Hz-0.6B-Base](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-0.6B-Base) (~1.2GB)

## Quick Start

### 1. Download the binary

```bash
# From GitLab package registry
curl -o qwen3-tts-server \
  "https://scovil.labtau.com/api/v4/projects/622/packages/generic/qwen3-tts-server/v0.5.2/qwen3-tts-server-v0.5.2-linux-x86_64"
chmod +x qwen3-tts-server
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
MODEL_DIR=models/0.6b-base PORT=8090 MAX_BATCH=8 ./qwen3-tts-server
```

### 4. Test

```bash
# Basic synthesis
curl -X POST http://localhost:8090/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"text": "Buenos días, ¿en qué puedo ayudarle?", "language": "spanish"}' \
  --output test.wav

# Health check
curl http://localhost:8090/health

# Metrics
curl http://localhost:8090/metrics
```

## API Reference

### `POST /v1/audio/speech`

Synthesize speech from text. Supports standard synthesis, voice cloning, and streaming.

#### Request body

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `text` | string | yes | — | Text to synthesize |
| `language` | string | no | `"spanish"` | Language: `spanish`/`es`, `english`/`en`, `french`/`fr` |
| `temperature` | float | no | `0.7` | Sampling temperature (0.0-1.0) |
| `stream` | bool | no | `false` | Enable chunked streaming response |
| `ref_audio` | string | no | — | Base64-encoded WAV for voice cloning |
| `ref_text` | string | no | — | Transcript of ref_audio (enables ICL mode for better quality) |

#### Response

- Content-Type: `audio/wav`
- Sample rate: 24000 Hz, 16-bit PCM, mono
- Headers: `x-rtf` (real-time factor), `x-ttfa-ms` (time to first audio in milliseconds)

#### Streaming response

When `stream: true`:
- Content-Type: `audio/wav`
- Transfer-Encoding: `chunked`
- Headers: `x-audio-format: pcm-s16le-24000-mono`, `x-ttfa-ms` (time to first audio)
- First chunk: 44-byte WAV header, then PCM data chunks (~800ms each)

#### Error responses

| Status | Body | Cause |
|--------|------|-------|
| 400 | `{"error": "<message>"}` | Invalid input: empty text, bad language, invalid ref_audio WAV |
| 413 | `{"error": "ref_audio exceeds ... limit"}` | ref_audio too large |
| 503 | `{"error": "Queue full"}` / `{"error": "Stream queue full"}` | All batch/stream slots occupied |
| 500 | `{"error": "<message>"}` | Synthesis failed (model error, voice clone failure) |

#### Examples

Standard synthesis:

```bash
curl -X POST http://localhost:8090/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello world", "language": "english"}' \
  --output hello.wav
```

Voice cloning (x_vector mode — speaker embedding only):

```bash
REF_B64=$(base64 -w0 reference.wav)
curl -X POST http://localhost:8090/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d "{\"text\": \"Buenos días\", \"language\": \"spanish\", \"ref_audio\": \"$REF_B64\"}" \
  --output cloned.wav
```

Voice cloning (ICL mode — higher quality, uses transcript):

```bash
REF_B64=$(base64 -w0 reference.wav)
curl -X POST http://localhost:8090/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d "{\"text\": \"Buenos días\", \"language\": \"spanish\", \"ref_audio\": \"$REF_B64\", \"ref_text\": \"Transcript of the reference audio.\"}" \
  --output cloned_icl.wav
```

Streaming:

```bash
curl -X POST http://localhost:8090/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"text": "Buenos días", "language": "spanish", "stream": true}' \
  --output stream.wav
```

### `GET /health`

```json
{"status": "ok", "queue_depth": 0, "max_batch": 8}
```

### `GET /metrics`

Prometheus text format:

```
tts_requests_total 142
tts_requests_streaming 23
tts_errors_total 0
tts_audio_seconds_total 891.2
tts_gen_seconds_total 312.4
tts_avg_rtf 2.85
tts_queue_depth 3
```

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_DIR` | `models/0.6b-base` | Path to Qwen3-TTS model directory |
| `MAX_BATCH` | `8` | Maximum batch size (16 fits on L4 23GB) |
| `MAX_WAIT_MS` | `200` | Max wait to fill batch before processing (ms) |
| `PORT` | `8090` | HTTP listen port |
| `RUST_LOG` | `info` | Log level (`debug`, `info`, `warn`, `error`) |
| `STREAM_MAX_BATCH` | `8` | Max concurrent streaming requests per batch |
| `STREAM_WAIT_MS` | `50` | Wait window to collect streaming batch (ms) |
| `STREAM_CHUNK_FRAMES` | `10` | Frames per streaming chunk (~800ms audio) |
| `MAX_REF_AUDIO_BYTES` | `10485760` | Max ref_audio size (10MB) |

## Voice Cloning

Two modes available:

- **x_vector** (no `ref_text`): Uses speaker embedding only. Captures timbre, fast inference.
- **ICL** (with `ref_text`): Uses speaker embedding + reference audio codes. Better prosody matching.

### Best practices

- **Reference audio**: 6-15 seconds of clean speech, 24kHz mono WAV
- **Transcript**: Use Whisper to generate accurate `ref_text` — improves similarity from ~0.75 to ~0.89
- **Language matching**: Clone in Spanish → synthesize in Spanish
- **Temperature**: 0.8 recommended for voice cloning
- **Speaker cache**: Same `ref_audio` bytes are cached automatically — second request is instant

### TTFA (Time To First Audio)

| Scenario | TTFA |
|----------|------|
| Streaming, no voice cloning | ~700ms |
| Streaming + voice cloning (first call, cold) | ~785ms |
| Streaming + voice cloning (same voice, cached) | ~700ms |
| Non-streaming + voice cloning (cached) | ~2800ms |

The server warms up CUDA kernels at startup (speaker encoder + transformer + vocoder). First real request has no compilation penalty.

## Deployment

### Systemd

```bash
sudo cp qwen3-tts-server.service /etc/systemd/system/
# Edit service file to match your paths
sudo systemctl daemon-reload
sudo systemctl enable --now qwen3-tts-server
sudo journalctl -u qwen3-tts-server -f
```

### Docker (not required)

The binary is statically linked with CUDA runtime. No container needed — just the binary + model files + NVIDIA driver.

## Build from Source

### Local compilation

Requires Rust, CUDA toolkit 12.x, CMake, clang, and pkg-config:

```bash
# Install dependencies (Ubuntu/Debian)
sudo apt install cmake pkg-config libssl-dev libasound2-dev libclang-dev clang

# Install Rust
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# Clone and build
git clone git@scovil.labtau.com:ccvass/ai-audio/qwen3-tts-server.git
cd qwen3-tts-server
cargo build --release --features cuda,flash-attn

# Binary at target/release/qwen3-tts-server
```

For flash-attn, the CUDA compute capability must match your GPU. Set `CUDA_COMPUTE_CAP` if needed:

```bash
CUDA_COMPUTE_CAP=89 cargo build --release --features cuda,flash-attn  # L4, L40S
CUDA_COMPUTE_CAP=80 cargo build --release --features cuda,flash-attn  # A100
CUDA_COMPUTE_CAP=90 cargo build --release --features cuda,flash-attn  # H100
```

Without flash-attn (simpler, slightly slower):

```bash
cargo build --release --features cuda
```

### Cross-compilation on Modal

For building on H100 targeting L4 (sm_89):

```bash
modal run modal_compile.py          # compile on H100
```

## Benchmarking

Scripts in `scripts/` for benchmarking against a running server (no Modal needed):

```bash
# Batch throughput + concurrent latency (1/2/4/8/16 requests)
python3 scripts/bench_server.py --url http://localhost:8090

# Voice cloning (x_vector mode)
python3 scripts/bench_voice_clone.py --ref reference.wav --output cloned.wav

# Voice cloning (ICL mode — higher quality)
python3 scripts/bench_voice_clone.py --ref reference.wav --ref-text "Transcript of reference." --output cloned_icl.wav

# Streaming TTFA (time to first audio)
python3 scripts/bench_streaming.py --url http://localhost:8090 --trials 5

# Custom concurrency levels
python3 scripts/bench_server.py --concurrency 1,4,8,16,32
```

Remote benchmarks on Modal (optional, for profiling on cloud GPUs):

```bash
modal run modal_flash_batch.py      # batch throughput on L4
modal run modal_profile.py          # per-phase profiling on L4
modal run modal_test.py             # unit tests on L4
```

## Architecture

- Axum HTTP server with dedicated batch engine thread
- `Arc<Qwen3TTS>` shared model weights across batch + streaming workers
- Batched transformer forward pass (N sequences per GPU call)
- Batched vocoder decoding (single pass for all sequences)
- Batched streaming worker (collects up to 8 concurrent streams)
- Adaptive `max_length` based on text word count (~6 frames/word)
- OOM recovery with automatic batch splitting

See [DEVELOPMENT.md](DEVELOPMENT.md) for full technical details, optimization history, and profiling data.
