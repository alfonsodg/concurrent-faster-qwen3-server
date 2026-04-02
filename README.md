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

| Batch | Throughput | Latency/req | Concurrent calls (real-time) |
|-------|-----------|-------------|------------------------------|
| 1 | 2.12x RT | 2.4s | 2 |
| 4 | 6.99x RT | 0.7s | 6 |
| 8 | 11.49x RT | 0.4s | 11 |
| 16 | 16.59x RT | 0.3s | 16 |

Streaming TTFA: 450ms. In a real call center scenario (conversational duty cycle ~10%), a single L4 can handle ~60-80 simultaneous calls.

## Requirements

- Linux x86_64
- NVIDIA GPU with CUDA 12.x and compute capability >= 8.9 (L4, L40S, A100, H100)
- Minimum 6GB VRAM (8GB+ recommended for batch > 4)
- Model: [Qwen/Qwen3-TTS-12Hz-0.6B-Base](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-0.6B-Base) (~1.2GB)

## Quick Start

### 1. Download the binary

```bash
curl --header "PRIVATE-TOKEN: <your-token>" \
  -o qwen3-tts-server \
  "https://scovil.labtau.com/api/v4/projects/622/packages/generic/qwen3-tts-server/0.3.2/qwen3-tts-server-v0.3.2-linux-x86_64"
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
- Headers: `x-rtf` (real-time factor)

#### Streaming response

When `stream: true`:
- Content-Type: `audio/wav`
- Transfer-Encoding: `chunked`
- Header: `x-audio-format: pcm-s16le-24000-mono`
- First chunk: 44-byte WAV header, then PCM data chunks (~800ms each)

#### Error responses

| Status | Body | Cause |
|--------|------|-------|
| 503 | `{"error": "Queue full"}` | All batch slots occupied |
| 500 | `{"error": "<message>"}` | Synthesis failed (model error, voice clone failure, etc.) |

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

Requires Modal account for H100 cross-compilation:

```bash
modal run modal_compile.py          # compile on H100 (targets L4 sm_89)
modal run modal_flash_batch.py      # benchmark on L4
modal run modal_test.py             # run unit tests
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
