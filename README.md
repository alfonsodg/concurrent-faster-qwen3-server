# qwen3-tts-server

High-performance Rust TTS server for Qwen3-TTS-12Hz-0.6B-Base. Batched inference with voice cloning, streaming, and flash-attention on NVIDIA GPUs.

## Performance (L4, 23GB)

| Batch | Throughput | Latency/req |
|-------|-----------|-------------|
| 1 | 2.12x RT | 2.4s |
| 8 | 11.49x RT | 0.4s |
| 16 | 16.59x RT | 0.3s |

Streaming TTFA: 450ms. VRAM: 2.7GB idle.

## Quick Start

### 1. Download the binary

From [GitLab Releases](https://scovil.labtau.com/ccvass/ai-audio/qwen3-tts-server/-/releases/v0.3.0):

```bash
curl --header "PRIVATE-TOKEN: <your-token>" \
  -o qwen3-tts-server \
  "https://scovil.labtau.com/api/v4/projects/622/packages/generic/qwen3-tts-server/0.3.2/qwen3-tts-server-v0.3.2-linux-x86_64"
chmod +x qwen3-tts-server
```

Requires NVIDIA GPU with CUDA 12.x (compiled for sm_89: L4, L40S, A100, H100).

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

Model size: ~1.2GB. Source: [Qwen/Qwen3-TTS-12Hz-0.6B-Base](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-0.6B-Base) on HuggingFace.

### 3. Run

```bash
MODEL_DIR=models/0.6b-base PORT=8090 MAX_BATCH=8 ./qwen3-tts-server
```

### 4. Test

```bash
curl -X POST http://localhost:8090/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"text": "Buenos días, ¿en qué puedo ayudarle?", "language": "spanish"}' \
  --output test.wav
```

## API

### `POST /v1/audio/speech`

```json
{
  "text": "Buenos días, ¿en qué puedo ayudarle?",
  "language": "spanish",
  "stream": false,
  "temperature": 0.7,
  "ref_audio": "<base64 WAV for voice cloning>",
  "ref_text": "Reference transcript"
}
```

Returns `audio/wav` with `x-rtf` header. With `"stream": true`, returns chunked WAV stream (TTFA ~450ms).

Languages: `spanish`/`es`, `english`/`en`, `french`/`fr`.

### `GET /health`

```json
{"status": "ok", "queue_depth": 0, "max_batch": 8}
```

### `GET /metrics`

Prometheus-compatible: `tts_requests_total`, `tts_avg_rtf`, `tts_queue_depth`, etc.

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_DIR` | `models/0.6b-base` | Path to Qwen3-TTS model |
| `MAX_BATCH` | `8` | Maximum batch size |
| `MAX_WAIT_MS` | `200` | Max wait to fill batch (ms) |
| `PORT` | `8090` | HTTP listen port |
| `RUST_LOG` | `info` | Log level |

## Systemd Service

```bash
sudo cp qwen3-tts-server.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now qwen3-tts-server
```

## Build from Source

Requires Modal H100 for flash-attn cross-compilation targeting L4 (sm_89):

```bash
modal run modal_compile.py          # compile on H100
modal run modal_flash_batch.py      # benchmark on L4
```

## Architecture

- Axum HTTP server with batch engine (dedicated thread)
- `Arc<Qwen3TTS>` shared model weights across batch + streaming workers
- Batched transformer forward pass (N sequences per GPU call)
- Batched vocoder decoding (single ONNX pass)
- Adaptive `max_length` for call center text (~6 frames/word)
- OOM recovery with automatic batch splitting

See [DEVELOPMENT.md](DEVELOPMENT.md) for full technical details.
