# TTS Model Comparison — Call Center Workloads

Comparative evaluation of TTS models tested on NVIDIA L4 (24GB) and L40S (48GB) for call center use cases. Focus on Spanish language, voice cloning, and concurrent throughput.

## Summary

| Model | Params | GPU | Batching | Best Throughput | Voice Clone | VRAM Idle | VRAM Peak | License |
|-------|--------|-----|----------|----------------|-------------|-----------|-----------|---------|
| **qwen3-tts-server** (ours) | 600M | L4 | ✅ Batch=16 | **16.59x RT** | ✅ ICL + x_vector | 2.7GB | ~5GB | MIT |
| **OmniVoice 0.6B** | 600M | L4 | ❌ Sequential | **6.8x RT** (seq) | ✅ Zero-shot | 1.9GB | 2.3GB | Apache 2.0 |
| **Multilingual-Exp 0.6B** | 600M | L4 | ✅ vLLM | **14.1x RT** @8 CCU | ✅ (poor accent) | 12.2GB | ~14GB | Open |
| **Higgs Audio V2 3B** | 3B | L40S | ✅ vLLM | **8.0x RT** @8 CCU | ✅ Good | 38.2GB | ~42GB | Apache 2.0 |
| **Voxtral 4B** | 4B | L40S | ✅ vLLM-Omni | **13.5x RT** @8 CCU | ❌ | 19.6GB (L4) | 36.9GB (L40S) | CC-BY-NC |
| **Kokoro 82M** | 82M | L4 | ❌ Single | **15x RT** | ✅ (via RVC) | ~0.3GB | ~3GB | Apache 2.0 |
| **Supertonic 2 66M** | 66M | CPU | ONNX threads | **68x RT** | ❌ (10 fixed) | 0 (CPU) | <1GB RAM | OpenRAIL |
| **MOSS-TTS 1.7B** | 1.7B | L4 | ❌ Single | **0.3x RT** | ✅ Good | 13.4GB | ~15GB | Apache 2.0 |
| **qts (GGUF)** | 600M | L4 | ❌ Single | **0.23x RT** (Q8) | ✅ | ~1.3GB | ~2GB | Apache 2.0 |

## Detailed Results

### qwen3-tts-server v0.3.2 (Rust/candle, flash-attn)

Our custom server. Batched autoregressive inference with shared model weights.

| Batch | Audio | Wall time | Throughput | Latency/req | VRAM |
|-------|-------|-----------|-----------|-------------|------|
| 1 | 5.0s | 2.4s | 2.12x RT | 2.4s | 2.7GB |
| 2 | 9.8s | 2.5s | 3.84x RT | 1.3s | ~3GB |
| 4 | 19.5s | 2.8s | 6.99x RT | 0.7s | ~3.5GB |
| 8 | 41.4s | 3.6s | 11.49x RT | 0.4s | ~4GB |
| 16 | 84.6s | 5.1s | 16.59x RT | 0.3s | ~5GB |

Streaming TTFA: 450ms. Voice cloning: works (ICL + x_vector modes).

### OmniVoice 0.6B (k2-fsa, diffusion LM)

Sequential only (no batching). 600+ languages. Steps=16 for speed, steps=32 for quality.

| Text length | Steps=16 | Steps=32 | VRAM |
|-------------|----------|----------|------|
| Short (68 chars) | 5.21x RT (0.53s) | 2.65x RT (1.05s) | 2.0GB |
| Medium (191 chars) | 11.16x RT (0.56s) | 5.86x RT (1.07s) | 2.1GB |
| Long (413 chars) | 19.36x RT (0.59s) | 10.6x RT (1.06s) | 2.3GB |

Voice cloning: 0.25x RT (12-15s) — not viable for real-time.

Concurrent throughput (sequential, no batching):

| Requests | Throughput |
|----------|-----------|
| 1 | 6.4x RT |
| 4 | 6.8x RT |
| 8 | 6.8x RT |

### Multilingual-Expressive 0.6B (vLLM)

| CCU | Throughput | VRAM |
|-----|-----------|------|
| 1 | 2.7x RT | 12.2GB |
| 4 | 8.8x RT | — |
| 8 | 14.1x RT | — |

Spanish accent: poor (Asian-influenced). Voice cloning does not capture Latin American accent.

### Higgs Audio V2 3B (vLLM, L40S only)

| CCU | Throughput | VRAM |
|-----|-----------|------|
| 1 | 1.7x RT | 38.2GB |
| 4 | 5.9x RT | — |
| 8 | 8.0x RT | — |

Good Spanish quality. Good voice cloning. Does not fit on L4.

### Voxtral 4B (vLLM-Omni, L40S)

| CCU | Throughput (L4) | Throughput (L40S) | VRAM |
|-----|----------------|-------------------|------|
| 1 | 1.5x RT | 2.5x RT | 19.6GB (L4) / 36.9GB (L40S) |
| 4 | 2.8x RT | 7.0x RT | — |
| 8 | 5.2x RT | 13.5x RT | — |

No voice cloning. License: CC-BY-NC (non-commercial).

### Kokoro 82M (single-stream)

| Metric | Value |
|--------|-------|
| Single stream | 15x RT |
| VRAM | ~3GB |
| Voice cloning | Via RVC pipeline (external) |

Very fast but no native batching or voice cloning.

### Supertonic 2 66M (CPU/ONNX)

| Metric | Value |
|--------|-------|
| Single stream | 35x RT |
| 4 threads | 59x RT |
| 8 threads | 68x RT |
| VRAM | 0 (CPU only) |

10 fixed voices, no voice cloning. Requires 44.1kHz, 10 steps, speed >= 1.1.

### qts — GGUF/ggml (Qwen3-TTS)

| Variant | Throughput | VRAM |
|---------|-----------|------|
| F16 (ggml) | 0.12x RT | — |
| Q8 (ggml) | 0.23x RT | — |

10x slower than candle. ggml CUDA backend is immature. Not viable.

### Models tested and discarded

| Model | Params | Reason |
|-------|--------|--------|
| CosyVoice3 0.5B | 500M | Not realtime (RTF 1.14), incoherent Spanish |
| Chatterbox ML 0.5B | 500M | Not realtime (RTF 1.15) |
| VyvoTTS LFM2 0.4B | 400M | 1.0x RT short Spanish, no voice clone |
| MOSS-TTS 1.7B | 1.7B | 0.3x RT without batching |
| nano-qwen3tts-vllm | 600M | OOM on L4 (needs 22GB) |
| NVIDIA Magpie TTS | 241M | OOM on L4 and L40S (codec decoder needs 21.3GB) |
| VibeVoice-Realtime 0.5B | 500M | No voice clone, 0.91x RT on L4 |
| NeuTTS Nano Spanish | 120M | 0.3x RT, truncated audio |
| LongCat-AudioDiT 1B | 1B | Spanish unintelligible (zh+en only) |
| XTTS-v2 (Coqui) | ~1B | Restrictive CPML license |

## Recommendation by use case

| Use case | Model | GPU | Why |
|----------|-------|-----|-----|
| **Call center L4 + voice clone** | qwen3-tts-server | L4 | 16.59x RT batch=16, voice clone ICL, 2.7GB VRAM, MIT |
| **Call center L4 + multilingual** | OmniVoice 0.6B | L4 | 600+ languages, 6.8x RT, 2.1GB VRAM, Apache 2.0 |
| **Call center L40S** | Higgs Audio V2 3B | L40S | 8x RT @8 CCU, good voice clone, Apache 2.0 |
| **Low latency single-stream** | Kokoro 82M | L4 | 15x RT, voice clone via RVC |
| **CPU/edge no GPU** | Supertonic 2 | CPU | 68x RT, 0 VRAM |
