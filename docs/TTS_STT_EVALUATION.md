# TTS/STT Model Comparison — Call Center Workloads

Comparative evaluation of TTS/STT models for call center use cases.
Focus on Spanish language, voice cloning, concurrent throughput, and L4 24GB deployment.

**Last updated:** 2026-04-04
**Full evaluation:** [EVALUACION_CONSOLIDADA.md](https://voicetest.apulab.info/EVALUACION_CONSOLIDADA.md) (30+ TTS models, 5+ STT models)

---

## TTS Summary

| Model | Params | GPU | Batching | Best Throughput | Voice Clone | VRAM Idle | License |
|-------|--------|-----|----------|----------------|-------------|-----------|---------|
| **qwen3-tts-server v0.4.7** (ours) | 600M | L4 | ✅ Batch=16 | **16.59x RT** | ✅ ICL + x_vector | 2.7GB | MIT |
| OmniVoice 0.6B | 600M | L4 | ❌ Sequential | 6.8x RT (seq) | ✅ Zero-shot | 1.9GB | Apache 2.0 |
| Higgs Audio V2 3B | 3B | L40S | ✅ vLLM | 8.0x RT @8 CCU | ✅ Good | 38.2GB | Apache 2.0 |
| Kokoro 82M | 82M | L4 | ❌ Single | 15x RT | ✅ (via RVC) | ~0.3GB | Apache 2.0 |
| Supertonic 2 66M | 66M | CPU | ONNX threads | 68x RT | ❌ (10 fixed) | 0 (CPU) | OpenRAIL |

### qwen3-tts-server v0.4.7 (Rust/candle, flash-attn)

Custom batched autoregressive inference server with shared model weights.

| Batch | Audio | Wall time | Throughput | Latency/req | VRAM |
|-------|-------|-----------|-----------|-------------|------|
| 1 | 5.0s | 2.4s | 2.12x RT | 2.4s | 2.7GB |
| 4 | 19.5s | 2.8s | 6.99x RT | 0.7s | ~3.5GB |
| 8 | 41.4s | 3.6s | 11.49x RT | 0.4s | ~4GB |
| 16 | 84.6s | 5.1s | 16.59x RT | 0.3s | ~5GB |

Streaming TTFA: 450ms. Voice cloning: ICL + x_vector modes.
Call center capacity (10% duty cycle): ~60-80 simultaneous calls on single L4.

### Models tested and discarded (30+)

| Model | Params | Reason |
|-------|--------|--------|
| MOSS-TTS 1.7B | 1.7B | 0.3x RT, 13.4GB VRAM — too slow |
| MOSS-TTS-Realtime 1.7B | 1.7B | TTFB 180ms but no batching, no serving layer |
| T5Gemma-TTS 4B | 4B | No Spanish (EN/CN/JP only) |
| Multilingual-Exp 0.6B | 600M | Poor Spanish accent (Asian-influenced) |
| Voxtral 4B | 4B | CC-BY-NC license, no voice clone |
| CosyVoice3 0.5B | 500M | Not realtime (RTF 1.14), incoherent Spanish |
| Chatterbox ML 0.5B | 500M | Not realtime (RTF 1.15) |
| nano-qwen3tts-vllm | 600M | OOM on L4 (needs 22GB) |
| NVIDIA Magpie TTS | 241M | OOM on L4 and L40S (codec needs 21.3GB) |
| qts (GGUF) | 600M | 0.23x RT — ggml CUDA immature |
| LongCat-AudioDiT 1B | 1B | Spanish unintelligible (zh+en only) |

Full list: 30+ models evaluated. See consolidated doc for details.

---

## STT Summary

| Model | Params | VRAM | Telephony 8kHz | Batching | Best Throughput | License |
|-------|--------|------|----------------|----------|----------------|---------|
| **faster-whisper large-v3-turbo** ⭐ | 809M | 1.2GB | ✅ Works | ❌ Sequential | 58x RT single | MIT |
| Parakeet TDT 0.6B v2 | 600M | 2.6GB | ❌ Fails | ✅ Native | 199x RT @8 CCU | Apache 2.0 |
| Cohere Transcribe 2B | 2B | 4.1GB | ❌ Fails | ✅ Native | 117x RT @8 CCU | Apache 2.0 |

**Production choice: faster-whisper** — only model robust with 8kHz telephony audio (G.711/G.729 codecs).

Cohere Transcribe is #1 on Open ASR Leaderboard (WER 5.42) but fails with 8kHz audio from phone networks. Same model `03-2026` confirmed in April 4 scan — no new version that fixes 8kHz issue.

---

## Recommendation

| Use case | Model | GPU | Why |
|----------|-------|-----|-----|
| **Call center L4 + voice clone** | qwen3-tts-server | L4 | 16.59x RT batch=16, voice clone, 2.7GB, MIT |
| **Call center L40S** | Higgs Audio V2 3B | L40S | 8x RT @8 CCU, good voice clone, Apache 2.0 |
| **STT telephony** | faster-whisper | L4 | Only robust with 8kHz audio |
| **STT 16kHz+ high throughput** | Parakeet TDT 0.6B | L4 | 199x RT, native punctuation |

---

## Monitoring

| Model | Why | Blocker |
|-------|-----|---------|
| MOSS-TTS-Realtime 1.7B | TTFB 180ms | No batching, no serving layer |
| Higgs Audio V2.5 1B | Would fit L4 | Weights not published yet |
| vLLM-Omni for Qwen3-TTS | Native batching | Pending evaluation |
