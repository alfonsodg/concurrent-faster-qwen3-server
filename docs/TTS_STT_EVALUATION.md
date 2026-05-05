# TTS/STT Model Comparison — Call Center Workloads

Comparative evaluation of TTS/STT models for call center use cases.
Focus on Spanish language, voice cloning, concurrent throughput, and L4 24GB deployment.

**Last updated:** 2026-05-05

---

## TTS Summary

| Model | Params | GPU | Batching | Best Throughput | Voice Clone | Streaming | VRAM | License |
|-------|--------|-----|----------|----------------|-------------|-----------|------|---------|
| **qwen3-tts-server v0.7.4** (ours) | 600M | L4 | ✅ Batch=16 | **16.59x RT** | ✅ ICL + x_vector | ✅ 325ms TTFA | 2.7GB | MIT |
| OmniVoice 0.6B (k2-fsa) | 600M | L4 | ❌ Sequential | 6.8x RT | ✅ Zero-shot | ❌ | 2.1GB | Apache 2.0 |
| VoxCPM2 2B (OpenBMB) | 2B | L4 | ✅ Nano-vLLM | 1.0x RT (L4) | ✅ Controllable | ✅ | ~8GB | Apache 2.0 |
| Multilingual-Exp 0.6B | 600M | L4 | ✅ vLLM | 14.1x RT @8 CCU | ✅ (bad accent) | ❌ | 12.2GB | Open |
| Kokoro 82M | 82M | L4 | ❌ Single | 15x RT | ✅ (via RVC) | ❌ | 0.3GB | Apache 2.0 |
| Supertonic 2 66M | 66M | CPU | ONNX threads | 68x RT | ❌ (10 fixed) | ❌ | 0 (CPU) | OpenRAIL |
| Higgs Audio V2 3B | 3B | L40S | ✅ vLLM | 8.0x RT @8 CCU | ✅ Good | ❌ | 38.2GB | Llama |
| Voxtral 4B | 4B | L40S | ✅ vLLM | 13.5x RT @8 CCU | ❌ | ❌ | 36.9GB | CC-BY-NC |

### qwen3-tts-server v0.7.4 (Rust/candle, flash-attn)

Custom batched autoregressive inference server with shared model weights.

**Batch throughput (L4, 0.6B):**

| Batch | Audio | Wall time | Throughput | Latency/req | VRAM |
|-------|-------|-----------|-----------|-------------|------|
| 1 | 5.0s | 2.4s | 2.12x RT | 2.4s | 2.7GB |
| 4 | 19.5s | 2.8s | 6.99x RT | 0.7s | ~3.5GB |
| 8 | 41.4s | 3.6s | 11.49x RT | 0.4s | ~4GB |
| 16 | 84.6s | 5.1s | 16.59x RT | 0.3s | ~5GB |

**Streaming with voice clone (L4, 0.6B, v0.7.4):**

| Phrase | Words | TTFA | Total | Audio | RTF |
|--------|-------|------|-------|-------|-----|
| Short | 6 | 331ms | 4.88s | 4.06s | 1.20x |
| Medium | 17 | 326ms | 6.25s | 4.96s | 1.26x |
| Long | 27 | 321ms | 7.03s | 5.63s | 1.25x |

Call center capacity (10% duty cycle): ~60-80 simultaneous calls on single L4.

**v0.7.x optimization history:**

| Version | Change | Impact |
|---------|--------|--------|
| v0.7.1 | Fixed-window vocoder O(1), cross-fade, adaptive max_length | Fixed fake streaming (10.67s → 4.56s) |
| v0.7.2 | Early stop on token repetition (rep=5) | Long phrases 28% faster |
| v0.7.3 | AtomicBool signal, rep=3, vocoder context 8→4 | Clone matches no-clone speed (~1.25x RT) |
| v0.7.4 | Stop generation on send failure | Stream close delay 4.3s → 0.4s |

### Models tested and discarded (30+)

| Model | Params | Reason |
|-------|--------|--------|
| MOSS-TTS 1.7B | 1.7B | 0.3x RT, 13.4GB VRAM — too slow |
| MOSS-TTS-Realtime 1.7B | 1.7B | TTFB 180ms but no batching, no serving layer |
| MOSS-TTS-Nano 100M | 100M | Package broken (pip install fails), API incompatible (2026-05-05) |
| T5Gemma-TTS 4B | 4B | No Spanish (EN/CN/JP only) |
| Voxtral 4B | 4B | CC-BY-NC license, no voice clone |
| CosyVoice3 0.5B | 500M | Not realtime (RTF 1.14), incoherent Spanish |
| Chatterbox ML 0.5B | 500M | Not realtime (RTF 1.15) |
| nano-qwen3tts-vllm | 600M | OOM on L4 (needs 22GB) |
| NVIDIA Magpie TTS | 241M | OOM on L4 and L40S (codec needs 21.3GB) |
| qts (GGUF) | 600M | 0.23x RT — ggml CUDA immature |
| LongCat-AudioDiT 1B | 1B | Spanish unintelligible (zh+en only) |
| VibeVoice-Realtime 0.5B | 500M | Barely realtime, no voice clone |
| VyvoTTS LFM2 0.4B | 400M | Better in English, no voice clone |
| LEMAS-TTS | 600M | Castilian phonemes only (not Latin American) |
| GLM-TTS (ZhipuAI) | ~1B | MIT, SOTA CER 0.89, but only ZH+EN — no Spanish |
| Fish Speech V1.5 | 4B | CC-BY-NC-SA-4.0 (non-commercial), exceeds L4 VRAM |
| Higgs Audio V2.5 1B | 1B | Weights gated (HF 401), Llama license, not self-hosteable |

---

## STT Summary

| Model | Params | VRAM | Telephony 8kHz | Batching | Best Throughput | Spanish Clinical | License |
|-------|--------|------|----------------|----------|----------------|-----------------|---------|
| **faster-whisper large-v3-turbo** | 809M | 1.2GB | ✅ Works | ❌ Sequential | 58x RT single | Good (1-2 errors/phrase) | MIT |
| **Cohere Transcribe 2B** | 2B | 4.5GB | ❌ Fails | ✅ Native | 14.8-43x RT | **Excellent (0 errors simple, 3 errors complex)** | Apache 2.0 |
| Parakeet TDT 0.6B v2 | 600M | 2.6GB | ❌ Fails | ✅ Native | 199x RT @8 CCU | N/A (EN only) | Apache 2.0 |
| Parakeet TDT 0.6B v3 | 600M | ~5GB | ? | ✅ Native | 21.6x RT (tested) | Regular (2 grave errors/phrase) | NVIDIA Community |
| Qwen3-ASR 0.6B | 900M | 2.2GB | ? | ✅ vLLM | 1-16x RT (tested) | ❌ Unusable (catastrophic errors) | Apache 2.0 |
| Qwen3-ASR 1.7B | 1.7B | 5GB | ? | ✅ vLLM | 8-13x RT (tested) | ❌ Unusable (catastrophic errors) | Apache 2.0 |
| GLM-ASR-Nano 1.5B | 1.5B | ~3GB | ? | ✅ | 145x RT (leaderboard) | N/A (EN/ZH/Cantonese only) | MIT |

### STT Spanish Clinical Evaluation (2026-05-02, L40S gpux51)

Test audio: espeak-ng es-419, medical terminology, 13-18s clips.

**Test 1 — Simple medical text:**
"Paciente masculino de 45 años con antecedentes de hipertensión arterial y diabetes mellitus tipo 2."

| Model | Result | Errors |
|-------|--------|--------|
| **Cohere Transcribe** | ✅ Perfect transcription | 0 |
| faster-whisper | "omeprazole" (anglicized), "día oral" (vía→día) | 2 |
| Parakeet v3 | "diabetes meícus" (mellitus), "omeprasol" | 2 |
| Qwen3-ASR 0.6B | "feria de alianzas" (hipertensión arterial) | catastrophic |

**Test 2-5 — Complex terminology (gadolinio, colangiopancreatografía, procalcitonina):**

| Model | Total errors (5 tests) | Grave errors |
|-------|----------------------|--------------|
| **Cohere Transcribe** | 6 | 3 |
| **faster-whisper** | 5 | 2 |
| Parakeet v3 | 8 | 5 |
| Qwen3-ASR | Not counted — completely unusable |

**Production choices:**

- **STT telephony 8kHz:** faster-whisper — only model robust with G.711/G.729 codecs
- **STT 16kHz+ clinical Spanish:** Cohere Transcribe — best accuracy for medical terminology
- **STT 16kHz+ high throughput (non-medical):** Parakeet TDT v3 — fastest but weak on specialized vocabulary

---

## Recommendation

| Use case | Model | GPU | Why |
|----------|-------|-----|-----|
| **Call center L4 + voice clone** | qwen3-tts-server | L4 | 16.59x RT batch, 1.25x RT streaming clone, 2.7GB, MIT |
| **Call center L40S** | Higgs Audio V2 3B | L40S | 8x RT @8 CCU, good voice clone, Apache 2.0 |
| **STT telephony 8kHz** | faster-whisper | L4 | Only robust with 8kHz audio |
| **STT 16kHz+ clinical** | Cohere Transcribe 2B | L4/L40S | Best Spanish medical accuracy, Apache 2.0 |
| **STT 16kHz+ high throughput** | Parakeet TDT 0.6B v3 | L4 | 199x RT, 25 EU languages, native punctuation |

---

## Monitoring

| Model | Why | Blocker |
|-------|-----|---------|
| VoxCPM2 2B | 30 langs, controllable clone, Apache 2.0 | 1.0x RT on L4, needs L40S |
| MOSS-TTS-Nano 100M | Apache 2.0, 20 langs, CPU, 48kHz | Package broken, API unstable |
| Higgs Audio V3 STT | Apache 2.0, 94 langs, beats Whisper V3 | boson_multimodal incompatible with transformers 4.47+ |
| Cohere Transcribe (8kHz) | Would replace faster-whisper if fixed | Fails with telephony 8kHz audio |
