# Evaluación Consolidada STT/TTS — Modelos de Referencia

**Fecha:** 2026-04-02
**Hardware:** NVIDIA L4 24GB, NVIDIA L40S 48GB
**Audio test:** Grabaciones reales de agentes call center (denise.wav 28.8s, español)
**Stack actual:** faster-whisper large-v3 (STT) + Faster-Qwen3-TTS 0.6B (TTS)

---

## TTS — Tabla comparativa completa

### Concurrencia y rendimiento

| Métrica | **OmniVoice 0.6B** 🆕 | **Higgs Audio V2 3B** ⭐ | **Voxtral 4B TTS** | **Multilingual-Expressive 0.6B** | **Kokoro-82M** | **Supertonic 2** | **Faster-Qwen3-TTS 0.6B** |
|---------|------------------------|--------------------------|---------------------|----------------------------------|----------------|-----------------|--------------------------|
| **Params** | 600M (Qwen3-0.6B) | 3B (5.77B total) | 4B | 600M | 82M | 66M | 600M |
| **VRAM (L4)** | **2.1GB** | ❌ No cabe | 19.6GB | 12.2GB (vLLM) | ~3GB | <1GB (CPU) | 2.6GB |
| **VRAM (L40S)** | — | 38.2GB | 36.9GB | — | — | — | — |
| **GPU mínima** | L4 24GB | L40S 48GB | L4 24GB (lento) / L40S | L4 24GB | L4 24GB | CPU | L4 24GB |
| **Licencia** | Apache 2.0 | Apache 2.0 | ❌ CC-BY-NC | Open | Apache 2.0 | OpenRAIL | MIT |
| **Español** | ✅ Buena | ✅ Buena | ✅ Buena | ❌ Acento asiático¹ | ✅ Básico | ✅ Buena² | ✅ Buena |
| **Voice clone** | ✅ Zero-shot nativo | ✅ Buena | ❌ No | ✅ (acento malo) | ✅ (vía RVC) | ❌ (10 fijas) | ✅ |
| **Voice design** | ✅ (género, pitch, acento) | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |
| **Idiomas** | **600+** | ~10 | 9 | ~20 | ~5 | ~10 | ~20 |
| **Non-verbal** | ✅ [laughter], [sigh]... | ❌ | ❌ | ✅ | ❌ | ❌ | ❌ |
| **Emoción** | ✅ (vía voice design) | ✅ Emergente | ❌ | ✅ | ❌ | ❌ | ❌ |
| **Batching** | ❌ Sequential only | ✅ vLLM | ✅ vLLM-omni | ✅ vLLM | ❌ Single | ONNX threads | ❌ Single |
| **Docker** | ❌ | ✅ bosonai/higgs-audio-vllm | ❌ (manual) | ✅ docker-compose | ❌ | ❌ | ❌ |

### Throughput — L4 24GB

| CCU | OmniVoice 0.6B 🆕 | Multilingual-Exp 0.6B | Voxtral 4B | Kokoro-82M | Supertonic 2 | Faster-Qwen3-TTS |
|-----|-------------------|----------------------|------------|------------|-------------|------------------|
| 1 | **6.4x RT** | 2.7x RT | 1.5x RT | 15x RT | 35x RT | 2.5x RT |
| 4 | **6.8x RT** (seq) | 8.8x RT | 2.8x RT | — | 59x RT | 2.5x RT |
| 8 | **6.8x RT** (seq) | **14.1x RT** | 5.2x RT | — | 68x RT | 2.5x RT |

### Throughput — L40S 48GB

| CCU | Higgs V2 3B | Voxtral 4B |
|-----|-------------|------------|
| 1 | 1.7x RT | 2.5x RT |
| 4 | 5.9x RT | 7.0x RT |
| 8 | **8.0x RT** | **13.5x RT** |

¹ Multilingual-TTS: voice clone no captura acento latinoamericano. Entrenado con datos Malaysia-AI.
² Supertonic 2: requiere 44.1kHz, 10 steps, speed≥1.1.

### Recomendación por caso de uso

| Caso de uso | Modelo | GPU | Por qué |
|-------------|--------|-----|---------|
| **Call center L4 + voice clone** 🆕 | OmniVoice 0.6B | L4 | 2.1GB VRAM, 6.8x RT, voice clone nativo, 600+ langs, Apache 2.0 |
| **Call center L40S + voice clone** | Higgs Audio V2 3B | L40S/A100 | Buena calidad español, voice clone, 8x RT a 8 CCU |
| **Call center L4 (alternativa)** | qwen3-tts-server (Rust) | L4 | 4.46x RT batched, voice clone ICL |
| **Baja latencia single-stream** | Kokoro-82M + RVC | L4 | 15x RT, voice clone vía RVC |
| **CPU/edge sin GPU** | Supertonic 2 | CPU | 42x RT, 0 VRAM, 44.1kHz |
| **Referencia (licencia NC)** | Voxtral 4B | L40S | 13.5x RT a 8 CCU, pero CC-BY-NC |

**Pendiente:** NVIDIA Magpie TTS en L40S, Higgs Audio V2.5 (1B) cuando se publique

### Pendientes de probar en L40S

| Modelo | Params | Español | Voice Clone | Batching | Licencia | Nota |
|--------|--------|---------|-------------|----------|----------|------|
| **nano-qwen3tts-vllm** | 0.6B | ✅ | ✅ | ✅ vLLM nativo | MIT | Requiere flash-attn (no compila en L40S por CUDA mismatch) |
| **Fish Audio S2 Pro** | 5B (4B+400M) | ✅ Tier 2 | ✅ | ✅ SGLang nativo | Research (comercial separada) | Licencia research, necesita aceptar términos en HF |
| **NVIDIA Magpie TTS** | 241M | ✅ es-US | ✅ (Zeroshot gated) | ✅ TensorRT | NVIDIA Open | ❌ OOM en L4 Y L40S — codec decoder TRT necesita 21.3GB extra, requiere A100 80GB |
| **XTTS-v2 (Coqui)** | ~1B | ✅ (17 idiomas) | ✅ (6s ref) | ❌ Single-stream | CPML (restrictiva) | Requiere Python <3.12 |
| **MOSS-TTS-GGUF 8B** | 8B | ✅ | ✅ | ❌ | Apache 2.0 | GGUF podría correr donde ONNX falló |

### Probados en L40S 48GB

| Modelo | VRAM | Throughput 8 CCU | Español | Voice Clone | Licencia | Veredicto |
|--------|------|-----------------|---------|-------------|----------|-----------|
| **Higgs Audio V2 3B** ⭐ | 38.2GB | **8.0x RT** | ✅ Buena | ✅ Buena | Apache 2.0 | Mejor opción L40S |
| **Voxtral 4B TTS** | 36.9GB | **13.5x RT** | ✅ Buena | ❌ No | ❌ CC-BY-NC | Solo referencia |

### Probados localmente (L4) — Resultados adicionales

| Modelo | VRAM | RTF | Español | Voice Clone | Nota |
|--------|------|-----|---------|-------------|------|
| **OmniVoice 0.6B (k2-fsa)** 🆕 | **2.1GB** | **0.08-0.34** (3-12x RT) | ✅ 600+ langs | ✅ Zero-shot nativo | Apache 2.0, Qwen3-0.6B base, voice design, non-verbal. Sin batching nativo |
| **VibeVoice-Realtime 0.5B** | ~3GB | 1.1x RT | ✅ Experimental | ❌ Voces embebidas | Apenas realtime en L4, sin batching, sin voice clone. No viable para call center |
| **MOSS-TTS-Local 1.7B** | 13.4GB | 0.3x RT | ✅ 20 idiomas | ✅ Buena calidad | Lento sin batching. Pendiente vLLM |
| **LEMAS-TTS (multilingual_grl)** | 2.3GB | 1.74-1.91x RT | ✅ Castellano (θ) | ✅ Zero-shot | Flow-matching, CC-BY-4.0. Fonemas castellanos (no latino). Sin batching nativo |
| **VyvoTTS LFM2 0.4B** 🆕 | 0.74GB | 1.8x RT (medio ES) | ⚠️ Mejor en inglés | ❌ No | MIT. LiquidAI LFM2 backbone, vLLM nativo (268 tok/s). 1.0x RT en texto corto ES. Sin voice clone |
| **QTS (Rust Qwen3-TTS GGUF)** | 0 (CPU) | 0.11x RT | ✅ (Qwen3-TTS) | ✅ | ggml CUDA no activa en L4. Solo CPU = inutilizable. Proyecto inmaduro (9 stars) |

### Pendientes

| Modelo | Estado |
|--------|--------|
| **nano-qwen3tts-vllm 1.7B** | El 0.6B tiene bug (tensor mismatch). Solo funciona con 1.7B (~20GB VRAM) |
| **VibeVoice-ASR 7B** | STT 60 min, 50+ idiomas, vLLM, speaker diarization |
| Higgs Audio V2.5 (1B) | No publicado aún. Cuando salga, probar en L40S |

### Conclusión (2026-04-02) — ACTUALIZADA

> **🆕 NUEVO CANDIDATO: OmniVoice 0.6B (k2-fsa)**
>
> Paper: [arXiv:2604.00688](https://arxiv.org/abs/2604.00688) (1 abril 2026)
> Repo: [github.com/k2-fsa/OmniVoice](https://github.com/k2-fsa/OmniVoice)
> HuggingFace: [k2-fsa/OmniVoice](https://huggingface.co/k2-fsa/OmniVoice)
>
> Diffusion Language Model NAR basado en Qwen3-0.6B. 600+ idiomas, Apache 2.0.
>
> **Benchmarks en L4 24GB (FP16):**
>
> | Test | Latencia | Audio | RTF | Speed | VRAM |
> |------|----------|-------|-----|-------|------|
> | Texto corto ES (68 chars) | 1.49s | 4.4s | 0.34 | 3.0x RT | 2,004 MB |
> | Texto medio ES (191 chars) | 1.61s | 12.1s | 0.13 | 7.5x RT | 2,110 MB |
> | Texto largo ES (413 chars) | 2.07s | 25.2s | 0.08 | 12.2x RT | 2,292 MB |
> | Voice clone (con ref_text) | 1.74-2.04s | 9.6s | 0.18-0.21 | 5-5.5x RT | 2,289 MB |
> | Voice clone (auto-whisper) | 4.54s | 5.0s | 0.91 | 1.1x RT | 3,832 MB |
> | 16 steps (medio ES) | 0.79s | 12.1s | 0.065 | 15.4x RT | 2,110 MB |
> | 8 steps (medio ES) | 0.43s | 12.1s | 0.035 | 28.2x RT | 2,110 MB |
>
> **Concurrencia (sin batching nativo):**
>
> | Modo | 1 req | 4 req | 8 req | VRAM |
> |------|-------|-------|-------|------|
> | Sequential | 6.4x RT | 6.8x RT | 6.8x RT | 2,085 MB |
> | Parallel threads | 6.4x RT | 5.4x RT | 3.4x RT | 2,779 MB |
>
> **Features:** Voice clone zero-shot ✅, Voice design (género/pitch/acento) ✅, Non-verbal [laughter] ✅, 600+ idiomas ✅, Pronunciación controlable ✅
>
> **Limitaciones:** Sin batching nativo (sequential only), sin streaming, proyecto nuevo (3 commits, 218 stars)
>
> **Audios de prueba:** https://voicetest.apulab.info/audio-bench/ (sección OmniVoice)
>
> **Nota sobre voice clone:** Siempre pasar `ref_text` para evitar carga de Whisper (+1.7GB VRAM, +3s latencia). Con ref_text: 5x RT y 2.3GB VRAM.

> **🚀 SOLUCIÓN PREVIA: qwen3-tts-server (Rust/candle)**
>
> Servidor TTS concurrente custom basado en [qwen3-tts-rs](https://github.com/TrevorS/qwen3-tts-rs).
> Repo: https://scovil.labtau.com/ccvass/ai-audio/qwen3-tts-server
>
> **Benchmarks en L4 24GB (Qwen3-TTS 0.6B, CUDA BF16, flash-attn, v0.3.2):**
>
> | Batch | Throughput | Por request | VRAM |
> |-------|-----------|-------------|------|
> | 1 | 2.12x RT | 2.4s | ~2.7GB |
> | 4 | 6.99x RT | 0.7s | ~3.5GB |
> | 8 | 11.49x RT | 0.4s | ~4GB |
> | **16** | **16.59x RT** | **0.3s** | ~5GB |
>
> **Features:** Voice clone (x_vector + ICL) ✅, Streaming TTFA 450ms ✅, API REST ✅, /metrics Prometheus ✅, OOM recovery ✅
>
> Optimizaciones: batched transformer forward, batched vocoder, batched code predictor, adaptive max_length, shared Arc model weights.

Después de evaluar 29+ modelos TTS en L4 y L40S:

| Opción | Modelo | GPU | Concurrencia | Voice Clone | Español |
|--------|--------|-----|-------------|-------------|---------|
| **🆕 E: L4 OmniVoice** | OmniVoice 0.6B (k2-fsa) | L4 24GB | **6.8x RT** (seq) | ✅ Zero-shot nativo | ✅ 600+ langs |
| **D: L4 custom server** | qwen3-tts-server (Qwen3-TTS 0.6B) | L4 24GB | ✅ **16.59x RT a 16 CCU** | ✅ ICL clone | ✅ (Qwen3) |
| **A: L40S** | Higgs Audio V2 3B | L40S 48GB | ✅ 8x RT a 8 CCU | ✅ Buena | ✅ Buena |
| **B: L4 single-stream** | Faster-Qwen3-TTS 0.6B | L4 24GB | ❌ 2.5x RT fijo | ✅ | ✅ Buena |
| **C: L4 sin voice clone** | Supertonic 2 | CPU | ✅ 68x RT | ❌ 10 fijas | ✅ Buena |

**Servidor QA L40S limpio.** Solo avalbot (7.7GB VRAM). Tests eliminados.

> **📌 NOTA: vLLM-Omni v0.18.0 (2026-03-28)**
> El fork [vllm-project/vllm-omni](https://github.com/vllm-project/vllm-omni) añade serving nativo de audio con continuous batching para: **Qwen3-TTS** (optimizaciones de TTFA, concurrencia, CUDA graphs, websocket streaming, voice upload), **Fish Speech S2 Pro** (voice cloning online), **Voxtral TTS** (8-step flow matching), **CosyVoice3 0.5B**, **MiMo-Audio**. API `/v1/audio/speech` con streaming WAV y speaker embeddings. Esto podría resolver el problema de concurrencia para Qwen3-TTS sin cambiar de modelo. Pendiente de evaluar.

> **📌 NOTA: TurboQuant — Compresión KV cache a 3 bits (Google, ICLR 2026)**
> [TurboQuant](https://arxiv.org/abs/2504.19874) comprime KV cache a 3 bits con zero accuracy loss y hasta 8x speedup en atención (H100). Compresión 4.9x (TQ3) / 3.8x (TQ4).
>
> **Implementaciones disponibles (2026-04-01):**
> | Proyecto | Tipo | Estado | Concurrencia |
> |----------|------|--------|-------------|
> | [mitkox/vllm-turboquant](https://github.com/mitkox/vllm-turboquant) | Fork vLLM 0.18.1rc1 completo | ✅ Compilable (369★, 73 forks) | ⚠️ Reportan 18 tok/s vs 3-5K normal — pendiente validar |
> | `pip install turboquant` (back2matching) | HuggingFace drop-in | ✅ Instalable | ❌ Rompe concurrencia en vLLM |
> | [0xSero/turboquant](https://github.com/0xSero/turboquant) | vLLM monkey-patch + Triton | ✅ Usable | ⚠️ Sin confirmar |
> | [ik_llama.cpp #1509](https://github.com/ikawrakow/ik_llama.cpp/issues/1509) | llama.cpp standalone | CPU ✅, CUDA pendiente | N/A (no serving) |
> | [TheTom/turboquant_plus](https://github.com/TheTom/turboquant_plus) | llama.cpp + Metal | ✅ Apple Silicon | ? |
> | [turbo-quant](https://lib.rs/crates/turbo-quant) | Crate Rust | ✅ Embeddings/KV | N/A |
>
> **Plan**: compilar `mitkox/vllm-turboquant` en Modal (H100) para evaluar con Qwen3.5 27B y validar si la concurrencia funciona. Si sí, aplicar a Qwen3 8B para liberar VRAM en L4 para TTS.
>
> **Notas prácticas** (del artículo [dev.to](https://dev.to/arshtechpro/turboquant-what-developers-need-to-know-about-googles-kv-cache-compression-eeg)):
> - 4-bit es el sweet spot (calidad ≈ FP16 en modelos 3B+)
> - 3-bit degrada en modelos <8B
> - Values más sensibles que Keys — dar más bits a V
> - Beneficio real a partir de 4K+ tokens de contexto
> - Residual window: mantener últimos 128-256 tokens en FP16
>
> **Probado 2026-04-01 en Modal H100 (Qwen2.5-3B, `pip install turboquant`):**
> | Métrica | FP16 | TQ4 | TQ3 |
> |---------|------|-----|-----|
> | tok/s | 36.9 | 31.2 | 34.3 |
> | VRAM peak | 5.79GB | 5.80GB | 5.81GB |
> | VRAM savings | — | 0 | 0 |
>
> **Resultado: zero VRAM savings, 15% más lento.** El paquete pip descomprime a FP16 para atención sin liberar datos comprimidos. Necesita kernels CUDA fusionados (atención directa sobre datos comprimidos) para ser útil. Confirma reporte de Laurent Laborde (18 tok/s en vLLM). **No usar en producción todavía — esperar integración nativa en llama.cpp/vLLM.**
>
> **Estado en llama.cpp (2026-04-01):**
> TurboQuant NO está mergeado en [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp) oficial. Existe como [Discussion #20969](https://github.com/ggml-org/llama.cpp/discussions/20969) (52 comentarios, 141 respuestas, muy activa). El desarrollo real ocurre en forks:
>
> **Fork principal: [TheTom/turboquant_plus](https://github.com/TheTom/turboquant_plus)** — Metal + CUDA, 1,351+ iteraciones
> - Uso: `--cache-type-k turbo3 --cache-type-v turbo3`
> - Metal (Apple Silicon): 98.7-99.5% velocidad de q8_0 a 32K contexto, PPL +1.1%
> - CUDA: kernels optimizados por Madreag. turbo2 **supera q8_0 en 5.4%** a 32K en RTX 5090
> - 4 GPUs validadas (SM86×2, SM89, SM120)
>
> **Hallazgos clave de la comunidad:**
> | Config | Compresión | PPL delta | Velocidad vs q8_0 |
> |--------|-----------|-----------|-------------------|
> | turbo3 K + turbo3 V | 5.12x | <3% | ~99% (Metal), ~100% (CUDA) |
> | turbo2 K + turbo2 V | 7.53x | ~5% | >100% a 32K+ (CUDA) |
> | q8_0 K + turbo3 V | ~3x | <1% | ~100% (config óptima calidad) |
>
> - V (values) compresión es "gratis" — cosine similarity 1.000 incluso a 2-bit
> - K (keys) es el cuello de botella de calidad — dar más bits a K
> - turbo4 tiene bug en algunos forks (PPL explota), turbo3 es seguro
> - Beneficio real a partir de 8K+ tokens de contexto
>
> **Impacto para AvalBot/OKBot:**
> Cuando TurboQuant se mergee en llama.cpp oficial → Ollama lo hereda automáticamente → Qwen3 8B con `turbo3` KV cache liberaría ~2-3GB VRAM en L4 para TTS concurrente. Monitorear Discussion #20969 para merge upstream.
>
> **Otros forks llama.cpp:**
> - [ik_llama.cpp #1509](https://github.com/ikawrakow/ik_llama.cpp/issues/1509): implementación standalone, CPU funcional, CUDA pendiente
> - Madreag: kernels CUDA optimizados integrados en turboquant_plus

### Modelos pendientes de monitorear (2026-04-02)

| Modelo | Params | Por qué interesa | Bloqueante |
|--------|--------|-----------------|------------|
| **Raon-Speech-9B** (KRAFTON) 🆕 | 9B | STT+TTS unificado, Qwen3 base, vLLM-Omni deploy | ❌ Solo en/ko, CC-BY-NC, 9B no cabe en L4 (RTF 0.27 en RTX 6000 Pro) |
| **Raon-OpenTTS-0.3B** (KRAFTON) 🆕 | 336M | Open-data (510K hrs), supera F5-TTS en WER | ❌ Solo inglés, CC-BY-NC |
| **Higgs Audio V2.5 1B** (Boson) | 1B | Cabría en L4, multilingüe, vLLM nativo | ❌ Pesos NO publicados ([issue #168](https://github.com/boson-ai/higgs-audio/issues/168)). Solo API vía Eigen AI Cloud |
| **LongCat-AudioDiT 1B** (Meituan) | 1B | MIT, SOTA voice cloning, NAR diffusion, 2.7GB VRAM BF16, RTF 6-12x en L4 | ❌ **Probado 2026-04-01: español ininteligible.** Solo zh+en. 3.8GB VRAM peak. Velocidad excelente pero idioma inutilizable |
| **LongCat-AudioDiT 3.5B** (Meituan) | 3.5B | MIT, supera Seed-TTS en voice cloning | ❌ Solo zh+en, ~10-14GB VRAM BF16. No probar — mismo problema de idioma que 1B |
| **Fish Audio S2 Pro** | ~1B | 80+ idiomas (incluye español), voice clone, control emocional inline, soporte vLLM-Omni | ❌ Licencia research-only (no comercial) |
| **MiMo-V2-TTS** (Xiaomi) | ? | 100M hrs entrenamiento, singing, RL multi-dimensional | ❌ API only, solo zh+en, sin weights públicos |
| **Qwen3.5-Omni** (Alibaba) | 30B MoE | Omnimodal con TTS nativo | ❌ Closed source, >>48GB VRAM |
| **IBM Granite Speech 3.3 8B** (STT) | 8B | Apache 2.0, multi-idioma, ASR+AST | ❌ 8B demasiado grande (~16GB VRAM) para L4 con servicios actuales |

### Modelos revisados y descartados (no probar)

| Modelo | Razón |
|--------|-------|
| Step-Audio-TTS-3B (StepFun) | Solo zh/en en benchmarks, sin español confirmado |
| Fish Speech 1.5 | ❌ Licencia CC-BY-NC-SA-4.0, español ~20k hrs (limitado) |
| neutts-air-dllm-bd8 | Fork no oficial de NeuTTS, sin español confirmado |
| QTS (Rust Qwen3-TTS) | ggml CUDA no activa, solo CPU (0.11x RT). 9 stars, 0 forks. Inmaduro |
| AutoArk-AI/GPA | Solo en/zh |
| VoiceCore (webbigdata) | Solo japonés |
| NeuTTS Nano Spanish (120M) | **Probado 2026-04-01:** 0.3x RT (no realtime), audio truncado (~2.7s para frases largas), 4.3GB VRAM. Licencia NeuTTS Open License 1.0 (no Apache). Descartado |
| KugelAudio (7B) | 23 EU langs, MIT, pero 7B (~14-16GB VRAM) no cabe en L4 con servicios. Sin info de concurrencia |
| FireRedTTS2 | 7 idiomas pero **sin español** (EN,ZH,JP,KO,FR,DE,RU). Descartado |
| Kyutai Pocket-TTS (100M) | ❌ Solo inglés. CPU-only 6x RT, voice clone, CC-BY-4.0. 538 likes, muy popular pero sin español |
| Kyutai STT-1B / STT-2.6B | ❌ Solo en/fr (1B) y solo en (2.6B). Streaming STT pero sin español |
| Soprano 80M (ekwek) | ❌ Solo inglés. 2000x RT en GPU, Apache 2.0. Ultra rápido pero sin español |
| SILMA F5-TTS 150M | ❌ Solo árabe+inglés. Apache 2.0, lightweight F5-TTS |
| Spark-TTS 0.5B | ❌ Solo en/zh |
| OuteTTS 0.3-1B | ❌ Sin español (en/jp/ko/zh/fr/de) |

> **Validación HuggingFace 2026-04-02:** Revisión exhaustiva de modelos TTS/STT recientes. No hay modelos nuevos relevantes para español que no estén evaluados. Cobertura: 29+ modelos TTS, 5+ modelos STT.

---

## STT — Tabla comparativa completa

### Especificaciones

| Métrica | **faster-whisper large-v3-turbo** ⭐ | **NVIDIA Parakeet TDT 0.6B** | **Cohere Transcribe** | **Voxtral Mini 4B Realtime** | **Granite 4.0-1B (IBM)** |
|---------|---------------------------------------|------------------------------|----------------------|------------------------------|-------------------------|
| **Params** | 809M | 600M | 2B | 4B | 1B |
| **VRAM** | 1.2GB | 2.6GB | 4.1GB | 8.9GB | 4.6GB |
| **Licencia** | MIT | Apache 2.0 | Apache 2.0 | Apache 2.0 | Apache 2.0 |
| **Español** | ✅ Bueno | ✅ Bueno (v3) | ✅ Excelente | ✅ Excelente | ❌ Alucinaciones |
| **Telefonía 8kHz** | ✅ Funciona | ❌ Falla | ❌ Falla | Pendiente | ✅ Funciona |
| **Batching** | ❌ Secuencial | ✅ Nativo | ✅ Nativo | ✅ vLLM realtime | ❌ |
| **Puntuación nativa** | ❌ | ✅ | ✅ | ✅ | ✅ |
| **Streaming** | ❌ | ❌ | ❌ | ✅ (websocket 480ms) | ❌ |

### Velocidad — Audio largo (~29s, español 16kHz)

| Modelo | alfonso 29.5s | denise 28.8s |
|--------|--------------|-------------|
| **Parakeet TDT 0.6B** | 153ms / **192x RT** | 186ms / **155x RT** |
| **faster-whisper int8** | 509ms / 58x RT | 721ms / 40x RT |
| **Cohere Transcribe** | 2439ms / 12x RT | 1077ms / 27x RT |
| **Voxtral Mini 4B** | 33357ms / 1x RT | 32543ms / 1x RT |

### Velocidad — Audio corto (español 16kHz)

| Modelo | 2s | 5s | 10s |
|--------|-----|-----|------|
| **Parakeet TDT 0.6B** | 108ms / 18.5x RT | 107ms / 46.9x RT | 114ms / 88x RT |
| **faster-whisper int8** | ~160ms / 12.4x RT | ~210ms / 23.7x RT | ~210ms / 48.1x RT |
| **Cohere Transcribe** | ~190ms / 10.5x RT | ~350ms / 14.2x RT | ~510ms / 19.5x RT |
| **Voxtral Mini 4B** | 3145ms / 0.6x RT | 6311ms / 0.8x RT | 11744ms / 0.9x RT |

### Concurrencia — Audio largo (29.5s)

| Modelo | 1 req | 4 req | 8 req |
|--------|-------|-------|-------|
| **Parakeet TDT 0.6B** | **198x RT** | **133x RT** | **199x RT** |
| **Cohere Transcribe** | 17x RT | 63x RT | 117x RT |
| **faster-whisper int8** | 27x RT | 27x RT | 27x RT |
| **Voxtral Mini 4B** | 0.9x RT | 3.4x RT | — |

### Concurrencia — Audio corto (2s)

| Modelo | 1 req | 4 req | 8 req |
|--------|-------|-------|-------|
| **Cohere Transcribe** | 10.5x RT | **34.1x RT** | **69.9x RT** |
| **faster-whisper int8** | **12.4x RT** | 12.3x RT | 12.3x RT |

### Recomendación STT

| Caso de uso | Modelo | Por qué |
|-------------|--------|---------|
| **Telefonía 8kHz (producción)** | faster-whisper large-v3 | Único robusto con audio 8kHz |
| **Audio 16kHz+ máximo throughput** | Parakeet TDT 0.6B | 199x RT a 8 req, puntuación nativa |
| **Audio 16kHz+ con concurrencia** | Cohere Transcribe | 117x RT a 8 CCU, mejor calidad |

> **⚠️ Cohere y Parakeet descartados para telefonía:** No funcionan con audio 8kHz (G.711/G.729). Parakeet estuvo en producción AvalBot brevemente (commit 5a044d4) y fue revertido a Whisper (commit 068249e).

> **📌 MAI-Transcribe-1 (Microsoft, 2 abril 2026):** SOTA en FLEURS 25 langs, 2.5x más rápido que Azure Fast, excelente en ruido. Pero es **solo API cloud** vía Microsoft Foundry ($0.36/hora). No hay pesos descargables ni modelo en HuggingFace. No evaluable localmente.

---

## Modelos TTS descartados

| Modelo | Params | Razón |
|--------|--------|-------|
| Multilingual-TTS 0.6B (Scicom) | 600M | Acento español no natural, voice clone no captura acento latinoamericano |
| CosyVoice3 0.5B | 500M | No realtime (RTF 1.14), audio incoherente en español |
| Chatterbox ML 0.5B | 500M | No realtime (RTF 1.15) |
| Chatterbox Multilingual 0.5B | 500M | RTF ~1.0 auto, 0.5x RT voice clone, 3.9GB VRAM. MIT. 23 langs con español. Apenas realtime, voice clone sub-realtime. No viable para call center |
| VyvoTTS LFM2 0.4B | 400M | 1.8x RT medio ES, 1.0x RT corto ES. 0.74GB VRAM. MIT. vLLM nativo (268 tok/s) pero lento en L4. Sin voice clone. Mejor en inglés |
| MOSS-TTS-Realtime ONNX | 1.7B | Solo CPU FP32, OOM después de 12 tokens |
| nano-qwen3tts-vllm | 600M | OOM en L4 (necesita 22GB, 3 procesos GPU) |
| NVIDIA Magpie TTS | 241M | OOM en L4 y L40S — codec decoder TRT necesita 21.3GB extra, requiere A100 80GB |
| Orpheus 3B (es_it) | 3B | Gated + necesita 19.8GB VRAM |
| VibeVoice-Realtime 0.5B (Microsoft) | 0.5B | ❌ Sin voice clone (voces embebidas). Español experimental, 0.91x RT en L4, MIT |
| Covo-Audio 7B | 7B | Academic only, solo en/zh |
| Maya1 3B | 3B | Solo inglés |
| VibeVoice 1.5B | 1.5B | Solo en/zh |
| Fish S2 Pro 5B | 5B | Licencia research only |
| TADA 3B-ml | 3B | Licencia Llama restrictiva |
| IndexTTS-2 | — | Solo en/zh |

---

## Audios de referencia

https://voicetest.apulab.info/audio-bench/

---

## Fine-tuning — Guía para mejorar acento español latinoamericano

### Objetivo

Los modelos TTS evaluados tienen buen español pero el acento no es 100% latinoamericano natural. Fine-tuning con LoRA permite mejorar prosodia y acento sin reentrenar el modelo completo.

### Datasets recomendados

| Dataset | Idioma | Horas | Notas |
|---------|--------|-------|-------|
| `mozilla-foundation/common_voice_17_0` (es) | Español | ~2000h | Variedad de acentos latam, filtrar por país |
| `OpenSpeechHub/common-voice-asr-clean` | Multi | Variable | Versión limpia de Common Voice |
| `amphion/Emilia-Dataset` | Multi (incl. es) | Masivo | Alta calidad, multilingüe |
| Grabaciones propias AvalBot/OKBot | Español PE | Custom | Dominio exacto del call center |
| `voices/denise_ref.wav`, `ok_mujer_5s_ref.wav` | Español PE | Refs | Speaker adaptation targets |

### Qwen3-TTS 0.6B (candidato L4)

LLM backbone basado en Qwen3. Acepta LoRA fine-tuning estándar.

```bash
# 1. Preparar datos
# Formato: pares (texto, audio_wav) con audio 24kHz mono
# Filtrar: SNR > 20dB, duración 3-15s, solo hablantes latam

# 2. Descargar Common Voice ES y filtrar
from datasets import load_dataset
ds = load_dataset("mozilla-foundation/common_voice_17_0", "es", split="train")
# Filtrar por locale: es_419 (latam), es_PE, es_MX, es_CO, es_AR

# 3. LoRA config
# Target modules: q_proj, v_proj del LLM backbone
# Rank: 16, alpha: 32, dropout: 0.05
# LR: 1e-5, epochs: 5-10
# Batch size: 4-8 (depende de VRAM disponible)

# 4. Evaluar con voces de referencia
# Comparar MOS y similitud de acento pre/post fine-tune
```

VRAM estimada para fine-tune: ~8-10GB (modelo 0.6B + LoRA + batch). Cabe en L4 con servicios apagados.

### Higgs Audio V2 3B (candidato L40S)

LLM backbone Llama-based. Mismo approach LoRA pero necesita L40S por tamaño.

```
# LoRA en capas del LLM, no tocar codec/vocoder
# VRAM fine-tune: ~20-25GB (modelo 3B + LoRA + gradientes)
# Usar L40S con 48GB
```

### faster-whisper STT (producción)

Ya funciona bien en español. Fine-tune opcional para mejorar en telefonía 8kHz:

```
# Whisper fine-tune con audio telefónico real
# Dataset: grabaciones reales del call center (8kHz, compresión telefónica)
# Esto mejoraría WER en el dominio específico de salud/seguros
```

### Notas importantes

- Fine-tune **NO resuelve** el problema de concurrencia (es arquitectural)
- Fine-tune **SÍ mejora**: acento, prosodia, pronunciación de términos de dominio
- Prioridad: primero resolver concurrencia (vLLM-Omni o Higgs V2.5), luego fine-tune
- Los datasets de `OpenSpeechHub` son versiones limpias de datasets públicos — útiles para evitar ruido en entrenamiento

---

### Referencia: catálogo de modelos TTS/STT open source

[awesome-ai-voice](https://github.com/wildminder/awesome-ai-voice) — Lista curada de modelos TTS, voice cloning, música y ASR open source (actualizada marzo 2026).

### Herramientas de referencia

| Herramienta | Descripción | URL |
|-------------|-------------|-----|
| **qwen3-TTS-studio** | UI Gradio profesional para Qwen3-TTS. Voice clone multi-sample, 9 voces preset, voice design, podcasts multi-speaker, Docker. 227★ | [github.com/bc-dunia/qwen3-TTS-studio](https://github.com/bc-dunia/qwen3-TTS-studio) |
| **qwen3-TTS-finetune-studio** | Framework de fine-tuning para Qwen3-TTS. Útil para mejorar acento español | [github.com/bc-dunia/qwen3-TTS-finetune-studio](https://github.com/bc-dunia/qwen3-TTS-finetune-studio) |

---

**Última actualización:** 2026-04-02 19:12
