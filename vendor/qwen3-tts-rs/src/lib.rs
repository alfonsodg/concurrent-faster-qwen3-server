//! # Qwen3-TTS
//!
//! Pure Rust inference for [Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS),
//! a high-quality text-to-speech model from Alibaba.
//!
//! ## Features
//!
//! - **CPU inference** with optional MKL/Accelerate for faster BLAS operations
//! - **CUDA** support for NVIDIA GPU acceleration
//! - **Metal** support for Apple Silicon
//! - **Streaming-friendly** architecture with incremental token generation
//! - **Voice cloning** via ECAPA-TDNN speaker encoder (Base models)
//! - **Auto-detection** of model variant from `config.json`
//!
//! ## Quick Start
//!
//! ```rust,ignore
//! use qwen3_tts::{Qwen3TTS, SynthesisOptions, auto_device};
//!
//! // Load model — variant auto-detected from config.json
//! let device = auto_device()?;
//! let model = Qwen3TTS::from_pretrained("path/to/model", device)?;
//!
//! // Synthesize speech with default settings
//! let audio = model.synthesize("Hello, world!", None)?;
//! audio.save("output.wav")?;
//!
//! // Or with custom options
//! let options = SynthesisOptions {
//!     temperature: 0.8,
//!     top_k: 30,
//!     ..Default::default()
//! };
//! let audio = model.synthesize("Custom settings!", Some(options))?;
//! ```
//!
//! ## Architecture
//!
//! The TTS pipeline consists of three stages:
//!
//! 1. **TalkerModel**: Transformer that generates semantic tokens from text
//!    autoregressively. Uses dual embeddings (text + codec) with MRoPE
//!    (multimodal rotary position encoding) across all variants.
//!
//! 2. **CodePredictor**: For each semantic token, generates 15 acoustic
//!    tokens using a 5-layer autoregressive decoder. The code predictor
//!    always has `hidden_size=1024` regardless of the talker size; 1.7B
//!    models use a `small_to_mtp_projection` layer to bridge the gap.
//!
//! 3. **Decoder12Hz**: Converts the 16-codebook codec tokens to audio
//!    waveform at 24kHz. Uses ConvNeXt blocks and transposed convolutions
//!    for upsampling. Shared across all model variants.
//!
//! ## Model Variants
//!
//! Five official variants exist in two size classes:
//!
//! | Variant | Size | Talker hidden | Speaker conditioning | HuggingFace ID |
//! |---------|------|---------------|---------------------|----------------|
//! | 0.6B Base | 1.8 GB | 1024 | Voice cloning (ECAPA-TDNN) | `Qwen/Qwen3-TTS-12Hz-0.6B-Base` |
//! | 0.6B CustomVoice | 1.8 GB | 1024 | 9 preset speakers | `Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice` |
//! | 1.7B Base | 3.9 GB | 2048 | Voice cloning (ECAPA-TDNN) | `Qwen/Qwen3-TTS-12Hz-1.7B-Base` |
//! | 1.7B CustomVoice | 3.9 GB | 2048 | 9 preset speakers | `Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice` |
//! | 1.7B VoiceDesign | 3.8 GB | 2048 | Text-described voices | `Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign` |
//!
//! **Base**: Includes a speaker encoder for voice cloning from reference audio.
//! Supports x_vector_only (speaker embedding) and ICL (in-context learning
//! with reference audio + text) modes.
//!
//! **CustomVoice**: 9 preset speakers (Serena, Vivian, Ryan, Aiden, etc.) with
//! no speaker encoder. Uses discrete speaker token IDs for voice selection.
//!
//! **VoiceDesign**: Creates novel voices from text descriptions (e.g.,
//! "a deep male voice"). No speaker encoder or preset speakers.
//!
//! All variants share the same speech tokenizer and decoder weights. The
//! code predictor architecture is identical (1024 hidden, 5 layers, 16 heads)
//! across all variants.
//!
//! ## Sample Rate
//!
//! Output audio is always 24kHz mono. Use [`audio::resample()`] if you need
//! a different sample rate.

pub mod audio;
pub mod generation;
#[cfg(feature = "hub")]
pub mod hub;
pub mod models;
pub mod profiling;
pub mod tokenizer;

use anyhow::Result;
use candle_core::{DType, Device, IndexOp, Tensor};
use serde::Serialize;
use std::collections::HashMap;
use std::path::Path;
use std::time::Instant;

use models::codec::{Decoder12Hz, Encoder12Hz};
use models::speaker::SpeakerEncoder;
use models::AnyKVCache;

// SynthesisTiming is defined above, already public.

/// Re-exports for convenience
pub use audio::AudioBuffer;
#[cfg(feature = "hub")]
pub use hub::ModelPaths;
pub use models::config::Qwen3TTSConfig;
// StreamingSession is defined in this module, exported as top-level type
pub use generation::SamplingContext;
pub use models::talker::{codec_tokens, special_tokens, tts_tokens, Language, Speaker};
pub use models::{
    CodePredictor, CodePredictorConfig, ModelType, ParsedModelConfig, SpeakerEncoderConfig,
    TalkerConfig, TalkerModel,
};

/// A sequence of codec frames, where each frame contains 16 codebook values
/// (1 semantic + 15 acoustic, formatted as `[semantic, acoustic_0..14]`).
pub type FrameCodes = Vec<Vec<u32>>;

/// Reference audio prompt for voice cloning.
///
/// Holds the speaker embedding and optional ICL (in-context learning) data.
/// Created via [`Qwen3TTS::create_voice_clone_prompt`].
pub struct VoiceClonePrompt {
    /// Speaker embedding from the ECAPA-TDNN encoder, shape `[enc_dim]` (typically 1024).
    pub speaker_embedding: Tensor,
    /// Reference audio codec codes for ICL mode, shape `[T, 16]`. `None` = x_vector_only mode.
    pub ref_codes: Option<Tensor>,
    /// Tokenized reference text for ICL mode.
    pub ref_text_ids: Option<Vec<u32>>,
}

/// Per-stage timing breakdown from a synthesis run.
#[derive(Debug, Clone, Serialize)]
pub struct SynthesisTiming {
    /// Time spent in the prefill phase (ms).
    pub prefill_ms: f64,
    /// Time spent in the autoregressive generation loop (ms).
    pub generation_ms: f64,
    /// Number of codec frames generated.
    pub generation_frames: usize,
    /// Time spent decoding codec frames to audio (ms).
    pub decode_ms: f64,
}

/// Main TTS interface using proper autoregressive pipeline.
///
/// Supports all 5 Qwen3-TTS model variants. Use [`model_type()`](Self::model_type)
/// to check which variant was loaded and [`supports_voice_cloning()`](Self::supports_voice_cloning)
/// / [`supports_preset_speakers()`](Self::supports_preset_speakers) to check capabilities.
pub struct Qwen3TTS {
    /// Talker model for semantic token generation
    talker: TalkerModel,
    /// Code predictor for acoustic token generation
    code_predictor: CodePredictor,
    /// 12Hz decoder for audio synthesis
    decoder: Decoder12Hz,
    /// Text tokenizer
    text_tokenizer: tokenizer::TextTokenizer,
    /// Speaker encoder for voice cloning (loaded when weights are present)
    speaker_encoder: Option<SpeakerEncoder>,
    /// Speech tokenizer encoder for ICL voice cloning (encodes reference audio → codes)
    speech_encoder: Option<Encoder12Hz>,
    /// Detected model variant (None if loaded without config.json)
    model_type: Option<ModelType>,
    /// Device to run inference on
    device: Device,
    /// Compute dtype for talker + code predictor (BF16 on CUDA, F32 otherwise)
    compute_dtype: DType,
}

// SAFETY: Qwen3TTS weights are read-only after construction.
// All mutable state (KV caches) is allocated per-request as local variables.
// CUDA operations serialize on the GPU stream, so concurrent &self calls are safe.
unsafe impl Send for Qwen3TTS {}
unsafe impl Sync for Qwen3TTS {}

impl Qwen3TTS {
    /// Load a model from a HuggingFace model ID or local path.
    ///
    /// Auto-detects the model variant (0.6B/1.7B, Base/CustomVoice/VoiceDesign)
    /// from `config.json` if present, falling back to weight inspection.
    ///
    /// The text tokenizer is resolved from `model_id/tokenizer.json` if present,
    /// otherwise downloaded from HuggingFace Hub. Use `tokenizer_id` to override.
    pub fn from_pretrained(model_id: &str, device: Device) -> Result<Self> {
        Self::from_pretrained_with_tokenizer(model_id, None, device)
    }

    /// Load a model with an explicit tokenizer source.
    ///
    /// `tokenizer_id` can be a local directory, a file path, or a HuggingFace
    /// model ID (e.g. `"Qwen/Qwen2-0.5B"`). If `None`, resolves from the
    /// model directory or falls back to the default tokenizer repo.
    pub fn from_pretrained_with_tokenizer(
        model_id: &str,
        tokenizer_id: Option<&str>,
        device: Device,
    ) -> Result<Self> {
        tracing::info!("Loading Qwen3-TTS from: {}", model_id);
        tracing::info!("Compute dtype: {:?}", compute_dtype_for_device(&device));

        // Try to parse config.json for auto-detection
        let config_path = Path::new(model_id).join("config.json");
        let parsed_config = if config_path.exists() {
            match ParsedModelConfig::from_file(&config_path) {
                Ok(cfg) => {
                    tracing::info!("Detected model variant: {}", cfg.label());
                    Some(cfg)
                }
                Err(e) => {
                    tracing::warn!(
                        "Failed to parse config.json, falling back to weight inspection: {}",
                        e
                    );
                    None
                }
            }
        } else {
            None
        };

        // Load text tokenizer
        let tok_source = tokenizer_id.unwrap_or(model_id);
        let text_tokenizer = tokenizer::TextTokenizer::from_pretrained(tok_source)?;

        // Load model weights
        let model_path = Path::new(model_id).join("model.safetensors");
        if !model_path.exists() {
            anyhow::bail!(
                "Model weights not found at {}. Please download the model first.",
                model_path.display()
            );
        }
        let weights = Self::load_weights(&model_path, &device)?;

        // Load speech tokenizer for decoder
        let st_path = Path::new(model_id).join("speech_tokenizer/model.safetensors");
        let st_weights = if st_path.exists() {
            Self::load_weights(&st_path, &device)?
        } else {
            // Fall back to looking in parent dir
            let alt_path = Path::new(model_id)
                .parent()
                .map(|p| p.join("speech_tokenizer/model.safetensors"));
            if let Some(p) = alt_path {
                if p.exists() {
                    Self::load_weights(&p, &device)?
                } else {
                    anyhow::bail!("Speech tokenizer weights not found");
                }
            } else {
                anyhow::bail!("Speech tokenizer weights not found");
            }
        };

        Self::build_from_components(
            &weights,
            &st_weights,
            text_tokenizer,
            parsed_config.as_ref(),
            &device,
        )
    }

    /// Load from pre-loaded weight tensors.
    ///
    /// Uses weight inspection for auto-detection. For config.json-based
    /// detection, use [`from_pretrained`](Self::from_pretrained) instead.
    pub fn from_weights(
        model_weights: &HashMap<String, Tensor>,
        decoder_weights: &HashMap<String, Tensor>,
        text_tokenizer: tokenizer::TextTokenizer,
        device: &Device,
    ) -> Result<Self> {
        Self::build_from_components(model_weights, decoder_weights, text_tokenizer, None, device)
    }

    /// Load from downloaded model paths.
    ///
    /// Use with [`ModelPaths::download`] for automatic model downloading.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// use qwen3_tts::{Qwen3TTS, ModelPaths, auto_device};
    ///
    /// let paths = ModelPaths::download(None)?;
    /// let device = auto_device()?;
    /// let model = Qwen3TTS::from_paths(&paths, device)?;
    /// ```
    #[cfg(feature = "hub")]
    pub fn from_paths(paths: &hub::ModelPaths, device: Device) -> Result<Self> {
        tracing::info!("Loading Qwen3-TTS from downloaded paths...");

        let text_tokenizer = tokenizer::TextTokenizer::from_file(&paths.tokenizer)?;
        let weights = Self::load_weights(&paths.model_weights, &device)?;
        let st_weights = Self::load_weights(&paths.decoder_weights, &device)?;

        Self::build_from_components(&weights, &st_weights, text_tokenizer, None, &device)
    }

    /// Shared builder: assembles all model components from pre-loaded weights.
    ///
    /// When `parsed_config` is `Some`, uses config.json dimensions and model type.
    /// When `None`, auto-detects the model variant from weight shapes.
    fn build_from_components(
        model_weights: &HashMap<String, Tensor>,
        decoder_weights: &HashMap<String, Tensor>,
        text_tokenizer: tokenizer::TextTokenizer,
        parsed_config: Option<&ParsedModelConfig>,
        device: &Device,
    ) -> Result<Self> {
        let compute_dtype = compute_dtype_for_device(device);

        // Build TalkerModel
        let talker_config = if let Some(cfg) = parsed_config {
            TalkerConfig::from_parsed(cfg)
        } else {
            Self::detect_talker_config(model_weights)?
        };
        let talker = TalkerModel::from_weights_with_config_dtype(
            model_weights,
            talker_config,
            device,
            compute_dtype,
        )?;

        // Build CodePredictor
        let cp_config = if let Some(cfg) = parsed_config {
            CodePredictorConfig::from_parsed(cfg)
        } else {
            let talker_hidden = talker.config().hidden_size;
            if talker_hidden != 1024 {
                CodePredictorConfig {
                    codec_embed_dim: Some(talker_hidden),
                    ..CodePredictorConfig::default()
                }
            } else {
                CodePredictorConfig::default()
            }
        };
        let cp_weights = Self::filter_weights(model_weights, "talker.code_predictor.");
        let cp_vb = candle_nn::VarBuilder::from_tensors(cp_weights, compute_dtype, device);
        let code_predictor = CodePredictor::new(cp_config, cp_vb)?;

        // Decoder (always F32 — convolutional, no attention)
        let decoder = Decoder12Hz::from_weights(decoder_weights, Default::default())?;

        // Speaker encoder (always F32, only present in Base models)
        let se_config = parsed_config.and_then(|c| c.speaker_encoder_config.clone());
        let speaker_encoder =
            Self::try_load_speaker_encoder(model_weights, se_config.as_ref(), device)?;

        // Speech encoder for ICL voice cloning
        let speech_encoder = Self::try_load_speech_encoder(decoder_weights, device)?;

        let model_type = parsed_config.map(|c| c.model_type);

        Ok(Self {
            talker,
            code_predictor,
            decoder,
            text_tokenizer,
            speaker_encoder,
            speech_encoder,
            model_type,
            device: device.clone(),
            compute_dtype,
        })
    }

    /// Detect talker config from weight shapes (fallback when no config.json).
    fn detect_talker_config(weights: &HashMap<String, Tensor>) -> Result<TalkerConfig> {
        let norm_weight = weights
            .get("talker.model.norm.weight")
            .ok_or_else(|| anyhow::anyhow!("Missing talker.model.norm.weight"))?;
        let hidden_size = norm_weight.dim(0)?;
        Ok(if hidden_size == 2048 {
            TalkerConfig::custom_voice()
        } else {
            TalkerConfig::default()
        })
    }

    /// Returns the detected model type, or `None` if loaded without config.json.
    pub fn model_type(&self) -> Option<&ModelType> {
        self.model_type.as_ref()
    }

    /// Whether this model supports voice cloning (Base models with speaker encoder).
    pub fn supports_voice_cloning(&self) -> bool {
        self.speaker_encoder.is_some()
    }

    /// Whether this model supports preset speaker selection (CustomVoice models).
    ///
    /// Returns `true` for CustomVoice, `false` for Base and VoiceDesign.
    /// When `model_type` is unknown (loaded without config.json), returns `true`
    /// as a permissive default.
    pub fn supports_preset_speakers(&self) -> bool {
        match &self.model_type {
            Some(ModelType::CustomVoice) => true,
            Some(ModelType::Base) | Some(ModelType::VoiceDesign) => false,
            None => true, // permissive when unknown
        }
    }

    /// Whether this model supports voice design (text-described voice conditioning).
    ///
    /// Returns `true` for VoiceDesign, `false` for all other variants.
    pub fn supports_voice_design(&self) -> bool {
        matches!(&self.model_type, Some(ModelType::VoiceDesign))
    }

    /// Synthesize speech from text with default voice (Ryan, English).
    ///
    /// Convenience wrapper around [`synthesize_with_voice`](Self::synthesize_with_voice).
    pub fn synthesize(&self, text: &str, options: Option<SynthesisOptions>) -> Result<AudioBuffer> {
        self.synthesize_with_voice(text, Speaker::Ryan, Language::English, options)
    }

    /// Synthesize speech with per-stage timing breakdown.
    ///
    /// Same as [`synthesize_with_voice`](Self::synthesize_with_voice) but also
    /// returns a [`SynthesisTiming`] with prefill, generation, and decode durations.
    /// Uses [`sync_device`] at timing boundaries for accurate GPU measurements.
    pub fn synthesize_with_timing(
        &self,
        text: &str,
        speaker: Speaker,
        language: Language,
        options: Option<SynthesisOptions>,
    ) -> Result<(AudioBuffer, SynthesisTiming)> {
        #[cfg(feature = "profiling")]
        let _span = tracing::info_span!("synthesize").entered();

        let options = options.unwrap_or_default();
        let mut sampling_ctx = generation::SamplingContext::new(options.seed);
        let input_ids = self.text_tokenizer.encode(text)?;
        let gen_config = options.to_gen_config();

        let (trailing_text_hidden, trailing_text_len, tts_pad_embed) =
            self.build_trailing_text(&input_ids)?;

        // -- Prefill --
        #[cfg(feature = "profiling")]
        let _prefill_span = tracing::info_span!("prefill").entered();

        sync_device(&self.device)?;
        let t_prefill = Instant::now();

        let mut kv_caches = self.talker.new_kv_caches(gen_config.max_new_tokens + 256);
        let (hidden, logits) =
            self.talker
                .prefill_custom_voice(&input_ids, speaker, language, &mut kv_caches)?;
        let prefill_len = hidden.dim(1)?;
        let offset = prefill_len;
        let last_hidden = hidden.i((.., prefill_len - 1..prefill_len, ..))?;

        sync_device(&self.device)?;
        let prefill_ms = t_prefill.elapsed().as_secs_f64() * 1000.0;

        #[cfg(feature = "profiling")]
        drop(_prefill_span);

        // -- Generation --
        let t_gen = Instant::now();

        let all_codes = self.generate_codes(
            &gen_config,
            &mut sampling_ctx,
            &mut kv_caches,
            offset,
            last_hidden,
            &logits,
            &trailing_text_hidden,
            trailing_text_len,
            &tts_pad_embed,
        )?;

        sync_device(&self.device)?;
        let generation_ms = t_gen.elapsed().as_secs_f64() * 1000.0;
        let generation_frames = all_codes.len();

        // -- Decode --
        #[cfg(feature = "profiling")]
        let _decode_span = tracing::info_span!("decode").entered();

        let t_decode = Instant::now();
        let audio = self.decode_codes(&all_codes)?;

        sync_device(&self.device)?;
        let decode_ms = t_decode.elapsed().as_secs_f64() * 1000.0;

        let timing = SynthesisTiming {
            prefill_ms,
            generation_ms,
            generation_frames,
            decode_ms,
        };

        Ok((audio, timing))
    }

    /// Build trailing text embeddings from input token IDs.
    ///
    /// Returns (trailing_text_hidden, trailing_text_len, tts_pad_embed).
    /// The trailing text is: remaining text tokens (all except first) projected + tts_eos.
    /// After trailing text is exhausted, tts_pad is used for each subsequent step.
    fn build_trailing_text(&self, input_ids: &[u32]) -> Result<(Tensor, usize, Tensor)> {
        let trailing_text_hidden = if input_ids.len() > 1 {
            let remaining_proj = self.talker.get_projected_text_embeddings(&input_ids[1..])?;
            let tts_eos_embed = self.talker.get_tts_eos_embed()?;
            Tensor::cat(&[&remaining_proj, &tts_eos_embed], 1)?
        } else {
            self.talker.get_tts_eos_embed()?
        };
        let trailing_text_len = trailing_text_hidden.dim(1)?;
        let tts_pad_embed = self.talker.get_tts_pad_embed()?;
        Ok((trailing_text_hidden, trailing_text_len, tts_pad_embed))
    }

    /// Core generation loop shared by all synthesis methods.
    ///
    /// Runs the autoregressive generation loop: for each frame, check EOS,
    /// generate acoustic codes via CodePredictor, build the residual VQ sum
    /// with trailing text fusion, and sample the next semantic token.
    ///
    /// Callers handle prefill (which varies by model variant) and post-processing
    /// (decode, ICL ref_codes prepending, etc.).
    #[allow(clippy::too_many_arguments)]
    fn generate_codes(
        &self,
        gen_config: &generation::GenerationConfig,
        sampling_ctx: &mut generation::SamplingContext,
        kv_caches: &mut [AnyKVCache],
        mut offset: usize,
        mut last_hidden: Tensor,
        initial_logits: &Tensor,
        trailing_text_hidden: &Tensor,
        trailing_text_len: usize,
        tts_pad_embed: &Tensor,
    ) -> Result<FrameCodes> {
        // Pre-build the token suppression mask once (reused every frame)
        let suppression_mask = generation::build_suppression_mask(
            codec_tokens::CODEC_VOCAB_SIZE,
            CODEC_EOS_TOKEN_ID,
            &self.device,
        )?;

        // GPU-side repetition penalty mask: [1, vocab] — updated incrementally
        // instead of transferring all generated tokens to CPU each frame.
        let vocab_size = codec_tokens::CODEC_VOCAB_SIZE;
        let mut penalty_mask = Tensor::zeros((1, vocab_size), DType::F32, &self.device)?;

        // Pre-allocate code predictor KV caches (reused + reset each frame)
        let mut cp_kv_caches = self.code_predictor.new_kv_caches();

        // Sample first semantic token
        let logits_2d = initial_logits.squeeze(1)?;
        let logits_2d = self.apply_generation_penalties_gpu(
            &logits_2d,
            &penalty_mask,
            gen_config,
            0,
            Some(&suppression_mask),
        )?;
        let mut semantic_token_tensor = generation::sample(&logits_2d, gen_config, sampling_ctx)?;
        tracing::trace!(target: "gpu_sync", "to_vec1 in generate_codes first token");
        let mut semantic_token: u32 = semantic_token_tensor.flatten_all()?.to_vec1::<u32>()?[0];
        // Update penalty mask with this token (O(1) CPU work)
        Self::update_penalty_mask(&mut penalty_mask, semantic_token, vocab_size)?;
        let mut token_count: usize = 1;

        // Accumulate frames as GPU tensors: Vec of [16] U32 tensors
        // Deferred to_vec1 at the end eliminates per-frame acoustic code sync.
        let mut gpu_frames: Vec<Tensor> = Vec::new();

        #[cfg(feature = "profiling")]
        let _gen_span = tracing::info_span!("generate_frames").entered();

        for frame_idx in 0..gen_config.max_new_tokens {
            if let Some(eos_id) = gen_config.eos_token_id {
                if semantic_token == eos_id {
                    break;
                }
            }

            // Embedding lookup using GPU-resident token tensor (no CPU→GPU roundtrip)
            let semantic_embed = self
                .talker
                .get_codec_embedding_from_tensor(&semantic_token_tensor)?;

            #[cfg(feature = "profiling")]
            let _cp_span = tracing::info_span!("code_predictor", frame = frame_idx).entered();

            let acoustic_codes_tensor = self.code_predictor.generate_acoustic_codes(
                &last_hidden,
                &semantic_embed,
                &mut cp_kv_caches,
            )?;

            #[cfg(feature = "profiling")]
            drop(_cp_span);

            // Build [16] frame tensor on GPU: [semantic_token, acoustic_0..14]
            let frame_tensor = Tensor::cat(
                &[&semantic_token_tensor.reshape(1)?, &acoustic_codes_tensor],
                0,
            )?;
            gpu_frames.push(frame_tensor);

            // Use GPU tensor directly for embedding lookup (avoids 15 CPU→GPU transfers)
            let acoustic_embed_sum = self
                .code_predictor
                .get_acoustic_embeddings_sum_from_tensor(&acoustic_codes_tensor)?;
            let summed = semantic_embed.add(&acoustic_embed_sum)?;

            let text_addition = if frame_idx < trailing_text_len {
                trailing_text_hidden.i((.., frame_idx..frame_idx + 1, ..))?
            } else {
                tts_pad_embed.clone()
            };
            let step_input = summed.add(&text_addition)?;

            #[cfg(feature = "profiling")]
            let _talker_span = tracing::info_span!("talker_step", frame = frame_idx).entered();

            let (h, new_logits) =
                self.talker
                    .generate_step_with_embed(&step_input, kv_caches, offset)?;
            offset += 1;
            last_hidden = h;

            #[cfg(feature = "profiling")]
            drop(_talker_span);

            #[cfg(feature = "profiling")]
            let _sample_span = tracing::info_span!("sampling", frame = frame_idx).entered();

            let logits_2d = new_logits.squeeze(1)?;
            let logits_2d = self.apply_generation_penalties_gpu(
                &logits_2d,
                &penalty_mask,
                gen_config,
                token_count,
                Some(&suppression_mask),
            )?;
            semantic_token_tensor = generation::sample(&logits_2d, gen_config, sampling_ctx)?;
            tracing::trace!(target: "gpu_sync", "to_vec1 in generate_codes sampling");
            semantic_token = semantic_token_tensor.flatten_all()?.to_vec1::<u32>()?[0];
            Self::update_penalty_mask(&mut penalty_mask, semantic_token, vocab_size)?;
            token_count += 1;
        }

        // Single GPU→CPU transfer: convert all accumulated GPU frames to FrameCodes
        self.gpu_frames_to_frame_codes(&gpu_frames)
    }

    /// Update the GPU-side penalty mask for a single token ID.
    ///
    /// Sets `penalty_mask[0, token_id] = 1.0` using slice_assign with a
    /// pre-built scalar. This is O(1) CPU work (no GPU→CPU transfer).
    fn update_penalty_mask(
        penalty_mask: &mut Tensor,
        token_id: u32,
        vocab_size: usize,
    ) -> Result<()> {
        let idx = token_id as usize;
        if idx < vocab_size {
            let one = Tensor::ones((1, 1), DType::F32, penalty_mask.device())?;
            *penalty_mask = penalty_mask.slice_assign(&[0..1, idx..idx + 1], &one)?;
        }
        Ok(())
    }

    /// Convert accumulated GPU frame tensors to FrameCodes via a single bulk transfer.
    fn gpu_frames_to_frame_codes(&self, gpu_frames: &[Tensor]) -> Result<FrameCodes> {
        if gpu_frames.is_empty() {
            return Ok(Vec::new());
        }
        // Stack all frames into [n_frames, 16], then single to_vec1
        let stacked = Tensor::stack(gpu_frames, 0)?; // [n_frames, 16]
        let n_frames = stacked.dim(0)?;
        let flat: Vec<u32> = stacked.flatten_all()?.to_vec1()?;
        let mut result = Vec::with_capacity(n_frames);
        for f in 0..n_frames {
            let start = f * 16;
            result.push(flat[start..start + 16].to_vec());
        }
        Ok(result)
    }

    /// Synthesize speech with a specific voice and language.
    ///
    /// Uses the correct generation loop: CustomVoice prefill, autoregressive
    /// semantic tokens, per-frame acoustic code prediction via CodePredictor,
    /// residual VQ summation, and trailing text fusion.
    ///
    /// # Arguments
    ///
    /// * `text` - Text to synthesize
    /// * `speaker` - Predefined speaker voice
    /// * `language` - Target language
    /// * `options` - Synthesis options (temperature, top_k, etc.)
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// use qwen3_tts::{Qwen3TTS, Speaker, Language, SynthesisOptions};
    ///
    /// let audio = model.synthesize_with_voice(
    ///     "Hello, world!",
    ///     Speaker::Ryan,
    ///     Language::English,
    ///     None,
    /// )?;
    /// audio.save("output.wav")?;
    /// ```
    pub fn synthesize_with_voice(
        &self,
        text: &str,
        speaker: Speaker,
        language: Language,
        options: Option<SynthesisOptions>,
    ) -> Result<AudioBuffer> {
        #[cfg(feature = "profiling")]
        let _span = tracing::info_span!("synthesize").entered();

        if let Some(ModelType::Base) = &self.model_type {
            tracing::warn!(
                "Using preset speaker {:?} on a Base model. Base models are trained for \
                 voice cloning, not preset speakers — output will have an unpredictable voice. \
                 Use synthesize_voice_clone() with reference audio instead.",
                speaker
            );
        } else if let Some(ModelType::VoiceDesign) = &self.model_type {
            tracing::warn!(
                "Using preset speaker {:?} on a VoiceDesign model. VoiceDesign models \
                 are trained for text-described voice creation, not preset speakers.",
                speaker
            );
        }

        let options = options.unwrap_or_default();
        let mut sampling_ctx = generation::SamplingContext::new(options.seed);
        let input_ids = self.text_tokenizer.encode(text)?;

        let gen_config = options.to_gen_config();

        let (trailing_text_hidden, trailing_text_len, tts_pad_embed) =
            self.build_trailing_text(&input_ids)?;

        // Prefill with CustomVoice format
        #[cfg(feature = "profiling")]
        let _prefill_span = tracing::info_span!("prefill").entered();

        let mut kv_caches = self.talker.new_kv_caches(gen_config.max_new_tokens + 256);
        let (hidden, logits) =
            self.talker
                .prefill_custom_voice(&input_ids, speaker, language, &mut kv_caches)?;
        let prefill_len = hidden.dim(1)?;
        let offset = prefill_len;
        let last_hidden = hidden.i((.., prefill_len - 1..prefill_len, ..))?;

        #[cfg(feature = "profiling")]
        drop(_prefill_span);

        let all_codes = self.generate_codes(
            &gen_config,
            &mut sampling_ctx,
            &mut kv_caches,
            offset,
            last_hidden,
            &logits,
            &trailing_text_hidden,
            trailing_text_len,
            &tts_pad_embed,
        )?;

        // Decode to audio
        #[cfg(feature = "profiling")]
        let _decode_span = tracing::info_span!("decode").entered();

        self.decode_codes(&all_codes)
    }

    /// Synthesize speech using a text-described voice (VoiceDesign model).
    ///
    /// Uses the same generation loop as [`Self::synthesize_with_voice`] but runs the
    /// VoiceDesign prefill instead of the predefined-speaker prefill. The voice
    /// is conditioned on a natural language description (e.g., "A cheerful young
    /// female voice with high pitch and energetic tone").
    ///
    /// The instruct text is tokenized with ChatML framing:
    /// `<|im_start|>user\n{instruct}<|im_end|>\n`
    ///
    /// # Arguments
    ///
    /// * `text` - Text to synthesize
    /// * `instruct` - Natural language voice description
    /// * `language` - Target language
    /// * `options` - Synthesis options (temperature, top_k, etc.)
    pub fn synthesize_voice_design(
        &self,
        text: &str,
        instruct: &str,
        language: Language,
        options: Option<SynthesisOptions>,
    ) -> Result<AudioBuffer> {
        #[cfg(feature = "profiling")]
        let _span = tracing::info_span!("synthesize").entered();

        if let Some(ref mt) = self.model_type {
            if *mt != ModelType::VoiceDesign {
                tracing::warn!(
                    "Using VoiceDesign synthesis on a {:?} model. This model was not trained \
                     for text-described voice conditioning — output may be unpredictable.",
                    mt
                );
            }
        }

        let options = options.unwrap_or_default();
        let mut sampling_ctx = generation::SamplingContext::new(options.seed);
        let input_ids = self.text_tokenizer.encode(text)?;

        // Tokenize instruct with ChatML user framing: <|im_start|>user\n{instruct}<|im_end|>\n
        let instruct_text = format!("<|im_start|>user\n{}<|im_end|>\n", instruct);
        let instruct_ids = self.text_tokenizer.encode(&instruct_text)?;

        let gen_config = options.to_gen_config();

        let (trailing_text_hidden, trailing_text_len, tts_pad_embed) =
            self.build_trailing_text(&input_ids)?;

        // Prefill with VoiceDesign format
        #[cfg(feature = "profiling")]
        let _prefill_span = tracing::info_span!("prefill").entered();

        let mut kv_caches = self.talker.new_kv_caches(gen_config.max_new_tokens + 256);
        let (hidden, logits) = self.talker.prefill_voice_design(
            &input_ids,
            &instruct_ids,
            language,
            &mut kv_caches,
        )?;
        let prefill_len = hidden.dim(1)?;
        let offset = prefill_len;
        let last_hidden = hidden.i((.., prefill_len - 1..prefill_len, ..))?;

        #[cfg(feature = "profiling")]
        drop(_prefill_span);

        let all_codes = self.generate_codes(
            &gen_config,
            &mut sampling_ctx,
            &mut kv_caches,
            offset,
            last_hidden,
            &logits,
            &trailing_text_hidden,
            trailing_text_len,
            &tts_pad_embed,
        )?;

        // Decode to audio
        #[cfg(feature = "profiling")]
        let _decode_span = tracing::info_span!("decode").entered();

        self.decode_codes(&all_codes)
    }

    /// Convert list of frame codes to tensor [batch, 16, num_frames]
    pub fn codes_to_tensor(&self, codes: &[Vec<u32>]) -> Result<Tensor> {
        codes_to_tensor(codes, &self.device)
    }

    /// Decode raw frame codes to audio.
    ///
    /// Takes a slice of frames (each frame is a `Vec<u32>` of 16 codebook values)
    /// and runs the 12Hz decoder to produce an audio waveform at 24kHz.
    pub fn decode_codes(&self, codes: &[Vec<u32>]) -> Result<AudioBuffer> {
        let tensor = self.codes_to_tensor(codes)?;
        self.decode_tensor(&tensor)
    }

    /// Decode a codes tensor `[1, 16, T]` to audio.
    fn decode_tensor(&self, codes: &Tensor) -> Result<AudioBuffer> {
        let waveform = self.decoder.decode(codes)?;
        AudioBuffer::from_tensor(waveform, 24000)
    }

    /// Synthesize speech using a cloned voice, returning raw codes alongside audio.
    ///
    /// Identical to [`synthesize_voice_clone`](Self::synthesize_voice_clone) but also
    /// returns the raw generated codes (`Vec<Vec<u32>>`) for debugging.
    /// Each inner `Vec<u32>` is one frame: `[semantic, acoustic_0..14]` (16 values).
    pub fn synthesize_voice_clone_debug(
        &self,
        text: &str,
        prompt: &VoiceClonePrompt,
        language: Language,
        options: Option<SynthesisOptions>,
    ) -> Result<(AudioBuffer, FrameCodes)> {
        #[cfg(feature = "profiling")]
        let _span = tracing::info_span!("synthesize").entered();

        let options = options.unwrap_or_default();
        let mut sampling_ctx = generation::SamplingContext::new(options.seed);
        let input_ids = self.text_tokenizer.encode(text)?;

        // Determine if ICL mode is active (ref_codes + ref_text present)
        let is_icl = prompt.ref_codes.is_some() && prompt.ref_text_ids.is_some();

        // ICL mode adjustments (matching mlx-audio):
        let repetition_penalty = if is_icl {
            options.repetition_penalty.max(ICL_MIN_REPETITION_PENALTY)
        } else {
            options.repetition_penalty
        };
        let max_new_tokens = if is_icl {
            // ICL generates frames for ref_text + target_text combined
            let ref_text_len = prompt.ref_text_ids.as_ref().map(|t| t.len()).unwrap_or(0);
            let total_tokens = input_ids.len() + ref_text_len;
            options
                .max_length
                .min(ICL_MIN_FRAMES.max(total_tokens * ICL_FRAMES_PER_TOKEN))
        } else {
            options.max_length
        };
        let mut gen_config = options.to_gen_config();
        gen_config.max_new_tokens = max_new_tokens;
        gen_config.repetition_penalty = repetition_penalty;
        // Prevent premature EOS in voice cloning — but don't force past natural stop
        if is_icl {
            gen_config.min_new_tokens = gen_config.min_new_tokens.max(input_ids.len().max(20));
        }

        // Cast speaker embedding to compute dtype (speaker encoder produces F32)
        let speaker_embed = prompt.speaker_embedding.to_dtype(self.compute_dtype)?;

        // Voice clone prefill (9 positions for ICL, 10 for x_vector_only)
        #[cfg(feature = "profiling")]
        let _prefill_span = tracing::info_span!("prefill").entered();

        // ICL prompt can be very large (ref_text + target_text + ref_codec_frames)
        let icl_extra = if is_icl {
            let ref_frames = prompt.ref_codes.as_ref().map(|c| c.dim(0).unwrap_or(0)).unwrap_or(0);
            let ref_text_len = prompt.ref_text_ids.as_ref().map(|t| t.len()).unwrap_or(0);
            ref_frames + ref_text_len + input_ids.len() + 16
        } else { 0 };
        let mut kv_caches = self.talker.new_kv_caches(gen_config.max_new_tokens + 256 + icl_extra);
        let (hidden, logits) = self.talker.prefill_voice_clone(
            &input_ids,
            &speaker_embed,
            language,
            is_icl,
            &mut kv_caches,
        )?;
        let prefill_len = hidden.dim(1)?;
        let mut offset = prefill_len;

        // Initialize last_hidden from prefill; updated by ICL block if active.
        let mut last_hidden = hidden.i((.., prefill_len - 1..prefill_len, ..))?;

        // ICL extension (if reference codes + text are provided)
        let (trailing_text_hidden, logits) = if let (Some(ref_codes), Some(ref_text_ids)) =
            (&prompt.ref_codes, &prompt.ref_text_ids)
        {
            let ref_codec_embeds = self.sum_ref_codec_embeddings(ref_codes)?;

            // In ICL mode, all text tokens go into the ICL prompt (Python:
            // text_id=input_id[:, 3:-5] passes ALL target text tokens).
            // In the non-ICL path the first text token is consumed by the prefill,
            // so only the remaining tokens go to trailing_text.
            let (icl_embed, icl_trailing) =
                self.talker
                    .build_icl_prompt(&input_ids, ref_text_ids, &ref_codec_embeds, false)?;

            let icl_len = icl_embed.dim(1)?;
            if icl_len > 0 {
                let mask = models::transformer::create_causal_mask(icl_len, offset, &self.device)?;

                let mut icl_hidden = icl_embed;
                for (i, layer) in self.talker.layers_iter().enumerate() {
                    icl_hidden = layer.forward(
                        &icl_hidden,
                        self.talker.rope(),
                        Some(&mask),
                        Some(&mut kv_caches[i]),
                        offset,
                    )?;
                }
                icl_hidden = self.talker.apply_norm(&icl_hidden)?;
                offset += icl_len;

                let last_icl_hidden = icl_hidden.i((.., icl_len - 1..icl_len, ..))?;
                let new_logits = self.talker.apply_codec_head(&last_icl_hidden)?;

                // Update last_hidden so the code predictor is conditioned on
                // the ICL context, not the stale prefill hidden state.
                last_hidden = last_icl_hidden;

                // ICL trailing contains remaining text tokens after overlay
                (icl_trailing, new_logits)
            } else {
                let trailing = self.build_default_trailing_text(&input_ids)?;
                (trailing, logits)
            }
        } else {
            let trailing = self.build_default_trailing_text(&input_ids)?;
            (trailing, logits)
        };

        #[cfg(feature = "profiling")]
        drop(_prefill_span);

        let trailing_text_len = trailing_text_hidden.dim(1)?;
        let tts_pad_embed = self.talker.get_tts_pad_embed()?;

        let all_codes = self.generate_codes(
            &gen_config,
            &mut sampling_ctx,
            &mut kv_caches,
            offset,
            last_hidden,
            &logits,
            &trailing_text_hidden,
            trailing_text_len,
            &tts_pad_embed,
        )?;

        // Prepend ref_codes for ICL decoder context (same fix as synthesize_voice_clone)
        #[cfg(feature = "profiling")]
        let _decode_span = tracing::info_span!("decode").entered();

        let audio = if let (Some(ref_codes), Some(ref_text_ids)) = (&prompt.ref_codes, &prompt.ref_text_ids) {
            let ref_frames = self.tensor_to_frame_codes(ref_codes)?;
            if all_codes.is_empty() {
                AudioBuffer::new(vec![], 24000)
            } else {
                // ICL decode: prepend ref_codes for vocoder continuity, then trim
                // the ref_audio portion using its known duration.
                let ref_audio_samples = ref_codes.dim(0)? * (24000 / 12); // 2000 samples per frame at 24kHz/12Hz

                let mut combined = ref_frames;
                combined.extend(all_codes.iter().cloned());
                let mut audio = self.decode_codes(&combined)?;

                // Trim exactly the ref_audio duration from the start
                let cut = ref_audio_samples.min(audio.samples.len());
                audio.samples = audio.samples[cut..].to_vec();
                audio
            }
        } else {
            self.decode_codes(&all_codes)?
        };
        Ok((audio, all_codes))
    }

    /// Get the device this model is running on
    pub fn device(&self) -> &Device {
        &self.device
    }

    /// Batched synthesis: process N sequences through a single forward pass per frame.
    ///
    /// Prefills are padded to the same length, then the autoregressive loop
    /// runs all sequences in a single batched forward pass per frame.
    pub fn synthesize_batch(
        &self,
        requests: &[(String, Language, Option<SynthesisOptions>)],
    ) -> Result<Vec<AudioBuffer>> {
        let prompts: Vec<Option<&VoiceClonePrompt>> = vec![None; requests.len()];
        self.synthesize_batch_with_voices(requests, &prompts)
    }

    /// Batched synthesis with optional per-request voice clone prompts.
    /// When `voice_prompts[i]` is `Some`, uses that speaker embedding;
    /// when `None`, uses the default Serena voice.
    pub fn synthesize_batch_with_voices(
        &self,
        requests: &[(String, Language, Option<SynthesisOptions>)],
        voice_prompts: &[Option<&VoiceClonePrompt>],
    ) -> Result<Vec<AudioBuffer>> {
        if requests.is_empty() {
            return Ok(vec![]);
        }
        if requests.len() == 1 {
            let (text, lang, opts) = &requests[0];
            if let Some(prompt) = voice_prompts.first().and_then(|p| *p) {
                return Ok(vec![self.synthesize_voice_clone(text, prompt, *lang, opts.clone())?]);
            }
            let audio = self.synthesize_with_voice(text, Speaker::Serena, *lang, opts.clone())?;
            return Ok(vec![audio]);
        }

        let n = requests.len();
        let opts0 = requests[0].2.clone().unwrap_or_default();
        let gen_config = opts0.to_gen_config();

        // Phase 1: Build prefill embeddings for each sequence
        // Pre-compute shared tensors once (identical across all requests)
        let role_prefix = self.talker.build_role_prefix_pub()?;
        let tts_text_embed = self.talker.build_tts_pad_bos_pub(5)?;

        // Pre-compute Serena codec embeddings (reused for non-voice-clone requests)
        let serena_codec_ids_base = Tensor::new(
            &[
                codec_tokens::CODEC_THINK, codec_tokens::CODEC_THINK_BOS,
                0u32, // placeholder for language — filled per request
                codec_tokens::CODEC_THINK_EOS, Speaker::Serena.token_id(),
                codec_tokens::CODEC_PAD, codec_tokens::CODEC_BOS,
            ],
            &self.device,
        )?;
        // Voice clone prefix tokens (without speaker)
        let vc_prefix_ids = Tensor::new(
            &[codec_tokens::CODEC_THINK, codec_tokens::CODEC_THINK_BOS, 0u32, codec_tokens::CODEC_THINK_EOS],
            &self.device,
        )?;
        let vc_suffix_ids = Tensor::new(
            &[codec_tokens::CODEC_PAD, codec_tokens::CODEC_BOS],
            &self.device,
        )?;
        let vc_suffix_embed = self.talker.codec_embedding_forward(&vc_suffix_ids)?.unsqueeze(0)?;

        let mut prefill_embeds: Vec<Tensor> = Vec::with_capacity(n);
        let mut all_input_ids: Vec<Vec<u32>> = Vec::with_capacity(n);

        for (idx, (text, lang, _opts)) in requests.iter().enumerate() {
            let input_ids = self.text_tokenizer.encode(text)?;

            let (codec_hidden, codec_bos_embed) = if let Some(prompt) = voice_prompts.get(idx).and_then(|p| *p) {
                // Voice clone path
                let mut pids: Vec<u32> = vc_prefix_ids.to_vec1()?;
                pids[2] = lang.token_id();
                let prefix_ids = Tensor::new(pids.as_slice(), &self.device)?;
                let prefix_embed = self.talker.codec_embedding_forward(&prefix_ids)?.unsqueeze(0)?;
                let speaker = prompt.speaker_embedding.to_dtype(self.compute_dtype)?
                    .reshape((1, 1, self.talker.config().hidden_size))?;
                let codec_embed = Tensor::cat(&[&prefix_embed, &speaker, &vc_suffix_embed], 1)?;
                let codec_first6 = codec_embed.i((.., ..6, ..))?;
                let ch = tts_text_embed.add(&codec_first6)?;
                let bos = codec_embed.i((.., 6..7, ..))?;
                (ch, bos)
            } else {
                // Serena path — only recompute language token
                let mut ids: Vec<u32> = serena_codec_ids_base.to_vec1()?;
                ids[2] = lang.token_id();
                let codec_ids = Tensor::new(ids.as_slice(), &self.device)?;
                let codec_embed = self.talker.codec_embedding_forward(&codec_ids)?.unsqueeze(0)?;
                let codec_first6 = codec_embed.i((.., ..6, ..))?;
                let ch = tts_text_embed.add(&codec_first6)?;
                let bos = codec_embed.i((.., 6..7, ..))?;
                (ch, bos)
            };

            let mut hidden = Tensor::cat(&[&role_prefix, &codec_hidden], 1)?;
            if let Some(combined) = self.talker.build_first_text_combined_pub(&input_ids, &codec_bos_embed)? {
                hidden = Tensor::cat(&[&hidden, &combined], 1)?;
            }

            prefill_embeds.push(hidden); // [1, seq_len_i, hidden]
            all_input_ids.push(input_ids);
        }

        // Pad all prefill embeddings to the same length
        let max_prefill_len = prefill_embeds.iter().map(|e| e.dim(1).unwrap_or(0)).max().unwrap_or(0);
        let hidden_size = prefill_embeds[0].dim(2)?;

        let mut padded_embeds: Vec<Tensor> = Vec::with_capacity(n);
        let mut attention_masks: Vec<Vec<f32>> = Vec::with_capacity(n);

        for embed in &prefill_embeds {
            let seq_len = embed.dim(1)?;
            let pad_len = max_prefill_len - seq_len;

            if pad_len > 0 {
                let padding = Tensor::zeros((1, pad_len, hidden_size), embed.dtype(), &self.device)?;
                padded_embeds.push(Tensor::cat(&[&padding, embed], 1)?); // left-pad
            } else {
                padded_embeds.push(embed.clone());
            }

            // Attention mask: 0 for padding (left), 1 for real tokens (right)
            let mut mask = vec![0.0f32; pad_len];
            mask.extend(vec![1.0f32; seq_len]);
            attention_masks.push(mask);
        }

        // Stack into batch: [N, max_prefill_len, hidden]
        let batched_embed = Tensor::cat(&padded_embeds, 0)?;

        // Build causal attention mask with padding: [N, 1, max_prefill_len, max_prefill_len]
        let attn_mask = self.build_batched_causal_mask(&attention_masks, max_prefill_len)?;

        // Phase 2: Batched prefill through transformer
        // Use pre-allocated KV caches with batch=N for zero-copy CUDA writes
        let max_gen = gen_config.max_new_tokens;
        let mut kv_caches = self.talker.new_kv_caches_batched(n, max_gen + max_prefill_len + 16);

        let mut hidden = batched_embed;
        for (i, layer) in self.talker.layers_iter().enumerate() {
            hidden = layer.forward(&hidden, self.talker.rope(), Some(&attn_mask), Some(&mut kv_caches[i]), 0)?;
        }
        hidden = self.talker.apply_norm(&hidden)?;

        // Extract last real position for each sequence
        let mut last_hiddens: Vec<Tensor> = Vec::with_capacity(n);
        let mut last_logits: Vec<Tensor> = Vec::with_capacity(n);
        for i in 0..n {
            let h_i = hidden.i(i..i + 1)?; // [1, max_prefill_len, hidden]
            let last_h = h_i.i((.., max_prefill_len - 1..max_prefill_len, ..))?;
            let logits = self.talker.apply_codec_head(&last_h)?;
            last_hiddens.push(last_h);
            last_logits.push(logits);
        }

        // Phase 3: Batched autoregressive generation
        let tts_pad_embed = self.talker.get_tts_pad_embed()?;
        let suppression_mask = generation::build_suppression_mask(
            codec_tokens::CODEC_VOCAB_SIZE, CODEC_EOS_TOKEN_ID, &self.device,
        )?;

        let mut all_codes: Vec<FrameCodes> = (0..n).map(|_| Vec::new()).collect();
        let mut done: Vec<bool> = vec![false; n];

        // Build trailing text for each sequence
        let mut all_trailing: Vec<Tensor> = Vec::with_capacity(n);
        let mut all_trailing_len: Vec<usize> = Vec::with_capacity(n);
        for ids in &all_input_ids {
            let trailing = self.build_default_trailing_text(ids)?;
            let tlen = trailing.dim(1)?;
            all_trailing.push(trailing);
            all_trailing_len.push(tlen);
        }

        // Batched initial token sampling (same greedy path as main loop)
        let init_logits = Tensor::cat(&last_logits, 0)?.squeeze(1)?.to_dtype(candle_core::DType::F32)?;
        let init_suppressed = generation::apply_token_suppression_with_mask(&init_logits, &suppression_mask)?;
        let init_tokens = init_suppressed.argmax(candle_core::D::Minus1)?;
        let init_ids: Vec<u32> = init_tokens.flatten_all()?.to_vec1()?;

        let mut semantic_tokens: Vec<Tensor> = Vec::with_capacity(n);
        let mut semantic_ids: Vec<u32> = Vec::with_capacity(n);
        for (i, &id) in init_ids.iter().enumerate().take(n) {
            semantic_tokens.push(init_tokens.i(i)?);
            semantic_ids.push(id);
        }

        // Pre-allocate zero tensor for done sequences (avoid per-frame allocation)
        let zero_input = Tensor::zeros((1, 1, hidden_size), self.compute_dtype, &self.device)?;

        let mut offset = max_prefill_len;

        for frame_idx in 0..gen_config.max_new_tokens {
            // Check EOS
            for i in 0..n {
                if !done[i] {
                    if let Some(eos_id) = gen_config.eos_token_id {
                        if semantic_ids[i] == eos_id { done[i] = true; }
                    }
                }
            }
            if done.iter().all(|&d| d) { break; }

            // BATCHED code predictor: all active sequences in one pass
            let active: Vec<usize> = (0..n).filter(|&i| !done[i]).collect();

            // Semantic embedding lookup per active sequence
            let active_semantic_embeds: Vec<Tensor> = active.iter()
                .map(|&i| self.talker.get_codec_embedding_from_tensor(&semantic_tokens[i]))
                .collect::<Result<Vec<_>>>()?;
            let active_hiddens: Vec<Tensor> = active.iter()
                .map(|&i| last_hiddens[i].clone())
                .collect();
            let batched_hiddens = Tensor::cat(&active_hiddens, 0)?;
            let batched_sem_embeds = Tensor::cat(&active_semantic_embeds, 0)?;

            let batched_acoustic = self.code_predictor.generate_acoustic_codes_batched(
                &batched_hiddens, &batched_sem_embeds,
            )?;

            // Build step inputs from batched results
            let mut step_inputs: Vec<Tensor> = Vec::with_capacity(n);
            let mut active_idx = 0;
            // Store frame tensors on GPU, defer CPU transfer
            let mut gpu_frames: Vec<Option<Tensor>> = vec![None; n];
            for i in 0..n {
                if done[i] {
                    step_inputs.push(zero_input.clone());
                    continue;
                }

                let acoustic_codes = &batched_acoustic[active_idx];
                let semantic_embed_i = &active_semantic_embeds[active_idx];
                active_idx += 1;

                // Keep frame on GPU — defer to_vec1 to end
                let frame_tensor = Tensor::cat(&[&semantic_tokens[i].reshape(1)?, acoustic_codes], 0)?;
                gpu_frames[i] = Some(frame_tensor);

                let acoustic_sum = self.code_predictor.get_acoustic_embeddings_sum_from_tensor(acoustic_codes)?;
                let summed = semantic_embed_i.add(&acoustic_sum)?;

                let text_addition = if frame_idx < all_trailing_len[i] {
                    all_trailing[i].i((.., frame_idx..frame_idx + 1, ..))?
                } else {
                    tts_pad_embed.clone()
                };
                step_inputs.push(summed.add(&text_addition)?);
            }

            // BATCHED FORWARD: [N, 1, hidden] through transformer
            let batched_input = Tensor::cat(&step_inputs, 0)?;
            let mut batched_hidden = batched_input;

            for (layer_idx, layer) in self.talker.layers_iter().enumerate() {
                batched_hidden = layer.forward(
                    &batched_hidden,
                    self.talker.rope(),
                    None,
                    Some(&mut kv_caches[layer_idx]),
                    offset,
                )?;
            }
            offset += 1;

            batched_hidden = self.talker.apply_norm(&batched_hidden)?;
            let batched_logits = self.talker.apply_codec_head(&batched_hidden)?;

            // Batched greedy sampling: apply suppression + argmax in one pass
            let logits_2d = batched_logits.squeeze(1)?.to_dtype(candle_core::DType::F32)?; // [N, vocab]
            let suppressed = generation::apply_token_suppression_with_mask(&logits_2d, &suppression_mask)?;
            let all_new_tokens = suppressed.argmax(candle_core::D::Minus1)?; // [N]

            // Update per-sequence state
            let mut new_tokens: Vec<Tensor> = Vec::with_capacity(n);
            #[allow(clippy::needless_range_loop)]
            for i in 0..n {
                let tok = all_new_tokens.i(i)?;
                new_tokens.push(tok.clone());
                last_hiddens[i] = batched_hidden.i(i..i + 1)?;
            }

            // Batched EOS check: stack all tokens → single GPU→CPU transfer
            let stacked = Tensor::stack(&new_tokens, 0)?.flatten_all()?;
            let all_ids: Vec<u32> = stacked.to_vec1()?;
            #[allow(clippy::manual_memcpy)]
            for i in 0..n {
                semantic_tokens[i] = new_tokens[i].clone();
                semantic_ids[i] = all_ids[i];
            }

            // Bulk transfer frame codes (single sync)
            if !gpu_frames.iter().all(|f| f.is_none()) {
                let mut to_transfer: Vec<(usize, &Tensor)> = Vec::new();
                #[allow(clippy::needless_range_loop)]
                for i in 0..n {
                    if let Some(ref frame) = gpu_frames[i] {
                        to_transfer.push((i, frame));
                    }
                }
                if !to_transfer.is_empty() {
                    let stacked_frames = Tensor::stack(
                        &to_transfer.iter().map(|(_, f)| *f).collect::<Vec<_>>(), 0
                    )?;
                    let all_frame_data: Vec<u32> = stacked_frames.flatten_all()?.to_vec1()?;
                    let codes_per_frame = 16;
                    for (idx, (i, _)) in to_transfer.iter().enumerate() {
                        let start = idx * codes_per_frame;
                        all_codes[*i].push(all_frame_data[start..start + codes_per_frame].to_vec());
                    }
                }
                for f in gpu_frames.iter_mut() { *f = None; }
            }
        }

        // Phase 4: Batched vocoder decode
        let non_empty: Vec<(usize, &Vec<Vec<u32>>)> = all_codes.iter().enumerate()
            .filter(|(_, c)| !c.is_empty()).collect();

        let mut results: Vec<AudioBuffer> = (0..n).map(|_| AudioBuffer::new(vec![], 24000)).collect();

        if non_empty.len() > 1 {
            // Batch decode: pad to max frames, stack [N, 16, T_max], single vocoder pass
            let max_frames = non_empty.iter().map(|(_, c)| c.len()).max().unwrap_or(0);
            let frame_lens: Vec<usize> = non_empty.iter().map(|(_, c)| c.len()).collect();

            let mut batch_data = vec![0i64; non_empty.len() * 16 * max_frames];
            for (batch_idx, (_, codes)) in non_empty.iter().enumerate() {
                for (frame, frame_codes) in codes.iter().enumerate() {
                    for (q, &code) in frame_codes.iter().enumerate() {
                        batch_data[batch_idx * 16 * max_frames + q * max_frames + frame] = code as i64;
                    }
                }
            }
            let batched_tensor = Tensor::from_vec(
                batch_data, (non_empty.len(), 16, max_frames), &self.device,
            )?;
            let waveform = self.decoder.decode(&batched_tensor)?; // [N, 1, total_samples]
            let samples_per_frame = 24000 / 12; // 2000 samples per frame at 24kHz/12Hz
            for (batch_idx, (orig_idx, _)) in non_empty.iter().enumerate() {
                let actual_samples = frame_lens[batch_idx] * samples_per_frame;
                let wav_i = waveform.i(batch_idx)?.flatten_all()?;
                let trim_len = actual_samples.min(wav_i.dim(0)?);
                let trimmed = wav_i.narrow(0, 0, trim_len)?;
                results[*orig_idx] = AudioBuffer::from_tensor(trimmed, 24000)?;
            }
        } else {
            for (orig_idx, codes) in &non_empty {
                results[*orig_idx] = self.decode_codes(codes)?;
            }
        }
        Ok(results)
    }

    /// Build a batched causal attention mask with padding support.
    /// Returns [N, 1, seq_len, seq_len] mask where padding positions are -inf.
    #[allow(clippy::needless_range_loop)]
    fn build_batched_causal_mask(
        &self,
        attention_masks: &[Vec<f32>],
        seq_len: usize,
    ) -> Result<Tensor> {
        let n = attention_masks.len();
        let mut mask_data = vec![0.0f32; n * seq_len * seq_len];

        for b in 0..n {
            for i in 0..seq_len {
                for j in 0..seq_len {
                    let idx = b * seq_len * seq_len + i * seq_len + j;
                    if j > i {
                        // Future position: mask
                        mask_data[idx] = f32::NEG_INFINITY;
                    } else if attention_masks[b][j] == 0.0 {
                        // Padding position: mask
                        mask_data[idx] = f32::NEG_INFINITY;
                    }
                }
            }
        }

        let mask = Tensor::from_vec(mask_data, (n, seq_len, seq_len), &self.device)?;
        Ok(mask.unsqueeze(1)?) // [N, 1, seq_len, seq_len]
    }

    /// Batched streaming synthesis: generates audio for N sequences simultaneously,
    /// sending decoded chunks through per-request channels every `chunk_frames` frames.
    ///
    /// Each sender receives `Ok(AudioBuffer)` chunks as they're generated,
    /// then the channel is dropped when generation completes.
    pub fn synthesize_batch_streaming(
        &self,
        requests: &[(String, Language, Option<SynthesisOptions>)],
        senders: &[std::sync::mpsc::Sender<AudioBuffer>],
        chunk_frames: usize,
        voice_prompts: &[Option<&VoiceClonePrompt>],
        stop_flags: &[&std::sync::atomic::AtomicBool],
    ) -> Result<()> {
        let n = requests.len();
        let reqs: Vec<(String, Language, Option<SynthesisOptions>)> = requests.to_vec();

        // Use per-request options or defaults
        let opts0 = reqs[0].2.clone().unwrap_or_default();
        let mut gen_config = opts0.to_gen_config();
        // Use max_length from options (set by caller via adaptive_max_length)
        // Safety cap: 120 frames ≈ 10s max audio — prevents runaway generation
        gen_config.max_new_tokens = gen_config.max_new_tokens.min(120);

        // Phase 1: Build prefill embeddings (with voice clone support)
        let role_prefix = self.talker.build_role_prefix_pub()?;
        let tts_text_embed = self.talker.build_tts_pad_bos_pub(5)?;
        let vc_prefix_ids = Tensor::new(
            &[codec_tokens::CODEC_THINK, codec_tokens::CODEC_THINK_BOS, 0u32, codec_tokens::CODEC_THINK_EOS],
            &self.device,
        )?;
        let vc_suffix_ids = Tensor::new(
            &[codec_tokens::CODEC_PAD, codec_tokens::CODEC_BOS],
            &self.device,
        )?;
        let vc_suffix_embed = self.talker.codec_embedding_forward(&vc_suffix_ids)?.unsqueeze(0)?;

        let mut prefill_embeds: Vec<Tensor> = Vec::with_capacity(n);
        let mut all_input_ids: Vec<Vec<u32>> = Vec::with_capacity(n);

        for (idx, (text, lang, _)) in reqs.iter().enumerate() {
            let input_ids = self.text_tokenizer.encode(text)?;

            let (codec_hidden, codec_bos_embed) = if let Some(prompt) = voice_prompts.get(idx).and_then(|p| *p) {
                // Voice clone path
                let mut pids: Vec<u32> = vc_prefix_ids.to_vec1()?;
                pids[2] = lang.token_id();
                let prefix_ids = Tensor::new(pids.as_slice(), &self.device)?;
                let prefix_embed = self.talker.codec_embedding_forward(&prefix_ids)?.unsqueeze(0)?;
                let speaker = prompt.speaker_embedding.to_dtype(self.compute_dtype)?
                    .reshape((1, 1, self.talker.config().hidden_size))?;
                let codec_embed = Tensor::cat(&[&prefix_embed, &speaker, &vc_suffix_embed], 1)?;
                let first6 = codec_embed.i((.., ..6, ..))?;
                (tts_text_embed.add(&first6)?, codec_embed.i((.., 6..7, ..))?)
            } else {
                // Default Serena path
                let codec_ids = Tensor::new(
                    &[codec_tokens::CODEC_THINK, codec_tokens::CODEC_THINK_BOS,
                      lang.token_id(), codec_tokens::CODEC_THINK_EOS,
                      Speaker::Serena.token_id(), codec_tokens::CODEC_PAD, codec_tokens::CODEC_BOS],
                    &self.device,
                )?;
                let codec_embed = self.talker.codec_embedding_forward(&codec_ids)?.unsqueeze(0)?;
                let first6 = codec_embed.i((.., ..6, ..))?;
                (tts_text_embed.add(&first6)?, codec_embed.i((.., 6..7, ..))?)
            };

            let mut hidden = Tensor::cat(&[&role_prefix, &codec_hidden], 1)?;
            if let Some(combined) = self.talker.build_first_text_combined_pub(&input_ids, &codec_bos_embed)? {
                hidden = Tensor::cat(&[&hidden, &combined], 1)?;
            }
            prefill_embeds.push(hidden);
            all_input_ids.push(input_ids);
        }

        // Pad prefills
        let max_prefill_len = prefill_embeds.iter().map(|e| e.dim(1).unwrap_or(0)).max().unwrap_or(0);
        let hidden_size = prefill_embeds[0].dim(2)?;
        let mut padded: Vec<Tensor> = Vec::with_capacity(n);
        let mut masks: Vec<Vec<f32>> = Vec::with_capacity(n);
        for embed in &prefill_embeds {
            let seq = embed.dim(1)?;
            let pad = max_prefill_len - seq;
            if pad > 0 {
                let p = Tensor::zeros((1, pad, hidden_size), embed.dtype(), &self.device)?;
                padded.push(Tensor::cat(&[&p, embed], 1)?);
            } else {
                padded.push(embed.clone());
            }
            let mut m = vec![0.0f32; pad];
            m.extend(vec![1.0f32; seq]);
            masks.push(m);
        }

        let batched_embed = Tensor::cat(&padded, 0)?;
        let attn_mask = self.build_batched_causal_mask(&masks, max_prefill_len)?;

        // Phase 2: Batched prefill
        let num_layers = self.talker.layers_iter().count();
        let mut kv_caches: Vec<models::transformer::AnyKVCache> = (0..num_layers)
            .map(|_| models::transformer::AnyKVCache::Concat(models::kv_cache::KVCache::new()))
            .collect();

        let mut hidden = batched_embed;
        for (i, layer) in self.talker.layers_iter().enumerate() {
            hidden = layer.forward(&hidden, self.talker.rope(), Some(&attn_mask), Some(&mut kv_caches[i]), 0)?;
        }
        hidden = self.talker.apply_norm(&hidden)?;

        let mut last_hiddens: Vec<Tensor> = Vec::with_capacity(n);
        let mut last_logits: Vec<Tensor> = Vec::with_capacity(n);
        for i in 0..n {
            let h_i = hidden.i(i..i + 1)?;
            let last_h = h_i.i((.., max_prefill_len - 1..max_prefill_len, ..))?;
            let logits = self.talker.apply_codec_head(&last_h)?;
            last_hiddens.push(last_h);
            last_logits.push(logits);
        }

        // Setup generation state
        let tts_pad_embed = self.talker.get_tts_pad_embed()?;
        let suppression_mask = generation::build_suppression_mask(
            codec_tokens::CODEC_VOCAB_SIZE, CODEC_EOS_TOKEN_ID, &self.device,
        )?;
        let mut done: Vec<bool> = vec![false; n];
        let mut penalty_masks: Vec<Tensor> = (0..n)
            .map(|_| Tensor::zeros((1, codec_tokens::CODEC_VOCAB_SIZE), DType::F32, &self.device))
            .collect::<std::result::Result<Vec<_>, _>>()?;
        let mut sampling_ctxs: Vec<generation::SamplingContext> = (0..n)
            .map(|_| generation::SamplingContext::new(None)).collect();
        let mut all_trailing: Vec<Tensor> = Vec::with_capacity(n);
        let mut all_trailing_len: Vec<usize> = Vec::with_capacity(n);
        for ids in &all_input_ids {
            let trailing = self.build_default_trailing_text(ids)?;
            let tlen = trailing.dim(1)?;
            all_trailing.push(trailing);
            all_trailing_len.push(tlen);
        }

        let mut semantic_tokens: Vec<Tensor> = Vec::with_capacity(n);
        let mut semantic_ids: Vec<u32> = Vec::with_capacity(n);
        for i in 0..n {
            let logits_2d = last_logits[i].squeeze(1)?;
            let logits_2d = self.apply_generation_penalties_gpu(
                &logits_2d, &penalty_masks[i], &gen_config, 0, Some(&suppression_mask),
            )?;
            let tok = generation::sample(&logits_2d, &gen_config, &mut sampling_ctxs[i])?;
            let id: u32 = tok.flatten_all()?.to_vec1::<u32>()?[0];
            Self::update_penalty_mask(&mut penalty_masks[i], id, codec_tokens::CODEC_VOCAB_SIZE)?;
            semantic_tokens.push(tok);
            semantic_ids.push(id);
        }

        let mut offset = max_prefill_len;

        // Frame buffers for chunked decode
        let mut frame_buffers: Vec<Vec<Vec<u32>>> = (0..n).map(|_| Vec::new()).collect();

        // Pre-allocate zero tensor for done sequences
        let zero_input = Tensor::zeros((1, 1, hidden_size), self.compute_dtype, &self.device)?;

        // Fixed-window vocoder context: decode chunk with last N context frames prepended,
        // trim context audio. O(1) per chunk, seamless boundaries.
        let vocoder_context_frames: usize = 4;
        let samples_per_frame: usize = 24000 / 12; // 2000
        let crossfade_len: usize = 48; // ~2ms at 24kHz — eliminates boundary clicks
        let mut prev_frames: Vec<Vec<Vec<u32>>> = (0..n).map(|_| Vec::new()).collect();
        let mut prev_tail: Vec<Vec<f32>> = (0..n).map(|_| Vec::new()).collect();

        // Early stopping: detect token repetition (model stuck in loop)
        let rep_threshold: usize = 3; // stop after 3 consecutive identical tokens
        let mut rep_counts: Vec<usize> = vec![0; n];
        let mut prev_ids: Vec<u32> = vec![u32::MAX; n];

        // Phase 3: Batched generation with streaming decode
        for frame_idx in 0..gen_config.max_new_tokens {
            for i in 0..n {
                if !done[i] {
                    // EOS detection
                    let is_eos = gen_config.eos_token_id.map_or(false, |eos| semantic_ids[i] == eos);
                    // Repetition detection
                    if semantic_ids[i] == prev_ids[i] {
                        rep_counts[i] += 1;
                    } else {
                        rep_counts[i] = 0;
                        prev_ids[i] = semantic_ids[i];
                    }
                    let is_stuck = rep_counts[i] >= rep_threshold;
                    let is_stopped = stop_flags.get(i).map_or(false, |f| f.load(std::sync::atomic::Ordering::Relaxed));

                    if is_eos || is_stuck || is_stopped {
                        done[i] = true;
                        // Flush buffered frames immediately (with context)
                        if !frame_buffers[i].is_empty() {
                            let ctx = &prev_frames[i];
                            let mut decode_input = ctx.clone();
                            decode_input.extend(frame_buffers[i].drain(..));
                            let ctx_samples = ctx.len() * samples_per_frame;
                            if let Ok(audio) = self.decode_codes(&decode_input) {
                                if ctx_samples < audio.samples.len() {
                                    let mut chunk = audio.samples[ctx_samples..].to_vec();
                                    let tail = &prev_tail[i];
                                    let fade = tail.len().min(chunk.len()).min(crossfade_len);
                                    for j in 0..fade {
                                        let t = (j + 1) as f32 / (fade + 1) as f32;
                                        chunk[j] = tail[tail.len() - fade + j] * (1.0 - t) + chunk[j] * t;
                                    }
                                    let _ = senders[i].send(AudioBuffer::new(chunk, audio.sample_rate));
                                }
                            }
                        }
                    }
                }
            }
            if done.iter().all(|&d| d) { break; }

            // BATCHED code predictor: all active sequences in one pass
            let active: Vec<usize> = (0..n).filter(|&i| !done[i]).collect();

            let active_semantic_embeds: Vec<Tensor> = active.iter()
                .map(|&i| self.talker.get_codec_embedding_from_tensor(&semantic_tokens[i]))
                .collect::<Result<Vec<_>>>()?;
            let active_hiddens: Vec<Tensor> = active.iter()
                .map(|&i| last_hiddens[i].clone()).collect();
            let batched_hiddens = Tensor::cat(&active_hiddens, 0)?;
            let batched_sem_embeds = Tensor::cat(&active_semantic_embeds, 0)?;

            let batched_acoustic = self.code_predictor.generate_acoustic_codes_batched(
                &batched_hiddens, &batched_sem_embeds,
            )?;

            let mut step_inputs: Vec<Tensor> = Vec::with_capacity(n);
            let mut active_idx = 0;
            for i in 0..n {
                if done[i] {
                    step_inputs.push(zero_input.clone());
                    continue;
                }

                let acoustic_codes = &batched_acoustic[active_idx];
                let semantic_embed_i = &active_semantic_embeds[active_idx];
                active_idx += 1;

                let frame_tensor = Tensor::cat(&[&semantic_tokens[i].reshape(1)?, acoustic_codes], 0)?;
                let frame_vec: Vec<u32> = frame_tensor.to_vec1()?;
                frame_buffers[i].push(frame_vec);

                let acoustic_sum = self.code_predictor.get_acoustic_embeddings_sum_from_tensor(acoustic_codes)?;
                let summed = semantic_embed_i.add(&acoustic_sum)?;
                let text_addition = if frame_idx < all_trailing_len[i] {
                    all_trailing[i].i((.., frame_idx..frame_idx + 1, ..))?
                } else { tts_pad_embed.clone() };
                step_inputs.push(summed.add(&text_addition)?);
            }

            // Batched forward
            let batched_input = Tensor::cat(&step_inputs, 0)?;
            let mut batched_hidden = batched_input;
            for (layer_idx, layer) in self.talker.layers_iter().enumerate() {
                batched_hidden = layer.forward(
                    &batched_hidden, self.talker.rope(), None, Some(&mut kv_caches[layer_idx]), offset,
                )?;
            }
            offset += 1;
            batched_hidden = self.talker.apply_norm(&batched_hidden)?;
            let batched_logits = self.talker.apply_codec_head(&batched_hidden)?;

            for i in 0..n {
                if done[i] { continue; }
                last_hiddens[i] = batched_hidden.i(i..i + 1)?;
                let logits_i = batched_logits.i(i..i + 1)?.squeeze(1)?;
                let logits_i = self.apply_generation_penalties_gpu(
                    &logits_i, &penalty_masks[i], &gen_config, frame_idx + 1, Some(&suppression_mask),
                )?;
                semantic_tokens[i] = generation::sample(&logits_i, &gen_config, &mut sampling_ctxs[i])?;
                semantic_ids[i] = semantic_tokens[i].flatten_all()?.to_vec1::<u32>()?[0];
                Self::update_penalty_mask(&mut penalty_masks[i], semantic_ids[i], codec_tokens::CODEC_VOCAB_SIZE)?;
            }

            // Decode and send chunks with fixed vocoder context window
            if (frame_idx + 1) % chunk_frames == 0 {
                for i in 0..n {
                    if done[i] || frame_buffers[i].is_empty() { continue; }
                    let ctx = &prev_frames[i];
                    let mut decode_input = ctx.clone();
                    decode_input.extend(frame_buffers[i].iter().cloned());
                    let ctx_samples = ctx.len() * samples_per_frame;

                    if let Ok(audio) = self.decode_codes(&decode_input) {
                        if ctx_samples < audio.samples.len() {
                            let mut chunk = audio.samples[ctx_samples..].to_vec();
                            // Cross-fade with previous chunk tail to eliminate boundary click
                            let tail = &prev_tail[i];
                            let fade = tail.len().min(chunk.len()).min(crossfade_len);
                            for j in 0..fade {
                                let t = (j + 1) as f32 / (fade + 1) as f32;
                                chunk[j] = tail[tail.len() - fade + j] * (1.0 - t) + chunk[j] * t;
                            }
                            if chunk.len() >= crossfade_len {
                                prev_tail[i] = chunk[chunk.len() - crossfade_len..].to_vec();
                                chunk.truncate(chunk.len() - crossfade_len);
                            }
                            if !chunk.is_empty() {
                                if senders[i].send(AudioBuffer::new(chunk, audio.sample_rate)).is_err() {
                                    done[i] = true; // forward thread closed — stop generating
                                }
                            }
                        }
                    }
                    // Keep last N frames as context for next chunk
                    prev_frames[i].extend(frame_buffers[i].drain(..));
                    let total = prev_frames[i].len();
                    if total > vocoder_context_frames {
                        prev_frames[i] = prev_frames[i][total - vocoder_context_frames..].to_vec();
                    }
                }
            }
        }

        // Flush remaining frames
        for i in 0..n {
            if !done[i] && !frame_buffers[i].is_empty() {
                let ctx = &prev_frames[i];
                let mut decode_input = ctx.clone();
                decode_input.extend(frame_buffers[i].iter().cloned());
                let ctx_samples = ctx.len() * samples_per_frame;
                if let Ok(audio) = self.decode_codes(&decode_input) {
                    if ctx_samples < audio.samples.len() {
                        let mut chunk = audio.samples[ctx_samples..].to_vec();
                        let tail = &prev_tail[i];
                        let fade = tail.len().min(chunk.len()).min(crossfade_len);
                        for j in 0..fade {
                            let t = (j + 1) as f32 / (fade + 1) as f32;
                            chunk[j] = tail[tail.len() - fade + j] * (1.0 - t) + chunk[j] * t;
                        }
                        let _ = senders[i].send(AudioBuffer::new(chunk, audio.sample_rate));
                    }
                }
            }
        }

        Ok(())
    }

    /// Create a streaming synthesis session with a specific voice and language.
    ///
    /// Returns an iterator that yields audio chunks as they are generated.
    /// Each chunk contains approximately `chunk_frames` frames worth of audio
    /// (default: 10 frames = ~800ms at 12.5 Hz frame rate).
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// use qwen3_tts::{Qwen3TTS, Speaker, Language, SynthesisOptions};
    ///
    /// let options = SynthesisOptions::default();
    /// for chunk in model.synthesize_streaming("Hello!", Speaker::Ryan, Language::English, options)? {
    ///     let audio = chunk?;
    ///     // Play or process audio chunk (each ~800ms)
    /// }
    /// ```
    pub fn synthesize_streaming(
        &self,
        text: &str,
        speaker: Speaker,
        language: Language,
        options: SynthesisOptions,
    ) -> Result<StreamingSession<'_>> {
        let input_ids = self.text_tokenizer.encode(text)?;
        StreamingSession::new(self, &input_ids, speaker, language, options)
    }

    /// Synthesize speech using a text-described voice (VoiceDesign model), streaming.
    ///
    /// Same as [`Self::synthesize_voice_design`] but returns a streaming session
    /// that yields audio chunks as they are generated.
    ///
    /// The instruct text is tokenized with ChatML framing:
    /// `<|im_start|>user\n{instruct}<|im_end|>\n`
    ///
    /// # Arguments
    ///
    /// * `text` - Text to synthesize
    /// * `instruct` - Natural language voice description (e.g., "A cheerful young female voice")
    /// * `language` - Target language
    /// * `options` - Synthesis options (temperature, top_k, chunk_frames, etc.)
    pub fn synthesize_voice_design_streaming(
        &self,
        text: &str,
        instruct: &str,
        language: Language,
        options: SynthesisOptions,
    ) -> Result<StreamingSession<'_>> {
        if let Some(ref mt) = self.model_type {
            if *mt != ModelType::VoiceDesign {
                tracing::warn!(
                    "Using VoiceDesign synthesis on a {:?} model. This model was not trained \
                     for text-described voice conditioning — output may be unpredictable.",
                    mt
                );
            }
        }

        let input_ids = self.text_tokenizer.encode(text)?;

        // Tokenize instruct with ChatML user framing: <|im_start|>user\n{instruct}<|im_end|>\n
        let instruct_text = format!("<|im_start|>user\n{}<|im_end|>\n", instruct);
        let instruct_ids = self.text_tokenizer.encode(&instruct_text)?;

        StreamingSession::new_voice_design(self, &input_ids, &instruct_ids, language, options)
    }

    // ── Voice cloning API ─────────────────────────────────────────────────

    /// Create a voice clone prompt from reference audio.
    ///
    /// When `ref_text` is `None`, produces an **x_vector_only** prompt (speaker
    /// embedding only). When `Some`, produces an **ICL** prompt (speaker embedding
    /// + reference audio codes + reference text) — requires a speech encoder.
    ///
    /// # Errors
    ///
    /// Returns an error if the speaker encoder is not loaded.
    pub fn create_voice_clone_prompt(
        &self,
        ref_audio: &AudioBuffer,
        ref_text: Option<&str>,
    ) -> Result<VoiceClonePrompt> {
        let encoder = self.speaker_encoder.as_ref().ok_or_else(|| {
            let hint = match &self.model_type {
                Some(ModelType::CustomVoice) => {
                    " CustomVoice models use preset speakers (synthesize_with_voice), \
                     not voice cloning. Use a Base model for voice cloning."
                }
                Some(ModelType::VoiceDesign) => {
                    " VoiceDesign models use text-described voices, not voice cloning. \
                     Use a Base model for voice cloning."
                }
                _ => {
                    " Ensure model weights contain `speaker_encoder.*` keys \
                     (only Base models include a speaker encoder)."
                }
            };
            anyhow::anyhow!("Speaker encoder not available.{}", hint)
        })?;

        // Resample to 24kHz if needed — both encoders assume 24kHz input
        let ref_audio_24k;
        let ref_audio = if ref_audio.sample_rate != 24000 {
            tracing::info!(
                "Resampling reference audio from {}Hz to 24000Hz",
                ref_audio.sample_rate
            );
            ref_audio_24k = audio::resample_to_24k(ref_audio)?;
            &ref_audio_24k
        } else {
            ref_audio
        };

        let speaker_embedding = encoder.encode(ref_audio)?; // [enc_dim]

        // ICL data: encode reference audio to codes and tokenize reference text
        let (ref_codes, ref_text_ids) = if let Some(text) = ref_text {
            let speech_enc = self.speech_encoder.as_ref().ok_or_else(|| {
                anyhow::anyhow!(
                    "ICL voice cloning requires a speech encoder, but it was not loaded. \
                     Ensure the speech tokenizer weights contain encoder keys, or use \
                     x_vector_only mode by passing ref_text=None."
                )
            })?;

            let codes = speech_enc.encode(ref_audio)?; // [T_frames, 16]
            let text_ids = self.text_tokenizer.encode(text)?;

            (Some(codes), Some(text_ids))
        } else {
            (None, None)
        };

        Ok(VoiceClonePrompt {
            speaker_embedding,
            ref_codes,
            ref_text_ids,
        })
    }

    /// Synthesize speech using a cloned voice.
    ///
    /// Uses the same generation loop as [`Self::synthesize_with_voice`] but runs the
    /// voice-clone prefill instead of the predefined-speaker prefill.
    ///
    /// When the prompt contains ICL data (ref_codes + ref_text_ids), the model
    /// is conditioned on reference audio/text to better reproduce the speaker's voice.
    pub fn synthesize_voice_clone(
        &self,
        text: &str,
        prompt: &VoiceClonePrompt,
        language: Language,
        options: Option<SynthesisOptions>,
    ) -> Result<AudioBuffer> {
        self.synthesize_voice_clone_debug(text, prompt, language, options)
            .map(|(audio, _codes)| audio)
    }

    /// Convert a ref_codes tensor `[T_frames, 16]` to `Vec<Vec<u32>>` frame format.
    fn tensor_to_frame_codes(&self, codes: &Tensor) -> Result<FrameCodes> {
        let (n_frames, n_codebooks) = codes.dims2()?;
        let codes_u32 = codes.to_dtype(DType::U32)?;
        let mut frames = Vec::with_capacity(n_frames);
        for f in 0..n_frames {
            let frame_tensor = codes_u32.i(f)?; // [16]
            let frame_vec: Vec<u32> = frame_tensor.to_vec1()?;
            debug_assert_eq!(frame_vec.len(), n_codebooks);
            frames.push(frame_vec);
        }
        Ok(frames)
    }

    /// Sum reference codec embeddings across all 16 codebook groups.
    ///
    /// For each frame:
    /// - Group 0 (semantic): `talker.codec_embedding(ref_codes[:, 0])`
    /// - Groups 1–15 (acoustic): `code_predictor.codec_embeddings[i-1](ref_codes[:, i])`
    /// - Sum all 16 → single embedding per frame
    ///
    /// # Arguments
    /// * `ref_codes` — shape `[T_frames, 16]` of i64 codes
    ///
    /// # Returns
    /// Tensor of shape `[1, T_frames, hidden_size]`
    fn sum_ref_codec_embeddings(&self, ref_codes: &Tensor) -> Result<Tensor> {
        // Group 0: semantic codes → talker.codec_embedding
        let semantic_codes = ref_codes.i((.., 0))?; // [T_frames]
        let semantic_codes = semantic_codes.to_dtype(candle_core::DType::U32)?;
        let summed = self.talker.get_codec_embedding_batch(&semantic_codes)?; // [1, T, hidden]

        // Groups 1-15: acoustic codes → code_predictor.embed_codes_for_group
        let mut summed = summed;
        for group in 1..16 {
            let group_codes = ref_codes.i((.., group))?; // [T_frames]
            let group_codes = group_codes.to_dtype(candle_core::DType::U32)?;
            let group_embed = self
                .code_predictor
                .embed_codes_for_group(group - 1, &group_codes)?; // [1, T, embed_dim]
            summed = summed.add(&group_embed)?;
        }

        Ok(summed)
    }

    /// Build default trailing text embeddings (for non-ICL mode).
    fn build_default_trailing_text(&self, input_ids: &[u32]) -> Result<Tensor> {
        let (hidden, _len, _pad) = self.build_trailing_text(input_ids)?;
        Ok(hidden)
    }

    /// Apply repetition penalty, token suppression, and min_new_tokens EOS suppression
    /// using a pre-built `[1, vocab]` penalty mask on GPU.
    ///
    /// The mask is updated incrementally via [`update_penalty_mask`] after each
    /// sampled token, eliminating the O(n) GPU→CPU transfer that grows with
    /// each frame.
    fn apply_generation_penalties_gpu(
        &self,
        logits: &Tensor,
        penalty_mask: &Tensor,
        config: &generation::GenerationConfig,
        token_count: usize,
        suppression_mask: Option<&generation::SuppressionMask>,
    ) -> Result<Tensor> {
        let logits = logits.to_dtype(DType::F32)?;

        // 1. Repetition penalty via pre-built GPU mask
        let logits = if config.repetition_penalty != 1.0 {
            generation::apply_repetition_penalty_with_mask(
                &logits,
                penalty_mask,
                config.repetition_penalty,
            )?
        } else {
            logits
        };

        // 2. Token suppression
        let logits = if let Some(mask) = suppression_mask {
            generation::apply_token_suppression_with_mask(&logits, mask)?
        } else {
            generation::apply_token_suppression(
                &logits,
                codec_tokens::CODEC_VOCAB_SIZE,
                CODEC_EOS_TOKEN_ID,
            )?
        };

        // 3. Min new tokens EOS suppression
        if token_count < config.min_new_tokens {
            if let Some(eos_id) = config.eos_token_id {
                let vocab = logits.dim(1)?;
                let batch = logits.dim(0)?;
                let mut mask_data = vec![0.0f32; vocab];
                mask_data[eos_id as usize] = 1.0;
                let eos_mask = Tensor::new(mask_data.as_slice(), logits.device())?
                    .unsqueeze(0)?
                    .broadcast_as((batch, vocab))?;
                let neg_inf = Tensor::new(&[f32::NEG_INFINITY], logits.device())?
                    .broadcast_as((batch, vocab))?;
                let zeros = Tensor::zeros((batch, vocab), DType::F32, logits.device())?;
                let is_eos = eos_mask.gt(&zeros)?;
                return Ok(is_eos.where_cond(&neg_inf, &logits)?);
            }
        }

        Ok(logits)
    }

    /// Returns `true` if a speech encoder is loaded (ICL voice cloning is available).
    pub fn has_speech_encoder(&self) -> bool {
        self.speech_encoder.is_some()
    }

    // ── Private helpers ─────────────────────────────────────────────────

    /// Attempt to load the speaker encoder from model weights.
    ///
    /// Returns `Ok(Some(encoder))` if `speaker_encoder.*` keys are found,
    /// `Ok(None)` if they are absent. When `config` is provided, uses the
    /// parsed enc_dim; otherwise falls back to defaults (enc_dim=1024).
    fn try_load_speaker_encoder(
        weights: &HashMap<String, Tensor>,
        config: Option<&SpeakerEncoderConfig>,
        device: &Device,
    ) -> Result<Option<SpeakerEncoder>> {
        let has_se_weights = weights.keys().any(|k| k.starts_with("speaker_encoder."));
        if !has_se_weights {
            return Ok(None);
        }

        let config = config.cloned().unwrap_or_default();
        tracing::info!(
            "Loading speaker encoder (ECAPA-TDNN, enc_dim={}) for voice cloning...",
            config.enc_dim
        );
        let se_weights = Self::filter_weights(weights, "speaker_encoder.");
        let se_vb = candle_nn::VarBuilder::from_tensors(se_weights, DType::F32, device);
        let encoder = SpeakerEncoder::new(config, se_vb)?;
        Ok(Some(encoder))
    }

    /// Attempt to load the speech encoder (Mimi) from speech tokenizer weights.
    ///
    /// The speech encoder encodes raw audio to 12Hz codec codes, needed for
    /// ICL voice cloning. Returns `Ok(None)` if encoder keys are absent or
    /// loading fails (non-fatal — ICL mode just won't be available).
    fn try_load_speech_encoder(
        weights: &HashMap<String, Tensor>,
        device: &Device,
    ) -> Result<Option<Encoder12Hz>> {
        // Check for encoder-related keys (either HF or candle format)
        let has_encoder_keys = weights
            .keys()
            .any(|k| k.starts_with("encoder.") || k.starts_with("encoder_transformer."));
        if !has_encoder_keys {
            return Ok(None);
        }

        tracing::debug!("Attempting to load speech encoder (Mimi) for ICL voice cloning...");
        match Encoder12Hz::from_weights(weights, device) {
            Ok(enc) => {
                tracing::info!("Loaded speech encoder — ICL voice cloning available");
                Ok(Some(enc))
            }
            Err(e) => {
                tracing::debug!(
                    "Speech encoder not available ({}). ICL voice cloning disabled.",
                    e
                );
                Ok(None)
            }
        }
    }

    /// Load weights from safetensors file.
    ///
    /// Tensors are loaded in their native dtype (typically BF16 for Qwen3-TTS).
    /// Each component's VarBuilder handles casting to its target dtype.
    fn load_weights(path: &Path, device: &Device) -> Result<HashMap<String, Tensor>> {
        Ok(candle_core::safetensors::load(path, device)?)
    }

    /// Filter weights by prefix, removing the prefix from keys.
    pub(crate) fn filter_weights(
        weights: &HashMap<String, Tensor>,
        prefix: &str,
    ) -> HashMap<String, Tensor> {
        weights
            .iter()
            .filter_map(|(k, v)| {
                k.strip_prefix(prefix)
                    .map(|stripped| (stripped.to_string(), v.clone()))
            })
            .collect()
    }
}

/// Convert a slice of codec frames into a tensor of shape `[1, 16, T]`.
///
/// Each frame must contain exactly 16 codebook values. The output layout is
/// `[q0_f0, q0_f1, ...], [q1_f0, q1_f1, ...]` matching the decoder's expectation.
pub fn codes_to_tensor(codes: &[Vec<u32>], device: &Device) -> Result<Tensor> {
    let num_frames = codes.len();
    if num_frames == 0 {
        return Ok(Tensor::zeros((1, 16, 0), DType::I64, device)?);
    }

    let mut data = vec![0i64; 16 * num_frames];
    for (frame, frame_codes) in codes.iter().enumerate() {
        for (q, &code) in frame_codes.iter().enumerate() {
            data[q * num_frames + frame] = code as i64;
        }
    }

    Ok(Tensor::from_vec(data, (1, 16, num_frames), device)?)
}

/// Return the recommended compute dtype for the given device.
///
/// Returns `BF16` for CUDA/Metal (lower memory, faster attention) and `F32` for CPU.
pub fn compute_dtype_for_device(device: &Device) -> DType {
    if device.is_cuda() || device.is_metal() {
        DType::BF16
    } else {
        DType::F32
    }
}

/// Force the GPU to complete all pending work before returning.
///
/// On CUDA/Metal, GPU operations are asynchronous — `Instant::now()` would
/// measure submission time, not completion time. This helper forces a sync
/// by creating a tiny tensor and reading it back to the CPU.
///
/// On CPU this is a no-op.
pub fn sync_device(device: &Device) -> Result<()> {
    match device {
        Device::Cpu => Ok(()),
        _ => {
            // Force a GPU→CPU sync by reading a scalar back
            let _: Vec<f32> = Tensor::zeros(1, DType::F32, device)?.to_vec1()?;
            Ok(())
        }
    }
}

/// The codec end-of-sequence token ID (2150).
///
/// Generation stops when this token is sampled. This is in the codec vocabulary
/// `[0, 3072)`, not the text vocabulary.
pub const CODEC_EOS_TOKEN_ID: u32 = codec_tokens::CODEC_EOS;

/// Number of audio samples per codec frame at 24kHz (1920 = 80ms per frame at 12Hz).
pub const SAMPLES_PER_FRAME: usize = 1920;

/// ICL mode: minimum frames to generate regardless of text length (matching mlx-audio)
const ICL_MIN_FRAMES: usize = 75;

/// ICL mode: estimated frames per input text token for max-length cap (matching mlx-audio)
const ICL_FRAMES_PER_TOKEN: usize = 6;

/// ICL mode: minimum repetition penalty to prevent degenerate loops (matching mlx-audio)
const ICL_MIN_REPETITION_PENALTY: f64 = 1.3;

/// Streaming synthesis session.
///
/// Yields audio chunks as they are generated. Use with
/// [`Qwen3TTS::synthesize_streaming`].
pub struct StreamingSession<'a> {
    model: &'a Qwen3TTS,
    config: generation::GenerationConfig,
    sampling_ctx: generation::SamplingContext,
    kv_caches: Vec<AnyKVCache>,
    offset: usize,
    last_hidden: Tensor,
    current_token: Option<u32>,
    current_token_tensor: Option<Tensor>,
    frames_generated: usize,
    frame_buffer: FrameCodes,
    chunk_frames: usize,
    done: bool,
    // Trailing text state for residual VQ + text fusion
    trailing_text_hidden: Tensor,
    trailing_text_len: usize,
    tts_pad_embed: Tensor,
    // GPU-side repetition penalty mask [1, vocab] — updated incrementally
    penalty_mask: Tensor,
    token_count: usize,
    // Pre-built suppression mask (reused every frame)
    suppression_mask: generation::SuppressionMask,
    // Pre-allocated code predictor KV caches (reused + reset each frame)
    cp_kv_caches: Vec<AnyKVCache>,
}

impl<'a> StreamingSession<'a> {
    fn new(
        model: &'a Qwen3TTS,
        input_ids: &[u32],
        speaker: Speaker,
        language: Language,
        options: SynthesisOptions,
    ) -> Result<Self> {
        let sampling_ctx = generation::SamplingContext::new(options.seed);
        let config = options.to_gen_config();

        let (trailing_text_hidden, trailing_text_len, tts_pad_embed) =
            model.build_trailing_text(input_ids)?;

        let mut kv_caches = model.talker.new_kv_caches(config.max_new_tokens + 256);
        let prefill_result =
            model
                .talker
                .prefill_custom_voice(input_ids, speaker, language, &mut kv_caches)?;

        Self::from_prefill(
            model,
            config,
            sampling_ctx,
            kv_caches,
            prefill_result,
            trailing_text_hidden,
            trailing_text_len,
            tts_pad_embed,
            options.chunk_frames,
        )
    }

    /// Create a streaming session using voice design (text-described voice).
    ///
    /// Uses `prefill_voice_design` instead of `prefill_custom_voice` to condition
    /// on a natural language voice description rather than a predefined speaker.
    fn new_voice_design(
        model: &'a Qwen3TTS,
        input_ids: &[u32],
        instruct_ids: &[u32],
        language: Language,
        options: SynthesisOptions,
    ) -> Result<Self> {
        let sampling_ctx = generation::SamplingContext::new(options.seed);
        let config = options.to_gen_config();

        let (trailing_text_hidden, trailing_text_len, tts_pad_embed) =
            model.build_trailing_text(input_ids)?;

        let mut kv_caches = model.talker.new_kv_caches(config.max_new_tokens + 256);
        let prefill_result =
            model
                .talker
                .prefill_voice_design(input_ids, instruct_ids, language, &mut kv_caches)?;

        Self::from_prefill(
            model,
            config,
            sampling_ctx,
            kv_caches,
            prefill_result,
            trailing_text_hidden,
            trailing_text_len,
            tts_pad_embed,
            options.chunk_frames,
        )
    }

    /// Shared post-prefill constructor.
    ///
    /// Extracts `last_hidden` from the prefill result, builds the suppression and
    /// penalty masks, samples the first semantic token, and assembles the session.
    #[allow(clippy::too_many_arguments)]
    fn from_prefill(
        model: &'a Qwen3TTS,
        config: generation::GenerationConfig,
        mut sampling_ctx: generation::SamplingContext,
        kv_caches: Vec<AnyKVCache>,
        prefill_result: (Tensor, Tensor),
        trailing_text_hidden: Tensor,
        trailing_text_len: usize,
        tts_pad_embed: Tensor,
        chunk_frames: usize,
    ) -> Result<Self> {
        let (hidden, logits) = prefill_result;
        let prefill_len = hidden.dim(1)?;
        let last_hidden = hidden.i((.., prefill_len - 1..prefill_len, ..))?;

        // Build suppression mask once for reuse across all frames
        let suppression_mask = generation::build_suppression_mask(
            codec_tokens::CODEC_VOCAB_SIZE,
            CODEC_EOS_TOKEN_ID,
            &model.device,
        )?;

        // Sample first token with full penalty pipeline
        let vocab_size = codec_tokens::CODEC_VOCAB_SIZE;
        let mut penalty_mask = Tensor::zeros((1, vocab_size), DType::F32, &model.device)?;
        let logits_2d = logits.squeeze(1)?;
        let logits_2d = model.apply_generation_penalties_gpu(
            &logits_2d,
            &penalty_mask,
            &config,
            0,
            Some(&suppression_mask),
        )?;
        let first_token = generation::sample(&logits_2d, &config, &mut sampling_ctx)?;
        let first_token_id: u32 = first_token.flatten_all()?.to_vec1::<u32>()?[0];
        Qwen3TTS::update_penalty_mask(&mut penalty_mask, first_token_id, vocab_size)?;

        let done = config.eos_token_id == Some(first_token_id);
        let cp_kv_caches = model.code_predictor.new_kv_caches();

        Ok(Self {
            model,
            config,
            sampling_ctx,
            kv_caches,
            offset: prefill_len,
            last_hidden,
            current_token: if done { None } else { Some(first_token_id) },
            current_token_tensor: if done { None } else { Some(first_token) },
            frames_generated: 0,
            frame_buffer: Vec::new(),
            chunk_frames,
            done,
            trailing_text_hidden,
            trailing_text_len,
            tts_pad_embed,
            penalty_mask,
            token_count: 1,
            suppression_mask,
            cp_kv_caches,
        })
    }

    /// Generate the next chunk of audio.
    ///
    /// Returns `Some(AudioBuffer)` for each chunk, or `None` when generation is complete.
    pub fn next_chunk(&mut self) -> Result<Option<AudioBuffer>> {
        if self.done {
            // Flush remaining buffer
            if !self.frame_buffer.is_empty() {
                let codes = self.model.codes_to_tensor(&self.frame_buffer)?;
                self.frame_buffer.clear();
                let audio = self.model.decoder.decode(&codes)?;
                return Ok(Some(AudioBuffer::from_tensor(audio, 24000)?));
            }
            return Ok(None);
        }

        // Generate frames until we have enough for a chunk
        while self.frame_buffer.len() < self.chunk_frames
            && self.frames_generated < self.config.max_new_tokens
        {
            let (token_id, token_tensor) =
                match (self.current_token, self.current_token_tensor.take()) {
                    (Some(id), Some(t)) => (id, t),
                    _ => {
                        self.done = true;
                        break;
                    }
                };

            // Embedding lookup using GPU-resident token tensor (no CPU→GPU roundtrip)
            let semantic_embed = self
                .model
                .talker
                .get_codec_embedding_from_tensor(&token_tensor)?;

            // Generate 15 acoustic codes (stays on GPU)
            let acoustic_codes_tensor = self.model.code_predictor.generate_acoustic_codes(
                &self.last_hidden,
                &semantic_embed,
                &mut self.cp_kv_caches,
            )?;

            // Build frame on GPU, then transfer for frame_buffer
            let semantic_t = Tensor::new(&[token_id], self.model.device())?;
            let frame_tensor = Tensor::cat(&[&semantic_t, &acoustic_codes_tensor], 0)?;
            let frame_codes: Vec<u32> = frame_tensor.to_vec1()?;
            self.frame_buffer.push(frame_codes);

            let frame_idx = self.frames_generated;
            self.frames_generated += 1;

            // Build residual VQ sum + trailing text for next step
            let acoustic_embed_sum = self
                .model
                .code_predictor
                .get_acoustic_embeddings_sum_from_tensor(&acoustic_codes_tensor)?;
            let summed = semantic_embed.add(&acoustic_embed_sum)?;

            let text_addition = if frame_idx < self.trailing_text_len {
                self.trailing_text_hidden
                    .i((.., frame_idx..frame_idx + 1, ..))?
            } else {
                self.tts_pad_embed.clone()
            };
            let step_input = summed.add(&text_addition)?;

            // Run talker step with fused embedding
            let (h, new_logits) = self.model.talker.generate_step_with_embed(
                &step_input,
                &mut self.kv_caches,
                self.offset,
            )?;
            self.offset += 1;
            self.last_hidden = h;

            // Sample next semantic token with repetition penalty + token suppression + min_new_tokens
            let logits_2d = new_logits.squeeze(1)?;
            let logits_2d = self.model.apply_generation_penalties_gpu(
                &logits_2d,
                &self.penalty_mask,
                &self.config,
                self.token_count,
                Some(&self.suppression_mask),
            )?;
            let next_token_tensor =
                generation::sample(&logits_2d, &self.config, &mut self.sampling_ctx)?;
            let next_token_id: u32 = next_token_tensor.flatten_all()?.to_vec1::<u32>()?[0];
            Qwen3TTS::update_penalty_mask(
                &mut self.penalty_mask,
                next_token_id,
                codec_tokens::CODEC_VOCAB_SIZE,
            )?;
            self.token_count += 1;

            if self.config.eos_token_id == Some(next_token_id) {
                self.current_token = None;
                self.current_token_tensor = None;
                self.done = true;
            } else {
                self.current_token = Some(next_token_id);
                self.current_token_tensor = Some(next_token_tensor);
            }
        }

        // Decode the buffered frames
        if self.frame_buffer.is_empty() {
            return Ok(None);
        }

        let codes = self.model.codes_to_tensor(&self.frame_buffer)?;
        self.frame_buffer.clear();
        let audio = self.model.decoder.decode(&codes)?;
        Ok(Some(AudioBuffer::from_tensor(audio, 24000)?))
    }

    /// Returns the total number of frames generated so far.
    pub fn frames_generated(&self) -> usize {
        self.frames_generated
    }

    /// Returns true if generation is complete.
    pub fn is_done(&self) -> bool {
        self.done && self.frame_buffer.is_empty()
    }
}

impl<'a> Iterator for StreamingSession<'a> {
    type Item = Result<AudioBuffer>;

    fn next(&mut self) -> Option<Self::Item> {
        match self.next_chunk() {
            Ok(Some(audio)) => Some(Ok(audio)),
            Ok(None) => None,
            Err(e) => Some(Err(e)),
        }
    }
}

/// Options for speech synthesis
#[derive(Debug, Clone)]
pub struct SynthesisOptions {
    /// Maximum number of frames to generate
    pub max_length: usize,
    /// Sampling temperature (higher = more random)
    pub temperature: f64,
    /// Top-k sampling
    pub top_k: usize,
    /// Top-p (nucleus) sampling
    pub top_p: f64,
    /// Repetition penalty (1.0 = disabled, 1.05 = Python default)
    pub repetition_penalty: f64,
    /// End-of-sequence token ID (defaults to codec EOS token 2150)
    pub eos_token_id: Option<u32>,
    /// Frames per streaming chunk (default: 10 = ~800ms)
    pub chunk_frames: usize,
    /// Minimum tokens before EOS is allowed (default: 2, matching Python)
    pub min_new_tokens: usize,
    /// Random seed for deterministic generation. `None` = non-deterministic.
    pub seed: Option<u64>,
}

impl SynthesisOptions {
    /// Convert to a [`GenerationConfig`](generation::GenerationConfig) for the generation loop.
    pub fn to_gen_config(&self) -> generation::GenerationConfig {
        generation::GenerationConfig {
            max_new_tokens: self.max_length,
            temperature: self.temperature,
            top_k: self.top_k,
            top_p: self.top_p,
            repetition_penalty: self.repetition_penalty,
            eos_token_id: self.eos_token_id,
            min_new_tokens: self.min_new_tokens,
        }
    }
}

impl Default for SynthesisOptions {
    fn default() -> Self {
        Self {
            max_length: 2048,
            temperature: 0.9,
            top_k: 50,
            top_p: 0.9,
            repetition_penalty: 1.05,
            eos_token_id: Some(CODEC_EOS_TOKEN_ID),
            chunk_frames: 10, // ~800ms per chunk at 12.5 Hz
            min_new_tokens: 2,
            seed: None,
        }
    }
}

/// Select the best available compute device for inference.
///
/// Checks for available hardware in order: CUDA → Metal → CPU.
/// Falls back to CPU if no GPU acceleration is available.
///
/// # Feature Flags
///
/// - `cuda`: Enables NVIDIA GPU support
/// - `metal`: Enables Apple Silicon GPU support
///
/// # Example
///
/// ```rust,ignore
/// let device = qwen3_tts::auto_device()?;
/// let model = Qwen3TTS::from_pretrained("path/to/model", device)?;
/// ```
pub fn auto_device() -> Result<Device> {
    #[cfg(feature = "cuda")]
    {
        if let Ok(device) = Device::cuda_if_available(0) {
            if device.is_cuda() {
                tracing::info!("Using CUDA device");
                return Ok(device);
            }
        }
    }

    #[cfg(feature = "metal")]
    {
        if let Ok(device) = Device::new_metal(0) {
            tracing::info!("Using Metal device");
            return Ok(device);
        }
    }

    tracing::info!("Using CPU device");
    Ok(Device::Cpu)
}

/// Parse a device string into a [`Device`].
///
/// Supported formats:
/// - `"auto"` — select best available via [`auto_device`]
/// - `"cpu"` — force CPU
/// - `"cuda"` or `"cuda:0"` — CUDA device 0
/// - `"cuda:N"` — CUDA device N
/// - `"metal"` — Apple Silicon GPU
///
/// # Errors
///
/// Returns an error if the device string is unrecognized, the requested
/// backend wasn't compiled in, or hardware initialization fails.
pub fn parse_device(device_str: &str) -> Result<Device> {
    match device_str.to_lowercase().as_str() {
        "auto" => auto_device(),
        "cpu" => Ok(Device::Cpu),
        s if s.starts_with("cuda") => {
            #[cfg(feature = "cuda")]
            {
                let ordinal: usize = if s == "cuda" {
                    0
                } else if let Some(idx) = s.strip_prefix("cuda:") {
                    idx.parse()
                        .map_err(|e| anyhow::anyhow!("invalid CUDA device index: {e}"))?
                } else {
                    0
                };
                Device::cuda_if_available(ordinal)
                    .map_err(|e| anyhow::anyhow!("failed to init CUDA device {ordinal}: {e}"))
            }
            #[cfg(not(feature = "cuda"))]
            anyhow::bail!("CUDA support not compiled in. Rebuild with: cargo build --features cuda")
        }
        "metal" => {
            #[cfg(feature = "metal")]
            {
                Device::new_metal(0)
                    .map_err(|e| anyhow::anyhow!("failed to init Metal device: {e}"))
            }
            #[cfg(not(feature = "metal"))]
            anyhow::bail!(
                "Metal support not compiled in. Rebuild with: cargo build --features metal"
            )
        }
        other => {
            anyhow::bail!("unknown device '{other}'. Supported: auto, cpu, cuda, cuda:N, metal")
        }
    }
}

/// Human-readable label for a [`Device`].
pub fn device_info(device: &Device) -> String {
    match device {
        Device::Cpu => "CPU".to_string(),
        Device::Cuda(_) => "CUDA".to_string(),
        Device::Metal(_) => "Metal".to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_synthesis_options_default() {
        let options = SynthesisOptions::default();
        assert_eq!(options.max_length, 2048);
        assert!((options.temperature - 0.9).abs() < 1e-6);
        assert_eq!(options.top_k, 50);
        assert!((options.top_p - 0.9).abs() < 1e-6);
        assert!((options.repetition_penalty - 1.05).abs() < 1e-6);
        assert_eq!(options.eos_token_id, Some(CODEC_EOS_TOKEN_ID));
        assert_eq!(options.chunk_frames, 10);
        assert_eq!(options.min_new_tokens, 2);
    }

    #[test]
    fn test_synthesis_options_custom() {
        let options = SynthesisOptions {
            max_length: 512,
            temperature: 0.5,
            top_k: 10,
            top_p: 0.8,
            repetition_penalty: 1.2,
            eos_token_id: Some(CODEC_EOS_TOKEN_ID),
            chunk_frames: 5,
            min_new_tokens: 0,
            seed: Some(42),
        };
        assert_eq!(options.max_length, 512);
        assert!((options.temperature - 0.5).abs() < 1e-6);
        assert_eq!(options.eos_token_id, Some(CODEC_EOS_TOKEN_ID));
        assert_eq!(options.chunk_frames, 5);
    }

    #[test]
    fn test_synthesis_options_clone() {
        let options = SynthesisOptions::default();
        let cloned = options.clone();
        assert_eq!(cloned.max_length, options.max_length);
        assert_eq!(cloned.top_k, options.top_k);
    }

    #[test]
    fn test_synthesis_options_debug() {
        let options = SynthesisOptions::default();
        let debug_str = format!("{:?}", options);
        assert!(debug_str.contains("max_length"));
        assert!(debug_str.contains("2048"));
    }

    #[test]
    fn test_auto_device() {
        // Should always succeed on CPU
        let device = auto_device().unwrap();
        // Just verify it returns a valid device
        assert!(
            matches!(device, Device::Cpu)
                || matches!(device, Device::Cuda(_))
                || matches!(device, Device::Metal(_))
        );
    }

    #[test]
    fn test_audio_buffer_reexport() {
        // Verify re-exports work
        let buffer = AudioBuffer::new(vec![0.0f32; 100], 24000);
        assert_eq!(buffer.sample_rate, 24000);
    }

    #[test]
    fn test_config_reexport() {
        // Verify config re-export works
        let config = Qwen3TTSConfig::default();
        assert_eq!(config.model_type, "qwen3_tts");
    }

    #[test]
    fn test_codes_to_tensor_empty() {
        let device = Device::Cpu;
        let codes: Vec<Vec<u32>> = vec![];
        let tensor = codes_to_tensor(&codes, &device).unwrap();
        assert_eq!(tensor.dims(), &[1, 16, 0]);
    }

    #[test]
    fn test_codes_to_tensor_single_frame() {
        let device = Device::Cpu;
        let codes = vec![vec![0u32; 16]];
        let tensor = codes_to_tensor(&codes, &device).unwrap();
        assert_eq!(tensor.dims(), &[1, 16, 1]);
    }

    #[test]
    fn test_codes_to_tensor_layout() {
        let device = Device::Cpu;
        // 2 frames, each with 16 codebooks
        let codes = vec![
            (0..16).map(|i| i as u32).collect::<Vec<_>>(), // frame 0
            (100..116).map(|i| i as u32).collect::<Vec<_>>(), // frame 1
        ];
        let tensor = codes_to_tensor(&codes, &device).unwrap();
        assert_eq!(tensor.dims(), &[1, 16, 2]);

        // Verify layout: tensor[0, q, frame] = codes[frame][q]
        let vals: Vec<i64> = tensor.flatten_all().unwrap().to_vec1().unwrap();
        // q=0: [frame0_q0, frame1_q0] = [0, 100]
        assert_eq!(vals[0], 0);
        assert_eq!(vals[1], 100);
        // q=1: [frame0_q1, frame1_q1] = [1, 101]
        assert_eq!(vals[2], 1);
        assert_eq!(vals[3], 101);
    }

    #[test]
    fn test_parse_device_cpu() {
        let device = parse_device("cpu").unwrap();
        assert!(matches!(device, Device::Cpu));
    }

    #[test]
    fn test_parse_device_auto() {
        let device = parse_device("auto").unwrap();
        // Should succeed regardless of hardware
        assert!(
            matches!(device, Device::Cpu)
                || matches!(device, Device::Cuda(_))
                || matches!(device, Device::Metal(_))
        );
    }

    #[test]
    fn test_parse_device_unknown() {
        let result = parse_device("tpu");
        assert!(result.is_err());
    }

    #[test]
    fn test_parse_device_case_insensitive() {
        let device = parse_device("CPU").unwrap();
        assert!(matches!(device, Device::Cpu));
    }

    #[test]
    fn test_device_info() {
        assert_eq!(device_info(&Device::Cpu), "CPU");
    }

    #[test]
    fn test_compute_dtype_for_device() {
        let dtype = compute_dtype_for_device(&Device::Cpu);
        assert_eq!(dtype, DType::F32);
    }

    #[test]
    fn test_update_penalty_mask() {
        let device = Device::Cpu;
        let vocab_size = 3072;
        let mut mask = Tensor::zeros((1, vocab_size), DType::F32, &device).unwrap();

        Qwen3TTS::update_penalty_mask(&mut mask, 42, vocab_size).unwrap();

        let vals: Vec<f32> = mask.flatten_all().unwrap().to_vec1().unwrap();
        assert_eq!(vals[42], 1.0);
        // Neighboring positions should be untouched
        assert_eq!(vals[41], 0.0);
        assert_eq!(vals[43], 0.0);
    }

    #[test]
    fn test_update_penalty_mask_out_of_range() {
        let device = Device::Cpu;
        let vocab_size = 3072;
        let mut mask = Tensor::zeros((1, vocab_size), DType::F32, &device).unwrap();

        // Token beyond vocab_size should be a no-op (no panic)
        Qwen3TTS::update_penalty_mask(&mut mask, 9999, vocab_size).unwrap();

        let sum: f32 = mask.sum_all().unwrap().to_scalar().unwrap();
        assert_eq!(sum, 0.0);
    }

    #[test]
    fn test_suppression_mask_deterministic() {
        let device = Device::Cpu;
        let vocab = codec_tokens::CODEC_VOCAB_SIZE;
        let mask1 = generation::build_suppression_mask(vocab, CODEC_EOS_TOKEN_ID, &device).unwrap();
        let mask2 = generation::build_suppression_mask(vocab, CODEC_EOS_TOKEN_ID, &device).unwrap();

        // Apply both masks to uniform logits and verify identical output
        let logits = Tensor::ones((1, vocab), DType::F32, &device).unwrap();
        let out1 = generation::apply_token_suppression_with_mask(&logits, &mask1).unwrap();
        let out2 = generation::apply_token_suppression_with_mask(&logits, &mask2).unwrap();
        let v1: Vec<f32> = out1.flatten_all().unwrap().to_vec1().unwrap();
        let v2: Vec<f32> = out2.flatten_all().unwrap().to_vec1().unwrap();
        assert_eq!(v1, v2);
    }
}
