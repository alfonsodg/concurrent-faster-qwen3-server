//! Batched inference engine for concurrent TTS generation.
//!
//! Collects multiple requests, runs prefill independently, then batches
//! the autoregressive generation loop so N sequences share a single
//! transformer forward pass per frame.

use anyhow::Result;
use qwen3_tts::{AudioBuffer, Language, Qwen3TTS, Speaker, SynthesisOptions};
use std::sync::Arc;
use std::time::Instant;
use tokio::sync::{mpsc, oneshot};
use tracing::{info, warn};

/// A pending TTS request waiting to be batched.
pub struct BatchRequest {
    pub text: String,
    pub language: Language,
    pub voice_clone: Option<VoiceCloneData>,
    pub options: SynthesisOptions,
    pub reply: oneshot::Sender<Result<BatchResult>>,
}

pub struct VoiceCloneData {
    pub ref_audio_path: std::path::PathBuf,
    pub ref_text: Option<String>,
}

impl Drop for VoiceCloneData {
    fn drop(&mut self) {
        if self.ref_audio_path.exists() {
            let _ = std::fs::remove_file(&self.ref_audio_path);
        }
    }
}

pub struct BatchResult {
    pub audio: AudioBuffer,
    pub gen_time_secs: f32,
}

/// Configuration for the batch engine.
pub struct BatchEngineConfig {
    /// Maximum batch size (number of sequences processed together)
    pub max_batch_size: usize,
    /// Maximum time to wait for a full batch before processing partial batch (ms)
    pub max_wait_ms: u64,
}

/// The batch engine runs on a dedicated thread, collecting requests
/// and processing them in batches.
pub struct BatchEngine;

/// Calculate adaptive max_length based on word count.
pub fn adaptive_max_length(text: &str) -> usize {
    let word_count = text.split_whitespace().count();
    ((word_count * 6) + 50).clamp(100, 512)
}

/// Build voice clone prompts for a batch, returning failed indices.
/// Shared between batch engine and streaming worker (#30).
pub fn build_voice_clone_prompts(
    model: &Qwen3TTS,
    voice_clones: &[Option<&VoiceCloneData>],
) -> (Vec<Option<qwen3_tts::VoiceClonePrompt>>, Vec<usize>) {
    let mut failed = Vec::new();
    let prompts = voice_clones.iter().enumerate().map(|(idx, vc)| {
        vc.and_then(|vc| {
            match qwen3_tts::AudioBuffer::load(&vc.ref_audio_path) {
                Ok(ref_buf) => match model.create_voice_clone_prompt(&ref_buf, vc.ref_text.as_deref()) {
                    Ok(p) => Some(p),
                    Err(e) => { warn!("Voice clone prompt failed: {e:#}"); failed.push(idx); None }
                },
                Err(e) => { warn!("Voice clone ref_audio load failed: {e:#}"); failed.push(idx); None }
            }
        })
    }).collect();
    (prompts, failed)
}

impl BatchEngine {
    /// Start the batch engine on a dedicated thread.
    /// Returns a sender for submitting requests.
    pub fn start(model: Arc<Qwen3TTS>, config: BatchEngineConfig) -> mpsc::Sender<BatchRequest> {
        let (tx, rx) = mpsc::channel::<BatchRequest>(config.max_batch_size * 4);

        std::thread::spawn(move || {
            if let Err(e) = Self::run_loop(rx, config, &model) {
                tracing::error!("Batch engine crashed: {e:#}");
            }
        });

        tx
    }

    fn run_loop(
        mut rx: mpsc::Receiver<BatchRequest>,
        config: BatchEngineConfig,
        model: &Qwen3TTS,
    ) -> Result<()> {
        info!("Batch engine ready, max_batch={}", config.max_batch_size);

        loop {
            // Collect a batch: wait for first request, then drain up to max_batch_size
            let mut batch: Vec<BatchRequest> = Vec::with_capacity(config.max_batch_size);

            // Block on first request
            match rx.blocking_recv() {
                Some(req) => batch.push(req),
                None => break, // channel closed
            }

            // Try to fill the batch within the wait window
            let deadline = Instant::now()
                + std::time::Duration::from_millis(config.max_wait_ms);

            while batch.len() < config.max_batch_size {
                let timeout = deadline.saturating_duration_since(Instant::now());
                if timeout.is_zero() {
                    break;
                }
                match rx.blocking_recv_timeout(timeout) {
                    Ok(req) => batch.push(req),
                    Err(_) => break, // timeout
                }
            }

            let batch_size = batch.len();
            info!(batch_size, "Processing batch");

            let t0 = Instant::now();

            if batch_size > 1 {
                // Build per-request voice clone prompts
                let vc_refs: Vec<Option<&VoiceCloneData>> = batch.iter().map(|r| r.voice_clone.as_ref()).collect();
                let (prompts, failed_indices) = build_voice_clone_prompts(model, &vc_refs);

                // Send errors for failed voice clone requests before batching
                for &idx in failed_indices.iter().rev() {
                    let req = batch.remove(idx);
                    let _ = req.reply.send(Err(anyhow::anyhow!("Voice clone failed")));
                }
                if batch.is_empty() { continue; }

                // Rebuild requests/prompts without failed entries
                let requests: Vec<(String, qwen3_tts::Language, Option<SynthesisOptions>)> = batch
                    .iter()
                    .map(|r| {
                        let mut opts = r.options.clone();
                        // Only skip adaptive cap for ICL (has ref_text); x_vector is fine with it
                        let is_icl = r.voice_clone.as_ref().map_or(false, |vc| vc.ref_text.is_some());
                        if !is_icl {
                            opts.max_length = adaptive_max_length(&r.text);
                        }
                        (r.text.clone(), r.language, Some(opts))
                    })
                    .collect();
                let prompts: Vec<Option<qwen3_tts::VoiceClonePrompt>> = {
                    let mut kept = Vec::new();
                    for (idx, p) in prompts.into_iter().enumerate() {
                        if !failed_indices.contains(&idx) { kept.push(p); }
                    }
                    kept
                };
                let prompt_refs: Vec<Option<&qwen3_tts::VoiceClonePrompt>> =
                    prompts.iter().map(|p| p.as_ref()).collect();

                match model.synthesize_batch_with_voices(&requests, &prompt_refs) {
                    Ok(audios) => {
                        let gen_time = t0.elapsed().as_secs_f32();
                        let per_req = gen_time / audios.len() as f32;
                        info!(batch_size, gen_time, per_req, "Batched forward complete");
                        for (req, audio) in batch.into_iter().zip(audios) {
                            let _ = req.reply.send(Ok(BatchResult {
                                audio,
                                gen_time_secs: per_req,
                            }));
                        }
                        let total = t0.elapsed().as_secs_f32();
                        info!(batch_size, total_secs = total, "Batch complete");
                        continue;
                    }
                    Err(e) => {
                        let err_msg = format!("{e:#}");
                        if err_msg.contains("out of memory") || err_msg.contains("OOM") {
                            // OOM: split batch in half and retry
                            let mid = batch.len() / 2;
                            warn!(batch_size, mid, "OOM in batch, splitting and retrying");
                            let second_half: Vec<BatchRequest> = batch.drain(mid..).collect();
                            Self::process_sequential(model, batch);
                            Self::process_sequential(model, second_half);
                            let total = t0.elapsed().as_secs_f32();
                            info!(total_secs = total, "OOM recovery complete (sequential fallback)");
                            continue;
                        }
                        warn!("Batched forward failed: {e:#}, falling back to sequential");
                    }
                }
            }

            // Fallback: sequential processing
            Self::process_sequential(model, batch);

            let total = t0.elapsed().as_secs_f32();
            info!(total_secs = total, "Sequential fallback complete");
        }

        Ok(())
    }

    fn process_sequential(model: &Qwen3TTS, batch: Vec<BatchRequest>) {
        for req in batch {
            let t_req = Instant::now();
            let result = Self::process_single(model, &req);
            let gen_time = t_req.elapsed().as_secs_f32();
            let _ = req.reply.send(result.map(|audio| BatchResult { audio, gen_time_secs: gen_time }));
        }
    }

    fn process_single(model: &Qwen3TTS, req: &BatchRequest) -> Result<AudioBuffer> {
        if let Some(vc) = &req.voice_clone {
            let ref_buf = AudioBuffer::load(&vc.ref_audio_path)?;
            let prompt = model.create_voice_clone_prompt(
                &ref_buf,
                vc.ref_text.as_deref(),
            )?;
            let mut opts = req.options.clone();
            // Cap x_vector max_length to prevent OOM on large models
            if vc.ref_text.is_none() {
                opts.max_length = adaptive_max_length(&req.text);
            }
            model.synthesize_voice_clone(
                &req.text,
                &prompt,
                req.language,
                Some(opts),
            )
        } else {
            model.synthesize_with_voice(
                &req.text,
                Speaker::Serena,
                req.language,
                Some(req.options.clone()),
            )
        }
    }
}

/// Blocking recv with timeout using OS-level sleep instead of busy-wait poll.
trait BlockingRecvTimeout<T> {
    fn blocking_recv_timeout(&mut self, timeout: std::time::Duration) -> Result<T, ()>;
}

impl<T> BlockingRecvTimeout<T> for mpsc::Receiver<T> {
    fn blocking_recv_timeout(&mut self, timeout: std::time::Duration) -> Result<T, ()> {
        let deadline = Instant::now() + timeout;
        loop {
            match self.try_recv() {
                Ok(val) => return Ok(val),
                Err(mpsc::error::TryRecvError::Empty) => {
                    let remaining = deadline.saturating_duration_since(Instant::now());
                    if remaining.is_zero() {
                        return Err(());
                    }
                    // Sleep with OS scheduler (no busy-wait), check every 10ms
                    std::thread::sleep(remaining.min(std::time::Duration::from_millis(10)));
                }
                Err(mpsc::error::TryRecvError::Disconnected) => return Err(()),
            }
        }
    }
}
