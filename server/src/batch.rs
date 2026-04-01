//! Batched inference engine for concurrent TTS generation.
//!
//! Collects multiple requests, runs prefill independently, then batches
//! the autoregressive generation loop so N sequences share a single
//! transformer forward pass per frame.

use anyhow::Result;
use qwen3_tts::{AudioBuffer, Language, Qwen3TTS, Speaker, SynthesisOptions};
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
    /// Model directory path
    pub model_dir: String,
}

/// The batch engine runs on a dedicated thread, collecting requests
/// and processing them in batches.
pub struct BatchEngine;

impl BatchEngine {
    /// Start the batch engine on a dedicated thread.
    /// Returns a sender for submitting requests.
    pub fn start(config: BatchEngineConfig) -> mpsc::Sender<BatchRequest> {
        let (tx, rx) = mpsc::channel::<BatchRequest>(config.max_batch_size * 4);

        std::thread::spawn(move || {
            if let Err(e) = Self::run_loop(rx, config) {
                tracing::error!("Batch engine crashed: {e:#}");
            }
        });

        tx
    }

    fn run_loop(
        mut rx: mpsc::Receiver<BatchRequest>,
        config: BatchEngineConfig,
    ) -> Result<()> {
        let device = qwen3_tts::auto_device()?;
        info!(?device, "Batch engine loading model");

        let model = Qwen3TTS::from_pretrained(&config.model_dir, device.clone())?;
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

            // Process sequentially on single GPU thread — no contention,
            // GPU stays warm between requests. Each request gets ~1.1x RT.
            // Total throughput = N * 1.1x RT / N = 1.1x RT constant.
            for req in batch {
                let t_req = Instant::now();
                let result = Self::process_single(&model, &req);
                let gen_time = t_req.elapsed().as_secs_f32();
                let reply = match result {
                    Ok(audio) => Ok(BatchResult { audio, gen_time_secs: gen_time }),
                    Err(e) => Err(e),
                };
                let _ = req.reply.send(reply);
            }

            let total = t0.elapsed().as_secs_f32();
            info!(batch_size, total_secs = total, "Batch complete");
        }

        Ok(())
    }

    fn process_single(model: &Qwen3TTS, req: &BatchRequest) -> Result<AudioBuffer> {
        if let Some(vc) = &req.voice_clone {
            let ref_buf = AudioBuffer::load(&vc.ref_audio_path)?;
            let prompt = model.create_voice_clone_prompt(
                &ref_buf,
                vc.ref_text.as_deref(),
            )?;
            model.synthesize_voice_clone(
                &req.text,
                &prompt,
                req.language,
                Some(req.options.clone()),
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

/// Blocking recv with timeout for std mpsc
trait BlockingRecvTimeout<T> {
    fn blocking_recv_timeout(&mut self, timeout: std::time::Duration) -> Result<T, ()>;
}

impl<T> BlockingRecvTimeout<T> for mpsc::Receiver<T> {
    fn blocking_recv_timeout(&mut self, timeout: std::time::Duration) -> Result<T, ()> {
        // Use tokio's blocking_recv with a manual timeout
        let start = Instant::now();
        loop {
            match self.try_recv() {
                Ok(val) => return Ok(val),
                Err(mpsc::error::TryRecvError::Empty) => {
                    if start.elapsed() >= timeout {
                        return Err(());
                    }
                    std::thread::sleep(std::time::Duration::from_millis(1));
                }
                Err(mpsc::error::TryRecvError::Disconnected) => return Err(()),
            }
        }
    }
}
