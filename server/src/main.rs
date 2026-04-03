mod batch;

use anyhow::Result;
use axum::{
    body::Body,
    extract::State,
    http::StatusCode,
    response::{IntoResponse, Response},
    routing::{get, post},
    Json, Router,
};
use base64::Engine;
use batch::{BatchEngine, BatchEngineConfig, BatchRequest, VoiceCloneData};
use hound::{SampleFormat, WavSpec, WavWriter};
use qwen3_tts::{Language, SynthesisOptions};
use serde::{Deserialize, Serialize};
use std::{io::Cursor, sync::Arc, sync::atomic::{AtomicU64, Ordering}};
use tokio::sync::{mpsc, oneshot, Semaphore};
use tracing::info;

struct Metrics {
    requests_total: AtomicU64,
    requests_streaming: AtomicU64,
    errors_total: AtomicU64,
    audio_seconds_total: AtomicU64, // stored as milliseconds
    gen_seconds_total: AtomicU64,   // stored as milliseconds
}

impl Metrics {
    fn new() -> Self {
        Self {
            requests_total: AtomicU64::new(0),
            requests_streaming: AtomicU64::new(0),
            errors_total: AtomicU64::new(0),
            audio_seconds_total: AtomicU64::new(0),
            gen_seconds_total: AtomicU64::new(0),
        }
    }
}

struct AppState {
    tx: mpsc::Sender<BatchRequest>,
    stream_tx: mpsc::Sender<StreamingRequest>,
    semaphore: Arc<Semaphore>,
    max_inflight: usize,
    max_batch: usize,
    metrics: Arc<Metrics>,
}

#[derive(Deserialize)]
struct SpeechRequest {
    text: String,
    #[serde(default = "default_language")]
    language: String,
    #[serde(default)]
    ref_audio: Option<String>,
    #[serde(default)]
    ref_text: Option<String>,
    #[serde(default)]
    temperature: Option<f64>,
    #[serde(default)]
    stream: Option<bool>,
}

fn default_language() -> String { "spanish".into() }

#[derive(Serialize)]
struct HealthResponse { status: &'static str, queue_depth: usize, max_batch: usize }

#[derive(Serialize)]
struct ErrorResponse { error: String }

fn parse_language(s: &str) -> Language {
    match s.to_lowercase().as_str() {
        "spanish" | "es" => Language::Spanish,
        "english" | "en" => Language::English,
        "french" | "fr" => Language::French,
        _ => Language::Spanish,
    }
}

fn audio_to_wav_bytes(samples: &[f32], sample_rate: u32) -> Result<Vec<u8>> {
    let spec = WavSpec { channels: 1, sample_rate, bits_per_sample: 16, sample_format: SampleFormat::Int };
    let mut buf = Cursor::new(Vec::new());
    let mut writer = WavWriter::new(&mut buf, spec)?;
    for &s in samples { writer.write_sample((s * 32767.0).clamp(-32768.0, 32767.0) as i16)?; }
    writer.finalize()?;
    Ok(buf.into_inner())
}

fn samples_to_pcm16(samples: &[f32]) -> Vec<u8> {
    let mut pcm = Vec::with_capacity(samples.len() * 2);
    for &s in samples {
        let v = (s * 32767.0).clamp(-32768.0, 32767.0) as i16;
        pcm.extend_from_slice(&v.to_le_bytes());
    }
    pcm
}

fn wav_header(sample_rate: u32, data_len: u32) -> Vec<u8> {
    let mut h = Vec::with_capacity(44);
    h.extend_from_slice(b"RIFF");
    h.extend_from_slice(&(36 + data_len).to_le_bytes());
    h.extend_from_slice(b"WAVE");
    h.extend_from_slice(b"fmt ");
    h.extend_from_slice(&16u32.to_le_bytes());
    h.extend_from_slice(&1u16.to_le_bytes()); // PCM
    h.extend_from_slice(&1u16.to_le_bytes()); // mono
    h.extend_from_slice(&sample_rate.to_le_bytes());
    h.extend_from_slice(&(sample_rate * 2).to_le_bytes()); // byte rate
    h.extend_from_slice(&2u16.to_le_bytes()); // block align
    h.extend_from_slice(&16u16.to_le_bytes()); // bits per sample
    h.extend_from_slice(b"data");
    h.extend_from_slice(&data_len.to_le_bytes());
    h
}

async fn health(State(state): State<Arc<AppState>>) -> Json<HealthResponse> {
    let queue = state.max_inflight.saturating_sub(state.semaphore.available_permits());
    Json(HealthResponse { status: "ok", queue_depth: queue, max_batch: state.max_batch })
}

async fn metrics(State(state): State<Arc<AppState>>) -> String {
    let m = &state.metrics;
    let audio_s = m.audio_seconds_total.load(Ordering::Relaxed) as f64 / 1000.0;
    let gen_s = m.gen_seconds_total.load(Ordering::Relaxed) as f64 / 1000.0;
    let avg_rtf = if gen_s > 0.0 { audio_s / gen_s } else { 0.0 };
    format!(
        "# HELP tts_requests_total Total synthesis requests\ntts_requests_total {}\n\
         # HELP tts_requests_streaming Total streaming requests\ntts_requests_streaming {}\n\
         # HELP tts_errors_total Total errors\ntts_errors_total {}\n\
         # HELP tts_audio_seconds_total Total audio generated (seconds)\ntts_audio_seconds_total {:.1}\n\
         # HELP tts_gen_seconds_total Total generation time (seconds)\ntts_gen_seconds_total {:.1}\n\
         # HELP tts_avg_rtf Average real-time factor\ntts_avg_rtf {:.2}\n\
         # HELP tts_queue_depth Current queue depth\ntts_queue_depth {}\n",
        m.requests_total.load(Ordering::Relaxed),
        m.requests_streaming.load(Ordering::Relaxed),
        m.errors_total.load(Ordering::Relaxed),
        audio_s, gen_s, avg_rtf,
        state.max_inflight.saturating_sub(state.semaphore.available_permits()),
    )
}

async fn synthesize(State(state): State<Arc<AppState>>, Json(req): Json<SpeechRequest>) -> Response {
    state.metrics.requests_total.fetch_add(1, Ordering::Relaxed);
    if req.stream.unwrap_or(false) {
        state.metrics.requests_streaming.fetch_add(1, Ordering::Relaxed);
        return synthesize_streaming(state, req).await;
    }

    let permit = match state.semaphore.clone().try_acquire_owned() {
        Ok(p) => p,
        Err(_) => return (StatusCode::SERVICE_UNAVAILABLE, Json(ErrorResponse { error: "Queue full".into() })).into_response(),
    };

    let voice_clone = decode_ref_audio(&req);

    let (reply_tx, reply_rx) = oneshot::channel();
    let batch_req = BatchRequest {
        text: req.text, language: parse_language(&req.language), voice_clone,
        options: SynthesisOptions { temperature: req.temperature.unwrap_or(0.7), ..SynthesisOptions::default() },
        reply: reply_tx,
    };

    if state.tx.send(batch_req).await.is_err() {
        drop(permit);
        return (StatusCode::INTERNAL_SERVER_ERROR, Json(ErrorResponse { error: "Engine down".into() })).into_response();
    }

    match reply_rx.await {
        Ok(Ok(result)) => {
            drop(permit);
            let duration = result.audio.samples.len() as f32 / result.audio.sample_rate as f32;
            let rtf = duration / result.gen_time_secs;
            state.metrics.audio_seconds_total.fetch_add((duration * 1000.0) as u64, Ordering::Relaxed);
            state.metrics.gen_seconds_total.fetch_add((result.gen_time_secs * 1000.0) as u64, Ordering::Relaxed);
            info!(duration, gen_time = result.gen_time_secs, rtf, "Done");
            match audio_to_wav_bytes(&result.audio.samples, result.audio.sample_rate) {
                Ok(wav) => (StatusCode::OK, [("content-type", "audio/wav"), ("x-rtf", &format!("{rtf:.2}"))], wav).into_response(),
                Err(e) => (StatusCode::INTERNAL_SERVER_ERROR, Json(ErrorResponse { error: format!("{e:#}") })).into_response(),
            }
        }
        Ok(Err(e)) => { drop(permit); state.metrics.errors_total.fetch_add(1, Ordering::Relaxed); (StatusCode::INTERNAL_SERVER_ERROR, Json(ErrorResponse { error: format!("{e:#}") })).into_response() }
        Err(_) => { drop(permit); state.metrics.errors_total.fetch_add(1, Ordering::Relaxed); (StatusCode::INTERNAL_SERVER_ERROR, Json(ErrorResponse { error: "Dropped".into() })).into_response() }
    }
}

async fn synthesize_streaming(state: Arc<AppState>, req: SpeechRequest) -> Response {
    let language = parse_language(&req.language);
    let text = req.text.clone();

    let (tx, rx) = mpsc::channel::<Result<Vec<u8>, String>>(32);

    // Submit to streaming worker
    let stream_req = StreamingRequest { text, language, temperature: req.temperature.unwrap_or(0.7), tx };
    if state.stream_tx.send(stream_req).await.is_err() {
        return (StatusCode::INTERNAL_SERVER_ERROR, Json(ErrorResponse { error: "Stream engine down".into() })).into_response();
    }

    let stream = tokio_stream::wrappers::ReceiverStream::new(rx);
    let body = Body::from_stream(stream.map(|r| r.map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))));

    Response::builder()
        .status(StatusCode::OK)
        .header("content-type", "audio/wav")
        .header("transfer-encoding", "chunked")
        .header("x-audio-format", "pcm-s16le-24000-mono")
        .body(body)
        .unwrap_or_else(|_| (StatusCode::INTERNAL_SERVER_ERROR, "stream build failed").into_response())
}

struct StreamingRequest {
    text: String,
    language: Language,
    temperature: f64,
    tx: mpsc::Sender<Result<Vec<u8>, String>>,
}

fn start_streaming_worker(model: Arc<qwen3_tts::Qwen3TTS>) -> mpsc::Sender<StreamingRequest> {
    let (tx, rx) = mpsc::channel::<StreamingRequest>(16);
    let rx = std::sync::Arc::new(std::sync::Mutex::new(rx));

    std::thread::spawn(move || {
        info!("Streaming worker ready (batched)");

        loop {
            // Block on first request
            let first = {
                let mut guard = rx.lock().unwrap_or_else(|e| { tracing::error!("streaming mutex poisoned, recovering"); e.into_inner() });
                match guard.blocking_recv() {
                    Some(r) => r,
                    None => break,
                }
            };

            // Collect more requests within 50ms for batching
            let mut batch = vec![first];
            let deadline = std::time::Instant::now() + std::time::Duration::from_millis(50);
            loop {
                if batch.len() >= 8 { break; }
                let mut guard = rx.lock().unwrap_or_else(|e| { tracing::error!("streaming mutex poisoned, recovering"); e.into_inner() });
                match guard.try_recv() {
                    Ok(r) => { batch.push(r); }
                    Err(_) => {
                        drop(guard);
                        if std::time::Instant::now() >= deadline { break; }
                        std::thread::sleep(std::time::Duration::from_millis(5));
                    }
                }
            }

            let n = batch.len();
            info!(batch_size = n, "Streaming batch");

            let requests: Vec<(String, qwen3_tts::Language, Option<qwen3_tts::SynthesisOptions>)> = batch.iter()
                .map(|r| (r.text.clone(), r.language,
                    Some(qwen3_tts::SynthesisOptions { temperature: r.temperature, ..Default::default() })
                )).collect();
            let (senders, receivers): (Vec<_>, Vec<_>) = (0..n)
                .map(|_| std::sync::mpsc::channel::<qwen3_tts::AudioBuffer>()).unzip();

            // Send WAV headers
            for req in &batch {
                let _ = req.tx.blocking_send(Ok(wav_header(24000, 0xFFFFFFFF)));
            }

            // Forward decoded audio chunks: std::sync → tokio channels as PCM
            let forwards: Vec<std::thread::JoinHandle<()>> = receivers.into_iter()
                .zip(batch.iter().map(|r| r.tx.clone()))
                .map(|(rx, tx)| {
                    std::thread::spawn(move || {
                        while let Ok(audio) = rx.recv() {
                            let pcm = samples_to_pcm16(&audio.samples);
                            if tx.blocking_send(Ok(pcm)).is_err() { break; }
                        }
                    })
                }).collect();

            // Run batched streaming (decodes + sends every 10 frames ~800ms)
            if let Err(e) = model.synthesize_batch_streaming(&requests, &senders, 10) {
                for req in &batch {
                    let _ = req.tx.blocking_send(Err(format!("{e}")));
                }
            }

            drop(senders);
            for f in forwards { let _ = f.join(); }
        }
    });

    tx
}

fn decode_ref_audio(req: &SpeechRequest) -> Option<VoiceCloneData> {
    req.ref_audio.as_ref().and_then(|b64| {
        let bytes = base64::engine::general_purpose::STANDARD.decode(b64).ok()?;
        let tmp = std::env::temp_dir().join(format!("ref_{}.wav",
            std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap_or_default().as_nanos()));
        match std::fs::write(&tmp, &bytes) {
            Ok(_) => Some(VoiceCloneData { ref_audio_path: tmp, ref_text: req.ref_text.clone() }),
            Err(e) => { tracing::error!("Failed to write ref_audio temp file: {e}"); None }
        }
    })
}

use tokio_stream::StreamExt;

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| "info".into()))
        .init();

    let model_dir = std::env::var("MODEL_DIR").unwrap_or_else(|_| "models/0.6b-base".into());
    let max_batch: usize = std::env::var("MAX_BATCH").ok().and_then(|v| v.parse().ok()).unwrap_or(8);
    let max_wait_ms: u64 = std::env::var("MAX_WAIT_MS").ok().and_then(|v| v.parse().ok()).unwrap_or(200);
    let port: u16 = std::env::var("PORT").ok().and_then(|v| v.parse().ok()).unwrap_or(8090);

    info!(model_dir = %model_dir, max_batch, max_wait_ms, port, "Starting qwen3-tts-server");

    let device = qwen3_tts::auto_device()?;
    info!(?device, "Loading shared model");
    let model = Arc::new(qwen3_tts::Qwen3TTS::from_pretrained(&model_dir, device)?);
    info!("Shared model loaded");

    let tx = BatchEngine::start(model.clone(), BatchEngineConfig { max_batch_size: max_batch, max_wait_ms });
    let stream_tx = start_streaming_worker(model.clone());
    let max_inflight = max_batch * 2;

    let state = Arc::new(AppState {
        tx, stream_tx, semaphore: Arc::new(Semaphore::new(max_inflight)), max_inflight, max_batch,
        metrics: Arc::new(Metrics::new()),
    });

    let app = Router::new()
        .route("/health", get(health))
        .route("/metrics", get(metrics))
        .route("/v1/audio/speech", post(synthesize))
        .with_state(state);

    let listener = tokio::net::TcpListener::bind(format!("0.0.0.0:{port}")).await?;
    info!("Listening on 0.0.0.0:{port}");
    axum::serve(listener, app).await?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;

    #[test]
    fn test_parse_language() {
        assert!(matches!(parse_language("spanish"), Language::Spanish));
        assert!(matches!(parse_language("es"), Language::Spanish));
        assert!(matches!(parse_language("english"), Language::English));
        assert!(matches!(parse_language("en"), Language::English));
        assert!(matches!(parse_language("french"), Language::French));
        assert!(matches!(parse_language("unknown"), Language::Spanish)); // default
    }

    #[test]
    fn test_wav_header_structure() {
        let h = wav_header(24000, 1000);
        assert_eq!(h.len(), 44);
        assert_eq!(&h[0..4], b"RIFF");
        assert_eq!(&h[8..12], b"WAVE");
        assert_eq!(&h[12..16], b"fmt ");
        assert_eq!(&h[36..40], b"data");
    }

    #[test]
    fn test_samples_to_pcm16() {
        let samples = vec![0.0f32, 1.0, -1.0, 0.5];
        let pcm = samples_to_pcm16(&samples);
        assert_eq!(pcm.len(), 8); // 4 samples * 2 bytes
        // 0.0 -> 0
        assert_eq!(i16::from_le_bytes([pcm[0], pcm[1]]), 0);
        // 1.0 -> 32767
        assert_eq!(i16::from_le_bytes([pcm[2], pcm[3]]), 32767);
        // -1.0 -> -32767
        assert_eq!(i16::from_le_bytes([pcm[4], pcm[5]]), -32767);
    }

    #[test]
    fn test_audio_to_wav_bytes() {
        let samples = vec![0.0f32; 100];
        let wav = audio_to_wav_bytes(&samples, 24000).unwrap();
        assert!(wav.len() > 44); // header + data
        assert_eq!(&wav[0..4], b"RIFF");
    }

    #[test]
    fn test_decode_ref_audio_valid_base64() {
        // Minimal WAV: just some bytes encoded as base64
        let bytes = vec![0u8; 44]; // fake WAV header
        let b64 = base64::engine::general_purpose::STANDARD.encode(&bytes);
        let req = SpeechRequest {
            text: "test".into(),
            language: "spanish".into(),
            ref_audio: Some(b64),
            ref_text: Some("ref".into()),
            temperature: None,
            stream: None,
        };
        let result = decode_ref_audio(&req);
        assert!(result.is_some());
        let vc = result.unwrap();
        assert!(vc.ref_audio_path.exists());
        assert_eq!(vc.ref_text, Some("ref".into()));
        // Drop should clean up
        let path = vc.ref_audio_path.clone();
        drop(vc);
        // VoiceCloneData is in batch module, Drop cleans up
    }

    #[test]
    fn test_decode_ref_audio_invalid_base64() {
        let req = SpeechRequest {
            text: "test".into(),
            language: "spanish".into(),
            ref_audio: Some("not-valid-base64!!!".into()),
            ref_text: None,
            temperature: None,
            stream: None,
        };
        assert!(decode_ref_audio(&req).is_none());
    }

    #[test]
    fn test_decode_ref_audio_none() {
        let req = SpeechRequest {
            text: "test".into(),
            language: "spanish".into(),
            ref_audio: None,
            ref_text: None,
            temperature: None,
            stream: None,
        };
        assert!(decode_ref_audio(&req).is_none());
    }

    #[test]
    fn test_voice_clone_data_drop_cleanup() {
        let tmp = std::env::temp_dir().join("test_vc_drop.wav");
        std::fs::write(&tmp, b"test").unwrap();
        assert!(tmp.exists());
        let vc = batch::VoiceCloneData {
            ref_audio_path: tmp.clone(),
            ref_text: None,
        };
        drop(vc);
        assert!(!tmp.exists(), "Drop should have deleted temp file");
    }

    #[test]
    fn test_voice_clone_data_drop_missing_file() {
        // Should not panic if file doesn't exist
        let vc = batch::VoiceCloneData {
            ref_audio_path: "/tmp/nonexistent_vc_test.wav".into(),
            ref_text: None,
        };
        drop(vc); // should not panic
    }

    #[test]
    fn test_metrics_struct() {
        let m = Metrics::new();
        assert_eq!(m.requests_total.load(std::sync::atomic::Ordering::Relaxed), 0);
        m.requests_total.fetch_add(5, std::sync::atomic::Ordering::Relaxed);
        assert_eq!(m.requests_total.load(std::sync::atomic::Ordering::Relaxed), 5);
    }
}
