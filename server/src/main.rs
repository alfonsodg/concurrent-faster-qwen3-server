mod batch;

use anyhow::Result;
use axum::{
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
use std::{io::Cursor, sync::Arc};
use tokio::sync::{mpsc, oneshot, Semaphore};
use tracing::info;

struct AppState {
    tx: mpsc::Sender<BatchRequest>,
    semaphore: Arc<Semaphore>,
    max_batch: usize,
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
}

fn default_language() -> String {
    "spanish".into()
}

#[derive(Serialize)]
struct HealthResponse {
    status: &'static str,
    queue_depth: usize,
    max_batch: usize,
}

#[derive(Serialize)]
struct ErrorResponse {
    error: String,
}

fn parse_language(s: &str) -> Language {
    match s.to_lowercase().as_str() {
        "spanish" | "es" => Language::Spanish,
        "english" | "en" => Language::English,
        "chinese" | "zh" => Language::Chinese,
        "french" | "fr" => Language::French,
        "japanese" | "ja" => Language::Japanese,
        "korean" | "ko" => Language::Korean,
        _ => Language::Spanish,
    }
}

fn audio_to_wav_bytes(samples: &[f32], sample_rate: u32) -> Result<Vec<u8>> {
    let spec = WavSpec {
        channels: 1,
        sample_rate,
        bits_per_sample: 16,
        sample_format: SampleFormat::Int,
    };
    let mut buf = Cursor::new(Vec::new());
    let mut writer = WavWriter::new(&mut buf, spec)?;
    for &s in samples {
        writer.write_sample((s * 32767.0).clamp(-32768.0, 32767.0) as i16)?;
    }
    writer.finalize()?;
    Ok(buf.into_inner())
}

async fn health(State(state): State<Arc<AppState>>) -> Json<HealthResponse> {
    let queue = state.max_batch - state.semaphore.available_permits();
    Json(HealthResponse {
        status: "ok",
        queue_depth: queue,
        max_batch: state.max_batch,
    })
}

async fn synthesize(
    State(state): State<Arc<AppState>>,
    Json(req): Json<SpeechRequest>,
) -> Response {
    let permit = match state.semaphore.clone().try_acquire_owned() {
        Ok(p) => p,
        Err(_) => {
            return (
                StatusCode::SERVICE_UNAVAILABLE,
                Json(ErrorResponse { error: "Queue full".into() }),
            ).into_response();
        }
    };

    // Handle voice clone ref audio
    let voice_clone = if let Some(b64) = &req.ref_audio {
        match base64::engine::general_purpose::STANDARD.decode(b64) {
            Ok(bytes) => {
                let tmp = std::env::temp_dir().join(format!(
                    "ref_{}.wav",
                    std::time::SystemTime::now()
                        .duration_since(std::time::UNIX_EPOCH)
                        .unwrap()
                        .as_nanos()
                ));
                if let Err(e) = std::fs::write(&tmp, &bytes) {
                    return (
                        StatusCode::INTERNAL_SERVER_ERROR,
                        Json(ErrorResponse { error: format!("Failed to write ref audio: {e}") }),
                    ).into_response();
                }
                Some(VoiceCloneData {
                    ref_audio_path: tmp,
                    ref_text: req.ref_text.clone(),
                })
            }
            Err(e) => {
                return (
                    StatusCode::BAD_REQUEST,
                    Json(ErrorResponse { error: format!("Invalid base64: {e}") }),
                ).into_response();
            }
        }
    } else {
        None
    };

    let (reply_tx, reply_rx) = oneshot::channel();
    let batch_req = BatchRequest {
        text: req.text,
        language: parse_language(&req.language),
        voice_clone,
        options: SynthesisOptions {
            temperature: req.temperature.unwrap_or(0.7),
            ..SynthesisOptions::default()
        },
        reply: reply_tx,
    };

    if state.tx.send(batch_req).await.is_err() {
        drop(permit);
        return (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ErrorResponse { error: "Engine shut down".into() }),
        ).into_response();
    }

    match reply_rx.await {
        Ok(Ok(result)) => {
            drop(permit);
            let duration = result.audio.samples.len() as f32 / result.audio.sample_rate as f32;
            let rtf = duration / result.gen_time_secs;
            info!(duration, gen_time = result.gen_time_secs, rtf, "Done");
            match audio_to_wav_bytes(&result.audio.samples, result.audio.sample_rate) {
                Ok(wav) => (
                    StatusCode::OK,
                    [
                        ("content-type", "audio/wav"),
                        ("x-rtf", &format!("{rtf:.2}")),
                        ("x-duration", &format!("{duration:.2}")),
                        ("x-gen-time", &format!("{:.2}", result.gen_time_secs)),
                    ],
                    wav,
                ).into_response(),
                Err(e) => (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    Json(ErrorResponse { error: format!("{e:#}") }),
                ).into_response(),
            }
        }
        Ok(Err(e)) => {
            drop(permit);
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(ErrorResponse { error: format!("{e:#}") }),
            ).into_response()
        }
        Err(_) => {
            drop(permit);
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(ErrorResponse { error: "Engine dropped request".into() }),
            ).into_response()
        }
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "info".into()),
        )
        .init();

    let model_dir = std::env::var("MODEL_DIR").unwrap_or_else(|_| "models/0.6b-base".into());
    let max_batch: usize = std::env::var("MAX_BATCH").ok().and_then(|v| v.parse().ok()).unwrap_or(8);
    let max_wait_ms: u64 = std::env::var("MAX_WAIT_MS").ok().and_then(|v| v.parse().ok()).unwrap_or(50);
    let port: u16 = std::env::var("PORT").ok().and_then(|v| v.parse().ok()).unwrap_or(8090);

    info!(model_dir = %model_dir, max_batch, max_wait_ms, port, "Starting qwen3-tts-server");

    // Start batch engine on dedicated thread
    let tx = BatchEngine::start(BatchEngineConfig {
        max_batch_size: max_batch,
        max_wait_ms,
        model_dir,
    });

    // Allow max_batch * 2 in-flight requests (batch + queue)
    let max_inflight = max_batch * 2;
    let state = Arc::new(AppState {
        tx,
        semaphore: Arc::new(Semaphore::new(max_inflight)),
        max_batch,
    });

    let app = Router::new()
        .route("/health", get(health))
        .route("/v1/audio/speech", post(synthesize))
        .with_state(state);

    let listener = tokio::net::TcpListener::bind(format!("0.0.0.0:{port}")).await?;
    info!("Listening on 0.0.0.0:{port}");
    axum::serve(listener, app).await?;
    Ok(())
}
