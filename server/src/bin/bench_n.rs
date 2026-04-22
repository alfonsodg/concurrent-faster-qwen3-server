use qwen3_tts::{Qwen3TTS, Language};
use std::time::Instant;

fn main() {
    let n: usize = std::env::args().nth(1).and_then(|s| s.parse().ok()).unwrap_or(4);
    let device = qwen3_tts::auto_device().unwrap();
    let model = Qwen3TTS::from_pretrained("models/0.6b-base", device).unwrap();
    let _ = model.synthesize("Hola", None);

    let requests: Vec<(String, Language, Option<qwen3_tts::SynthesisOptions>)> = (0..n)
        .map(|i| (format!("Buenos días, asistente {}. ¿En qué puedo ayudarle?", i+1), Language::Spanish, None))
        .collect();

    let t0 = Instant::now();
    match model.synthesize_batch(&requests) {
        Ok(audios) => {
            let elapsed = t0.elapsed().as_secs_f32();
            let total_audio: f32 = audios.iter().map(|a| a.samples.len() as f32 / a.sample_rate as f32).sum();
            println!("Batch {}: {:.1}s audio, {:.1}s wall, {:.2}x RT, {:.1}s/req",
                n, total_audio, elapsed, total_audio / elapsed, elapsed / n as f32);
        }
        Err(e) => println!("Batch {n}: FAILED: {e:#}"),
    }
}
