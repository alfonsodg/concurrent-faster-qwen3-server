use qwen3_tts::{Qwen3TTS, Language};
use std::time::Instant;

fn main() {
    let device = qwen3_tts::auto_device().unwrap();
    let model = Qwen3TTS::from_pretrained("models/0.6b-base", device).unwrap();
    let _ = model.synthesize("Hola", None);
    println!("Ready\n");

    // Profile batch=4 with timing breakdown
    let requests: Vec<(String, Language, Option<qwen3_tts::SynthesisOptions>)> = (0..4)
        .map(|i| (format!("Buenos días, asistente {}. ¿En qué puedo ayudarle?", i+1), Language::Spanish, None))
        .collect();

    let t0 = Instant::now();
    let audios = model.synthesize_batch(&requests).unwrap();
    let elapsed = t0.elapsed().as_secs_f32();
    let total_audio: f32 = audios.iter().map(|a| a.samples.len() as f32 / a.sample_rate as f32).sum();
    println!("Batch 4: {:.1}s audio, {:.1}s wall, {:.2}x RT\n", total_audio, elapsed, total_audio / elapsed);

    // Now profile single request to measure code predictor vs talker
    // Use synthesize_with_timing if available, otherwise just time overall
    let t0 = Instant::now();
    let (audio, timing) = model.synthesize_with_timing(
        "Buenos días, le habla el asistente virtual del centro de atención al cliente.",
        qwen3_tts::Speaker::Serena,
        Language::Spanish,
        None,
    ).unwrap();
    let dur = audio.samples.len() as f32 / audio.sample_rate as f32;
    println!("Single: {:.2}s audio in {:.2}s", dur, t0.elapsed().as_secs_f32());
    println!("Timing: {:?}", timing);
}
