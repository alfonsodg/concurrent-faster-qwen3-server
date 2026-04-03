use qwen3_tts::{Qwen3TTS, Language, SynthesisOptions};
use std::time::Instant;

fn main() {
    let device = qwen3_tts::auto_device().unwrap();
    let model = Qwen3TTS::from_pretrained("models/0.6b-base", device).unwrap();
    println!("Model loaded");

    let _ = model.synthesize("Hola", None);
    println!("Warmup done");

    // Adaptive max_length benchmark
    for n in [1, 2, 4, 8, 16] {
        let text = "Buenos días, le habla el asistente número uno. ¿En qué puedo ayudarle?";
        let word_count = text.split_whitespace().count();
        let adaptive_max = ((word_count * 6) + 50).min(512).max(100);

        let requests: Vec<(String, Language, Option<SynthesisOptions>)> = (0..n)
            .map(|i| (
                format!("Buenos días, le habla el asistente número {}. ¿En qué puedo ayudarle?", i+1),
                Language::Spanish,
                Some(SynthesisOptions { max_length: adaptive_max, ..SynthesisOptions::default() }),
            ))
            .collect();

        let t0 = Instant::now();
        match model.synthesize_batch(&requests) {
            Ok(audios) => {
                let elapsed = t0.elapsed().as_secs_f32();
                let total_audio: f32 = audios.iter()
                    .map(|a| a.samples.len() as f32 / a.sample_rate as f32).sum();
                println!("Batch {:>2} (max={}): {:.1}s audio, {:>5.1}s wall, {:.2}x RT, {:.1}s/req",
                    n, adaptive_max, total_audio, elapsed, total_audio / elapsed, elapsed / n as f32);
            }
            Err(e) => println!("Batch {n} (max={adaptive_max}): FAILED: {e:#}"),
        }
    }

    // Sequential baseline
    let t0 = Instant::now();
    let mut total_audio = 0.0f32;
    for i in 0..4 {
        let text = format!("Buenos días, le habla el asistente número {}. ¿En qué puedo ayudarle?", i+1);
        let a = model.synthesize(&text, None).unwrap();
        total_audio += a.samples.len() as f32 / a.sample_rate as f32;
    }
    let elapsed = t0.elapsed().as_secs_f32();
    println!("Seq   4: {:.1}s audio, {:>5.1}s wall, {:.2}x RT", total_audio, elapsed, total_audio / elapsed);

    // Streaming TTFA
    let t0 = Instant::now();
    let mut stream = model.synthesize_streaming(
        "Buenos días, le habla el asistente.", qwen3_tts::Speaker::Serena,
        Language::Spanish, SynthesisOptions::default(),
    ).unwrap();
    if let Ok(Some(_)) = stream.next_chunk() {
        println!("\nStreaming TTFA: {:.0}ms", t0.elapsed().as_millis());
    }
}
