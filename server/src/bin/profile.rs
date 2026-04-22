use qwen3_tts::{Qwen3TTS, Language, SynthesisOptions};
use std::time::Instant;

fn main() {
    let device = qwen3_tts::auto_device().unwrap();
    let model = Qwen3TTS::from_pretrained("models/0.6b-base", device).unwrap();
    let _ = model.synthesize("Hola", None);
    println!("Ready\n");

    // Detailed timing with sync points
    let text = "Buenos días, le habla el asistente virtual del centro de atención al cliente.";

    // Run 3 times, take last (warm)
    for run in 0..3 {
        let t0 = Instant::now();
        let (audio, timing) = model.synthesize_with_timing(
            text, qwen3_tts::Speaker::Serena, Language::Spanish, None,
        ).unwrap();
        let dur = audio.samples.len() as f32 / audio.sample_rate as f32;
        if run == 2 {
            println!("Single: {:.2}s audio in {:.2}s", dur, t0.elapsed().as_secs_f32());
            println!("  Prefill:    {:.1}ms", timing.prefill_ms);
            println!("  Generation: {:.1}ms ({} frames, {:.1}ms/frame)",
                timing.generation_ms, timing.generation_frames,
                timing.generation_ms / timing.generation_frames as f64);
            println!("  Decode:     {:.1}ms", timing.decode_ms);
        }
    }

    // Now measure code predictor vs transformer by running batch=1 with
    // our batch path (which has separate code predictor and transformer steps)
    // vs single path. The difference tells us the overhead.

    // Batch sizes with adaptive max_length
    for n in [1, 4, 8, 16] {
        let opts = SynthesisOptions { max_length: 122, ..SynthesisOptions::default() };
        let requests: Vec<(String, Language, Option<SynthesisOptions>)> = (0..n)
            .map(|i| (
                format!("Buenos días, le habla el asistente número {}. ¿En qué puedo ayudarle?", i+1),
                Language::Spanish, Some(opts.clone()),
            )).collect();

        // Warmup
        let _ = model.synthesize_batch(&requests);

        let t0 = Instant::now();
        match model.synthesize_batch(&requests) {
            Ok(audios) => {
                let elapsed = t0.elapsed().as_secs_f32();
                let total_audio: f32 = audios.iter()
                    .map(|a| a.samples.len() as f32 / a.sample_rate as f32).sum();
                let frames = total_audio / (1.0/12.0); // approx frames
                println!("\nBatch {:>2}: {:.1}s audio, {:.2}s wall, {:.2}x RT, {:.0}ms/req, ~{:.0} frames",
                    n, total_audio, elapsed, total_audio / elapsed, elapsed * 1000.0 / n as f32, frames);
            }
            Err(e) => println!("\nBatch {n}: FAILED: {e:#}"),
        }
    }

    println!("\nStreaming TTFA:");
    let t0 = Instant::now();
    let mut stream = model.synthesize_streaming(
        "Buenos días, le habla el asistente.", qwen3_tts::Speaker::Serena,
        Language::Spanish, SynthesisOptions::default(),
    ).unwrap();
    if let Ok(Some(_)) = stream.next_chunk() {
        println!("  {:.0}ms", t0.elapsed().as_millis());
    }
}
