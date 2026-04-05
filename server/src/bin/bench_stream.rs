use qwen3_tts::{Qwen3TTS, Language, AudioBuffer};
use std::sync::mpsc;
use std::time::Instant;

fn main() {
    let device = qwen3_tts::auto_device().unwrap();
    let model = Qwen3TTS::from_pretrained("models/0.6b-base", device).unwrap();
    println!("Model loaded");
    let _ = model.synthesize("Hola", None);
    println!("Warmup done");

    for n in [2, 4, 8] {
        let requests: Vec<(String, Language, Option<qwen3_tts::SynthesisOptions>)> = (0..n)
            .map(|i| (format!("Buenos días, le habla el asistente número {}. ¿En qué puedo ayudarle?", i+1), Language::Spanish, None))
            .collect();

        let mut senders = Vec::new();
        let mut receivers = Vec::new();
        for _ in 0..n {
            let (tx, rx) = mpsc::channel::<AudioBuffer>();
            senders.push(tx);
            receivers.push(rx);
        }

        let t0 = Instant::now();
        let ttfa = std::sync::Arc::new(std::sync::Mutex::new(vec![0.0f64; n]));

        // Spawn receiver threads to measure TTFA
        let mut handles = Vec::new();
        for i in 0..n {
            let rx = receivers.remove(0);
            let t0_clone = t0;
            let ttfa_clone = ttfa.clone();
            handles.push(std::thread::spawn(move || {
                let mut total_samples = 0usize;
                let mut first = true;
                while let Ok(audio) = rx.recv() {
                    if first {
                        ttfa_clone.lock().unwrap()[i] = t0_clone.elapsed().as_secs_f64();
                        first = false;
                    }
                    total_samples += audio.samples.len();
                }
                total_samples as f32 / 24000.0
            }));
        }

        // Run batched streaming
        let chunk_frames = 5; // ~400ms audio per chunk
        if let Err(e) = model.synthesize_batch_streaming(&requests, &senders, chunk_frames, &vec![None; requests.len()]) {
            println!("Batch stream {n}: FAILED: {e:#}");
            continue;
        }
        drop(senders); // close channels

        let mut total_audio = 0.0f32;
        for h in handles {
            total_audio += h.join().unwrap();
        }
        let elapsed = t0.elapsed().as_secs_f32();
        let ttfa_vals = ttfa.lock().unwrap();
        let max_ttfa = ttfa_vals.iter().cloned().fold(0.0f64, f64::max);
        let avg_ttfa = ttfa_vals.iter().sum::<f64>() / n as f64;

        println!("Batch stream {:>2}: {:.1}s audio, {:.1}s wall, {:.2}x RT, TTFA avg={:.0}ms max={:.0}ms",
            n, total_audio, elapsed, total_audio / elapsed, avg_ttfa * 1000.0, max_ttfa * 1000.0);
    }
}
