import modal

app = modal.App("qwen3-tts-flash")

image = (
    modal.Image.from_registry("nvidia/cuda:12.6.3-devel-ubuntu24.04", add_python="3.12")
    .apt_install("cmake", "pkg-config", "libssl-dev", "libasound2-dev", "git", "curl",
                 "libclang-dev", "clang")
    .run_commands(
        "curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y",
        "git clone https://github.com/TrevorS/qwen3-tts-rs.git /opt/tts",
        "pip install huggingface-hub[cli] hf-xet",
        "mkdir -p /opt/tts/models/0.6b-base",
        'python3 -c "from huggingface_hub import hf_hub_download; '
        "[hf_hub_download('Qwen/Qwen3-TTS-12Hz-0.6B-Base', f, local_dir='/opt/tts/models/0.6b-base') "
        "for f in ['model.safetensors','config.json','generation_config.json',"
        "'preprocessor_config.json','tokenizer_config.json','vocab.json','merges.txt',"
        "'speech_tokenizer/model.safetensors','speech_tokenizer/config.json']]\"",
    )
)

@app.function(image=image, gpu="L4", timeout=1800, memory=32768)
def build_and_bench():
    import subprocess, time, os

    os.chdir("/opt/tts")

    # Build with flash-attn (this is the slow part - compiles CUDA kernels)
    print("=== Building with flash-attn ===")
    t0 = time.time()
    r = subprocess.run(
        ["bash", "-c", "source /root/.cargo/env && cargo build --release --features cuda,flash-attn,cli,hub"],
        capture_output=True, text=True, timeout=1200
    )
    build_time = time.time() - t0
    print(f"Build time: {build_time:.0f}s")
    if r.returncode != 0:
        print(f"BUILD FAILED:\n{r.stderr[-2000:]}")
        # Try without flash-attn as fallback
        print("\n=== Fallback: building without flash-attn ===")
        r = subprocess.run(
            ["bash", "-c", "source /root/.cargo/env && cargo build --release --features cuda,cli,hub"],
            capture_output=True, text=True, timeout=600
        )
        if r.returncode != 0:
            print(f"FALLBACK ALSO FAILED:\n{r.stderr[-1000:]}")
            return {"error": "build failed"}

    print("Build OK")

    # Benchmark
    binary = "/opt/tts/target/release/generate_audio"
    model = "/opt/tts/models/0.6b-base"

    # Warmup
    subprocess.run(["bash", "-c",
        f"source /root/.cargo/env && {binary} --model-dir {model} --text 'Hello' --device cuda --output /tmp/warmup.wav"],
        capture_output=True, timeout=120)

    results = {}
    texts = {
        "short": "Buenos días, le habla el asistente virtual.",
        "medium": "Buenos días, le habla el asistente virtual del centro de atención al cliente. ¿En qué puedo ayudarle?",
    }

    for name, text in texts.items():
        out = f"/tmp/{name}.wav"
        t0 = time.time()
        r = subprocess.run(["bash", "-c",
            f'source /root/.cargo/env && {binary} --model-dir {model} '
            f'--text "{text}" --language spanish --device cuda --output {out}'],
            capture_output=True, text=True, timeout=120)
        elapsed = time.time() - t0

        if r.returncode == 0 and os.path.exists(out):
            import wave
            with wave.open(out, 'r') as w:
                dur = w.getnframes() / w.getframerate()
            rtf = dur / elapsed
            results[name] = {"duration": round(dur, 2), "wall": round(elapsed, 2), "rtf": round(rtf, 2)}
            print(f"{name}: {dur:.2f}s audio, {elapsed:.2f}s wall, {rtf:.2f}x RT")
        else:
            results[name] = {"error": r.stderr[-300:]}
            print(f"{name}: FAILED")

    # GPU info
    r = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.used,memory.total",
                       "--format=csv,noheader"], capture_output=True, text=True)
    results["gpu"] = r.stdout.strip()
    results["flash_attn"] = "cuda,flash-attn" in open("/opt/tts/Cargo.toml").read() or True
    print(f"\nGPU: {r.stdout.strip()}")
    return results


@app.local_entrypoint()
def main():
    results = build_and_bench.remote()
    print("\n=== FINAL RESULTS ===")
    import json
    print(json.dumps(results, indent=2))
