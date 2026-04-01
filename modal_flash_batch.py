import modal

app = modal.App("qwen3-tts-flash-batch")

image = (
    modal.Image.from_registry("nvidia/cuda:12.6.3-devel-ubuntu24.04", add_python="3.12")
    .apt_install("cmake", "pkg-config", "libssl-dev", "libasound2-dev", "git", "curl",
                 "libclang-dev", "clang")
    .run_commands(
        "curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y",
        "git clone https://github.com/TrevorS/qwen3-tts-rs.git /opt/upstream",
        "pip install huggingface-hub[cli] hf-xet",
        "mkdir -p /opt/models/0.6b-base",
        'python3 -c "from huggingface_hub import hf_hub_download; '
        "[hf_hub_download('Qwen/Qwen3-TTS-12Hz-0.6B-Base', f, local_dir='/opt/models/0.6b-base') "
        "for f in ['model.safetensors','config.json','generation_config.json',"
        "'preprocessor_config.json','tokenizer_config.json','vocab.json','merges.txt',"
        "'speech_tokenizer/model.safetensors','speech_tokenizer/config.json']]\"",
    )
)

vol = modal.Volume.from_name("tts-build-cache", create_if_missing=True)

@app.function(image=image, gpu="L4", timeout=1800, memory=32768, volumes={"/cache": vol})
def build_and_bench():
    import subprocess, time, os, shutil

    # Copy our repo from volume cache or use upstream
    if os.path.exists("/cache/qwen3-tts-server"):
        print("Using cached repo")
        if not os.path.exists("/opt/tts"):
            shutil.copytree("/cache/qwen3-tts-server", "/opt/tts")
    else:
        print("Using upstream qwen3-tts-rs")
        shutil.copytree("/opt/upstream", "/opt/tts")

    os.chdir("/opt/tts")

    # Build with flash-attn
    print("=== Building with flash-attn ===")
    t0 = time.time()
    r = subprocess.run(
        ["bash", "-c", "source /root/.cargo/env && cargo build --release --features cuda,flash-attn,cli,hub"],
        capture_output=True, text=True, timeout=1200
    )
    print(f"Build: {time.time()-t0:.0f}s, rc={r.returncode}")
    if r.returncode != 0:
        print(f"STDERR: {r.stderr[-500:]}")
        return {"error": "flash-attn build failed"}

    binary = "/opt/tts/target/release/generate_audio"
    model = "/opt/models/0.6b-base"

    # Warmup
    subprocess.run(["bash", "-c",
        f"source /root/.cargo/env && {binary} --model-dir {model} --text 'Hola' --device cuda --output /tmp/w.wav"],
        capture_output=True, timeout=120)
    print("Warmup done")

    # Single stream benchmark
    results = {}
    text = "Buenos días, le habla el asistente virtual del centro de atención. ¿En qué puedo ayudarle?"

    for trial in range(3):
        out = f"/tmp/t{trial}.wav"
        t0 = time.time()
        subprocess.run(["bash", "-c",
            f'source /root/.cargo/env && {binary} --model-dir {model} '
            f'--text "{text}" --language spanish --device cuda --output {out}'],
            capture_output=True, timeout=120)
        elapsed = time.time() - t0
        import wave
        if os.path.exists(out):
            with wave.open(out, 'r') as w:
                dur = w.getnframes() / w.getframerate()
            print(f"Trial {trial}: {dur:.2f}s audio, {elapsed:.2f}s wall, {dur/elapsed:.2f}x RT")
            results[f"single_{trial}"] = {"dur": round(dur,2), "wall": round(elapsed,2), "rtf": round(dur/elapsed,2)}

    # Concurrent benchmark (process pool since upstream doesn't have our batching)
    import concurrent.futures
    def run_one(i):
        out = f"/tmp/c{i}.wav"
        t0 = time.time()
        subprocess.run(["bash", "-c",
            f'source /root/.cargo/env && {binary} --model-dir {model} '
            f'--text "Buenos días, asistente {i}. ¿En qué puedo ayudarle?" '
            f'--language spanish --device cuda --output {out}'],
            capture_output=True, timeout=120)
        elapsed = time.time() - t0
        if os.path.exists(out):
            import wave
            with wave.open(out, 'r') as w:
                dur = w.getnframes() / w.getframerate()
            return {"dur": dur, "wall": elapsed}
        return {"error": "no output"}

    for n in [2, 4]:
        t0 = time.time()
        with concurrent.futures.ThreadPoolExecutor(max_workers=n) as ex:
            futs = [ex.submit(run_one, i) for i in range(n)]
            res = [f.result() for f in futs]
        wall = time.time() - t0
        total_audio = sum(r.get("dur",0) for r in res)
        print(f"Concurrent {n}: {total_audio:.1f}s audio, {wall:.1f}s wall, {total_audio/wall:.2f}x RT")
        results[f"conc_{n}"] = {"total_audio": round(total_audio,1), "wall": round(wall,1), "throughput": round(total_audio/wall,2)}

    r = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.used,memory.total",
                       "--format=csv,noheader"], capture_output=True, text=True)
    results["gpu"] = r.stdout.strip()
    return results


@app.local_entrypoint()
def main():
    results = build_and_bench.remote()
    print("\n=== RESULTS ===")
    import json
    print(json.dumps(results, indent=2))
