import modal, time, json

app = modal.App("qwen3-tts-h100")

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

@app.function(image=image, gpu="H100", timeout=1800, memory=65536)
def bench():
    import subprocess, os
    os.chdir("/opt/tts")
    model = "/opt/tts/models/0.6b-base"
    text = "Buenos días, le habla el asistente virtual del centro de atención. ¿En qué puedo ayudarle?"
    results = {}

    for mode, features in [("cuda_only", "cuda,cli,hub"), ("flash_attn", "cuda,flash-attn,cli,hub")]:
        print(f"\n=== Building {mode} ===")
        t0 = time.time()
        r = subprocess.run(
            ["bash", "-c", f"source /root/.cargo/env && cargo build --release --features {features}"],
            capture_output=True, text=True, timeout=1200
        )
        bt = time.time() - t0
        print(f"Build: {bt:.0f}s, rc={r.returncode}")
        if r.returncode != 0:
            print(f"FAILED: {r.stderr[-300:]}")
            results[mode] = {"error": "build failed"}
            continue

        binary = "/opt/tts/target/release/generate_audio"

        # Warmup
        subprocess.run(["bash", "-c",
            f'source /root/.cargo/env && {binary} --model-dir {model} --text "Hola" --device cuda --output /tmp/w.wav'],
            capture_output=True, timeout=60)

        # 3 trials
        rtfs = []
        for i in range(3):
            out = f"/tmp/{mode}_{i}.wav"
            t0 = time.time()
            subprocess.run(["bash", "-c",
                f'source /root/.cargo/env && {binary} --model-dir {model} '
                f'--text "{text}" --language spanish --device cuda --output {out}'],
                capture_output=True, timeout=60)
            elapsed = time.time() - t0
            if os.path.exists(out):
                import wave
                with wave.open(out, 'r') as w:
                    dur = w.getnframes() / w.getframerate()
                rtf = dur / elapsed
                rtfs.append(rtf)
                print(f"  {mode} trial {i}: {dur:.2f}s audio, {elapsed:.2f}s wall, {rtf:.2f}x RT")

        if rtfs:
            avg = sum(rtfs) / len(rtfs)
            results[mode] = {"avg_rtf": round(avg, 2), "trials": [round(r, 2) for r in rtfs]}

        # Clean build for next mode
        subprocess.run(["bash", "-c", "source /root/.cargo/env && cargo clean"], capture_output=True)

    # Compare
    if "cuda_only" in results and "flash_attn" in results:
        base = results["cuda_only"].get("avg_rtf", 0)
        flash = results["flash_attn"].get("avg_rtf", 0)
        if base > 0:
            improvement = (flash - base) / base * 100
            results["improvement_pct"] = round(improvement, 1)
            print(f"\n=== Flash-attn improvement: {improvement:.1f}% ===")

    r = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
                      capture_output=True, text=True)
    results["gpu"] = r.stdout.strip()
    return results

@app.local_entrypoint()
def main():
    r = bench.remote()
    print("\n=== RESULTS ===")
    print(json.dumps(r, indent=2))
