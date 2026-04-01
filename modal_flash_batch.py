import modal

app = modal.App("qwen3-tts-flash-batch")

image = (
    modal.Image.from_registry("nvidia/cuda:12.6.3-devel-ubuntu24.04", add_python="3.12")
    .apt_install("cmake", "pkg-config", "libssl-dev", "libasound2-dev", "git", "curl",
                 "libclang-dev", "clang")
    .run_commands(
        "curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y",
        "pip install huggingface-hub[cli] hf-xet",
        "mkdir -p /src/models/0.6b-base",
        'python3 -c "from huggingface_hub import hf_hub_download; '
        "[hf_hub_download('Qwen/Qwen3-TTS-12Hz-0.6B-Base', f, local_dir='/src/models/0.6b-base') "
        "for f in ['model.safetensors','config.json','generation_config.json',"
        "'preprocessor_config.json','tokenizer_config.json','vocab.json','merges.txt',"
        "'speech_tokenizer/model.safetensors','speech_tokenizer/config.json']]\"",
    )
    .add_local_dir(".", remote_path="/src", ignore=[
        "target", "models", "__pycache__", ".git",
        "*.wav", "*.gguf", "*.onnx", "*.pyc",
    ])
)


@app.function(image=image, gpu="L4", timeout=1800, memory=32768)
def build_and_bench():
    import subprocess, time, os

    os.chdir("/src")

    print("=== Building optimized fork with flash-attn ===")
    env = {**os.environ, "CUDA_COMPUTE_CAP": "89"}
    t0 = time.time()
    r = subprocess.run(
        ["bash", "-c", "source /root/.cargo/env && cargo build --release --features cuda,flash-attn"],
        capture_output=True, text=True, timeout=1200, env=env
    )
    build_time = time.time() - t0
    print(f"Build: {build_time:.0f}s, rc={r.returncode}")
    if r.returncode != 0:
        print(f"STDERR: {r.stderr[-2000:]}")
        return {"error": "build failed", "stderr": r.stderr[-2000:]}

    # bench_batch uses relative path "models/0.6b-base"
    print("\n=== Running bench_batch ===")
    r = subprocess.run(
        ["/src/target/release/bench_batch"],
        capture_output=True, text=True, timeout=600, cwd="/src"
    )
    print(r.stdout if r.stdout else "")
    if r.stderr:
        print(f"STDERR: {r.stderr[-500:]}")

    gpu = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.used,memory.total",
                         "--format=csv,noheader"], capture_output=True, text=True)
    print(f"GPU: {gpu.stdout.strip()}")

    return {"output": r.stdout, "gpu": gpu.stdout.strip(), "build_time": round(build_time)}


@app.local_entrypoint()
def main():
    results = build_and_bench.remote()
    print("\n=== RESULTS ===")
    import json
    print(json.dumps(results, indent=2, ensure_ascii=False))
