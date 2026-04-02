import modal

app = modal.App("qwen3-tts-profile")

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
def profile():
    import subprocess, time, os
    os.chdir("/src")
    env = {**os.environ, "CUDA_COMPUTE_CAP": "89"}
    r = subprocess.run(
        ["bash", "-c", "source /root/.cargo/env && cargo build --release --features cuda,flash-attn"],
        capture_output=True, text=True, timeout=1200, env=env
    )
    if r.returncode != 0:
        return {"error": r.stderr[-1000:]}

    r = subprocess.run(["/src/target/release/profile"], capture_output=True, text=True, timeout=300, cwd="/src")
    print(r.stdout)
    if r.stderr:
        print(f"STDERR: {r.stderr[-500:]}")
    return {"output": r.stdout}


@app.local_entrypoint()
def main():
    results = profile.remote()
    print(results.get("output", results.get("error", "unknown")))
