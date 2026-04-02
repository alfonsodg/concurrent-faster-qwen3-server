"""Test ICL voice cloning (with ref_text) on Modal L4."""
import modal

app = modal.App("qwen3-tts-icl-test")

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


@app.function(image=image, gpu="L4", timeout=1200, memory=32768)
def test_icl():
    import subprocess, os, json, time, base64, urllib.request
    os.chdir("/src")
    env = {**os.environ, "CUDA_COMPUTE_CAP": "89"}

    r = subprocess.run(
        ["bash", "-c", "source /root/.cargo/env && cargo build --release --features cuda,flash-attn"],
        capture_output=True, text=True, timeout=1200, env=env
    )
    if r.returncode != 0:
        return {"error": r.stderr[-500:]}

    server = subprocess.Popen(
        ["/src/target/release/qwen3-tts-server"],
        env={**os.environ, "MODEL_DIR": "/src/models/0.6b-base", "PORT": "8090",
             "MAX_BATCH": "1", "RUST_LOG": "info"},
        stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    time.sleep(40)

    # Generate ref audio
    req = urllib.request.Request("http://localhost:8090/v1/audio/speech",
        data=json.dumps({"text": "Hola, buenos días, mi nombre es Enrique.", "language": "spanish"}).encode(),
        headers={"Content-Type": "application/json"})
    resp = urllib.request.urlopen(req, timeout=60)
    ref_wav = resp.read()
    print(f"Ref audio: {len(ref_wav)} bytes")

    # Test ICL voice cloning (with ref_text)
    ref_b64 = base64.b64encode(ref_wav).decode()
    try:
        req = urllib.request.Request("http://localhost:8090/v1/audio/speech",
            data=json.dumps({
                "text": "Buenos días, le llamo para ofrecerle una mejora de plan.",
                "language": "spanish",
                "ref_audio": ref_b64,
                "ref_text": "Hola, buenos días, mi nombre es Enrique.",
            }).encode(),
            headers={"Content-Type": "application/json"})
        resp = urllib.request.urlopen(req, timeout=120)
        result = resp.read()
        dur = (len(result) - 44) / (24000 * 2)
        print(f"ICL Voice cloning: OK ({len(result)} bytes, {dur:.1f}s audio)")
    except Exception as e:
        print(f"ICL Voice cloning: FAILED - {e}")

    # Test x_vector_only (without ref_text)
    try:
        req = urllib.request.Request("http://localhost:8090/v1/audio/speech",
            data=json.dumps({
                "text": "Buenos días, le llamo para ofrecerle una mejora de plan.",
                "language": "spanish",
                "ref_audio": ref_b64,
            }).encode(),
            headers={"Content-Type": "application/json"})
        resp = urllib.request.urlopen(req, timeout=120)
        result = resp.read()
        dur = (len(result) - 44) / (24000 * 2)
        print(f"x_vector Voice cloning: OK ({len(result)} bytes, {dur:.1f}s audio)")
    except Exception as e:
        print(f"x_vector Voice cloning: FAILED - {e}")

    server.terminate()
    _, stderr = server.communicate(timeout=5)
    err = stderr.decode()
    if "shape" in err.lower() or "error" in err.lower():
        print(f"\nServer errors:\n{err[-1000:]}")
    return {"ok": True}


@app.local_entrypoint()
def main():
    r = test_icl.remote()
    print(r)
