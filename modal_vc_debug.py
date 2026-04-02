"""Test voice cloning on Modal L4 to reproduce shape bug."""
import modal

app = modal.App("qwen3-tts-vc-debug")

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
def debug_vc():
    import subprocess, os, json
    os.chdir("/src")
    env = {**os.environ, "CUDA_COMPUTE_CAP": "89"}

    # Build
    r = subprocess.run(
        ["bash", "-c", "source /root/.cargo/env && cargo build --release --features cuda,flash-attn"],
        capture_output=True, text=True, timeout=1200, env=env
    )
    if r.returncode != 0:
        return {"error": "build failed", "stderr": r.stderr[-1000:]}

    # Start server in background
    server = subprocess.Popen(
        ["/src/target/release/qwen3-tts-server"],
        env={**os.environ, "MODEL_DIR": "/src/models/0.6b-base", "PORT": "8090",
             "MAX_BATCH": "1", "RUST_LOG": "debug"},
        stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )

    import time
    time.sleep(30)  # wait for model load

    # Test 1: normal synthesis
    import urllib.request
    try:
        req = urllib.request.Request("http://localhost:8090/v1/audio/speech",
            data=json.dumps({"text": "Hola mundo", "language": "spanish"}).encode(),
            headers={"Content-Type": "application/json"})
        resp = urllib.request.urlopen(req, timeout=30)
        normal_size = len(resp.read())
        print(f"Normal synthesis: OK ({normal_size} bytes)")
    except Exception as e:
        print(f"Normal synthesis: FAILED - {e}")

    # Generate ref audio first
    try:
        req = urllib.request.Request("http://localhost:8090/v1/audio/speech",
            data=json.dumps({"text": "Esta es una referencia de audio para clonación de voz.", "language": "spanish"}).encode(),
            headers={"Content-Type": "application/json"})
        resp = urllib.request.urlopen(req, timeout=60)
        ref_wav = resp.read()
        with open("/tmp/ref.wav", "wb") as f:
            f.write(ref_wav)
        print(f"Ref audio generated: {len(ref_wav)} bytes")
    except Exception as e:
        print(f"Ref audio: FAILED - {e}")
        ref_wav = None

    # Test 2: voice cloning
    if ref_wav:
        import base64
        ref_b64 = base64.b64encode(ref_wav).decode()
        try:
            req = urllib.request.Request("http://localhost:8090/v1/audio/speech",
                data=json.dumps({
                    "text": "Buenos días, esta es una prueba de clonación.",
                    "language": "spanish",
                    "ref_audio": ref_b64,
                }).encode(),
                headers={"Content-Type": "application/json"})
            resp = urllib.request.urlopen(req, timeout=60)
            vc_size = len(resp.read())
            print(f"Voice cloning: OK ({vc_size} bytes)")
        except Exception as e:
            print(f"Voice cloning: FAILED - {e}")
            # Get server stderr for error details
            time.sleep(2)

    server.terminate()
    _, stderr = server.communicate(timeout=5)
    stderr_text = stderr.decode()[-3000:]
    print(f"\n=== SERVER STDERR (last 3000 chars) ===\n{stderr_text}")

    return {"stderr": stderr_text}


@app.local_entrypoint()
def main():
    r = debug_vc.remote()
    print(r.get("error", "done"))
