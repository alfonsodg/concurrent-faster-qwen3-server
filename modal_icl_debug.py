"""Debug ICL voice clone warm-up leak with real ref audio."""
import modal

app = modal.App("qwen3-tts-icl-debug")

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
        "curl -s -o /tmp/ok-mujer.wav https://okbot.apulab.info/static/debug-audio/ok-mujer.wav",
    )
    .add_local_dir(".", remote_path="/src", ignore=[
        "target", "models", "__pycache__", ".git",
        "*.wav", "*.gguf", "*.onnx", "*.pyc",
    ])
)


@app.function(image=image, gpu="L4", timeout=1200, memory=32768)
def debug():
    import subprocess, os, json, time, base64, urllib.request, wave
    os.chdir("/src")
    env = {**os.environ, "CUDA_COMPUTE_CAP": "89"}

    r = subprocess.run(
        ["bash", "-c", "source /root/.cargo/env && cargo build --release --features cuda,flash-attn"],
        capture_output=True, text=True, timeout=1200, env=env
    )
    if r.returncode != 0:
        print(f"BUILD FAILED: {r.stderr[-500:]}")
        return

    # Start server
    server = subprocess.Popen(
        ["/src/target/release/qwen3-tts-server"],
        env={**os.environ, "MODEL_DIR": "/src/models/0.6b-base", "PORT": "8090",
             "MAX_BATCH": "1", "RUST_LOG": "debug"},
        stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    time.sleep(45)

    ref_wav = open("/tmp/ok-mujer.wav", "rb").read()
    ref_b64 = base64.b64encode(ref_wav).decode()
    ref_text = "Hola, buenos días, mi nombre es la asistente virtual del centro de atención al cliente, ¿en qué puedo ayudarle?"

    # Test 1: without cloning
    print("=== Test 1: No cloning ===")
    req = urllib.request.Request("http://localhost:8090/v1/audio/speech",
        data=json.dumps({"text": "Hola, buenas tardes", "language": "spanish"}).encode(),
        headers={"Content-Type": "application/json"})
    t0 = time.time()
    resp = urllib.request.urlopen(req, timeout=60)
    data = resp.read()
    dur = (len(data) - 44) / (24000 * 2)
    print(f"  No clone: {dur:.2f}s audio, {time.time()-t0:.2f}s wall")

    # Test 2: with cloning (ICL, ref_text)
    print("\n=== Test 2: ICL clone ===")
    req = urllib.request.Request("http://localhost:8090/v1/audio/speech",
        data=json.dumps({
            "text": "Hola, buenas tardes",
            "language": "spanish",
            "ref_audio": ref_b64,
            "ref_text": ref_text,
        }).encode(),
        headers={"Content-Type": "application/json"})
    t0 = time.time()
    try:
        resp = urllib.request.urlopen(req, timeout=120)
        data_clone = resp.read()
        dur_clone = (len(data_clone) - 44) / (24000 * 2)
        print(f"  ICL clone: {dur_clone:.2f}s audio, {time.time()-t0:.2f}s wall")
        print(f"  Excess: {dur_clone - dur:.2f}s (should be ~0)")
    except Exception as e:
        print(f"  ICL clone: FAILED - {e}")
        try:
            print(f"  Response body: {e.read().decode()[:500]}")
        except:
            pass
        dur_clone = -1

    # Test 3: x_vector only (no ref_text)
    print("\n=== Test 3: x_vector clone ===")
    req = urllib.request.Request("http://localhost:8090/v1/audio/speech",
        data=json.dumps({
            "text": "Hola, buenas tardes",
            "language": "spanish",
            "ref_audio": ref_b64,
        }).encode(),
        headers={"Content-Type": "application/json"})
    t0 = time.time()
    try:
        resp = urllib.request.urlopen(req, timeout=120)
        data_xv = resp.read()
        dur_xv = (len(data_xv) - 44) / (24000 * 2)
        print(f"  x_vector: {dur_xv:.2f}s audio, {time.time()-t0:.2f}s wall")
    except Exception as e:
        print(f"  x_vector: FAILED - {e}")
        dur_xv = -1

    # Get server logs
    server.terminate()
    _, stderr = server.communicate(timeout=5)
    logs = stderr.decode()
    # Find ICL debug lines
    for line in logs.split('\n'):
        if 'ICL' in line or 'ref_frame' in line or 'cut' in line:
            print(f"  LOG: {line.strip()}")

    print(f"\nSummary: no_clone={dur:.2f}s, icl={dur_clone:.2f}s, xvector={dur_xv:.2f}s, excess={dur_clone-dur:.2f}s")


@app.local_entrypoint()
def main():
    debug.remote()
