"""Run server unit tests on Modal (no GPU needed)."""
import modal

app = modal.App("qwen3-tts-tests")

image = (
    modal.Image.from_registry("nvidia/cuda:12.6.3-devel-ubuntu24.04", add_python="3.12")
    .apt_install("cmake", "pkg-config", "libssl-dev", "libasound2-dev", "git", "curl",
                 "libclang-dev", "clang")
    .run_commands(
        "curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y",
    )
    .add_local_dir(".", remote_path="/src", ignore=[
        "target", "models", "__pycache__", ".git",
        "*.wav", "*.gguf", "*.onnx", "*.pyc",
    ])
)


@app.function(image=image, gpu="L4", timeout=1200, memory=32768)
def run_tests():
    import subprocess, os
    os.chdir("/src")
    env = {**os.environ, "CUDA_COMPUTE_CAP": "89"}
    r = subprocess.run(
        ["bash", "-c", "source /root/.cargo/env && cargo test -p qwen3-tts-server -- --nocapture"],
        capture_output=True, text=True, timeout=900, env=env
    )
    print(r.stdout[-3000:] if r.stdout else "")
    if r.stderr:
        # Filter out compilation noise, show test results
        lines = r.stderr.split('\n')
        for l in lines:
            if 'test ' in l or 'FAILED' in l or 'passed' in l or 'error' in l.lower():
                print(l)
    return {"rc": r.returncode, "stdout": r.stdout[-2000:], "stderr": r.stderr[-2000:]}


@app.local_entrypoint()
def main():
    r = run_tests.remote()
    print(f"\nExit code: {r['rc']}")
