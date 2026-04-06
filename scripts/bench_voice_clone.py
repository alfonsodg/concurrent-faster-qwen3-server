#!/usr/bin/env python3
"""Benchmark voice cloning on qwen3-tts-server.

Usage:
    python3 scripts/bench_voice_clone.py --ref reference.wav [--ref-text "transcript"] [--url http://localhost:8090]
"""
import argparse, time, json, base64, sys
from urllib.request import Request, urlopen

def main():
    parser = argparse.ArgumentParser(description="Benchmark voice cloning")
    parser.add_argument("--url", default="http://localhost:8090")
    parser.add_argument("--ref", required=True, help="Path to reference WAV file")
    parser.add_argument("--ref-text", default=None, help="Transcript of reference audio (enables ICL mode)")
    parser.add_argument("--text", default="Buenos días, le llamo para ofrecerle una mejora de plan. ¿Tiene un momento?")
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--output", default=None, help="Save last output to this WAV file")
    args = parser.parse_args()

    ref_wav = open(args.ref, "rb").read()
    ref_b64 = base64.b64encode(ref_wav).decode()
    ref_dur = (len(ref_wav) - 44) / (24000 * 2)
    mode = "ICL" if args.ref_text else "x_vector"

    print(f"Server: {args.url}")
    print(f"Reference: {args.ref} ({ref_dur:.1f}s)")
    print(f"Mode: {mode}")
    print(f"Text: {args.text[:60]}...")
    print()

    body = {"text": args.text, "language": "spanish", "ref_audio": ref_b64}
    if args.ref_text:
        body["ref_text"] = args.ref_text

    results = []
    for i in range(args.trials):
        req = Request(f"{args.url}/v1/audio/speech",
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"})
        t0 = time.time()
        try:
            resp = urlopen(req, timeout=120)
            data = resp.read()
            elapsed = time.time() - t0
            dur = (len(data) - 44) / (24000 * 2)
            rtf = dur / elapsed
            results.append({"elapsed": elapsed, "dur": dur, "rtf": rtf})
            print(f"  Trial {i+1}: {dur:.2f}s audio, {elapsed:.2f}s wall, {rtf:.2f}x RT")
            if args.output and i == args.trials - 1:
                open(args.output, "wb").write(data)
                print(f"  Saved: {args.output}")
        except Exception as e:
            print(f"  Trial {i+1}: FAILED - {e}")

    if results:
        avg_rtf = sum(r["rtf"] for r in results) / len(results)
        avg_lat = sum(r["elapsed"] for r in results) / len(results)
        print(f"\nAverage: {avg_rtf:.2f}x RT, {avg_lat:.2f}s latency ({mode} mode)")

if __name__ == "__main__":
    main()
