#!/usr/bin/env python3
"""Streaming TTFA (Time To First Audio) benchmark.

Usage:
    python3 scripts/bench_streaming.py [--url http://localhost:8090] [--trials 5]
"""
import argparse, time, json, sys
from urllib.request import Request, urlopen

def measure_ttfa(url, text, language="spanish"):
    body = json.dumps({"text": text, "language": language, "stream": True}).encode()
    req = Request(f"{url}/v1/audio/speech", data=body, headers={"Content-Type": "application/json"})
    t0 = time.time()
    resp = urlopen(req, timeout=60)
    # Read WAV header (44 bytes) + first PCM chunk
    header = resp.read(44)
    first_chunk = resp.read(4000)  # ~83ms of audio at 24kHz 16-bit
    ttfa = (time.time() - t0) * 1000
    # Drain rest
    total = len(header) + len(first_chunk)
    while True:
        chunk = resp.read(8192)
        if not chunk:
            break
        total += len(chunk)
    total_time = time.time() - t0
    dur = (total - 44) / (24000 * 2)
    return {"ttfa_ms": ttfa, "total_s": total_time, "audio_dur": dur}

def main():
    parser = argparse.ArgumentParser(description="Streaming TTFA benchmark")
    parser.add_argument("--url", default="http://localhost:8090")
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--text", default="Buenos días, le habla el asistente virtual del centro de atención al cliente. ¿En qué puedo ayudarle?")
    args = parser.parse_args()

    # Warmup
    measure_ttfa(args.url, "Hola")

    print(f"Streaming TTFA benchmark ({args.trials} trials)\n")
    results = []
    for i in range(args.trials):
        r = measure_ttfa(args.url, args.text)
        results.append(r)
        print(f"  Trial {i+1}: TTFA {r['ttfa_ms']:.0f}ms | Total {r['total_s']:.2f}s | Audio {r['audio_dur']:.2f}s")

    avg_ttfa = sum(r["ttfa_ms"] for r in results) / len(results)
    p95_ttfa = sorted(r["ttfa_ms"] for r in results)[int(len(results) * 0.95)]
    print(f"\nAverage TTFA: {avg_ttfa:.0f}ms | P95: {p95_ttfa:.0f}ms")

if __name__ == "__main__":
    main()
