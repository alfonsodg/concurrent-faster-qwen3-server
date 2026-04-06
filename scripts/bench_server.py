#!/usr/bin/env python3
"""Benchmark qwen3-tts-server: single request latency and batch throughput.

Usage:
    python3 scripts/bench_server.py [--url http://localhost:8090] [--concurrency 1,2,4,8,16]
"""
import argparse, time, json, concurrent.futures, statistics, sys
from urllib.request import Request, urlopen
from urllib.error import URLError

def synthesize(url, text, language="spanish", stream=False):
    body = json.dumps({"text": text, "language": language, "stream": stream}).encode()
    req = Request(f"{url}/v1/audio/speech", data=body, headers={"Content-Type": "application/json"})
    t0 = time.time()
    resp = urlopen(req, timeout=120)
    data = resp.read()
    elapsed = time.time() - t0
    rtf = float(resp.headers.get("x-rtf", 0))
    dur = (len(data) - 44) / (24000 * 2) if len(data) > 44 else 0
    return {"elapsed": elapsed, "audio_dur": dur, "rtf": rtf, "size": len(data), "status": resp.status}

def main():
    parser = argparse.ArgumentParser(description="Benchmark qwen3-tts-server")
    parser.add_argument("--url", default="http://localhost:8090")
    parser.add_argument("--concurrency", default="1,2,4,8,16")
    parser.add_argument("--text", default="Buenos días, le habla el asistente virtual del centro de atención al cliente. ¿En qué puedo ayudarle?")
    parser.add_argument("--warmup", type=int, default=2)
    args = parser.parse_args()

    # Health check
    try:
        resp = urlopen(f"{args.url}/health", timeout=5)
        health = json.loads(resp.read())
        print(f"Server: {args.url} | max_batch={health['max_batch']}\n")
    except Exception as e:
        print(f"Server not reachable at {args.url}: {e}")
        sys.exit(1)

    # Warmup
    print(f"Warmup ({args.warmup} requests)...")
    for _ in range(args.warmup):
        synthesize(args.url, "Hola")

    # Single request benchmark (3 trials)
    print("\n=== Single Request ===")
    trials = [synthesize(args.url, args.text) for _ in range(3)]
    avg_elapsed = statistics.mean(t["elapsed"] for t in trials)
    avg_dur = statistics.mean(t["audio_dur"] for t in trials)
    print(f"  Audio: {avg_dur:.2f}s | Latency: {avg_elapsed:.2f}s | RTF: {avg_dur/avg_elapsed:.2f}x RT")

    # Concurrent benchmark
    print("\n=== Concurrent Requests ===")
    print(f"{'N':>4} | {'Total audio':>11} | {'Wall time':>9} | {'Throughput':>10} | {'Lat/req':>8} | {'P95 lat':>8}")
    print("-" * 70)

    for n in [int(x) for x in args.concurrency.split(",")]:
        def run_one(i):
            text = f"Buenos días, le habla el asistente número {i+1}. ¿En qué puedo ayudarle?"
            return synthesize(args.url, text)

        t0 = time.time()
        with concurrent.futures.ThreadPoolExecutor(max_workers=n) as ex:
            results = list(ex.map(run_one, range(n)))
        wall = time.time() - t0

        total_audio = sum(r["audio_dur"] for r in results)
        lats = sorted(r["elapsed"] for r in results)
        p95 = lats[int(len(lats) * 0.95)] if len(lats) > 1 else lats[0]
        throughput = total_audio / wall
        errors = sum(1 for r in results if r["status"] != 200)

        suffix = f" ({errors} errors)" if errors else ""
        print(f"{n:>4} | {total_audio:>9.1f}s | {wall:>7.2f}s | {throughput:>8.2f}x RT | {wall/n:>6.2f}s | {p95:>6.2f}s{suffix}")

    # Metrics
    print("\n=== Server Metrics ===")
    resp = urlopen(f"{args.url}/metrics", timeout=5)
    print(resp.read().decode())

if __name__ == "__main__":
    main()
