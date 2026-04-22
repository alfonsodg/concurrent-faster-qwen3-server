# API Usage Guide

How to generate audio with qwen3-tts-server. Covers pre-recorded batch generation and real-time streaming for telephony.

## Quick Reference

```
POST /v1/audio/speech     — synthesize text to audio
POST /v1/embeddings/preload — pre-encode voice for fast cloning
GET  /health              — server status
GET  /metrics             — Prometheus metrics
```

## 1. Basic Synthesis (Pre-recorded)

Generate a complete WAV file. Best for IVR prompts, voicemail, batch audio generation.

```bash
curl -X POST http://SERVER:8090/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"text": "Buenos días, bienvenido a nuestro centro de atención.", "language": "spanish"}' \
  --output greeting.wav
```

Response: complete WAV file (24kHz, 16-bit, mono). Latency ~4s for a typical phrase.

## 2. Streaming Synthesis (Real-time)

Audio starts arriving in ~230ms. Essential for telephony and live conversations.

```bash
curl -X POST http://SERVER:8090/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"text": "Buenos días, bienvenido.", "language": "spanish", "stream": true}' \
  --output stream.wav
```

Response: chunked WAV. First 44 bytes = WAV header, then PCM chunks (~250ms each).

Headers returned:
- `x-ttfa-ms`: time to first audio chunk (ms)
- `x-audio-format`: `pcm-s16le-24000-mono`

## 3. Voice Cloning

Clone any voice from a reference audio. Two steps: preload once, then use the voice_id.

### Step 1: Preload voice (once)

```bash
# Reference audio: 6-15s clean speech, WAV format
REF_B64=$(base64 -w0 reference_voice.wav)

curl -X POST http://SERVER:8090/v1/embeddings/preload \
  -H "Content-Type: application/json" \
  -d "{\"ref_audio\": \"$REF_B64\", \"voice_id\": \"agent-maria\"}"
```

Response:
```json
{"voice_id": "agent-maria", "cached": false}
```

The voice embedding stays in memory until the server restarts. Preload again after restart.

### Step 2: Synthesize with cloned voice

```bash
# Pre-recorded (batch)
curl -X POST http://SERVER:8090/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"text": "Su pago ha sido procesado.", "language": "spanish", "voice_id": "agent-maria"}' \
  --output cloned.wav

# Streaming (real-time)
curl -X POST http://SERVER:8090/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"text": "Su pago ha sido procesado.", "language": "spanish", "voice_id": "agent-maria", "stream": true}' \
  --output cloned_stream.wav
```

### Reference audio best practices

- Duration: 6-15 seconds of clean speech
- Format: WAV, mono, 24kHz preferred (auto-resampled if different)
- Content: natural conversational speech, no music or background noise
- Language: match the target language (clone in Spanish for Spanish synthesis)
- Size limit: 10MB max (base64 encoded)

## 4. Batch Generation (Multiple Files)

Generate multiple audio files efficiently. The server batches concurrent requests automatically.

```bash
#!/bin/bash
SERVER="http://SERVER:8090"
VOICE="agent-maria"

# Preload voice once
REF_B64=$(base64 -w0 reference_voice.wav)
curl -s -X POST "$SERVER/v1/embeddings/preload" \
  -H "Content-Type: application/json" \
  -d "{\"ref_audio\": \"$REF_B64\", \"voice_id\": \"$VOICE\"}"

# Generate multiple files in parallel (server batches automatically)
declare -A TEXTS=(
  ["greeting"]="Buenos días, bienvenido a nuestro centro de atención al cliente."
  ["hold"]="Por favor espere un momento mientras lo transferimos con un agente."
  ["goodbye"]="Gracias por comunicarse con nosotros. Que tenga un excelente día."
  ["payment"]="Su pago ha sido procesado exitosamente. El número de confirmación es el siguiente."
  ["schedule"]="Su cita ha sido programada. Le enviaremos un recordatorio por correo electrónico."
)

for name in "${!TEXTS[@]}"; do
  curl -s -X POST "$SERVER/v1/audio/speech" \
    -H "Content-Type: application/json" \
    -d "{\"text\": \"${TEXTS[$name]}\", \"language\": \"spanish\", \"voice_id\": \"$VOICE\"}" \
    --output "${name}.wav" &
done
wait
echo "All files generated."
```

## 5. Python Integration

### Streaming client

```python
import json, wave
from urllib.request import Request, urlopen

def synthesize_stream(text, voice_id=None, server="http://localhost:8090"):
    body = {"text": text, "language": "spanish", "stream": True}
    if voice_id:
        body["voice_id"] = voice_id
    req = Request(
        f"{server}/v1/audio/speech",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    resp = urlopen(req, timeout=60)
    wav_header = resp.read(44)  # WAV header
    pcm_data = resp.read()      # PCM audio
    return pcm_data

def save_wav(pcm_data, path):
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(24000)
        w.writeframes(pcm_data)

# Usage
pcm = synthesize_stream("Hola, ¿cómo está?", voice_id="agent-maria")
save_wav(pcm, "output.wav")
```

### Preload voice

```python
import json, base64
from urllib.request import Request, urlopen

def preload_voice(ref_path, voice_id, server="http://localhost:8090"):
    with open(ref_path, "rb") as f:
        ref_b64 = base64.b64encode(f.read()).decode()
    body = json.dumps({"ref_audio": ref_b64, "voice_id": voice_id}).encode()
    req = Request(
        f"{server}/v1/embeddings/preload",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    return json.loads(urlopen(req, timeout=60).read())

# Usage
preload_voice("maria.wav", "agent-maria")
```

## 6. Request Parameters

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `text` | string | required | Text to synthesize |
| `language` | string | `"spanish"` | `spanish`/`es`, `english`/`en`, `french`/`fr` |
| `temperature` | float | `0.7` | Sampling randomness (0.0-1.0). Lower = more consistent |
| `stream` | bool | `false` | Enable chunked streaming |
| `voice_id` | string | — | Preloaded voice ID |
| `ref_audio` | string | — | Base64 WAV for inline voice cloning (prefer preload) |
| `ref_text` | string | — | Transcript of ref_audio (enables ICL mode, better quality) |

## 7. When to Use Each Mode

| Use case | Mode | Why |
|----------|------|-----|
| IVR prompts, voicemail | Batch (no stream) | Generate once, play many times |
| Live phone calls | Streaming + voice_id | Low latency, consistent voice |
| Batch audio generation | Batch, parallel requests | Server batches for max throughput |
| Testing/development | Either | Stream for quick iteration |

## 8. Performance Tips

- **Always preload voices** — sending `ref_audio` on every request wastes ~100ms on encoding
- **Use streaming for telephony** — 230ms TTFA vs 4s batch latency
- **Parallel batch requests** — the server batches up to 12 concurrent requests into one GPU pass
- **Temperature 0.7** — good balance of naturalness and consistency. Use 0.5 for more robotic but predictable output
- **Keep text under 50 words** — longer texts generate more frames and take longer. Split into sentences for streaming

## 9. Voice Clone: Sentence Splitting (Critical)

With voice cloning, the model generates fewer frames than without clone. Long texts (40+ words) get truncated — the audio cuts off before the text finishes. This is model behavior, not a server bug.

**The fix: split text into sentences before calling the TTS server.**

Each sentence should be under 20 words. Generate each one separately, then concatenate.

### Python example

```python
import re, wave, io

def split_sentences(text):
    """Split text into sentences at punctuation boundaries."""
    parts = re.split(r'(?<=[.!?;:])\s+', text.strip())
    # Further split long sentences at commas
    result = []
    for part in parts:
        if len(part.split()) > 20:
            sub = re.split(r'(?<=,)\s+', part)
            result.extend(sub)
        else:
            result.append(part)
    return [s.strip() for s in result if s.strip()]

def synthesize_long_text(text, voice_id, server="http://localhost:8090"):
    """Generate audio for long text with voice clone using sentence splitting."""
    import json
    from urllib.request import Request, urlopen

    sentences = split_sentences(text)
    all_pcm = bytearray()

    for sentence in sentences:
        body = json.dumps({
            "text": sentence,
            "language": "spanish",
            "stream": True,
            "voice_id": voice_id,
        }).encode()
        req = Request(
            f"{server}/v1/audio/speech",
            data=body,
            headers={"Content-Type": "application/json"},
        )
        resp = urlopen(req, timeout=60)
        resp.read(44)  # skip WAV header
        all_pcm.extend(resp.read())

    # Save concatenated audio
    return bytes(all_pcm)

# Usage
pcm = synthesize_long_text(
    "Le informo que hemos recibido su solicitud de crédito. "
    "El departamento de análisis la revisará en un plazo de tres a cinco días hábiles. "
    "¿Desea que le enviemos una notificación por correo electrónico?",
    voice_id="agent-maria",
)
save_wav(pcm, "long_text.wav")
```

### Bash example

```bash
#!/bin/bash
SERVER="http://localhost:8090"
VOICE="agent-maria"
OUTPUT="full_audio.wav"

# Split at sentence boundaries, generate each, concatenate
SENTENCES=(
  "Le informo que hemos recibido su solicitud de crédito."
  "El departamento de análisis la revisará en un plazo de tres a cinco días hábiles."
  "¿Desea que le enviemos una notificación por correo electrónico?"
)

# Generate each sentence
for i in "${!SENTENCES[@]}"; do
  curl -s -X POST "$SERVER/v1/audio/speech" \
    -H "Content-Type: application/json" \
    -d "{\"text\": \"${SENTENCES[$i]}\", \"language\": \"spanish\", \"stream\": true, \"voice_id\": \"$VOICE\"}" \
    --output "/tmp/part_${i}.wav"
done

# Concatenate with sox
sox /tmp/part_*.wav "$OUTPUT"
rm /tmp/part_*.wav
```

### Why this happens

The Qwen3-TTS model with voice clone (x_vector speaker embedding) changes the token distribution. The model tends to generate EOS or repetitive tokens earlier than without clone. Observed behavior:

| Text length | Without clone | With clone |
|-------------|--------------|------------|
| Short (<15 words) | Full audio | Full audio |
| Medium (15-25 words) | Full audio | Full audio |
| Long (25-40 words) | Full audio | May truncate |
| Very long (40+ words) | Full audio | Truncates ~50% |

The server's `adaptive_max_length` and `rep_threshold` help but cannot fully compensate for the model's behavior. Sentence splitting is the reliable solution.
