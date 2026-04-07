# TTS Backend Portability Plan

This document defines the implementation steps to evolve `qwen3-tts-server` from a Qwen3-specific server into a multi-backend TTS server.

## Goal

Support additional TTS engines (beyond Qwen3) without breaking current API behavior and performance characteristics for existing Qwen3 deployments.

## Current Constraint Summary

- Server code is tightly coupled to `qwen3_tts::Qwen3TTS`.
- Batch and streaming paths call Qwen-specific methods directly.
- Voice behavior assumes Qwen speaker/token semantics.
- Request schema mixes generic and Qwen-specific capabilities.

## Implementation Steps

### 1. Define a backend abstraction layer

Create a trait-based interface for synthesis engines and keep Qwen implementation behind it.

Deliverables:

- New trait (for example `TtsBackend`) with methods for:
  - non-streaming synth
  - streaming synth (or explicit unsupported capability)
  - optional voice-clone preparation
  - backend capability introspection
- Backend-agnostic request/response structures for internal processing.

Acceptance:

- Server compiles with all direct `Qwen3TTS` references removed from route handlers.

### 2. Add capability model (feature flags per backend)

Introduce explicit capability checks instead of implicit assumptions.

Deliverables:

- Capability struct/enums, e.g.:
  - supports_streaming
  - supports_voice_clone
  - supports_voice_design
  - supported_languages policy
- Runtime validation layer that rejects unsupported request combinations with clear errors.

Acceptance:

- Requests requiring unsupported features fail predictably with structured API errors.

### 3. Implement Qwen adapter as first backend

Wrap existing Qwen behavior inside a backend adapter to preserve current functionality.

Deliverables:

- `QwenBackend` implementing the new trait.
- Mapping between generic request types and current Qwen calls:
  - `synthesize_batch_with_voices`
  - `synthesize_batch_streaming`
  - `create_voice_clone_prompt`
- Preserve current defaults and performance paths.

Acceptance:

- Existing Qwen behavior remains functionally equivalent after adapter extraction.

### 4. Refactor batch engine to generic backend contract

Make `BatchEngine` operate on trait objects instead of Qwen concrete types.

Deliverables:

- Replace `Arc<Qwen3TTS>` usage with `Arc<dyn TtsBackend + Send + Sync>`.
- Keep batch queueing semantics and max wait strategy unchanged.
- Normalize backend-specific errors into a common error type.

Acceptance:

- Batch path no longer depends on Qwen symbols directly.

### 5. Refactor streaming worker to generic backend contract

Decouple streaming worker from Qwen-specific streaming function signatures.

Deliverables:

- Generic streaming request pipeline using backend streaming method.
- Explicit fallback path when backend has no streaming support.
- Keep current headers and streaming format contract (or version them if changed).

Acceptance:

- Streaming path compiles and runs through backend abstraction.

### 6. Split API schema into generic + optional extension fields

Keep compatibility while preparing for non-Qwen engines.

Deliverables:

- Core fields: text, language, stream, temperature.
- Optional extension object for backend-specific options (e.g. voice clone params).
- Validation that prevents invalid cross-backend option combinations.

Acceptance:

- API can express backend-neutral requests without exposing Qwen internals.

### 7. Add backend selection mechanism

Allow selecting backend via configuration.

Deliverables:

- Env var/config option such as `TTS_BACKEND=qwen3|...`.
- Backend factory at startup.
- Clear startup logs showing selected backend and capabilities.

Acceptance:

- Server starts with selected backend and fails fast on invalid configuration.

### 8. Add second backend (pilot)

Integrate one non-Qwen backend to validate architecture.

Deliverables:

- New adapter implementation for chosen engine.
- Mapping for synth + optional streaming.
- Capability matrix updated in docs.

Acceptance:

- At least one non-Qwen backend serves `/v1/audio/speech`.

### 9. Testing and regression coverage

Protect both legacy Qwen behavior and new abstraction.

Deliverables:

- Unit tests for capability validation and backend factory.
- Integration tests for:
  - successful synth
  - unsupported feature rejection
  - queue saturation response
  - streaming behavior per capability
- Golden-path tests for Qwen compatibility.

Acceptance:

- CI covers backend-neutral contracts and backend-specific adapters.

### 10. Documentation and operational rollout

Document migration and runtime usage.

Deliverables:

- README update with multi-backend architecture and config examples.
- Capability matrix by backend.
- Rollout plan:
  - phase 1: Qwen adapter under abstraction (no behavior change)
  - phase 2: pilot second backend
  - phase 3: production enablement

Acceptance:

- Operators can configure backend choice with documented behavior and known limits.

## Suggested Execution Order

1. Steps 1–3 (abstraction + Qwen adapter)  
2. Steps 4–5 (batch/stream refactor)  
3. Steps 6–7 (API/config hardening)  
4. Step 8 (second backend pilot)  
5. Steps 9–10 (tests + docs rollout)

## Risks to Watch

- Throughput regressions from abstraction overhead in hot paths.
- Semantic mismatch for voice options across engines.
- Streaming chunk format differences between backends.
- Error contract drift if adapters return inconsistent failure modes.

## Done Criteria

- Server supports Qwen through adapter with no regression in current behavior.
- At least one additional backend is functional in the same HTTP API.
- Capability validation and error contracts are deterministic.
- CI and docs cover multi-backend operation.
