# Security

Charon is local-first by design.

- **No network egress** except read-only GitHub REST API calls for the repository you
  point it at (public PR metadata). Nothing about you goes anywhere else.
- **No telemetry, no tracking.**
- **Credentials** (an optional GitHub token to lift rate limits) are read from your
  environment or local `gh` CLI at runtime and used only in a request header — never
  logged, written to disk, or committed.
- **Inference is local** via scikit-learn. Wire in an LLM panel and you point it at your
  own Ollama — Charon refuses to phone home.
- **Verdicts are advisory** — a draft triage, never binding — and each can be signed with
  a local key for a replayable, tamper-evident trail.

Found an issue? Open a report on this repo.
