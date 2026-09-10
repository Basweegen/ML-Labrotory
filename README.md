# ML Laboratory

A local-first, security-hardened LLM workbench: multi-slot chat with
streaming replies, live model management, an AI code editor with diff
review, a training neural profiler, voice in/out, secret-leak guarding,
and a full activity audit trail — all against your own Ollama server.

## Run it

Requirements: Rust 1.85+, a running Ollama at `http://localhost:11434`.

    cargo run --release

The release binary is ~14MB. A validated desktop launcher lives at
`~/.local/share/applications/ml-lab.desktop` after first setup.

## Highlights

- Chat slots with roles, streaming, stop, retry, broadcast-to-all, compare view
- Outbound secret scanner (blocks API keys from ever reaching a model)
- Loopback-only Ollama URL by default; explicit toggle for LAN servers
- Per-slot transcripts (clipboard / markdown), history with search + rename
- Neural profiler that trains on your real usage; voice dictation + read-aloud
- Keyboard-first: Ctrl+=/-/0 zoom, Alt+1/2/3 or F1–F3 slots, F4 cycle, Ctrl+Enter broadcast

## Licensing (dual-license)

ML Lab is free software AND a commercial product. Pick one:

1. **Community edition — AGPL-3.0-or-later** (`LICENSES/AGPL-3.0.txt`).
   Free for personal, educational, and research use. If you run a modified
   copy on a server, or embed it anywhere, you must publish your source.
2. **Commercial license** (`LICENSE-COMMERCIAL.md`).
   For companies that want to run, modify, or embed ML Lab without AGPL
   copyleft obligations. Contact the copyright holder to purchase.

Copyright (C) 2026 Sean M. Stow.
