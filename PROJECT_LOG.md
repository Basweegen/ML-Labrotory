<!-- Copyright 2026 Sean M. Stow. All rights reserved. -->
# ML Laboratory - Project Execution & Security Log

## System Identity & Posture
- **Developer:** Sean M. Stow
- **Role:** Quantum Computing Programmer & Cyber Security Specialist
- **Security Baseline:** Post-Quantum security mindset, defensive security, loopback isolation (`127.0.0.1`), strict permissions (`0o700` directories, `0o600` files), zero outbound telemetry, proactive entropy & secret scanning.
- **Architectural Paradigm:** Swarm intelligence & ecosystem-based multi-agent workbench.

---

## 1. Pre-Implementation Scan & Audit
- **Source Inspection:**
  - `src/security.rs`: Outbound secret filter with 15+ credential signatures (AWS, GitHub, OpenAI, Anthropic, xAI, Slack, Google, Stripe, PEM private keys, Bearer tokens) plus a Shannon entropy tripwire ($\ge 4.7$ bits/byte). Verified clean and functional.
  - `src/ollama/api.rs`: REST API client strictly targeting loopback `127.0.0.1:11434` by default with `allow_remote` toggle gating LAN/WAN access.
  - `src/storage.rs`: Local Sled key-value database stored under `~/.local/share/ai-dashboard/storage`. Session sizes capped at `MAX_SESSION_MESSAGES` (500) and `MAX_MESSAGE_CHARS` (50,000) to prevent unbounded heap allocation and memory leaks.
- **Identified Gaps:**
  - Absence of automated system launcher (`.desktop`) and application icon, preventing search via desktop environment Super key.
  - Missing standalone `setup.sh` and `start.sh` automation scripts for turnkey deployment and background Ollama service verification.
  - Absence of explicit `0o700` / `0o600` directory & database permission enforcement on startup.

---

## 2. Tasks & Status
- [x] [COMPLETE] Build release binary (`cargo build --release`) -> 15 MB stripped binary produced at `target/release/ai-dashboard`.
- [x] [COMPLETE] Create vector application icon (`resources/icon.svg`) and install into XDG hierarchy (`scalable`, `32x32`, `48x48`, `64x64`, `128x128`, `256x256`).
- [x] [COMPLETE] Create FreeDesktop launcher (`ml-labrotory.desktop`) with Super key search keywords (`ml`, `labrotory`, `laboratory`, `llm`, `ollama`).
- [x] [COMPLETE] Create `setup.sh` turnkey installation script with desktop database and KDE cache refreshes.
- [x] [COMPLETE] Create `start.sh` defensive runtime script with Ollama healthcheck and permission hardening.
- [x] [COMPLETE] Post-implementation validation (desktop file validation: 0 errors/warnings, database update, icon cache rebuild, sycoca cache rebuild).
- [x] [COMPLETE] Fix Ollama model loading bug caused by null values in model metadata (`blackgrg26/WORMGPT-14:latest`).
- [x] [COMPLETE] Fix CLI fallback parser in `src/ollama/cli.rs` (remove unsupported `--json` flag).

---

## 3. Post-Implementation Validation & Security Verification
- **Binary Integrity:** `target/release/ai-dashboard` verified: 15 MB, stripped, executable.
- **Desktop Specification:** `desktop-file-validate ~/.local/share/applications/ml-labrotory.desktop` passed with 0 errors and 0 warnings.
- **Desktop Search Integration:**
  - Updated desktop database (`update-desktop-database ~/.local/share/applications`).
  - Rebuilt KDE Sycoca index (`kbuildsycoca6 --noincremental`).
  - Search trigger: Pressing `Super` key and entering `ml labrotory`, `ml laboratory`, `ollama`, or `llm` activates the launcher.
- **Security & Permissions Hardening:**
  - `start.sh` actively enforces `chmod 700` on data directory and `chmod 600` on persistent session databases.
  - Proactive check for Ollama running on loopback `127.0.0.1:11434` before GUI launch.

---

## 4. Diagnostics & Remediation: Ollama Model Loading
- **Root Cause Analysis:**
  - When querying `/api/tags`, local model `blackgrg26/WORMGPT-14:latest` returned `"families": null`.
  - Serde's default `#[serde(default)]` attribute only applies when keys are completely absent, not when value is explicitly `null`. Consequently, Serde aborted with `invalid type: null, expected a sequence`, causing `list_models()` to fail completely and rendering the UI model list empty.
  - In `src/ollama/cli.rs`, `ollama list --json` failed because Ollama CLI does not support `--json`.
- **Defensive Solutions Applied:**
  - Implemented `deserialize_null_default` deserializer in `src/ollama/api.rs` to map `null` values to `Default::default()` across all fields in `Model` and `ModelDetails`.
  - Re-architected `list_models()` to parse each model item individually, guaranteeing that a single corrupted or experimental model manifest in Ollama never blocks loading the remaining models.
  - Refactored `src/ollama/cli.rs` to parse standard whitespace-delimited table output from `ollama list` as a robust fallback.
  - Added unit test `parse_model_with_null_families` in `src/ollama/api.rs` (all 39 tests passing).
  - Recompiled release binary and launched updated instance.

---

## 5. Security Audit, AI Chat Integrity & Multi-Model Selector Controls (2026-09-11)
- **Pre-Implementation Vulnerability & Error Scan:**
  - `src/ui/app.rs`: Found compilation failure on `visuals.extreme_highlights` due to egui 0.36 API update (field replaced with `extreme_bg_color`).
  - `src/ui/relay.rs`: Identified compilation failure from chaining `.fill()` on `egui::Response` instead of `egui::Button`.
  - `src/ui/relay.rs`: Identified asynchronous lifecycle error where `api.chat_stream` was called with mismatched stream expectations and captured `self` into a `'static` future via `rt.spawn`.
  - `src/ui/chat.rs`: Uncovered logic bug in `send_prompt` where user messages were pushed to `self.messages` before history collection, causing the same message to be sent to Ollama duplicated back-to-back.
  - `src/ui/app.rs`: Identified overly rigid RAM check in `try_assign_model` that hard-blocked model selection whenever total used memory exceeded budget, locking users out of local inference.
  - `src/ui/models.rs`: Missing copyright header and missing automatic tab navigation upon clicking "Select" for a model.
- **Tasks & Status:**
  - [x] [COMPLETE] Fix compilation error in `src/ui/app.rs` (`extreme_bg_color`).
  - [x] [COMPLETE] Fix button styling and asynchronous `chat_stream` pipeline in `src/ui/relay.rs`.
  - [x] [COMPLETE] Fix prompt duplication bug in `src/ui/chat.rs` (`send_prompt`).
  - [x] [COMPLETE] Add direct inline interactive model ComboBox in Chat header.
  - [x] [COMPLETE] Convert `try_assign_model` from hard blocking to defensive warning so models can always run.
  - [x] [COMPLETE] Implement collapsible/adjustable multi-model selector (`[⊟ Collapse Slots]` / `[⊞ Multi-Model Slots]`).
  - [x] [COMPLETE] Add automatic transition to Chat tab when selecting a model from the Models tab.
  - [x] [COMPLETE] Add missing copyright headers across all modified files.
  - [x] [COMPLETE] Verify zero external telemetry, strict loopback binding, and `0o700`/`0o600` permissions.
  - [x] [COMPLETE] Run test suite: 40/40 unit tests passing.
- **Defensive Posture & Post-Implementation Validation:**
  - **Memory Safety:** In-flight tasks properly aborted via `abort()`. History and audit queues strictly bounded.
  - **SSRF & Loopback Isolation:** Ollama client rejects non-loopback IPs by default. No external HTTP endpoints configured.
  - **Permissions:** Data directory enforced `0o700`, persistent storage files enforced `0o600`.

