#!/bin/sh
# Reproducible release build for ML Lab. Logs to target/build.log.
# Usage: sh scripts/release.sh
set -eu
cd "$(dirname "$0")/.."
cargo build --release 2>&1 | tee target/build.log
BIN=target/release/ai-dashboard
ls -la "$BIN"
sha256sum "$BIN" | tee "$BIN.sha256"
file "$BIN"
