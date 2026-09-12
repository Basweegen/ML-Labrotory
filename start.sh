#!/usr/bin/env bash
# Copyright 2026 Sean M. Stow. All rights reserved.
#
# ML Laboratory - Defensive Runtime Launcher
# Security: Loopback isolation, strict 0o700/0o600 permissions, Ollama health verification.

set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BIN_PATH="${SCRIPT_DIR}/target/release/ai-dashboard"
DATA_DIR="${HOME}/.local/share/ai-dashboard"

# 1. Enforce strict permissions on local database and logs (Defense in Depth)
if [[ -d "${DATA_DIR}" ]]; then
    chmod 700 "${DATA_DIR}" 2>/dev/null || true
    find "${DATA_DIR}" -type d -exec chmod 700 {} + 2>/dev/null || true
    find "${DATA_DIR}" -type f -exec chmod 600 {} + 2>/dev/null || true
fi

# Ensure GUI display variables are populated if running from subshell
if [[ -z "${WAYLAND_DISPLAY}" && -e "/run/user/$(id -u)/wayland-0" ]]; then
    export WAYLAND_DISPLAY="wayland-0"
fi
if [[ -z "${DISPLAY}" ]]; then
    export DISPLAY=":0"
fi

# 2. Check Ollama API loopback availability
OLLAMA_HOST="127.0.0.1"
OLLAMA_PORT="11434"

check_ollama() {
    curl -s --max-time 1 "http://${OLLAMA_HOST}:${OLLAMA_PORT}/api/version" >/dev/null 2>&1
}

if ! check_ollama; then
    echo "[ML Laboratory] Ollama service not detected on ${OLLAMA_HOST}:${OLLAMA_PORT}."
    
    # Check if ollama binary exists
    if command -v ollama >/dev/null 2>&1; then
        echo "[ML Laboratory] Initiating Ollama background server (Vulkan GPU enabled)..."
        export OLLAMA_IGPU_ENABLE=1
        nohup ollama serve > /dev/null 2>&1 &
        
        # Wait up to 5 seconds for Ollama to accept connections
        for _ in {1..10}; do
            if check_ollama; then
                echo "[ML Laboratory] Ollama successfully initialized."
                break
            fi
            sleep 0.5
        done
    else
        echo "[ML Laboratory WARNING] 'ollama' command not found in PATH."
        if command -v notify-send >/dev/null 2>&1; then
            notify-send "ML Laboratory Warning" "Ollama is not running. Start Ollama to enable AI inferences." -u normal -i ml-labrotory || true
        fi
    fi
fi

# 3. Verify release binary exists
if [[ ! -x "${BIN_PATH}" ]]; then
    echo "[ML Laboratory] Release binary not found at ${BIN_PATH}. Initiating automated build..."
    "${SCRIPT_DIR}/setup.sh"
fi

# 4. Launch ML Laboratory
exec "${BIN_PATH}" "$@"
