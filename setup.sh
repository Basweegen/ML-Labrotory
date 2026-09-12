#!/usr/bin/env bash
# Copyright 2026 Sean M. Stow. All rights reserved.
#
# ML Laboratory - Turnkey Installer & System Integrator
# Sets up binaries, desktop entry, vector icons, and system launcher indexes.

set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BIN_PATH="${SCRIPT_DIR}/target/release/ai-dashboard"
DESKTOP_DIR="${HOME}/.local/share/applications"
ICON_SCALABLE_DIR="${HOME}/.local/share/icons/hicolor/scalable/apps"
ICON_PNG_DIR="${HOME}/.local/share/icons/hicolor/128x128/apps"

echo "=== [ML Laboratory] System Integration & Setup ==="

# 1. Dependency Validation
echo "==> Verifying core dependencies..."
if ! command -v cargo >/dev/null 2>&1; then
    echo "ERROR: Cargo is not installed. Please install Rust 1.85+." >&2
    exit 1
fi

if ! command -v ollama >/dev/null 2>&1; then
    echo "WARNING: Ollama not found in PATH. ML Laboratory requires Ollama for local LLM inference." >&2
else
    echo "✔ Ollama detected at $(command -v ollama)"
fi

# 2. Build Release Binary
echo "==> Compiling optimized release binary..."
cd "${SCRIPT_DIR}"
cargo build --release

if [[ ! -f "${BIN_PATH}" ]]; then
    echo "ERROR: Compilation failed; ${BIN_PATH} not produced." >&2
    exit 1
fi
chmod +x "${BIN_PATH}"
chmod +x "${SCRIPT_DIR}/start.sh"
chmod +x "${SCRIPT_DIR}/setup.sh"
echo "✔ Release binary compiled successfully: $(ls -lh "${BIN_PATH}" | awk '{print $5}')"

# 3. Install Application Icon
echo "==> Installing icons into XDG icon hierarchy..."
mkdir -p "${ICON_SCALABLE_DIR}"
mkdir -p "${ICON_PNG_DIR}"

cp "${SCRIPT_DIR}/resources/icon.svg" "${ICON_SCALABLE_DIR}/ml-labrotory.svg"

# If conversion tools are available, rasterize PNG
if command -v rsvg-convert >/dev/null 2>&1; then
    rsvg-convert -w 128 -h 128 "${SCRIPT_DIR}/resources/icon.svg" -o "${ICON_PNG_DIR}/ml-labrotory.png"
elif command -v magick >/dev/null 2>&1; then
    magick -background none -density 300 "${SCRIPT_DIR}/resources/icon.svg" -resize 128x128 "${ICON_PNG_DIR}/ml-labrotory.png" 2>/dev/null || true
elif command -v convert >/dev/null 2>&1; then
    convert -background none -density 300 "${SCRIPT_DIR}/resources/icon.svg" -resize 128x128 "${ICON_PNG_DIR}/ml-labrotory.png" 2>/dev/null || true
fi
echo "✔ Icons registered at ${ICON_SCALABLE_DIR}/ml-labrotory.svg"

# 4. Install Desktop Launcher
echo "==> Registering desktop launcher..."
mkdir -p "${DESKTOP_DIR}"
cp "${SCRIPT_DIR}/ml-labrotory.desktop" "${DESKTOP_DIR}/ml-labrotory.desktop"
chmod 644 "${DESKTOP_DIR}/ml-labrotory.desktop"

# Validate desktop file
if command -v desktop-file-validate >/dev/null 2>&1; then
    desktop-file-validate "${DESKTOP_DIR}/ml-labrotory.desktop" || echo "Validation warning."
fi

# Update desktop databases and cache
if command -v update-desktop-database >/dev/null 2>&1; then
    update-desktop-database "${DESKTOP_DIR}" 2>/dev/null || true
fi

if command -v gtk-update-icon-cache >/dev/null 2>&1; then
    gtk-update-icon-cache -f -t "${HOME}/.local/share/icons/hicolor" 2>/dev/null || true
fi

# KDE Plasma Sycoca cache rebuild (ensures Super key search immediately picks up the new launcher)
if command -v kbuildsycoca6 >/dev/null 2>&1; then
    kbuildsycoca6 2>/dev/null || true
elif command -v kbuildsycoca5 >/dev/null 2>&1; then
    kbuildsycoca5 2>/dev/null || true
fi

echo "✔ Desktop entry installed to ${DESKTOP_DIR}/ml-labrotory.desktop"
echo "=== [ML Laboratory] Setup COMPLETE ==="
echo "You can now press the Super key and search for 'ml labrotory' or 'ML Laboratory' to launch!"
