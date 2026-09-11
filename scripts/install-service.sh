#!/bin/sh
# Install ML Lab as a user-level systemd service: starts at login and
# restarts after crashes. GUI apps need display env, so this imports the
# live session's DISPLAY/Wayland vars into the user manager first.
# Usage: sh scripts/install-service.sh
set -eu
cd "$(dirname "$0")/.."
BIN="$(pwd)/target/release/ai-dashboard"
if [ ! -x "$BIN" ]; then
    echo "Build first: sh scripts/release.sh (no binary at $BIN)" >&2
    exit 1
fi
UNIT="$HOME/.config/systemd/user/ml-lab.service"
mkdir -p "$(dirname "$UNIT")"
cat > "$UNIT" <<EOF
[Unit]
Description=ML Laboratory (local LLM workbench)
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
ExecStart=$BIN
Restart=on-failure
RestartSec=5
StandardOutput=append:%h/.local/share/ml-lab/service.log
StandardError=inherit

[Install]
WantedBy=default.target
EOF
mkdir -p "$HOME/.local/share/ml-lab"
# Best effort: hand the user manager the display env of THIS session so the
# service can open windows. Re-run after each login if autostart shows no window.
systemctl --user import-environment DISPLAY WAYLAND_DISPLAY XAUTHORITY XDG_RUNTIME_DIR 2>/dev/null || true
systemctl --user daemon-reload
systemctl --user enable --now ml-lab.service
systemctl --user status ml-lab.service --no-pager | head -12
echo "Logs: ~/.local/share/ml-lab/service.log and <data-dir>/ai-dashboard/crashes.log"
echo "Stop: systemctl --user stop ml-lab.service | Disable: systemctl --user disable ml-lab.service"
