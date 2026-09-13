#!/bin/bash
# Brainwave IME 启动脚本（后台模式 + 开机启动安装）

# launchd 启动时 PATH 极简（仅 /usr/bin:/bin:/usr/sbin:/sbin），
# 找不到 brew 等 Homebrew 安装的命令。显式补 PATH 让脚本在交互 shell 和 launchd 下行为一致。
export PATH="/opt/homebrew/bin:/opt/homebrew/sbin:/usr/local/bin:$PATH"

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT_DIR"
VENV_PY="$ROOT_DIR/venv/bin/python"
VENV_PIP="$ROOT_DIR/venv/bin/pip"
WHISPER_VENV_DIR="$ROOT_DIR/whisper_backend/venv"
WHISPER_VENV_PY="$WHISPER_VENV_DIR/bin/python"
WHISPER_VENV_PIP="$WHISPER_VENV_DIR/bin/pip"

LOG_DIR="$HOME/Library/Logs/Brainwave IME"
LOG_FILE="$LOG_DIR/brainwave_ime.log"
PLIST_PATH="$HOME/Library/LaunchAgents/com.brainwave.ime.plist"
SCRIPT_PATH="$ROOT_DIR/start_ime.sh"

install_autostart() {
    mkdir -p "$(dirname "$PLIST_PATH")" "$LOG_DIR"
    cat > "$PLIST_PATH" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.brainwave.ime</string>
    <key>ProgramArguments</key>
    <array>
        <string>/bin/bash</string>
        <string>$SCRIPT_PATH</string>
    </array>
    <key>RunAtLoad</key>
    <true/>
    <key>KeepAlive</key>
    <true/>
    <key>ProcessType</key>
    <string>Standard</string>
    <key>StandardOutPath</key>
    <string>$LOG_FILE</string>
    <key>StandardErrorPath</key>
    <string>$LOG_FILE</string>
</dict>
</plist>
EOF
    /bin/launchctl bootout "gui/$UID/com.brainwave.ime" 2>/dev/null || true
    /bin/launchctl bootstrap "gui/$UID" "$PLIST_PATH"
    /bin/launchctl enable "gui/$UID/com.brainwave.ime" 2>/dev/null || true
    echo "Autostart installed. LaunchAgent: $PLIST_PATH"
    exit 0
}

uninstall_autostart() {
    /bin/launchctl bootout "gui/$UID/com.brainwave.ime" 2>/dev/null || true
    rm -f "$PLIST_PATH"
    echo "Autostart removed."
    exit 0
}

case "$1" in
    --install-autostart)
        install_autostart
        ;;
    --uninstall-autostart)
        uninstall_autostart
        ;;
esac

# 检查虚拟环境
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# 检查依赖
if ! "$VENV_PY" -c "import pyaudio" 2>/dev/null; then
    echo "Installing dependencies..."
    # macOS 需要先安装 portaudio
    if command -v brew &> /dev/null; then
        brew install portaudio 2>/dev/null || true
    fi
    "$VENV_PIP" install -r requirements.txt
fi

# Whisper transcription runs in a separate interpreter/dependency boundary.
# It never imports the stable Echo provider/session modules and does not install
# the desktop-owner dependencies (rumps/PyAudio/Quartz) into this sidecar venv.
if [ ! -x "$WHISPER_VENV_PY" ]; then
    echo "Creating isolated Whisper backend virtual environment..."
    "$VENV_PY" -m venv "$WHISPER_VENV_DIR"
fi
if ! "$WHISPER_VENV_PY" -c "import fastapi, numpy, scipy, websockets" 2>/dev/null; then
    echo "Installing isolated Whisper backend dependencies..."
    "$WHISPER_VENV_PIP" install -r "$ROOT_DIR/whisper_backend/requirements.txt"
fi

# Load environment variables and API keys from .env.
# Copy .env.example to .env and fill in your keys.
# launcher.py does a preflight check; OPENAI_API_KEY is required.
if [ -f "$ROOT_DIR/.env" ]; then
    set -a
    . "$ROOT_DIR/.env"
    set +a
fi

export BRAINWAVE_ENFORCE_SINGLE_INSTANCE=1
export PYTHONUNBUFFERED=1

# launcher.py owns one exact shell PID plus exact backend child handles/PID files.
# Do not scan commands or kill by basename here: that can target the wrong
# backend/runtime and violates the single-owner transaction boundary.

if [ -n "${LAUNCH_JOB_NAME:-}" ] || [ "${XPC_SERVICE_NAME:-}" = "com.brainwave.ime" ]; then
    exec "$VENV_PY" launcher.py
fi

mkdir -p "$LOG_DIR"
nohup "$VENV_PY" launcher.py >> "$LOG_FILE" 2>&1 &
PID=$!

echo "Brainwave IME started in background (PID: $PID)"
echo "Log: $LOG_FILE"
