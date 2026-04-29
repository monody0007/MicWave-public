#!/bin/bash
# Brainwave IME 启动脚本（后台模式 + 开机启动安装）

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT_DIR"
VENV_PY="$ROOT_DIR/venv/bin/python"
VENV_PIP="$ROOT_DIR/venv/bin/pip"

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

# Load environment variables and API keys from .env.
# Copy .env.example to .env and fill in your keys.
# launcher.py does a preflight check; OPENAI_API_KEY is required for the
# default REALTIME_PROVIDER=openai path, xai/google keys are optional.
if [ -f "$ROOT_DIR/.env" ]; then
    set -a
    . "$ROOT_DIR/.env"
    set +a
fi

export BRAINWAVE_ENFORCE_SINGLE_INSTANCE=1
export PYTHONUNBUFFERED=1

# 清理可能残留的孤儿进程（launcher 被 SIGKILL 时 server 会残留）
_cleanup_stale() {
    local pids
    pids=$(ps -x -o pid=,command= | grep -E "(launcher\.py|realtime_server\.py)" | grep "$ROOT_DIR" | grep -v grep | awk '{print $1}')
    if [ -n "$pids" ]; then
        echo "Cleaning up stale processes: $pids"
        echo "$pids" | xargs kill 2>/dev/null || true
        sleep 1
        echo "$pids" | xargs kill -9 2>/dev/null || true
    fi
}
_cleanup_stale

if [ -n "${LAUNCH_JOB_NAME:-}" ] || [ "${XPC_SERVICE_NAME:-}" = "com.brainwave.ime" ]; then
    exec "$VENV_PY" launcher.py
fi

mkdir -p "$LOG_DIR"
nohup "$VENV_PY" launcher.py >> "$LOG_FILE" 2>&1 &
PID=$!

echo "Brainwave IME started in background (PID: $PID)"
echo "Log: $LOG_FILE"
