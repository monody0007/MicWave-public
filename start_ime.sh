#!/bin/bash
# EchoWave IME 启动脚本（后台模式 + 开机启动安装）

# launchd 启动时 PATH 极简（仅 /usr/bin:/bin:/usr/sbin:/sbin），
# 找不到 brew 及 start_ime.local.sh 可能用到的 Homebrew 命令，会导致 key 拉空 → preflight 失败。
# 显式补 PATH 让脚本在交互 shell 和 launchd 下行为一致。
export PATH="/opt/homebrew/bin:/opt/homebrew/sbin:/usr/local/bin:$PATH"

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT_DIR"
VENV_PY="$ROOT_DIR/venv/bin/python"
VENV_PIP="$ROOT_DIR/venv/bin/pip"
WHISPER_VENV_DIR="$ROOT_DIR/whisper_backend/venv"
WHISPER_VENV_PY="$WHISPER_VENV_DIR/bin/python"
WHISPER_VENV_PIP="$WHISPER_VENV_DIR/bin/pip"
# micwave-core 的来源:公开快照把 micwave_core/ 包放在本目录,原地 import 即可;
# 开发布局下它是同级 checkout(可用 MICWAVE_CORE_DIR 指别处),以 -e 链进两个 venv。
MICWAVE_CORE_DIR="${MICWAVE_CORE_DIR:-$ROOT_DIR/../micwave-core}"
# 建 venv 用 opt 路径:经 venv/bin/python 建会把 Cellar 里带补丁版本的目录写进 pyvenv.cfg,brew cleanup 后 venv 就断
BASE_PY="/opt/homebrew/opt/python@3.13/bin/python3.13"
[ -x "$BASE_PY" ] || BASE_PY="python3"  # 没有 Homebrew python@3.13 的机器用 PATH 里的 python3

LOG_DIR="$HOME/Library/Logs/EchoWave IME"
LOG_FILE="$LOG_DIR/echowave_ime.log"
PLIST_PATH="$HOME/Library/LaunchAgents/com.echowave.ime.plist"
SCRIPT_PATH="$ROOT_DIR/start_ime.sh"

install_autostart() {
    mkdir -p "$(dirname "$PLIST_PATH")" "$LOG_DIR"
    cat > "$PLIST_PATH" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.echowave.ime</string>
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
    /bin/launchctl bootout "gui/$UID/com.echowave.ime" 2>/dev/null || true
    /bin/launchctl bootstrap "gui/$UID" "$PLIST_PATH"
    /bin/launchctl enable "gui/$UID/com.echowave.ime" 2>/dev/null || true
    echo "Autostart installed. LaunchAgent: $PLIST_PATH"
    exit 0
}

uninstall_autostart() {
    /bin/launchctl bootout "gui/$UID/com.echowave.ime" 2>/dev/null || true
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

# 两个 venv 共用:已链到 $MICWAVE_CORE_DIR/src 就跳过,快照里自带 micwave_core/ 也跳过。
link_micwave_core() {
    local py_bin="$1" pip_bin="$2" label="$3"
    [ -f "$ROOT_DIR/micwave_core/__init__.py" ] && return 0
    "$py_bin" -c 'import pathlib, sys, micwave_core
expected = pathlib.Path(sys.argv[1]).resolve() / "src"
actual = pathlib.Path(micwave_core.__file__).resolve()
sys.exit(0 if actual.is_relative_to(expected) else 1)' "$MICWAVE_CORE_DIR" 2>/dev/null && return 0
    echo "Linking editable micwave-core into $label venv..."
    "$pip_bin" install --quiet --no-deps --no-build-isolation -e "$MICWAVE_CORE_DIR"
}

# 检查虚拟环境
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    "$BASE_PY" -m venv venv
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
link_micwave_core "$VENV_PY" "$VENV_PIP" "EchoWave"

# Whisper transcription runs in a separate interpreter/dependency boundary.
# It never imports desktop-owner dependencies (rumps/PyAudio/Quartz) into this sidecar venv.
if [ ! -x "$WHISPER_VENV_PY" ]; then
    echo "Creating isolated Whisper backend virtual environment..."
    "$BASE_PY" -m venv "$WHISPER_VENV_DIR"
fi
if ! "$WHISPER_VENV_PY" -c "import fastapi, numpy, scipy, websockets" 2>/dev/null; then
    echo "Installing isolated Whisper backend dependencies..."
    "$WHISPER_VENV_PIP" install -r "$ROOT_DIR/whisper_backend/requirements.txt"
fi
link_micwave_core "$WHISPER_VENV_PY" "$WHISPER_VENV_PIP" "Whisper backend"

# 加载 .env（模板见 .env.example）；key 也可以不放这里，见下方 start_ime.local.sh
if [ -f .env ]; then
    export $(grep -v '^#' .env | grep '=' | xargs)
fi

export BRAINWAVE_ENFORCE_SINGLE_INSTANCE=1
export PYTHONUNBUFFERED=1

# launcher.py owns one exact shell PID plus exact backend child handles/PID files.
# Do not scan commands or kill by basename here: that can target the wrong
# backend/runtime and violates the single-owner transaction boundary.

# 本机私有的 key 注入（不入库，*.sh 已被 .gitignore 忽略）：存在 start_ime.local.sh 就 source 它，
# 由它 export OPENAI_API_KEY / GOOGLE_API_KEY（例如从密钥管理器取），覆盖 .env 与调用方环境里的同名值。
# 缺失的 key 由 launcher.py preflight 决定是否 fatal
# (OpenAI 是唯一的 realtime provider，只需 OPENAI_API_KEY；GOOGLE_API_KEY 供可选 LLM 后处理)
if [ -f "$ROOT_DIR/start_ime.local.sh" ]; then
    source "$ROOT_DIR/start_ime.local.sh"
fi

if [ -n "${LAUNCH_JOB_NAME:-}" ] || [ "${XPC_SERVICE_NAME:-}" = "com.echowave.ime" ]; then
    exec "$VENV_PY" launcher.py
fi

mkdir -p "$LOG_DIR"
nohup "$VENV_PY" launcher.py >> "$LOG_FILE" 2>&1 &
PID=$!

echo "EchoWave IME started in background (PID: $PID)"
echo "Log: $LOG_FILE"
