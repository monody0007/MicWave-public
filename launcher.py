#!/usr/bin/env python3
"""
Brainwave IME 启动器（开发模式）
"""

import atexit
import os
import signal
import sys
import subprocess
import threading
import time
import traceback

# 模块级引用，供信号处理器和 atexit 访问
_server_proc = None


def _cleanup_server():
    """终止 server 子进程（信号处理器 + atexit 共用）"""
    global _server_proc
    proc = _server_proc
    if proc is None:
        return
    _server_proc = None  # 防止重复清理
    print("[Launcher] Cleaning up server process...")
    try:
        proc.terminate()
        proc.wait(timeout=5)
    except Exception:
        try:
            proc.kill()
        except Exception:
            pass


def _signal_handler(signum, frame):
    """捕获 SIGTERM/SIGINT，确保 server 子进程被清理"""
    sig_name = signal.Signals(signum).name
    print(f"[Launcher] Received {sig_name}, shutting down...")
    _cleanup_server()
    # rumps 的 NSApplication 事件循环会拦截 SystemExit，必须用 os._exit 强制退出
    os._exit(128 + signum)

# 项目目录/资源目录
PROJECT_DIR = os.path.abspath(os.path.dirname(__file__))
RESOURCE_DIR = PROJECT_DIR
APP_SUPPORT_DIR = os.path.expanduser("~/Library/Application Support/Brainwave IME")

_SINGLE_INSTANCE_PATTERNS = [
    PROJECT_DIR,
    "launcher.py",
    "ime_menubar.py",
    "realtime_server.py",
]

def _should_enforce_single_instance() -> bool:
    return os.getenv("BRAINWAVE_ENFORCE_SINGLE_INSTANCE") == "1"

def _kill_previous_processes():
    if not _should_enforce_single_instance():
        return
    if sys.platform != "darwin":
        return

    try:
        output = subprocess.check_output(
            ["/bin/ps", "-x", "-o", "pid=,command="],
            text=True
        )
    except Exception as exc:
        print(f"[Launcher] Failed to list processes: {exc}")
        return

    current_pid = os.getpid()
    matches = []

    for line in output.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            pid_str, cmd = line.split(None, 1)
            pid = int(pid_str)
        except ValueError:
            continue

        if pid == current_pid or pid == os.getppid():
            continue

        if any(pattern in cmd for pattern in _SINGLE_INSTANCE_PATTERNS):
            matches.append((pid, cmd))

    if not matches:
        return

    print(f"[Launcher] Found {len(matches)} existing Brainwave processes. Stopping...")
    for pid, cmd in matches:
        try:
            print(f"[Launcher] Terminating PID {pid}: {cmd}")
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            continue
        except Exception as exc:
            print(f"[Launcher] Failed to terminate PID {pid}: {exc}")

    time.sleep(0.8)
    for pid, cmd in matches:
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            continue
        except Exception as exc:
            print(f"[Launcher] Failed to kill PID {pid}: {exc}")

def load_env():
    """加载环境变量"""
    env_paths = [
        os.path.join(APP_SUPPORT_DIR, ".env"),
        os.path.join(PROJECT_DIR, ".env"),
        os.path.join(RESOURCE_DIR, ".env"),
    ]

    for env_path in env_paths:
        if os.path.exists(env_path):
            with open(env_path) as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#') and '=' in line:
                        key, value = line.split('=', 1)
                        os.environ[key] = value
            print(f"[Launcher] Loaded env from {env_path}")
            break
    else:
        # 没有找到任何 .env 文件，尝试从 .env.example 生成
        example_path = os.path.join(PROJECT_DIR, ".env.example")
        target_path = os.path.join(PROJECT_DIR, ".env")
        if os.path.exists(example_path):
            import shutil
            shutil.copy2(example_path, target_path)
            print(f"[Launcher] WARNING: No .env found. Created {target_path} from .env.example")
            print("[Launcher] WARNING: Please fill in your API keys in .env")
        else:
            print("[Launcher] WARNING: No .env or .env.example found")


def preflight_check() -> bool:
    """启动前自检：验证关键配置并输出生效值摘要。返回 False 表示致命错误。"""
    ok = True
    provider = os.getenv("REALTIME_PROVIDER", "openai")

    # ── 1. API key 检查 ──
    if provider == "openai":
        key = os.getenv("OPENAI_API_KEY", "")
        if not key or key.startswith("your_"):
            print("[Preflight] FATAL: OPENAI_API_KEY is missing or placeholder")
            ok = False
    elif provider == "xai":
        key = os.getenv("XAI_API_KEY", "")
        if not key or key.startswith("your_"):
            print("[Preflight] FATAL: XAI_API_KEY is missing or placeholder")
            ok = False

    # ── 2. 关键 session 管理参数（带默认值兜底） ──
    session_params = {
        "BRAINWAVE_KEEP_PROVIDER_SESSION": ("1", "复用 provider session"),
        "BRAINWAVE_PROVIDER_SESSION_MAX_TURNS": ("8", "单 session 最大轮次"),
        "BRAINWAVE_PROVIDER_SESSION_MAX_AGE_SEC": ("7200", "单 session 最大存活秒数（2小时）"),
        "BRAINWAVE_INCLUDE_INSTRUCTIONS_EACH_RESPONSE": ("1", "每次 response 附带 instructions"),
        "BRAINWAVE_IDLE_WS_RECONNECT_SEC": (None, "空闲 WS 重连阈值秒"),
    }

    print("[Preflight] ── Session 防漂移配置 ──")
    for key, (default, desc) in session_params.items():
        effective = os.getenv(key, default)
        source = "env" if key in os.environ else "default"
        display = effective if effective is not None else "(inherit MAX_AGE)"
        print(f"[Preflight]   {key} = {display}  ({source}) — {desc}")

    # ── 3. 关键防护开关告警 ──
    if os.getenv("BRAINWAVE_KEEP_PROVIDER_SESSION", "1") == "1":
        max_turns = os.getenv("BRAINWAVE_PROVIDER_SESSION_MAX_TURNS", "8")
        include_inst = os.getenv("BRAINWAVE_INCLUDE_INSTRUCTIONS_EACH_RESPONSE", "1")
        if max_turns == "0":
            print("[Preflight] WARNING: MAX_TURNS=0 (unlimited) — session 漂移风险高")
        if include_inst != "1":
            print("[Preflight] WARNING: INCLUDE_INSTRUCTIONS_EACH_RESPONSE 未开启 — 漂移风险高")

    # ── 4. 其他关键参数 ──
    print(f"[Preflight] ── Provider 配置 ──")
    print(f"[Preflight]   REALTIME_PROVIDER = {provider}")
    if provider == "openai":
        print(f"[Preflight]   OPENAI_REALTIME_MODEL = {os.getenv('OPENAI_REALTIME_MODEL', 'gpt-realtime-mini-2025-12-15')}")
        print(f"[Preflight]   OPENAI_REALTIME_MODALITIES = {os.getenv('OPENAI_REALTIME_MODALITIES', 'text')}")

    if ok:
        print("[Preflight] ✓ All checks passed")
    else:
        print("[Preflight] ✗ Fatal errors detected — cannot start")

    return ok

def run_server_foreground():
    """在当前进程启动服务器"""
    os.chdir(RESOURCE_DIR)
    from realtime_server import app
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=23456)

def is_server_running():
    """检查服务器是否运行"""
    import socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    result = sock.connect_ex(('127.0.0.1', 23456))
    sock.close()
    return result == 0

def start_server_subprocess():
    """用子进程启动服务器"""
    env = os.environ.copy()
    venv_python = os.path.join(PROJECT_DIR, 'venv', 'bin', 'python')
    server_script = os.path.join(PROJECT_DIR, 'realtime_server.py')
    proc = subprocess.Popen(
        [venv_python, server_script],
        cwd=PROJECT_DIR,
        env=env,
    )
    print(f"[Launcher] Server process started: PID {proc.pid}")
    return proc

def run_menubar_directly():
    """直接运行菜单栏代码"""
    # 添加项目目录到路径
    if RESOURCE_DIR not in sys.path:
        sys.path.insert(0, RESOURCE_DIR)
    os.chdir(RESOURCE_DIR)

    # 等待服务器
    print("[Launcher] Waiting for server...")
    for i in range(20):
        if is_server_running():
            print("[Launcher] Server is ready!")
            break
        time.sleep(0.5)
    else:
        print("[Launcher] Warning: Server may not be ready")

    print("[Launcher] Starting menubar app...")

    # 导入并运行
    try:
        from ime_menubar import run_menubar
        run_menubar()
    except Exception as e:
        print(f"[Launcher] Error starting menubar: {e}")
        traceback.print_exc()

def main():
    print("[Launcher] Starting Brainwave IME...")

    if "--server" in sys.argv:
        load_env()
        if not preflight_check():
            print("[Launcher] Aborting due to preflight failures.")
            sys.exit(1)
        run_server_foreground()
        return

    load_env()
    if not preflight_check():
        print("[Launcher] Aborting due to preflight failures.")
        sys.exit(1)
    _kill_previous_processes()

    # 注册信号处理器和 atexit 兜底
    signal.signal(signal.SIGTERM, _signal_handler)
    signal.signal(signal.SIGINT, _signal_handler)
    atexit.register(_cleanup_server)

    global _server_proc

    # 启动服务器（如果没运行）
    if not is_server_running():
        _server_proc = start_server_subprocess()
        time.sleep(2)

    # 运行菜单栏应用
    try:
        run_menubar_directly()
    except KeyboardInterrupt:
        print("[Launcher] Interrupted")
    except Exception as e:
        print(f"[Launcher] Error: {e}")
        traceback.print_exc()
    finally:
        _cleanup_server()

if __name__ == "__main__":
    main()
