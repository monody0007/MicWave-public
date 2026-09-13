# MicWave (Brainwave IME)

macOS 菜单栏语音输入法，按下快捷键说话，转写后的文字自动粘贴到当前焦点应用。底层使用 OpenAI Realtime API 转写。

## 项目状态

这是一个 personal project 的公开快照，随私有主线定期更新；不提供支持，也不接受 PR。代码以 MIT 协议放出，任何人可以 fork 自行修改。

## 功能特点

- 双后端转写：稳定的 Echo 后端（OpenAI Realtime，`gpt-realtime-2.1-mini`）与隔离的 Whisper 后端（`gpt-4o-transcribe`），菜单栏一键切换
- 低延迟：tuned 过的 upload chunking 和 session reuse，`fast` preset 下 stop-to-response 通常在 300ms 以内
- 即时按键反馈：开始/停止录音提示音在按键瞬间播放（进程内预加载音效，无子进程延迟）
- 菜单栏常驻：低存在感的状态图标，热键 `Cmd+\`` 启停（默认）
- 可选自启动：通过 LaunchAgent 注册开机自启

## 系统要求

- macOS（在 Apple Silicon + 最近版本的 Sonoma / Sequoia 上测过）
- Python 3.11+
- 一个能用的麦克风
- 一个 OpenAI API key

## 快速上手

```bash
# 1. Clone 仓库
git clone https://github.com/monody0007/MicWave-public.git
cd MicWave-public

# 2. 配置 secrets
cp .env.example .env
# 编辑 .env，填入 OPENAI_API_KEY (必填)

# 3. 启动（自动建 venv、装依赖、注册 LaunchAgent）
./start_ime.sh --install-autostart
```

## macOS 权限授权

首次运行时 macOS 会要求手动授予三项权限。打开 `System Settings → Privacy & Security`，把当前用来跑 launcher.py 的 Python 解释器（或终端进程）加入下面三个清单：

1. **Accessibility（辅助功能）**：用于模拟 `Cmd+V` 粘贴
2. **Input Monitoring（输入监控）**：用于全局热键捕获
3. **Microphone（麦克风）**：音频采集

授权后重启 `./start_ime.sh` 即可生效。

## 使用方法

菜单栏图标出现后，按 `Cmd+\`` 开始录音。再次按下停止录音，转写结果会粘贴到当前焦点应用。菜单栏第一行显示当前后端（`EchoWave` / `WhisperWave`），空闲时点击可切换。

日志路径：`~/Library/Logs/Brainwave IME/brainwave_ime.log`

## 配置项

所有运行时参数都是环境变量。完整清单见 [`.env.example`](.env.example)。最常用的几个：

| 变量名                                     | 默认值                | 用途                          |
|--------------------------------------------|-----------------------|-------------------------------|
| `OPENAI_REALTIME_MODEL`                    | `gpt-realtime-2.1-mini` | Echo 后端选用的 realtime model |
| `BRAINWAVE_LATENCY_PRESET`                 | `fast`                | `fast` \| `balanced`          |
| `BRAINWAVE_KEEP_PROVIDER_SESSION`          | `1`                   | 复用 WS session（更快）       |
| `BRAINWAVE_PROVIDER_SESSION_MAX_TURNS`     | `8`                   | 累计 N 轮后轮换 session       |
| `BRAINWAVE_PROVIDER_SESSION_MAX_AGE_SEC`   | `7200`                | 累计 N 秒后轮换 session       |
| `BRAINWAVE_{START,STOP,COMPLETE,ERROR}_SOUND` | `Tink`/`Morse`/`Bottle`/`Basso` | 各提示音，`none` 关闭 |

## 卸载

```bash
./start_ime.sh --uninstall-autostart
pkill -f launcher.py
rm -rf venv whisper_backend/venv
```

## License

MIT，详见 [`LICENSE`](LICENSE)。

---

## English Summary

A macOS menu-bar voice-to-text input method powered by the OpenAI Realtime API. **Snapshot** — periodically synced from the private mainline; no support, no PRs. See sections above (Chinese) for full setup; commands and code are language-agnostic. MIT licensed.
