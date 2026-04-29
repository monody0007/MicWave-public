#!/usr/bin/env python3
"""
Brainwave IME - macOS 菜单栏应用
在菜单栏显示状态图标，支持快捷键录音
使用 macOS 原生 API 监听快捷键，避免 pynput 的线程问题
"""

import asyncio
import json
import os
import queue
import signal
import subprocess
import sys
import threading
import time
import wave
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Optional
import numpy as np
import scipy.signal
import websockets
import pyaudio
from transcript_merge import merge_incremental_text

# macOS 原生 API
from Quartz import (
    CGEventTapCreate, CGEventTapEnable, CGEventMaskBit,
    kCGEventKeyDown,
    kCGHeadInsertEventTap, kCGSessionEventTap,
    CGEventGetIntegerValueField, kCGKeyboardEventKeycode,
    CGEventGetFlags, kCGEventFlagMaskCommand,
    CGEventCreateKeyboardEvent, CGEventPost, CGEventSetFlags,
    CGEventSetType, kCGEventNull,
    CFRunLoopAddSource, CFRunLoopGetCurrent, CFRunLoopRun,
    CFMachPortCreateRunLoopSource, kCFRunLoopCommonModes
)

try:
    from Quartz import AXIsProcessTrustedWithOptions, kAXTrustedCheckOptionPrompt
    HAS_AX_TRUST = True
except Exception:
    HAS_AX_TRUST = False
    AXIsProcessTrustedWithOptions = None
    kAXTrustedCheckOptionPrompt = None

try:
    from Quartz import CGPreflightListenEventAccess, CGRequestListenEventAccess
    HAS_LISTEN_EVENT = True
except Exception:
    HAS_LISTEN_EVENT = False
    CGPreflightListenEventAccess = None
    CGRequestListenEventAccess = None

try:
    import rumps
    HAS_RUMPS = True
except ImportError:
    HAS_RUMPS = False
    print("Warning: rumps not installed. Run: pip install rumps")

try:
    from AppKit import (
        NSPanel,
        NSButton,
        NSFont,
        NSPasteboard,
        NSWindowStyleMaskTitled,
        NSWindowStyleMaskClosable,
        NSWindowStyleMaskUtilityWindow,
        NSBackingStoreBuffered,
        NSTextAlignmentLeft
    )
    HAS_APPKIT = True
except Exception:
    HAS_APPKIT = False
    NSPanel = None
    NSButton = None
    NSFont = None
    NSPasteboard = None
    NSWindowStyleMaskTitled = None
    NSWindowStyleMaskClosable = None
    NSWindowStyleMaskUtilityWindow = None
    NSBackingStoreBuffered = None
    NSTextAlignmentLeft = None

PROJECT_DIR = os.path.abspath(os.path.dirname(__file__))
PASTEBOARD_TEXT_TYPE = "public.utf8-plain-text"
KEYCODE_V = 9

def _accessibility_is_trusted(prompt: bool = False) -> Optional[bool]:
    if not HAS_AX_TRUST:
        return None
    try:
        options = {kAXTrustedCheckOptionPrompt: bool(prompt)}
        return bool(AXIsProcessTrustedWithOptions(options))
    except Exception as exc:
        print(f"[Access] Failed to check accessibility trust: {exc}")
        return None


def _input_monitoring_is_allowed(prompt: bool = False) -> Optional[bool]:
    if not HAS_LISTEN_EVENT:
        return None
    try:
        if prompt:
            return bool(CGRequestListenEventAccess())
        return bool(CGPreflightListenEventAccess())
    except Exception as exc:
        print(f"[Access] Failed to check input monitoring access: {exc}")
        return None


@dataclass
class Config:
    server_host: str = "localhost"
    server_port: int = 23456
    sample_rate: int = 48000
    target_sample_rate: int = 24000
    latency_preset: str = os.getenv("BRAINWAVE_LATENCY_PRESET", "fast").lower()
    chunk_size: int = int(os.getenv("BRAINWAVE_PYAUDIO_CHUNK_SIZE", "0"))
    upload_chunk_ms: int = int(os.getenv("BRAINWAVE_UPLOAD_CHUNK_MS", "0"))
    channels: int = 1
    hotkey_keycode: int = 50  # ` 键的 keycode
    provider: str = "openai"
    model: str = os.getenv("OPENAI_REALTIME_MODEL", "gpt-realtime-mini-2025-12-15")
    stop_tail_wait_min_ms: int = int(os.getenv("BRAINWAVE_STOP_TAIL_WAIT_MIN_MS", "260"))
    stop_tail_wait_max_ms: int = int(os.getenv("BRAINWAVE_STOP_TAIL_WAIT_MAX_MS", "400"))
    stop_tail_wait_guard_ms: int = int(os.getenv("BRAINWAVE_STOP_TAIL_WAIT_GUARD_MS", "30"))
    idle_ws_reconnect_sec: int = int(
        os.getenv(
            "BRAINWAVE_IDLE_WS_RECONNECT_SEC",
            os.getenv("BRAINWAVE_PROVIDER_SESSION_MAX_AGE_SEC", "36000"),
        )
    )


class IMEState(Enum):
    IDLE = "idle"
    RECORDING = "recording"
    PROCESSING = "processing"
    DISCONNECTED = "disconnected"


class RecordingMode(Enum):
    OPTIMIZED = "optimized"


class AudioProcessor:
    def __init__(self, source_rate: int = 48000, target_rate: int = 24000):
        self.source_rate = source_rate
        self.target_rate = target_rate

    def resample(self, audio_data: bytes) -> bytes:
        pcm_data = np.frombuffer(audio_data, dtype=np.int16)
        float_data = pcm_data.astype(np.float32) / 32768.0
        resampled = scipy.signal.resample_poly(
            float_data, self.target_rate, self.source_rate
        )
        resampled_int16 = (resampled * 32768.0).clip(-32768, 32767).astype(np.int16)
        return resampled_int16.tobytes()


class BrainwaveIMECore:
    """输入法核心逻辑"""

    def __init__(
        self,
        config: Config,
        on_state_change=None,
        on_transcript=None,
        on_transcript_complete=None
    ):
        self.config = config
        self.state = IMEState.DISCONNECTED
        self.recording_mode = None
        self.audio_processor = AudioProcessor(
            config.sample_rate, config.target_sample_rate
        )
        self.pyaudio_instance = pyaudio.PyAudio()
        self.audio_stream = None
        self.audio_buffer = []
        self._audio_buffer_samples = 0
        self.ws = None
        self.ws_connected = False
        self.hotkey_active = False
        self.loop = None
        self.transcript = ""
        self.on_state_change = on_state_change
        self.on_transcript = on_transcript
        self.on_transcript_complete = on_transcript_complete
        self._receive_task = None
        self._session_started = False
        self._session_task = None
        self._session_prompt_mode = "optimize"
        self._force_ws_refresh_before_turn = False
        self._last_turn_completed_ts = None
        self._last_turn_completed_wall_ts = None
        self._ws_connected_wall_ts = None
        self._turn_id = 0
        self._active_turn_id = None
        self._hotkey_down_ts = None
        self._recording_started_ts = None
        self._stop_pressed_ts = None
        self._response_done_ts = None
        self._last_audio_callback_ts = None
        if self.config.upload_chunk_ms <= 0 or self.config.chunk_size <= 0:
            preset_defaults = {
                "balanced": (160, 3840),
                "fast": (120, 2880),
            }
            default_upload_ms, default_chunk_frames = preset_defaults.get(
                self.config.latency_preset,
                preset_defaults["balanced"],
            )
            if self.config.upload_chunk_ms <= 0:
                self.config.upload_chunk_ms = default_upload_ms
            if self.config.chunk_size <= 0:
                self.config.chunk_size = default_chunk_frames
        self._chunk_size_frames = max(512, int(self.config.chunk_size))
        self._audio_callback_period_sec = self._chunk_size_frames / float(self.config.sample_rate)
        self._upload_chunk_samples = max(
            1,
            int(self.config.target_sample_rate * self.config.upload_chunk_ms / 1000.0),
        )
        self._upload_chunk_bytes = self._upload_chunk_samples * 2
        self._local_turn_audio = bytearray()
        self._max_local_audio_bytes = int(
            os.getenv("BRAINWAVE_LOCAL_AUDIO_MAX_BYTES", str(48_000 * 300))
        )
        self._recent_audio_cache_enabled = os.getenv(
            "BRAINWAVE_RECENT_AUDIO_CACHE_ENABLED",
            "1",
        ) == "1"
        recent_audio_cache_limit_raw = os.getenv(
            "BRAINWAVE_RECENT_AUDIO_CACHE_LIMIT",
            "0",
        )
        try:
            # 0 means unlimited (keep all recordings)
            self._recent_audio_cache_limit = max(0, int(recent_audio_cache_limit_raw))
        except ValueError:
            print(
                "[IME] Invalid BRAINWAVE_RECENT_AUDIO_CACHE_LIMIT="
                f"{recent_audio_cache_limit_raw!r}, fallback to 0 (unlimited)"
            )
            self._recent_audio_cache_limit = 0
        self._recent_audio_cache_dir = os.path.expanduser(
            os.getenv(
                "BRAINWAVE_RECENT_AUDIO_CACHE_DIR",
                "~/Library/Application Support/Brainwave IME/recent_audio",
            )
        )
        self._recent_audio_enqueued_turn_id = None
        self._recent_audio_queue = queue.Queue(
            maxsize=max(8, self._recent_audio_cache_limit * 2) if self._recent_audio_cache_limit else 0
        )
        self._recent_audio_worker_stop = threading.Event()
        self._recent_audio_worker = None
        if self._recent_audio_cache_enabled:
            self._start_recent_audio_worker()

    def _compute_idle_ws_age_sec(self) -> tuple[Optional[float], str]:
        if self._last_turn_completed_wall_ts is not None:
            idle_sec = max(0.0, time.time() - self._last_turn_completed_wall_ts)
            return idle_sec, "last_turn_completed"
        if self._ws_connected_wall_ts is not None:
            idle_sec = max(0.0, time.time() - self._ws_connected_wall_ts)
            return idle_sec, "ws_connected"
        return None, "unknown"

    def _should_refresh_ws_before_turn(self) -> bool:
        threshold_sec = self.config.idle_ws_reconnect_sec
        if threshold_sec <= 0 or not self.ws_connected:
            return False
        idle_sec, _ = self._compute_idle_ws_age_sec()
        if idle_sec is None:
            return False
        return idle_sec >= float(threshold_sec)

    async def _refresh_ws_for_idle_hygiene(self):
        if self._receive_task:
            self._receive_task.cancel()
            try:
                await self._receive_task
            except asyncio.CancelledError:
                pass
            self._receive_task = None
        if self.ws:
            try:
                await self.ws.close()
            except Exception:
                pass
            self.ws = None
        self.ws_connected = False
        self._session_started = False
        self._ws_connected_wall_ts = None

    def _close_audio_stream(self):
        if not self.audio_stream:
            return
        try:
            self.audio_stream.stop_stream()
        except Exception:
            pass
        try:
            self.audio_stream.close()
        except Exception:
            pass
        self.audio_stream = None

    def _trim_local_audio(self):
        overflow = len(self._local_turn_audio) - self._max_local_audio_bytes
        if overflow <= 0:
            return
        if overflow % 2 != 0:
            overflow += 1
        del self._local_turn_audio[:overflow]

    def _archive_failed_turn_audio(self, reason: str) -> Optional[str]:
        if not self._local_turn_audio:
            return None
        try:
            archive_dir = os.path.join(
                os.path.expanduser("~"),
                "Library",
                "Application Support",
                "Brainwave IME",
                "failed_audio"
            )
            os.makedirs(archive_dir, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            file_path = os.path.join(archive_dir, f"turn_{timestamp}_{reason}.wav")
            with wave.open(file_path, "wb") as wav_file:
                wav_file.setnchannels(self.config.channels)
                wav_file.setsampwidth(2)  # int16 PCM
                wav_file.setframerate(self.config.target_sample_rate)
                wav_file.writeframes(bytes(self._local_turn_audio))
            return file_path
        except Exception as exc:
            print(f"[IME] Failed to archive local audio: {exc}")
            return None

    def _start_recent_audio_worker(self):
        if self._recent_audio_worker and self._recent_audio_worker.is_alive():
            return
        self._recent_audio_worker_stop.clear()
        self._recent_audio_worker = threading.Thread(
            target=self._recent_audio_worker_loop,
            daemon=True,
            name="recent-audio-cache-worker",
        )
        self._recent_audio_worker.start()

    def _enqueue_recent_audio_task(self, task: Optional[dict]) -> bool:
        try:
            self._recent_audio_queue.put_nowait(task)
            return True
        except queue.Full:
            # Drop the oldest pending task to keep enqueue non-blocking.
            try:
                dropped = self._recent_audio_queue.get_nowait()
                self._recent_audio_queue.task_done()
                if dropped is None:
                    # Preserve stop sentinel if it was dequeued.
                    self._recent_audio_queue.put_nowait(None)
            except Exception:
                pass
            try:
                self._recent_audio_queue.put_nowait(task)
                return True
            except Exception:
                return False

    def _archive_recent_turn_audio(self, outcome: str) -> bool:
        if not self._recent_audio_cache_enabled:
            return False
        if not self._local_turn_audio:
            return False
        if (
            self._active_turn_id is not None
            and self._recent_audio_enqueued_turn_id == self._active_turn_id
        ):
            return False

        task = {
            "turn_id": self._active_turn_id,
            "outcome": outcome,
            "audio_bytes": bytes(self._local_turn_audio),
            "channels": self.config.channels,
            "sample_rate": self.config.target_sample_rate,
        }
        enqueued = self._enqueue_recent_audio_task(task)
        if enqueued and self._active_turn_id is not None:
            self._recent_audio_enqueued_turn_id = self._active_turn_id
        if not enqueued:
            print("[IME] Recent audio queue is full, dropping newest cached audio task")
        return enqueued

    def _recent_audio_worker_loop(self):
        while not self._recent_audio_worker_stop.is_set():
            try:
                task = self._recent_audio_queue.get(timeout=0.2)
            except queue.Empty:
                continue

            try:
                if task is None:
                    break
                recent_audio_path = self._write_recent_turn_audio(task)
                if recent_audio_path:
                    print(f"[IME] Recent audio cached: {recent_audio_path}")
            except Exception as exc:
                print(f"[IME] Recent audio worker error: {exc}")
            finally:
                self._recent_audio_queue.task_done()

    def _write_recent_turn_audio(self, task: dict) -> Optional[str]:
        turn_id = task.get("turn_id")
        outcome = str(task.get("outcome", "unknown"))
        audio_bytes = task.get("audio_bytes")
        channels = int(task.get("channels", self.config.channels))
        sample_rate = int(task.get("sample_rate", self.config.target_sample_rate))

        if not audio_bytes:
            return None

        safe_outcome = "".join(
            ch if ch.isalnum() or ch in {"-", "_"} else "_"
            for ch in outcome
        ).strip("_") or "unknown"
        turn_label = (
            f"T{turn_id}"
            if turn_id is not None
            else "Tunknown"
        )

        try:
            os.makedirs(self._recent_audio_cache_dir, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            file_name = f"{timestamp}_{turn_label}_{safe_outcome}.wav"
            file_path = os.path.join(self._recent_audio_cache_dir, file_name)
            with wave.open(file_path, "wb") as wav_file:
                wav_file.setnchannels(channels)
                wav_file.setsampwidth(2)  # int16 PCM
                wav_file.setframerate(sample_rate)
                wav_file.writeframes(audio_bytes)
            self._prune_recent_audio_cache()
            return file_path
        except Exception as exc:
            print(f"[IME] Failed to archive recent turn audio: {exc}")
            return None

    def _prune_recent_audio_cache(self):
        if self._recent_audio_cache_limit == 0:
            return  # 0 = unlimited, keep all recordings
        try:
            wav_files = []
            for name in os.listdir(self._recent_audio_cache_dir):
                if not name.lower().endswith(".wav"):
                    continue
                path = os.path.join(self._recent_audio_cache_dir, name)
                try:
                    mtime = os.path.getmtime(path)
                except OSError:
                    continue
                wav_files.append((mtime, path))
            if len(wav_files) <= self._recent_audio_cache_limit:
                return
            wav_files.sort(key=lambda item: item[0], reverse=True)
            for _, stale_path in wav_files[self._recent_audio_cache_limit:]:
                try:
                    os.remove(stale_path)
                except Exception as exc:
                    print(f"[IME] Failed to prune cached audio {stale_path}: {exc}")
        except Exception as exc:
            print(f"[IME] Failed to prune recent audio cache: {exc}")

    def _start_session_task(self):
        if not self.loop:
            return
        if self._session_task and not self._session_task.done():
            return
        self._session_task = asyncio.run_coroutine_threadsafe(
            self._ensure_session_started(self._session_prompt_mode),
            self.loop
        )

    def _clear_audio_buffer(self):
        self.audio_buffer.clear()
        self._audio_buffer_samples = 0

    def _append_audio_buffer(self, chunk: bytes):
        self.audio_buffer.append(chunk)
        self._audio_buffer_samples += len(chunk) // 2

    def _compute_stop_tail_wait_sec(self) -> float:
        min_wait_sec = max(0.0, self.config.stop_tail_wait_min_ms / 1000.0)
        max_wait_sec = max(min_wait_sec, self.config.stop_tail_wait_max_ms / 1000.0)
        guard_sec = max(0.0, self.config.stop_tail_wait_guard_ms / 1000.0)

        target_wait_sec = self._audio_callback_period_sec + guard_sec
        if self._last_audio_callback_ts is None:
            adaptive_wait_sec = target_wait_sec
        else:
            elapsed_sec = time.perf_counter() - self._last_audio_callback_ts
            adaptive_wait_sec = max(0.0, target_wait_sec - elapsed_sec)

        return min(max_wait_sec, max(min_wait_sec, adaptive_wait_sec))

    def _set_state(self, new_state: IMEState):
        if self.state != new_state:
            old_state = self.state
            self.state = new_state
            if (
                new_state == IMEState.DISCONNECTED
                and old_state in (IMEState.RECORDING, IMEState.PROCESSING)
            ):
                # Unexpected disconnect during an active turn: surface it with a clear sound cue.
                self._play_sound("Basso")
            if new_state in (IMEState.IDLE, IMEState.DISCONNECTED):
                self.recording_mode = None
            print(f"[IME] State: {old_state.value} -> {new_state.value}")
            if self.on_state_change:
                self.on_state_change(new_state)

    def _play_sound(self, sound_name: str):
        try:
            # Launch immediately in caller thread to reduce scheduling jitter,
            # then reap in background to avoid zombie processes.
            proc = subprocess.Popen(
                ['afplay', f'/System/Library/Sounds/{sound_name}.aiff'],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            threading.Thread(target=proc.wait, daemon=True).start()
        except Exception as exc:
            print(f"[IME] Sound play error ({sound_name}): {exc}")

    async def connect_websocket(self):
        uri = f"ws://{self.config.server_host}:{self.config.server_port}/api/v1/ws"
        try:
            self.ws = await websockets.connect(uri)
            self.ws_connected = True
            self._ws_connected_wall_ts = time.time()
            if self.state in (IMEState.DISCONNECTED, IMEState.IDLE):
                self._set_state(IMEState.IDLE)
            print(f"[IME] Connected to {uri}")
            if self._receive_task is None or self._receive_task.done():
                self._receive_task = asyncio.create_task(self.receive_messages())
            return True
        except Exception as e:
            print(f"[IME] Connection failed: {e}")
            self.ws_connected = False
            self._ws_connected_wall_ts = None
            self._set_state(IMEState.DISCONNECTED)
            return False

    async def disconnect_websocket(self):
        if self._receive_task:
            self._receive_task.cancel()
            try:
                await self._receive_task
            except asyncio.CancelledError:
                pass
            self._receive_task = None
        if self.ws:
            await self.ws.close()
            self.ws = None
        self.ws_connected = False
        self._ws_connected_wall_ts = None
        self._set_state(IMEState.DISCONNECTED)

    async def receive_messages(self):
        try:
            async for message in self.ws:
                if isinstance(message, str):
                    data = json.loads(message)
                    await self._handle_message(data)
        except websockets.exceptions.ConnectionClosed:
            print("[IME] Connection closed")
            self.ws_connected = False
            self.ws = None
            self._ws_connected_wall_ts = None
            self._session_started = False
            if self.state in (IMEState.RECORDING, IMEState.PROCESSING):
                print("[IME] Connection lost during active turn, keep local recording and retry session.")
                self._start_session_task()
            else:
                self._set_state(IMEState.DISCONNECTED)
        except asyncio.CancelledError:
            pass
        except Exception as e:
            print(f"[IME] Receive error: {e}")
            self.ws_connected = False
            self.ws = None
            self._ws_connected_wall_ts = None
            self._session_started = False
            if self.state in (IMEState.RECORDING, IMEState.PROCESSING):
                self._start_session_task()
            else:
                self._set_state(IMEState.DISCONNECTED)
        finally:
            self._receive_task = None

    async def _handle_message(self, data: dict):
        msg_type = data.get("type")
        print(f"[IME] Received message: {msg_type}")
        if not self._is_message_for_active_turn(msg_type, data):
            return

        if msg_type == "status":
            status = data.get("status")
            if status == "connected":
                self._session_started = True
            if status == "idle" and self.state == IMEState.PROCESSING:
                self._last_turn_completed_ts = time.perf_counter()
                self._last_turn_completed_wall_ts = time.time()
                self._response_done_ts = time.perf_counter()
                if self._stop_pressed_ts is not None:
                    stop_to_response_done_ms = (self._response_done_ts - self._stop_pressed_ts) * 1000
                    print(f"[Perf][T{self._active_turn_id}] stop_to_response_done_ms={stop_to_response_done_ms:.1f}")
                last_mode = self.recording_mode
                self._archive_recent_turn_audio("completed")
                self._session_started = False
                self._set_state(IMEState.IDLE)
                if self.transcript:
                    # 先播放声音，立即给用户反馈
                    self._play_sound("Morse")
                    if self.on_transcript_complete:
                        try:
                            self.on_transcript_complete(
                                self.transcript,
                                last_mode,
                                self.config.provider,
                                self.config.model
                            )
                        except Exception as exc:
                            print(f"[IME] Transcript complete callback error: {exc}")
                    # 然后输入文字（这个操作有延迟）
                    await self._input_text(self.transcript)

        elif msg_type == "text":
            content = data.get("content", "")
            is_new = data.get("isNewResponse", False)
            if is_new and self.state not in (IMEState.RECORDING, IMEState.PROCESSING):
                print(f"[IME] Dropping reset text while state={self.state.value}")
                return
            if not is_new and self.state != IMEState.PROCESSING:
                print(f"[IME] Dropping text delta while state={self.state.value}")
                return
            if is_new:
                self.transcript = content
            else:
                self.transcript = merge_incremental_text(self.transcript, content)
            if self.on_transcript:
                self.on_transcript(self.transcript)

        elif msg_type == "error":
            print(f"[IME] Error: {data.get('content')}")
            if self.state == IMEState.RECORDING:
                # Keep recording locally; retry session in background.
                had_started_session = self._session_started
                self._session_started = False
                if had_started_session:
                    self._play_sound("Basso")
                else:
                    print("[IME] Session warmup failed before first successful start; retrying silently.")
                self._start_session_task()
            elif self.state == IMEState.PROCESSING:
                self._play_sound("Basso")
                archive_path = self._archive_failed_turn_audio("processing_error")
                if archive_path:
                    print(f"[IME] Local audio archived: {archive_path}")
                self._archive_recent_turn_audio("processing_error")
                self._set_state(IMEState.IDLE)
            else:
                self._set_state(IMEState.IDLE)

    async def _input_text(self, text: str):
        if not text:
            return
        try:
            old_clipboard = self._get_clipboard_text_native()
            paste_mode = "native"
            pasted = False

            if old_clipboard is not None and self._set_clipboard_text_native(text):
                await asyncio.sleep(0.01)
                pasted = self._trigger_cmd_v_native()
            else:
                paste_mode = "fallback"

            if not pasted:
                # Fallback path keeps current behavior for environments where
                # native pasteboard/event APIs are unavailable.
                if old_clipboard is None:
                    proc = subprocess.run(
                        ['pbpaste'],
                        capture_output=True,
                        text=True,
                        check=False,
                        timeout=2,
                    )
                    old_clipboard = proc.stdout
                proc = subprocess.Popen(['pbcopy'], stdin=subprocess.PIPE)
                proc.communicate(text.encode('utf-8'))
                await asyncio.sleep(0.03)
                paste_script = '''
                tell application "System Events"
                    keystroke "v" using command down
                end tell
                '''
                subprocess.run(['osascript', '-e', paste_script], check=False, timeout=3)
                paste_mode = "fallback"

            if self._stop_pressed_ts is not None:
                stop_to_cursor_paste_ms = (time.perf_counter() - self._stop_pressed_ts) * 1000
                print(
                    f"[Perf][T{self._active_turn_id}] "
                    f"stop_to_cursor_paste_ms={stop_to_cursor_paste_ms:.1f}"
                )

            # 异步恢复剪贴板，避免阻塞当前转录回传链路
            asyncio.create_task(
                self._restore_clipboard_if_unchanged(
                    pasted_text=text,
                    old_clipboard=old_clipboard or "",
                    use_native=(paste_mode == "native"),
                    delay_sec=0.3,
                )
            )

            preview = f"{text[:50]}..." if len(text) > 50 else text
            print(f"[IME] Input ({paste_mode}, chars={len(text)}): {preview}")
            if self._response_done_ts is not None:
                response_done_to_paste_done_ms = (time.perf_counter() - self._response_done_ts) * 1000
                print(f"[Perf][T{self._active_turn_id}] response_done_to_paste_done_ms={response_done_to_paste_done_ms:.1f}")
        except Exception as e:
            print(f"[IME] Input error: {e}")

    async def _restore_clipboard_if_unchanged(
        self,
        pasted_text: str,
        old_clipboard: str,
        use_native: bool = True,
        delay_sec: float = 0.3,
    ):
        try:
            await asyncio.sleep(max(0.0, delay_sec))
            if use_native:
                current_clipboard = self._get_clipboard_text_native()
                if current_clipboard is None:
                    use_native = False

            if not use_native:
                proc = subprocess.run(
                    ['pbpaste'],
                    capture_output=True,
                    text=True,
                    check=False,
                    timeout=2,
                )
                current_clipboard = proc.stdout

            # 仅当剪贴板仍是本次注入文本时再恢复，避免覆盖用户后续复制。
            if current_clipboard != pasted_text:
                return

            if use_native and self._set_clipboard_text_native(old_clipboard):
                return

            proc = subprocess.Popen(['pbcopy'], stdin=subprocess.PIPE)
            proc.communicate(old_clipboard.encode('utf-8'))
        except Exception as e:
            print(f"[IME] Clipboard restore error: {e}")

    def _normalize_turn_id(self, raw_turn_id) -> Optional[int]:
        if raw_turn_id is None:
            return None
        try:
            return int(raw_turn_id)
        except (TypeError, ValueError):
            return None

    def _is_message_for_active_turn(self, msg_type: str, data: dict) -> bool:
        if msg_type not in {"status", "text", "error"}:
            return True
        msg_turn_id = self._normalize_turn_id(data.get("turn_id"))
        if data.get("turn_id") is not None and msg_turn_id is None:
            print(f"[IME] Ignoring message with invalid turn_id: {data.get('turn_id')!r}")
            return False
        if msg_turn_id is None or self._active_turn_id is None:
            return True
        if msg_turn_id != self._active_turn_id:
            print(
                f"[IME] Ignoring stale {msg_type} message for turn {msg_turn_id}, "
                f"active turn is {self._active_turn_id}"
            )
            return False
        return True

    def _get_clipboard_text_native(self) -> Optional[str]:
        if not HAS_APPKIT or NSPasteboard is None:
            return None
        try:
            pasteboard = NSPasteboard.generalPasteboard()
            if pasteboard is None:
                return None
            value = pasteboard.stringForType_(PASTEBOARD_TEXT_TYPE)
            if value is None:
                return ""
            return str(value)
        except Exception:
            return None

    def _set_clipboard_text_native(self, text: str) -> bool:
        if not HAS_APPKIT or NSPasteboard is None:
            return False
        try:
            pasteboard = NSPasteboard.generalPasteboard()
            if pasteboard is None:
                return False
            pasteboard.clearContents()
            return bool(pasteboard.setString_forType_(text, PASTEBOARD_TEXT_TYPE))
        except Exception:
            return False

    def _trigger_cmd_v_native(self) -> bool:
        try:
            key_down = CGEventCreateKeyboardEvent(None, KEYCODE_V, True)
            key_up = CGEventCreateKeyboardEvent(None, KEYCODE_V, False)
            if key_down is None or key_up is None:
                return False
            CGEventSetFlags(key_down, kCGEventFlagMaskCommand)
            CGEventSetFlags(key_up, kCGEventFlagMaskCommand)
            CGEventPost(kCGSessionEventTap, key_down)
            CGEventPost(kCGSessionEventTap, key_up)
            return True
        except Exception:
            return False

    def start_recording(self):
        if self.state not in (IMEState.IDLE,):
            return

        self._turn_id += 1
        self._active_turn_id = self._turn_id
        self._force_ws_refresh_before_turn = self._should_refresh_ws_before_turn()
        if self._force_ws_refresh_before_turn:
            idle_sec, idle_basis = self._compute_idle_ws_age_sec()
            if idle_sec is not None:
                basis_msg = "last turn completed" if idle_basis == "last_turn_completed" else "ws connected"
                print(
                    f"[IME] Idle threshold reached ({idle_sec:.1f}s >= {self.config.idle_ws_reconnect_sec}s, "
                    f"basis={basis_msg}), refreshing websocket before this turn."
                )
            else:
                print(
                    "[IME] Idle threshold reached, refreshing websocket before this turn."
                )
        self.recording_mode = RecordingMode.OPTIMIZED
        prompt_mode = "optimize"
        self._session_prompt_mode = prompt_mode
        self._session_started = False

        # Play start cue first so perceived audio feedback does not lag behind state icon.
        self._play_sound("Tink")
        self._set_state(IMEState.RECORDING)
        self.transcript = ""
        self._clear_audio_buffer()
        self._last_audio_callback_ts = None
        self._local_turn_audio = bytearray()
        print(
            f"[Perf][T{self._active_turn_id}] latency_preset={self.config.latency_preset} "
            f"upload_chunk_ms={self.config.upload_chunk_ms} "
            f"(samples={self._upload_chunk_samples}), callback_chunk_frames={self._chunk_size_frames}"
        )

        try:
            self.audio_stream = self.pyaudio_instance.open(
                format=pyaudio.paInt16,
                channels=self.config.channels,
                rate=self.config.sample_rate,
                input=True,
                frames_per_buffer=self._chunk_size_frames,
                stream_callback=self._audio_callback
            )
            self.audio_stream.start_stream()
            self._recording_started_ts = time.perf_counter()
            if self._hotkey_down_ts is not None:
                hotkey_to_recording_ms = (self._recording_started_ts - self._hotkey_down_ts) * 1000
                print(f"[Perf][T{self._active_turn_id}] hotkey_to_recording_ms={hotkey_to_recording_ms:.1f}")
        except Exception as e:
            print(f"[IME] Audio error: {e}")
            self._set_state(IMEState.IDLE)
            return

        self._start_session_task()

    async def _ensure_session_started(self, prompt_mode: str):
        retry_interval_sec = 0.3
        if self._force_ws_refresh_before_turn:
            self._force_ws_refresh_before_turn = False
            await self._refresh_ws_for_idle_hygiene()
        while self.state in (IMEState.RECORDING, IMEState.PROCESSING) and not self._session_started:
            if not self.ws_connected:
                connected = await self.connect_websocket()
                if not connected:
                    await asyncio.sleep(retry_interval_sec)
                    continue
            try:
                await self.ws.send(json.dumps({
                    "type": "start_recording",
                    "provider": self.config.provider,
                    "model": self.config.model if self.config.provider == "openai" else None,
                    "prompt_mode": prompt_mode,
                    "input_sample_rate": self.config.target_sample_rate,
                    "turn_id": self._active_turn_id,
                }))
                self._session_started = True
                return
            except Exception as exc:
                print(f"[IME] Failed to start recording session, retrying: {exc}")
                self.ws_connected = False
                self.ws = None
                await asyncio.sleep(retry_interval_sec)

    def _audio_callback(self, in_data, frame_count, time_info, status):
        if self.state in (IMEState.RECORDING, IMEState.PROCESSING):
            self._last_audio_callback_ts = time.perf_counter()
            resampled = self.audio_processor.resample(in_data)
            self._local_turn_audio.extend(resampled)
            self._trim_local_audio()
            self._append_audio_buffer(resampled)

            if (
                self._audio_buffer_samples >= self._upload_chunk_samples
                and self.ws_connected
                and self._session_started
            ):
                combined = b''.join(self.audio_buffer)
                send_buffer = combined[:self._upload_chunk_bytes]
                remaining = combined[self._upload_chunk_bytes:]
                self._clear_audio_buffer()
                if remaining:
                    self._append_audio_buffer(remaining)
                if self.loop:
                    asyncio.run_coroutine_threadsafe(
                        self.ws.send(send_buffer), self.loop
                    )
        return (None, pyaudio.paContinue)

    def stop_recording(self):
        if self.state != IMEState.RECORDING:
            return

        self._stop_pressed_ts = time.perf_counter()
        # Ensure each processing turn starts from a clean transcript buffer.
        self.transcript = ""
        self._set_state(IMEState.PROCESSING)

        if self.loop:
            asyncio.run_coroutine_threadsafe(self._async_stop(), self.loop)

    async def _async_stop(self):
        # 自适应等待尾音：至少等待一个回调周期+保护时间，且保留保守下限。
        tail_wait_sec = self._compute_stop_tail_wait_sec()
        print(f"[Perf][T{self._active_turn_id}] stop_tail_wait_ms={tail_wait_sec * 1000:.1f}")
        await asyncio.sleep(tail_wait_sec)

        self._close_audio_stream()

        # Give session startup a last chance before declaring upload failure.
        if not self._session_started and self._session_task and not self._session_task.done():
            try:
                await asyncio.wait_for(asyncio.wrap_future(self._session_task), timeout=1.5)
            except Exception:
                pass

        # 发送剩余的缓冲音频
        if self.audio_buffer:
            combined = b''.join(self.audio_buffer)
            if combined and self.ws_connected and self._session_started:
                await self.ws.send(combined)
            self._clear_audio_buffer()

        # 立即发送停止信号（不需要额外等待，WebSocket是顺序的）
        if self.ws_connected and self._session_started:
            await self.ws.send(json.dumps({
                "type": "stop_recording",
                "turn_id": self._active_turn_id,
            }))
        else:
            archive_path = self._archive_failed_turn_audio("upload_unavailable")
            if archive_path:
                print(f"[IME] Local audio archived: {archive_path}")
            self._archive_recent_turn_audio("upload_unavailable")
            self._play_sound("Basso")
            self._set_state(IMEState.IDLE)

    def on_hotkey_press(self):
        """快捷键按下 - 切换录音状态"""
        if self.state == IMEState.IDLE:
            self._hotkey_down_ts = time.perf_counter()
            self.start_recording()
        elif self.state == IMEState.RECORDING:
            self.stop_recording()
        # 如果是 PROCESSING 或 DISCONNECTED 状态则忽略

    def on_hotkey_down(self):
        """快捷键按下 (兼容旧接口)"""
        self.on_hotkey_press()

    def on_hotkey_up(self):
        """快捷键释放 (切换模式下不需要处理)"""
        pass

    def cleanup(self):
        self._close_audio_stream()
        if self._recent_audio_worker:
            self._recent_audio_worker_stop.set()
            if not self._enqueue_recent_audio_task(None):
                try:
                    _ = self._recent_audio_queue.get_nowait()
                    self._recent_audio_queue.task_done()
                except Exception:
                    pass
                try:
                    self._recent_audio_queue.put_nowait(None)
                except Exception:
                    pass
            self._recent_audio_worker.join(timeout=0.5)
        self.pyaudio_instance.terminate()


if HAS_RUMPS:
    class BrainwaveIMEApp(rumps.App):
        """macOS 菜单栏应用"""

        STATE_ICONS = {
            IMEState.IDLE: "🎤",
            IMEState.RECORDING: "🟣",
            IMEState.PROCESSING: "⏳",
            IMEState.DISCONNECTED: "⚫",
        }
        STATUS_GUIDE = [
            ("idle", IMEState.IDLE, None, "🎤 Idle - Ready (not recording)"),
            ("recording_optimized", IMEState.RECORDING, None, "🟣 Recording - Listening"),
            ("processing", IMEState.PROCESSING, None, "⏳ Processing - Transcribing"),
            ("disconnected", IMEState.DISCONNECTED, None, "⚫ Disconnected - Server offline"),
        ]
        PROVIDER_OPTIONS = [
            ("openai", "OpenAI"),
            ("xai", "xAI"),
        ]
        MODEL_OPTIONS = [
            ("gpt-realtime-mini-2025-12-15", "GPT Real Time Mini"),
            ("gpt-realtime", "GPT Realtime"),
        ]

        def __init__(self):
            super().__init__("⚫", quit_button=None)
            self.config = Config()
            self.core = BrainwaveIMECore(
                self.config,
                on_state_change=self._on_state_change,
                on_transcript=self._on_transcript,
                on_transcript_complete=self._on_transcript_complete
            )

            self.provider_labels = {key: label for key, label in self.PROVIDER_OPTIONS}
            self.model_labels = {key: label for key, label in self.MODEL_OPTIONS}
            self.status_item = rumps.MenuItem("Status: Disconnected", callback=self._noop)
            self.status_guide_header = rumps.MenuItem("States", callback=self._noop)
            self.status_guide_items = {}
            self.status_guide_labels = {}
            for key, state, mode, label in self.STATUS_GUIDE:
                self.status_guide_labels[key] = label
                item = rumps.MenuItem(label, callback=self._noop)
                self.status_guide_items[key] = item
            self.reconnect_item = rumps.MenuItem("Reconnect", callback=self.reconnect)

            self.provider_menu = rumps.MenuItem("Provider")
            self.provider_items = {}
            for key, label in self.PROVIDER_OPTIONS:
                item = rumps.MenuItem(label, callback=self._provider_selected)
                item._provider_key = key
                self.provider_menu.add(item)
                self.provider_items[key] = item

            self.model_menu = rumps.MenuItem("Model")
            self.model_items = {}
            for key, label in self.MODEL_OPTIONS:
                item = rumps.MenuItem(label, callback=self._model_selected)
                item._model_key = key
                self.model_menu.add(item)
                self.model_items[key] = item

            self.hotkeys_item = rumps.MenuItem("Hotkey: Cmd+` (Optimized)", callback=None)

            self._recent_limit = 5
            self._pending_transcripts = queue.Queue()
            self._recent_transcripts = []
            self._history_dir = self._resolve_history_dir()
            self._history_file = os.path.join(self._history_dir, "transcripts.jsonl")
            self._recent_item = rumps.MenuItem("Recent", callback=self._open_recent_panel)
            self._history_item = rumps.MenuItem("History", callback=self._open_history_folder)
            self._recent_panel = None
            self._recent_panel_buttons = []

            self._load_recent_history()

            self.menu = [
                self.status_item,
                self.status_guide_header,
                *list(self.status_guide_items.values()),
                self.reconnect_item,
                None,
                self.provider_menu,
                self.model_menu,
                self.hotkeys_item,
                None,
                self._recent_item,
                self._history_item,
                None,
                rumps.MenuItem("Quit", callback=self.quit_app),
            ]

            self.loop = None
            self.loop_thread = None
            self.event_tap = None
            self.event_tap_thread = None
            self._last_state = IMEState.DISCONNECTED
            self._sync_status_menu(self.core.state)
            self._sync_provider_menu()
            self._sync_model_menu()
            self._accessibility_warning_needed = False
            self._accessibility_warning_sent = False
            self._input_monitoring_warning_needed = False
            self._input_monitoring_warning_sent = False
            trusted = _accessibility_is_trusted(prompt=True)
            if trusted is False:
                self._accessibility_warning_needed = True
            input_allowed = _input_monitoring_is_allowed(prompt=True)
            if input_allowed is False:
                self._input_monitoring_warning_needed = True

            # 用定时器轮询状态变化，确保 UI 在主线程更新
            self._state_timer = rumps.Timer(self._poll_state, 0.2)
            self._state_timer.start()

            print("[App] BrainwaveIMEApp initialized, starting auto-connect in 1 second...")

            # 启动后自动连接
            threading.Timer(1.0, self._auto_connect).start()

        def _auto_connect(self):
            self._connect()
            # 启动快捷键监听
            self._start_event_tap()

        def _start_event_tap(self):
            """启动 macOS 原生事件监听"""
            def event_tap_thread():
                def callback(proxy, event_type, event, refcon):
                    try:
                        # macOS 会在 tap 响应慢时自动禁用，收到此事件时重新启用
                        if event_type == 0xFFFFFFFE:  # kCGEventTapDisabledByTimeout
                            print("[EventTap] Re-enabling event tap (was disabled by timeout)")
                            CGEventTapEnable(self.event_tap, True)
                            return event

                        # 只处理按下事件
                        if event_type != kCGEventKeyDown:
                            return event

                        keycode = CGEventGetIntegerValueField(event, kCGKeyboardEventKeycode)
                        flags = CGEventGetFlags(event)
                        cmd_pressed = bool(flags & kCGEventFlagMaskCommand)

                        # 监听 Cmd + ` (keycode 50) - 触发优化模式
                        if keycode == self.config.hotkey_keycode and cmd_pressed:
                            print(f"[EventTap] Optimize hotkey pressed, current state: {self.core.state.value}")
                            # 直接调用核心方法（线程安全，因为只修改状态）
                            self.core.on_hotkey_press()
                            # 标记为 Null 事件，避免系统默认切换窗口导致光标跳走
                            try:
                                CGEventSetType(event, kCGEventNull)
                            except Exception as exc:
                                print(f"[EventTap] Failed to nullify event: {exc}")
                            return event
                    except Exception as e:
                        print(f"[EventTap] Error: {e}")
                    return event

                # 仅监听按下事件（Cmd+`）
                mask = CGEventMaskBit(kCGEventKeyDown)
                self.event_tap = CGEventTapCreate(
                    kCGSessionEventTap,
                    kCGHeadInsertEventTap,
                    0,  # 0 = active tap
                    mask,
                    callback,
                    None
                )

                if self.event_tap is None:
                    print("[EventTap] Failed to create event tap. Need accessibility permission!")
                    self._accessibility_warning_needed = True
                    self._input_monitoring_warning_needed = True
                    return

                # 添加到 run loop
                source = CFMachPortCreateRunLoopSource(None, self.event_tap, 0)
                CFRunLoopAddSource(CFRunLoopGetCurrent(), source, kCFRunLoopCommonModes)
                CGEventTapEnable(self.event_tap, True)
                print("[EventTap] Started listening for Cmd+`")
                CFRunLoopRun()

            self.event_tap_thread = threading.Thread(target=event_tap_thread, daemon=True)
            self.event_tap_thread.start()

        def _show_accessibility_warning(self):
            message = (
                "Accessibility permission is required for the hotkey. "
                "Open System Settings > Privacy & Security > Accessibility to re-enable Brainwave IME. "
                "If it already looks enabled, run: tccutil reset Accessibility com.brainwave.ime"
            )
            print(f"[Access] {message}")
            if HAS_RUMPS:
                rumps.notification(
                    "Brainwave IME",
                    "Accessibility Required",
                    "Open System Settings > Privacy & Security > Accessibility."
                )
            try:
                subprocess.run(
                    ["open", "x-apple.systempreferences:com.apple.preference.security?Privacy_Accessibility"],
                    check=False
                )
            except Exception as exc:
                print(f"[Access] Failed to open Accessibility settings: {exc}")

        def _show_input_monitoring_warning(self):
            message = (
                "Input Monitoring permission may be required for the hotkey. "
                "Open System Settings > Privacy & Security > Input Monitoring to re-enable Brainwave IME."
            )
            print(f"[Access] {message}")
            if HAS_RUMPS:
                rumps.notification(
                    "Brainwave IME",
                    "Input Monitoring Required",
                    "Open System Settings > Privacy & Security > Input Monitoring."
                )
            try:
                subprocess.run(
                    ["open", "x-apple.systempreferences:com.apple.preference.security?Privacy_ListenEvent"],
                    check=False
                )
            except Exception as exc:
                print(f"[Access] Failed to open Input Monitoring settings: {exc}")

        def hotkeyDown_(self, sender):
            """快捷键按下 (从主线程调用)"""
            self.core.on_hotkey_down()

        def hotkeyUp_(self, sender):
            """快捷键释放 (从主线程调用)"""
            self.core.on_hotkey_up()

        def _noop(self, _=None):
            """用于菜单的空回调，保持菜单项可读"""
            pass

        def _current_state_icon(self, state: IMEState) -> str:
            return self.STATE_ICONS.get(state, "🎤")

        def _model_label(self) -> str:
            return self.model_labels.get(self.config.model, self.config.model)

        def _sync_status_menu(self, state: IMEState):
            icon = self._current_state_icon(state)
            self.status_item.title = f"Status: {icon} {state.value.capitalize()} | Model: {self._model_label()}"
            self.reconnect_item._menuitem.setEnabled_(state == IMEState.DISCONNECTED)
            for key, menu_state, mode, _label in self.STATUS_GUIDE:
                item = self.status_guide_items[key]
                base_label = self.status_guide_labels.get(key, _label)
                is_current = menu_state == state and (mode is None or self.core.recording_mode == mode)
                marker = "•" if is_current else " "
                item.title = f"{marker} {base_label}"
                item.state = 0

        def _sync_provider_menu(self):
            for key, item in self.provider_items.items():
                item.state = 1 if key == self.config.provider else 0

        def _sync_model_menu(self):
            for key, item in self.model_items.items():
                item.state = 1 if key == self.config.model else 0

        def _provider_selected(self, sender):
            provider_key = getattr(sender, "_provider_key", None)
            if provider_key:
                self.set_provider(provider_key)

        def _model_selected(self, sender):
            model_key = getattr(sender, "_model_key", None)
            if model_key:
                self.set_model(model_key)

        def set_provider(self, provider_key: str):
            if provider_key not in self.provider_labels:
                print(f"[UI] Unknown provider: {provider_key}")
                return
            if self.config.provider == provider_key:
                return
            self.config.provider = provider_key
            self._sync_provider_menu()
            self._sync_status_menu(self.core.state)

        def set_model(self, model_key: str):
            if model_key not in self.model_labels:
                print(f"[UI] Unknown model: {model_key}")
                return
            if self.config.model == model_key:
                return
            self.config.model = model_key
            self._sync_model_menu()
            self._sync_status_menu(self.core.state)

        def reconnect(self, _):
            if self.core.state != IMEState.DISCONNECTED:
                return
            self._connect()

        def _poll_state(self, timer):
            """定时器回调 - 在主线程检查并更新 UI"""
            if self._accessibility_warning_needed and not self._accessibility_warning_sent:
                self._accessibility_warning_sent = True
                self._show_accessibility_warning()
            if self._input_monitoring_warning_needed and not self._input_monitoring_warning_sent:
                self._input_monitoring_warning_sent = True
                self._show_input_monitoring_warning()
            self._drain_pending_transcripts()
            current_state = self.core.state
            if current_state != self._last_state:
                self._last_state = current_state
                icon = self._current_state_icon(current_state)
                print(f"[UI] State changed: {current_state.value}, updating icon to: {icon}")
                self.title = icon
                self._sync_status_menu(current_state)
                print(f"[UI] Title is now: {self.title}")

        def _on_state_change(self, state: IMEState):
            """状态变更回调 - 状态会被定时器轮询更新到 UI"""
            # 不在这里更新 UI，由 _poll_state 定时器处理
            pass

        def _on_transcript(self, text: str):
            pass

        def _on_transcript_complete(self, text: str, recording_mode, provider: str, model: str):
            if not text or not text.strip():
                return
            entry = {
                "ts": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "text": text,
                "mode": recording_mode.value if recording_mode else None,
                "provider": provider,
                "model": model
            }
            self._pending_transcripts.put(entry)

        def _resolve_history_dir(self) -> str:
            custom_dir = os.getenv("BRAINWAVE_HISTORY_DIR")
            if custom_dir:
                return os.path.expanduser(custom_dir)
            return os.path.join(
                os.path.expanduser("~"),
                "Library",
                "Application Support",
                "Brainwave IME"
            )

        def _load_recent_history(self):
            self._recent_transcripts = []
            try:
                os.makedirs(self._history_dir, exist_ok=True)
                if not os.path.exists(self._history_file):
                    return
                with open(self._history_file, "r", encoding="utf-8") as handle:
                    for line in handle:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            record = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        text = record.get("text", "")
                        if not text:
                            continue
                        self._recent_transcripts.append(record)
                if len(self._recent_transcripts) > self._recent_limit:
                    self._recent_transcripts = self._recent_transcripts[-self._recent_limit:]
                self._recent_transcripts.reverse()
            except Exception as exc:
                print(f"[History] Failed to load history: {exc}")

        def _append_history(self, entry: dict):
            try:
                os.makedirs(self._history_dir, exist_ok=True)
                with open(self._history_file, "a", encoding="utf-8") as handle:
                    handle.write(json.dumps(entry, ensure_ascii=False) + "\n")
            except Exception as exc:
                print(f"[History] Failed to write history: {exc}")

        def _drain_pending_transcripts(self):
            updated = False
            while True:
                try:
                    entry = self._pending_transcripts.get_nowait()
                except queue.Empty:
                    break
                self._append_history(entry)
                self._recent_transcripts.insert(0, entry)
                if len(self._recent_transcripts) > self._recent_limit:
                    self._recent_transcripts = self._recent_transcripts[:self._recent_limit]
                updated = True
            if updated:
                self._update_recent_panel()

        def _format_recent_title(self, entry: dict, index: int) -> str:
            raw_text = entry.get("text", "").strip()
            preview = " ".join(raw_text.splitlines()).strip()
            if len(preview) > 80:
                preview = preview[:77] + "..."
            timestamp = entry.get("ts", "")
            time_label = ""
            if timestamp:
                parts = timestamp.split(" ")
                if len(parts) > 1:
                    time_label = parts[1][:5]
                else:
                    time_label = timestamp[:5]
            prefix = f"{time_label} | " if time_label else ""
            return f"{prefix}{preview}" if preview else f"{prefix}(empty)"

        def _update_recent_panel(self):
            if not self._recent_panel_buttons:
                return
            for idx, button in enumerate(self._recent_panel_buttons):
                if idx < len(self._recent_transcripts):
                    entry = self._recent_transcripts[idx]
                    button.setTitle_(self._format_recent_title(entry, idx))
                    button.setEnabled_(True)
                else:
                    button.setTitle_("(empty)")
                    button.setEnabled_(False)

        def _open_history_folder(self, _):
            try:
                os.makedirs(self._history_dir, exist_ok=True)
                subprocess.run(["open", self._history_dir], check=False)
            except Exception as exc:
                print(f"[History] Failed to open history folder: {exc}")

        def _open_recent_panel(self, _):
            self._drain_pending_transcripts()
            if not HAS_APPKIT:
                rumps.alert(
                    "Recent",
                    "AppKit not available; cannot show recent panel.",
                    ok="OK"
                )
                return
            if self._recent_panel is None:
                self._build_recent_panel()
            self._update_recent_panel()
            try:
                self._recent_panel.makeKeyAndOrderFront_(None)
            except Exception as exc:
                print(f"[History] Failed to show recent panel: {exc}")

        def _build_recent_panel(self):
            width = 560
            row_height = 54
            padding = 12
            height = padding * 2 + row_height * self._recent_limit
            style = NSWindowStyleMaskTitled | NSWindowStyleMaskClosable | NSWindowStyleMaskUtilityWindow
            panel = NSPanel.alloc().initWithContentRect_styleMask_backing_defer_(
                ((0, 0), (width, height)),
                style,
                NSBackingStoreBuffered,
                False
            )
            panel.setTitle_("Recent")
            panel.setFloatingPanel_(True)
            panel.setHidesOnDeactivate_(False)
            panel.setReleasedWhenClosed_(False)
            try:
                panel.center()
            except Exception:
                pass

            content = panel.contentView()
            self._recent_panel_buttons = []
            for idx in range(self._recent_limit):
                y = height - padding - (idx + 1) * row_height
                button = NSButton.alloc().initWithFrame_(
                    ((padding, y), (width - padding * 2, row_height - 6))
                )
                button.setAlignment_(NSTextAlignmentLeft)
                button.setFont_(NSFont.systemFontOfSize_(13))
                button.setTarget_(self)
                button.setAction_("recentPanelButtonClicked:")
                button.setTag_(idx)
                button.setTitle_("(empty)")
                button.setEnabled_(False)
                content.addSubview_(button)
                self._recent_panel_buttons.append(button)
            self._recent_panel = panel

        def recentPanelButtonClicked_(self, sender):
            try:
                index = int(sender.tag())
            except Exception:
                return
            if index < 0 or index >= len(self._recent_transcripts):
                return
            text = self._recent_transcripts[index].get("text", "")
            if text:
                self._copy_to_clipboard(text)
            if self._recent_panel:
                self._recent_panel.orderOut_(None)

        def _copy_to_clipboard(self, text: str):
            try:
                proc = subprocess.Popen(['pbcopy'], stdin=subprocess.PIPE)
                proc.communicate(text.encode('utf-8'))
            except Exception as exc:
                print(f"[History] Clipboard copy failed: {exc}")

        def _terminate_server_processes(self):
            try:
                output = subprocess.check_output(
                    ["/bin/ps", "-x", "-o", "pid=,command="],
                    text=True
                )
            except Exception as exc:
                print(f"[App] Failed to list processes: {exc}")
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

                if pid == current_pid:
                    continue

                if "realtime_server.py" in cmd and PROJECT_DIR in cmd:
                    matches.append((pid, cmd))

            if not matches:
                return

            print(f"[App] Stopping {len(matches)} server process(es)...")
            for pid, cmd in matches:
                try:
                    print(f"[App] Terminating PID {pid}: {cmd}")
                    os.kill(pid, signal.SIGTERM)
                except ProcessLookupError:
                    continue
                except Exception as exc:
                    print(f"[App] Failed to terminate PID {pid}: {exc}")

            time.sleep(0.5)
            for pid, cmd in matches:
                try:
                    os.kill(pid, signal.SIGKILL)
                except ProcessLookupError:
                    continue
                except Exception as exc:
                    print(f"[App] Failed to kill PID {pid}: {exc}")

        def _connect(self):
            if self.loop is None:
                self.loop = asyncio.new_event_loop()
                self.loop_thread = threading.Thread(
                    target=self.loop.run_forever,
                    daemon=True
                )
                self.loop_thread.start()
                self.core.loop = self.loop

            asyncio.run_coroutine_threadsafe(
                self.core.connect_websocket(),
                self.loop
            )

        def quit_app(self, _):
            self._terminate_server_processes()
            if self.loop:
                try:
                    future = asyncio.run_coroutine_threadsafe(
                        self.core.disconnect_websocket(),
                        self.loop
                    )
                    future.result(timeout=2)
                except Exception as exc:
                    print(f"[App] Failed to disconnect websocket cleanly: {exc}")
                self.loop.call_soon_threadsafe(self.loop.stop)
            self.core.cleanup()
            rumps.quit_application()


def run_menubar():
    """运行菜单栏应用"""
    if not HAS_RUMPS:
        print("Error: rumps not installed")
        sys.exit(1)

    app = BrainwaveIMEApp()
    app.run()


def run_cli():
    """运行命令行版本"""
    config = Config()
    core = BrainwaveIMECore(
        config,
        on_state_change=lambda s: print(f"[State] {s.value}"),
        on_transcript=lambda t: print(f"[Transcript] {t[:50]}...")
    )

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    core.loop = loop

    print("Brainwave IME (CLI Mode)")
    print("Hotkey: Cmd+` (Optimized)")
    print("Press Ctrl+C to quit")

    try:
        loop.run_forever()
    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        core.cleanup()


if __name__ == "__main__":
    if "--cli" in sys.argv:
        run_cli()
    else:
        run_menubar()
