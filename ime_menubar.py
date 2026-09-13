#!/usr/bin/env python3
"""
EchoWave IME - macOS 菜单栏应用
在菜单栏显示状态图标，支持快捷键录音
使用 macOS 原生 API 监听快捷键，避免 pynput 的线程问题
"""

import asyncio
import concurrent.futures
import json
import os
import queue
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Optional
import numpy as np
import scipy.signal
import websockets
import pyaudio
from audio_spool import AudioSpoolError, TurnAudioSpool
from backend_runtime import (
    BACKEND_ECHO,
    BACKEND_WHISPER,
    BackendCoordinator,
    BackendDescriptor,
    BackendRuntimeError,
    BackendSelectionStore,
    BackendSupervisor,
    StartupDecision,
    build_default_descriptors,
)
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
    import objc
    from AppKit import (
        NSPanel,
        NSButton,
        NSColor,
        NSFont,
        NSPointInRect,
        NSPasteboard,
        NSSound,
        NSTextField,
        NSTrackingActiveAlways,
        NSTrackingArea,
        NSTrackingAssumeInside,
        NSTrackingInVisibleRect,
        NSTrackingMouseEnteredAndExited,
        NSEventTrackingRunLoopMode,
        NSView,
        NSViewHeightSizable,
        NSViewWidthSizable,
        NSWindowStyleMaskTitled,
        NSWindowStyleMaskClosable,
        NSWindowStyleMaskUtilityWindow,
        NSBackingStoreBuffered,
        NSTextAlignmentLeft
    )
    from Foundation import NSRunLoop
    HAS_APPKIT = True
except Exception:
    HAS_APPKIT = False
    objc = None
    NSPanel = None
    NSButton = None
    NSColor = None
    NSFont = None
    NSPointInRect = None
    NSPasteboard = None
    NSSound = None
    NSTextField = None
    NSTrackingActiveAlways = None
    NSTrackingArea = None
    NSTrackingAssumeInside = None
    NSTrackingInVisibleRect = None
    NSTrackingMouseEnteredAndExited = None
    NSEventTrackingRunLoopMode = None
    NSView = None
    NSViewHeightSizable = None
    NSViewWidthSizable = None
    NSWindowStyleMaskTitled = None
    NSWindowStyleMaskClosable = None
    NSWindowStyleMaskUtilityWindow = None
    NSBackingStoreBuffered = None
    NSTextAlignmentLeft = None
    NSRunLoop = None

PROJECT_DIR = os.path.abspath(os.path.dirname(__file__))
PASTEBOARD_TEXT_TYPE = "public.utf8-plain-text"
KEYCODE_V = 9
_DEFAULT_AUDIO_QUEUE_MAX_FRAMES = 1000
_BACKEND_SWITCH_VIEW_WIDTH = 240.0
_BACKEND_SWITCH_VIEW_HEIGHT = 24.0


if HAS_APPKIT:
    class BackendSwitchMenuView(NSView):
        """Sticky custom content for the backend menu row.

        NSMenu owns tracking while this view receives the click, so no standard
        NSMenuItem action fires and the popup remains open during the async
        backend transaction.
        """

        def configureWithOwner_(self, owner):
            self._owner = owner
            self._interaction_enabled = False
            self._dismiss_on_exit_armed = False
            self._dismissal_sent = False
            self._tracking_area = None

            label = NSTextField.alloc().initWithFrame_(
                (
                    (12.0, 2.0),
                    (
                        _BACKEND_SWITCH_VIEW_WIDTH - 24.0,
                        _BACKEND_SWITCH_VIEW_HEIGHT - 4.0,
                    ),
                )
            )
            label.setEditable_(False)
            label.setSelectable_(False)
            label.setBezeled_(False)
            label.setBordered_(False)
            label.setDrawsBackground_(False)
            label.setFont_(NSFont.menuFontOfSize_(13.0))
            label.setAutoresizingMask_(NSViewWidthSizable | NSViewHeightSizable)
            self.addSubview_(label)
            self._label = label
            self.updateTrackingAreas()

        def setBackendLabel_enabled_(self, label, enabled):
            label = str(label)
            enabled = bool(enabled)
            self._label.setStringValue_(label)
            self._label.setTextColor_(
                NSColor.controlTextColor()
                if enabled
                else NSColor.disabledControlTextColor()
            )
            self.setAccessibilityLabel_(label)
            self.setAccessibilityEnabled_(enabled)
            self._interaction_enabled = enabled
            self.setNeedsDisplay_(True)

        def setDismissOnExitArmed_(self, armed):
            self._dismiss_on_exit_armed = bool(armed)
            self._dismissal_sent = False

        def hitTest_(self, point):
            # Keep the child text field from consuming the click. The whole row
            # is one control and mouseUp_ below is its only action path.
            return self if NSPointInRect(point, self.bounds()) else None

        def mouseDown_(self, _event):
            # Consuming mouse-down prevents the event from becoming a standard
            # menu command, which would end NSMenu tracking immediately.
            pass

        def mouseUp_(self, _event):
            if not self._interaction_enabled:
                return
            owner = getattr(self, "_owner", None)
            callback = getattr(owner, "_backend_switch_selected", None)
            if callback is not None:
                callback(None)

        def updateTrackingAreas(self):
            tracking_area = getattr(self, "_tracking_area", None)
            if tracking_area is not None:
                self.removeTrackingArea_(tracking_area)
            objc.super(BackendSwitchMenuView, self).updateTrackingAreas()
            options = (
                NSTrackingMouseEnteredAndExited
                | NSTrackingActiveAlways
                | NSTrackingInVisibleRect
                | NSTrackingAssumeInside
            )
            tracking_area = NSTrackingArea.alloc().initWithRect_options_owner_userInfo_(
                self.bounds(),
                options,
                self,
                None,
            )
            self.addTrackingArea_(tracking_area)
            self._tracking_area = tracking_area

        def mouseExited_(self, _event):
            if not self._dismiss_on_exit_armed or self._dismissal_sent:
                return
            self._dismissal_sent = True
            self._dismiss_on_exit_armed = False
            owner = getattr(self, "_owner", None)
            cancel_tracking = getattr(owner, "_cancel_backend_menu_tracking", None)
            if cancel_tracking is not None:
                cancel_tracking()

        def viewDidMoveToWindow(self):
            objc.super(BackendSwitchMenuView, self).viewDidMoveToWindow()
            if self.window() is None:
                self.setDismissOnExitArmed_(False)
else:
    BackendSwitchMenuView = None


def _register_timer_for_menu_tracking(timer, *, run_loop=None):
    """Register rumps' existing native timer in menu-tracking run-loop mode."""
    if not HAS_APPKIT:
        return None
    native_timer = getattr(timer, "_nstimer", None)
    if native_timer is None:
        raise RuntimeError(
            "rumps Timer bridge changed: started timer has no _nstimer"
        )
    target_run_loop = run_loop or NSRunLoop.currentRunLoop()
    add_timer = getattr(target_run_loop, "addTimer_forMode_", None)
    if add_timer is None:
        raise RuntimeError(
            "AppKit run-loop bridge changed: addTimer_forMode_ is unavailable"
        )
    add_timer(native_timer, NSEventTrackingRunLoopMode)
    return native_timer


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
    backend_key: str = BACKEND_ECHO
    provider: str = "openai"
    model: str = os.getenv("OPENAI_REALTIME_MODEL", "gpt-realtime-2.1-mini")
    start_sound: str = os.getenv("BRAINWAVE_START_SOUND", "Tink")
    stop_sound: str = os.getenv("BRAINWAVE_STOP_SOUND", "Morse")
    complete_sound: str = os.getenv("BRAINWAVE_COMPLETE_SOUND", "Bottle")
    error_sound: str = os.getenv("BRAINWAVE_ERROR_SOUND", "Basso")
    stop_tail_wait_min_ms: int = int(os.getenv("BRAINWAVE_STOP_TAIL_WAIT_MIN_MS", "150"))
    stop_tail_wait_max_ms: int = int(os.getenv("BRAINWAVE_STOP_TAIL_WAIT_MAX_MS", "150"))
    stop_tail_wait_guard_ms: int = int(os.getenv("BRAINWAVE_STOP_TAIL_WAIT_GUARD_MS", "20"))
    processing_timeout_sec: int = int(os.getenv("BRAINWAVE_PROCESSING_TIMEOUT_SEC", "30"))
    processing_hard_timeout_sec: int = int(
        os.getenv("BRAINWAVE_PROCESSING_HARD_TIMEOUT_SEC", "120")
    )
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
        on_transcript_complete=None,
        on_processing_delay=None,
    ):
        self.config = config
        self.state = IMEState.DISCONNECTED
        self.recording_mode = None
        self.audio_processor = AudioProcessor(
            config.sample_rate, config.target_sample_rate
        )
        self._pyaudio_executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="pyaudio",
        )
        self._pyaudio_executor_shutdown = False
        try:
            self.pyaudio_instance = self._pyaudio_executor.submit(
                pyaudio.PyAudio
            ).result()
        except Exception:
            self._pyaudio_executor.shutdown(wait=True, cancel_futures=True)
            self._pyaudio_executor_shutdown = True
            raise
        self._audio_rebuild_in_progress = False
        self.audio_stream = None
        self.audio_buffer = []
        self._audio_buffer_samples = 0
        self._audio_queue: Optional[queue.Queue] = None
        self._audio_consumer_task: Optional[asyncio.Task] = None
        self._audio_drained: asyncio.Event = asyncio.Event()
        self._audio_consumer_paused: bool = True
        self._audio_drop_count: int = 0
        self._audio_backpressure_failure_count: int = 0
        self._audio_ingestion_failed_reason: Optional[str] = None
        self._audio_failure_lock = threading.RLock()
        self._audio_abort_task: Optional[asyncio.Task] = None
        self._capture_queue_max_frames = max(
            1,
            int(
                os.getenv(
                    "BRAINWAVE_AUDIO_CAPTURE_QUEUE_MAX_FRAMES",
                    str(_DEFAULT_AUDIO_QUEUE_MAX_FRAMES),
                )
            ),
        )
        self.ws = None
        self.ws_connected = False
        self._connection_generation = 0
        self.hotkey_active = False
        self.loop = None
        self.transcript = ""
        self.on_state_change = on_state_change
        self.on_transcript = on_transcript
        self.on_transcript_complete = on_transcript_complete
        self.on_processing_delay = on_processing_delay
        self._receive_task = None
        # Two-state session lifecycle (task 0436 S2):
        #   _start_requested  — start_recording message has been sent
        #   _server_connected — matching-turn `status=connected` ack received
        # Audio is only uploaded (and removed from the local upload buffer) once
        # _server_connected is True, so a confirmation failure never drops the
        # already-recorded PCM — it stays buffered and is retransmitted after the
        # rebuild finally connects.
        self._start_requested = False
        # `_server_connected` is a property backed by this Event (task 0436
        # R3-M1) so the stop path can await the matching-turn `connected` ack as
        # a single source of truth instead of a separate bool that can drift.
        self._server_connected_event: asyncio.Event = asyncio.Event()
        self._server_connected = False
        # Bounded backoff for the confirmation-failure retry loop so a failing
        # session start during recording cannot become a retry storm (S2).
        self._session_start_attempts = 0
        self._max_session_start_attempts = max(
            1, int(os.getenv("BRAINWAVE_SESSION_START_MAX_ATTEMPTS", "5"))
        )
        self._session_start_backoff_base_sec = float(
            os.getenv("BRAINWAVE_SESSION_START_BACKOFF_BASE_SEC", "0.5")
        )
        self._session_start_backoff_cap_sec = float(
            os.getenv("BRAINWAVE_SESSION_START_BACKOFF_CAP_SEC", "5.0")
        )
        # Confirmation budget the stop path waits for a slow-but-legitimate
        # session start to reach the matching-turn `connected` ack before
        # declaring upload failure (task 0436 R3-M1). Aligned to the server
        # confirmation contract (5s + one 5s resend) plus margin. The old 1.5s
        # hard wait cancelled a ~3s start that would otherwise have succeeded.
        self._stop_session_connect_deadline_sec = max(
            0.0,
            float(os.getenv("BRAINWAVE_STOP_SESSION_CONNECT_DEADLINE_SEC", "11.0")),
        )
        self._session_task = None
        self._session_prompt_mode = "optimize"
        self._force_ws_refresh_before_turn = False
        self._last_turn_completed_ts = None
        self._last_turn_completed_wall_ts = None
        self._ws_connected_wall_ts = None
        self._turn_id = 0
        self._active_turn_id = None
        self._recording_generation = 0
        self._active_recording_token = None
        self._hotkey_down_ts = None
        self._recording_started_ts = None
        self._stop_pressed_ts = None
        self._response_done_ts = None
        self._processing_entered_ts = None
        self._processing_warning_sent = False
        self._last_audio_callback_ts = None
        self._current_stream_turn_id = None
        self._active_turn_backend = self.config.backend_key
        self._active_turn_provider = self.config.provider
        self._active_turn_model = self.config.model
        self._terminal_turn_ids: set[int] = set()
        self._failed_turn_ids: set[int] = set()
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
        self._max_upload_backlog_bytes = max(
            self._upload_chunk_bytes * 2,
            int(
                os.getenv(
                    "BRAINWAVE_MAX_UPLOAD_BACKLOG_BYTES",
                    str(48_000 * 120),
                )
            ),
        )
        self._audio_spool_dir = os.path.expanduser(
            os.getenv(
                "BRAINWAVE_AUDIO_SPOOL_DIR",
                "~/Library/Application Support/EchoWave/audio_spool",
            )
        )
        self._turn_audio_spool: Optional[TurnAudioSpool] = None
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

    @property
    def _server_connected(self) -> bool:
        """True once the matching-turn ``connected`` ack has arrived (task 0436
        R3-M1 SSOT). Backed by ``_server_connected_event`` so the stop path can
        await it without a separate bool drifting out of sync.
        """
        return self._server_connected_event.is_set()

    @_server_connected.setter
    def _server_connected(self, value: bool) -> None:
        if value:
            self._server_connected_event.set()
        else:
            self._server_connected_event.clear()

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
        self._start_requested = False
        self._server_connected = False
        self._ws_connected_wall_ts = None

    def _close_audio_stream_sync(self):
        stream = self.audio_stream
        self.audio_stream = None
        self._close_audio_stream_blocking(stream)

    def _close_audio_stream_blocking(self, stream):
        if not stream:
            return
        try:
            stream.stop_stream()
        except Exception:
            pass
        try:
            stream.close()
        except Exception:
            pass

    async def _close_audio_stream(self):
        stream = self.audio_stream
        self.audio_stream = None
        if not stream:
            return
        loop = self.loop or asyncio.get_running_loop()
        await loop.run_in_executor(
            self._pyaudio_executor,
            self._close_audio_stream_blocking,
            stream,
        )

    def _replace_pyaudio_instance_blocking(self) -> None:
        """Replace PortAudio on its single owner thread with no live stream."""
        self._close_audio_stream_sync()
        old_instance = self.pyaudio_instance
        self.pyaudio_instance = None
        if old_instance is not None:
            old_instance.terminate()
        self.pyaudio_instance = pyaudio.PyAudio()

    def _terminate_pyaudio_instance_blocking(self) -> None:
        """Idempotently close the stream and terminate the current instance."""
        self._close_audio_stream_sync()
        instance = self.pyaudio_instance
        self.pyaudio_instance = None
        if instance is not None:
            instance.terminate()

    def _open_audio_stream_with_recovery_blocking(
        self,
        recording_token: tuple[int, int],
        audio_callback,
    ):
        """Open once, rebuild PortAudio after failure, then retry once."""
        for attempt in (1, 2):
            if not self._recording_token_is_current(recording_token):
                return None
            try:
                if self.pyaudio_instance is None:
                    self.pyaudio_instance = pyaudio.PyAudio()
                stream = self.pyaudio_instance.open(
                    format=pyaudio.paInt16,
                    channels=self.config.channels,
                    rate=self.config.sample_rate,
                    input=True,
                    frames_per_buffer=self._chunk_size_frames,
                    stream_callback=audio_callback,
                    start=False,
                )
            except Exception as exc:
                if attempt == 2:
                    print(
                        "[IME] Audio open failed after PortAudio rebuild "
                        f"(attempt 2/2): {exc}"
                    )
                    raise
                print(
                    "[IME] Audio open failed (attempt 1/2), rebuilding "
                    f"PortAudio device table: {exc}"
                )
                if not self._recording_token_is_current(recording_token):
                    return None
                self._close_audio_stream_sync()
                if not self._recording_token_is_current(recording_token):
                    return None
                try:
                    self._replace_pyaudio_instance_blocking()
                except Exception as rebuild_exc:
                    print(
                        "[IME] PortAudio rebuild failed before audio open "
                        f"(attempt 2/2): {rebuild_exc}"
                    )
                    raise
                if not self._recording_token_is_current(recording_token):
                    return None
                continue

            if not self._recording_token_is_current(recording_token):
                self._close_audio_stream_blocking(stream)
                return None
            if attempt == 2:
                print(
                    "[IME] PortAudio rebuilt, audio stream opened on attempt 2/2"
                )
            return stream
        return None

    async def rebuild_audio_device_table(self) -> bool:
        """Attempt a Restart Service refresh while capture is inactive.

        ``False`` means the state guard rejected the attempt. Once an allowed
        attempt starts, PortAudio failure is logged and treated as best-effort
        so the websocket/backend restart can still recover independently.
        """
        if (
            self.state not in (IMEState.IDLE, IMEState.DISCONNECTED)
            or self._audio_rebuild_in_progress
        ):
            return False
        self._audio_rebuild_in_progress = True
        try:
            loop = self.loop or asyncio.get_running_loop()
            await loop.run_in_executor(
                self._pyaudio_executor,
                self._replace_pyaudio_instance_blocking,
            )
            print("[IME] Restart Service PortAudio rebuild succeeded (attempt 1/1)")
            return True
        except Exception as exc:
            print(
                "[IME] Restart Service PortAudio rebuild failed "
                f"(attempt 1/1): {exc}"
            )
            return True
        finally:
            self._audio_rebuild_in_progress = False

    def _allocate_recording_token(self) -> tuple[int, int]:
        self._recording_generation += 1
        token = (self._recording_generation, int(self._active_turn_id))
        self._active_recording_token = token
        return token

    def _invalidate_recording_token(self) -> None:
        self._active_recording_token = None

    def _recording_token_is_current(self, token: tuple[int, int]) -> bool:
        return (
            self._active_recording_token == token
            and self._active_turn_id == token[1]
            and self.state == IMEState.RECORDING
        )

    def _finish_abandoned_audio_start(self) -> None:
        """Close a token-only cancellation without clobbering a real stop/new turn."""
        if self.state != IMEState.RECORDING or self._active_recording_token is not None:
            return
        self._audio_consumer_paused = True
        self._audio_drained.set()
        self._cleanup_turn_spool()
        self._set_state(IMEState.IDLE)

    def _ensure_audio_pipeline(self):
        if self._audio_queue is None:
            # PyAudio owns the producer thread. A real thread-safe queue avoids
            # scheduling one event-loop callback per frame. The queue is bounded
            # for memory safety; saturation aborts the turn explicitly and
            # preserves the crossing frame instead of silently dropping it.
            self._audio_queue = queue.Queue(maxsize=self._capture_queue_max_frames)
        if self._audio_consumer_task is None or self._audio_consumer_task.done():
            self._audio_consumer_task = asyncio.create_task(
                self._audio_consumer_loop()
            )

    def _begin_turn_spool(self) -> None:
        self._cleanup_turn_spool()
        self._turn_audio_spool = TurnAudioSpool(
            self._audio_spool_dir,
            turn_id=self._active_turn_id,
            sample_rate=self.config.target_sample_rate,
            channels=self.config.channels,
        )

    def _cleanup_turn_spool(self) -> None:
        spool = self._turn_audio_spool
        self._turn_audio_spool = None
        if spool is not None:
            spool.cleanup()

    def _archive_failed_turn_audio(self, reason: str) -> Optional[str]:
        spool = self._turn_audio_spool
        if spool is None or spool.byte_count == 0:
            return None
        try:
            archive_dir = os.path.join(
                os.path.expanduser("~"),
                "Library",
                "Application Support",
                "EchoWave",
                "failed_audio"
            )
            return str(spool.archive_wav(archive_dir, outcome=reason))
        except Exception as exc:
            print(f"[IME] Failed to archive local audio: {exc}")
            return None

    def _archive_recent_turn_audio(self, outcome: str) -> bool:
        if not self._recent_audio_cache_enabled:
            return False
        spool = self._turn_audio_spool
        if spool is None or spool.byte_count == 0:
            return False
        if (
            self._active_turn_id is not None
            and self._recent_audio_enqueued_turn_id == self._active_turn_id
        ):
            return False
        try:
            path = spool.archive_wav(self._recent_audio_cache_dir, outcome=outcome)
            print(f"[IME] Recent audio cached: {path}")
            self._prune_recent_audio_cache()
        except AudioSpoolError as exc:
            print(f"[IME] Failed to archive recent turn audio: {exc}")
            return False
        if self._active_turn_id is not None:
            self._recent_audio_enqueued_turn_id = self._active_turn_id
        return True

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
        if self._session_task and not self._session_task.done():
            return
        self._session_task = asyncio.create_task(
            self._ensure_session_started(self._session_prompt_mode)
        )

    def _clear_audio_buffer(self):
        self.audio_buffer.clear()
        self._audio_buffer_samples = 0

    def _append_audio_buffer(self, chunk: bytes):
        self.audio_buffer.append(chunk)
        self._audio_buffer_samples += len(chunk) // 2
        if self._audio_buffer_samples * 2 > self._max_upload_backlog_bytes:
            raise AudioSpoolError(
                "upload backlog exceeded explicit memory budget; full turn remains on disk"
            )

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
            if new_state == IMEState.PROCESSING:
                self._processing_entered_ts = time.perf_counter()
                self._processing_warning_sent = False
            else:
                self._processing_entered_ts = None
            if (
                new_state == IMEState.DISCONNECTED
                and old_state in (IMEState.RECORDING, IMEState.PROCESSING)
            ):
                self._play_sound(getattr(getattr(self, "config", None), "error_sound", Config.error_sound))
            if new_state in (IMEState.IDLE, IMEState.DISCONNECTED):
                self.recording_mode = None
            print(f"[IME] State: {old_state.value} -> {new_state.value}")
            if self.on_state_change:
                self.on_state_change(new_state)

    def apply_backend_descriptor(self, descriptor: BackendDescriptor) -> None:
        """Point the next turn at one probed backend without sharing internals."""
        if self.state not in (IMEState.IDLE, IMEState.DISCONNECTED):
            raise RuntimeError(f"cannot repoint backend while state={self.state.value}")
        self.config.backend_key = descriptor.key
        self.config.server_host = descriptor.host
        self.config.server_port = descriptor.port
        self.config.provider = descriptor.provider
        self.config.model = descriptor.model

    async def repoint_backend(self, descriptor: BackendDescriptor) -> bool:
        if self.state not in (IMEState.IDLE, IMEState.DISCONNECTED):
            return False
        previous = (
            self.config.backend_key,
            self.config.server_host,
            self.config.server_port,
            self.config.provider,
            self.config.model,
        )
        await self.disconnect_websocket()
        self.apply_backend_descriptor(descriptor)
        target_error = None
        try:
            connected = bool(await self.connect_websocket())
        except Exception as exc:
            connected = False
            target_error = exc
        if connected:
            return True

        (
            self.config.backend_key,
            self.config.server_host,
            self.config.server_port,
            self.config.provider,
            self.config.model,
        ) = previous
        try:
            previous_connected = bool(await self.connect_websocket())
        except Exception as exc:
            raise BackendRuntimeError(
                f"target backend connect failed and previous endpoint restore raised: {exc}"
            ) from exc
        if not previous_connected:
            raise BackendRuntimeError(
                "target backend connect failed and previous endpoint could not be verified"
            ) from target_error
        if target_error is not None:
            print(
                "[Backend] Target endpoint connect raised; previous endpoint "
                f"was restored: {target_error}"
            )
        return False

    def _apply_provider(self, key: str):
        if self.config.provider == key:
            return
        self.config.provider = key
        print(f"[IME] Provider set to: {key}")

    def _apply_model(self, key: str):
        if self.config.model == key:
            return
        self.config.model = key
        print(f"[IME] Model set to: {key}")

    def _play_sound(self, sound_name: str):
        if not sound_name:
            return
        sound_name = str(sound_name).strip()
        if not sound_name or sound_name.lower() == "none":
            return

        if NSSound is not None:
            try:
                # setdefault also keeps lazy initialization safe for concurrent callers.
                lock = self.__dict__.setdefault("_sound_lock", threading.Lock())
                with lock:
                    if getattr(self, "_sound_objects", None) is None:
                        self._sound_objects: dict = {}
                    sound = self._sound_objects.get(sound_name)
                    if sound is None:
                        path = f"/System/Library/Sounds/{sound_name}.aiff"
                        try:
                            sound = (
                                NSSound.alloc().initWithContentsOfFile_byReference_(path, False)
                                if os.path.isfile(path) else None
                            )
                        except Exception:
                            sound = None
                        if sound is None:
                            print(f"[IME] Sound unavailable ({sound_name}); skipping")
                            return
                        self._sound_objects[sound_name] = sound
                    sound.stop()
                    if not sound.play():
                        raise RuntimeError("NSSound.play() returned False")
            except Exception as exc:
                print(f"[IME] Sound play error ({sound_name}): {exc}")
            return

        print(f"[IME] NSSound unavailable; using afplay ({sound_name})")
        try:
            running_loop = asyncio.get_running_loop()
        except RuntimeError:
            running_loop = None

        loop = running_loop or self.loop
        if loop is None or not loop.is_running():
            print(f"[IME] Sound skipped ({sound_name}): asyncio loop is not running")
            return

        def schedule():
            asyncio.ensure_future(self._play_sound_async(sound_name))

        try:
            if running_loop is loop:
                schedule()
            else:
                loop.call_soon_threadsafe(schedule)
        except Exception as exc:
            print(f"[IME] Sound play error ({sound_name}): {exc}")

    def warm_up_sounds(self):
        if NSSound is None:
            return
        try:
            sound_name = str(self.config.start_sound or "").strip()
            if not sound_name or sound_name.lower() == "none":
                return
            path = f"/System/Library/Sounds/{sound_name}.aiff"
            if not os.path.isfile(path):
                return
            sound = NSSound.alloc().initWithContentsOfFile_byReference_(path, False)
            if sound is None:
                return
            try:
                sound.setVolume_(0.0)
                sound.play()
                time.sleep(0.2)
            finally:
                try:
                    sound.stop()
                finally:
                    sound.setVolume_(1.0)
        except Exception:
            pass

    async def _play_sound_async(self, sound_name: str):
        try:
            proc = await asyncio.create_subprocess_exec(
                'afplay',
                f'/System/Library/Sounds/{sound_name}.aiff',
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            asyncio.create_task(self._reap_sound_process(proc, sound_name))
        except Exception as exc:
            print(f"[IME] Sound play error ({sound_name}): {exc}")

    async def _reap_sound_process(self, proc, sound_name: str):
        try:
            await proc.wait()
        except Exception as exc:
            print(f"[IME] Sound reap error ({sound_name}): {exc}")

    async def connect_websocket(self):
        self._ensure_audio_pipeline()
        uri = f"ws://{self.config.server_host}:{self.config.server_port}/api/v1/ws"
        try:
            websocket = await websockets.connect(uri)
            self._connection_generation += 1
            generation = self._connection_generation
            self.ws = websocket
            self.ws_connected = True
            self._ws_connected_wall_ts = time.time()
            if self.state not in (IMEState.RECORDING, IMEState.PROCESSING):
                self._set_state(IMEState.IDLE)
            print(f"[IME] Connected to {uri}")
            if self._receive_task is None or self._receive_task.done():
                self._receive_task = asyncio.create_task(
                    self.receive_messages(websocket, generation)
                )
            return True
        except Exception as e:
            print(f"[IME] Connection failed: {e}")
            self.ws_connected = False
            self._ws_connected_wall_ts = None
            if self.state not in (IMEState.RECORDING, IMEState.PROCESSING):
                self._set_state(IMEState.DISCONNECTED)
            return False

    async def disconnect_websocket(self):
        self._connection_generation += 1
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
        self._ws_connected_wall_ts = None
        self._set_state(IMEState.DISCONNECTED)

    async def _fail_active_turn_transport_loss(self, reason: str) -> None:
        """Fail closed when a local socket disappears before a terminal event.

        A new WebSocket owns a new server session and cannot prove that the old
        session received every uploaded byte or its stop frame. The disk spool
        is therefore archived once and the turn is forbidden from pasting.
        """

        turn_id = self._active_turn_id
        if turn_id is None:
            self._set_state(IMEState.DISCONNECTED)
            return
        if turn_id in self._failed_turn_ids:
            self._set_state(IMEState.DISCONNECTED)
            return
        try:
            await self._stop_and_drain_audio_capture()
        except Exception as exc:
            print(f"[IME] Audio drain during transport failure raised: {exc}")
        self._failed_turn_ids.add(turn_id)
        archive_path = self._archive_failed_turn_audio(reason)
        if archive_path:
            print(f"[IME] Local audio archived: {archive_path}")
        self._archive_recent_turn_audio(reason)
        self._start_requested = False
        self._server_connected = False
        self._play_sound(getattr(getattr(self, "config", None), "error_sound", Config.error_sound))
        self._set_state(IMEState.DISCONNECTED)

    async def receive_messages(self, websocket=None, generation: Optional[int] = None):
        websocket = websocket or self.ws
        if generation is None:
            generation = self._connection_generation
        try:
            async for message in websocket:
                if generation != self._connection_generation:
                    return
                if isinstance(message, str):
                    data = json.loads(message)
                    await self._handle_message(data, connection_generation=generation)
        except websockets.exceptions.ConnectionClosed:
            if generation != self._connection_generation:
                return
            print("[IME] Connection closed")
            self.ws_connected = False
            self.ws = None
            self._ws_connected_wall_ts = None
            self._start_requested = False
            self._server_connected = False
            self._receive_task = None
            if self.state == IMEState.RECORDING:
                print(
                    "[IME] Connection lost while recording; preserving local "
                    "capture and rebuilding the same turn session."
                )
                self._start_session_task()
            elif self.state == IMEState.PROCESSING:
                print("[IME] Connection lost before terminal; failing turn with full local archive.")
                await self._fail_active_turn_transport_loss(
                    "transport_closed_before_terminal"
                )
            else:
                self._set_state(IMEState.DISCONNECTED)
        except asyncio.CancelledError:
            pass
        except Exception as e:
            if generation != self._connection_generation:
                return
            print(f"[IME] Receive error: {e}")
            self.ws_connected = False
            self.ws = None
            self._ws_connected_wall_ts = None
            self._start_requested = False
            self._server_connected = False
            self._receive_task = None
            if self.state == IMEState.RECORDING:
                print(
                    "[IME] Receive error while recording; preserving local "
                    "capture and rebuilding the same turn session."
                )
                self._start_session_task()
            elif self.state == IMEState.PROCESSING:
                await self._fail_active_turn_transport_loss(
                    "transport_error_before_terminal"
                )
            else:
                self._set_state(IMEState.DISCONNECTED)
        finally:
            if generation == self._connection_generation:
                self._receive_task = None

    async def _handle_message(
        self,
        data: dict,
        *,
        connection_generation: Optional[int] = None,
    ):
        if (
            connection_generation is not None
            and connection_generation != self._connection_generation
        ):
            print(
                f"[IME] Ignoring stale connection generation "
                f"{connection_generation} (active={self._connection_generation})"
            )
            return
        msg_type = data.get("type")
        print(f"[IME] Received message: {msg_type}")
        if not self._is_message_for_active_turn(msg_type, data):
            return

        if msg_type == "status":
            status = data.get("status")
            if status == "connected":
                # Matching-turn ack: the server has an active provider session,
                # so buffered audio is now safe to upload (task 0436 S2).
                self._server_connected = True
                self._session_start_attempts = 0
            if status == "idle" and self.state == IMEState.PROCESSING:
                turn_id = self._active_turn_id
                if turn_id is not None and turn_id in self._terminal_turn_ids:
                    print(f"[IME] Ignoring duplicate terminal for turn {turn_id}")
                    return
                if (
                    self._audio_drop_count != 0
                    or self._audio_ingestion_failed_reason is not None
                    or (turn_id is not None and turn_id in self._failed_turn_ids)
                ):
                    if turn_id is not None:
                        self._failed_turn_ids.add(turn_id)
                    archive_path = self._archive_failed_turn_audio("incomplete_audio")
                    if archive_path:
                        print(f"[IME] Local audio archived: {archive_path}")
                    self._archive_recent_turn_audio("incomplete_audio")
                    self._play_sound(getattr(getattr(self, "config", None), "error_sound", Config.error_sound))
                    self._set_state(IMEState.IDLE)
                    return
                if turn_id is not None:
                    self._terminal_turn_ids.add(turn_id)
                self._last_turn_completed_ts = time.perf_counter()
                self._last_turn_completed_wall_ts = time.time()
                self._response_done_ts = time.perf_counter()
                if self._stop_pressed_ts is not None:
                    stop_to_response_done_ms = (self._response_done_ts - self._stop_pressed_ts) * 1000
                    print(f"[Perf][T{self._active_turn_id}] stop_to_response_done_ms={stop_to_response_done_ms:.1f}")
                last_mode = self.recording_mode
                self._archive_recent_turn_audio("completed")
                self._start_requested = False
                self._server_connected = False
                self._set_state(IMEState.IDLE)
                if self.transcript:
                    # 先播放声音，立即给用户反馈
                    self._play_sound(getattr(getattr(self, "config", None), "complete_sound", Config.complete_sound))
                    if self.on_transcript_complete:
                        try:
                            self.on_transcript_complete(
                                self.transcript,
                                last_mode,
                                self._active_turn_provider,
                                self._active_turn_model,
                                self._active_turn_backend,
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
                # Keep recording locally; retry session in background. Only alert
                # (Basso) if we had a working connected session; a warmup/
                # confirmation failure before the first connect retries silently.
                had_connected_session = self._server_connected
                self._start_requested = False
                self._server_connected = False
                if had_connected_session:
                    self._play_sound(getattr(getattr(self, "config", None), "error_sound", Config.error_sound))
                else:
                    print("[IME] Session warmup failed before first successful start; retrying silently.")
                self._start_session_task()
            elif self.state == IMEState.PROCESSING:
                if self._active_turn_id is not None:
                    self._failed_turn_ids.add(self._active_turn_id)
                self._play_sound(getattr(getattr(self, "config", None), "error_sound", Config.error_sound))
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
        active_turn_owns_transport = (
            self._active_turn_id is not None
            and self.state in (IMEState.RECORDING, IMEState.PROCESSING)
        )
        if msg_turn_id is None and active_turn_owns_transport:
            print(
                f"[IME] Ignoring unscoped {msg_type} while turn "
                f"{self._active_turn_id} owns the transport"
            )
            return False
        if msg_turn_id is None or self._active_turn_id is None:
            return True
        if msg_turn_id in self._failed_turn_ids:
            print(f"[IME] Ignoring {msg_type} for failed turn {msg_turn_id}")
            return False
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

    async def _handle_hotkey(self):
        """快捷键按下 - 在 asyncio loop 内切换录音状态"""
        if self.state == IMEState.RECORDING:
            await self._stop_recording()
        elif self.state == IMEState.IDLE:
            self._hotkey_down_ts = time.perf_counter()
            await self._start_recording()
        # 如果是 PROCESSING 或 DISCONNECTED 状态则忽略

    def _enqueue_audio(self, data, ts, turn_id) -> bool:
        if self._audio_queue is None:
            self._mark_audio_ingestion_failure(
                "capture_queue_unavailable",
                crossing_item=(data, ts, turn_id),
            )
            return False
        with self._audio_failure_lock:
            if self._audio_ingestion_failed_reason is not None:
                # The callback has already been told to abort. Any impossible
                # extra callback is counted as a real drop and the turn remains
                # failed; it can never paste a partial transcript.
                self._audio_drop_count += 1
                return False
        try:
            self._audio_queue.put_nowait((data, ts, turn_id))
            return True
        except queue.Full:
            self._audio_backpressure_failure_count += 1
            self._mark_audio_ingestion_failure(
                "capture_backpressure_exhausted",
                crossing_item=(data, ts, turn_id),
            )
            return False

    def _mark_audio_ingestion_failure(self, reason: str, crossing_item=None) -> None:
        with self._audio_failure_lock:
            if self._audio_ingestion_failed_reason is not None:
                return
            self._audio_ingestion_failed_reason = reason
        print(f"[IME] Audio ingestion failed explicitly: {reason}")
        loop = self.loop
        if loop is None or not loop.is_running():
            return

        def schedule_abort():
            if self._audio_abort_task is None or self._audio_abort_task.done():
                self._audio_abort_task = asyncio.create_task(
                    self._abort_audio_ingestion(crossing_item, reason)
                )

        try:
            loop.call_soon_threadsafe(schedule_abort)
        except RuntimeError:
            pass

    async def _abort_audio_ingestion(self, crossing_item, reason: str) -> None:
        """Drain every accepted frame, archive the spool, and forbid paste."""
        self._invalidate_recording_token()
        await self._close_audio_stream()
        if self._audio_queue is not None:
            if crossing_item is not None:
                # Queue saturation is terminal, but the first crossing frame is
                # still conserved in order once the consumer frees one slot.
                await asyncio.to_thread(self._audio_queue.put, crossing_item)
            self._audio_drained.clear()
            await asyncio.to_thread(self._audio_queue.put, None)
            await self._audio_drained.wait()
        self._audio_consumer_paused = True
        if self._active_turn_id is not None:
            self._failed_turn_ids.add(self._active_turn_id)
        archive_path = self._archive_failed_turn_audio(reason)
        if archive_path:
            print(f"[IME] Local audio archived: {archive_path}")
        self._archive_recent_turn_audio(reason)
        self._play_sound(getattr(getattr(self, "config", None), "error_sound", Config.error_sound))
        await self.disconnect_websocket()

    async def _audio_consumer_loop(self):
        while True:
            try:
                item = await asyncio.to_thread(self._audio_queue.get, True, 0.2)
            except queue.Empty:
                continue
            try:
                if item is None:
                    self._audio_drained.set()
                    continue

                chunk, ts, turn_id = item

                if self._audio_consumer_paused or turn_id != self._active_turn_id:
                    continue

                self._last_audio_callback_ts = ts
                resampled = self.audio_processor.resample(chunk)
                if self._turn_audio_spool is None:
                    raise AudioSpoolError("turn spool is unavailable")
                self._turn_audio_spool.append(resampled)
                if self._audio_ingestion_failed_reason is not None:
                    # Once the turn has failed closed, drain every already
                    # accepted frame into the disk SSOT without growing the
                    # in-memory upload backlog any further.
                    continue
                self._append_audio_buffer(resampled)

                while (
                    self._audio_buffer_samples >= self._upload_chunk_samples
                    and self.ws
                    and self.ws_connected
                    and self._server_connected
                ):
                    combined = b''.join(self.audio_buffer)
                    send_buffer = combined[:self._upload_chunk_bytes]
                    remaining = combined[self._upload_chunk_bytes:]
                    try:
                        await self.ws.send(send_buffer)
                    except Exception as exc:
                        print(f"[IME] Audio send error: {exc}")
                        # The send buffer remains locally queued. Rebuild the
                        # selected backend session and retry; never report a
                        # successful turn after discarding this PCM.
                        self.ws_connected = False
                        self._start_requested = False
                        self._server_connected = False
                        self._start_session_task()
                        break
                    self._clear_audio_buffer()
                    if remaining:
                        self._append_audio_buffer(remaining)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                print(f"[IME] Audio consumer error: {exc}")
                self._mark_audio_ingestion_failure(
                    f"audio_consumer_{type(exc).__name__}"
                )
            finally:
                self._audio_queue.task_done()

    async def _start_recording(self):
        if self.state not in (IMEState.IDLE,) or self._audio_rebuild_in_progress:
            return

        self._ensure_audio_pipeline()
        self._turn_id += 1
        self._active_turn_id = self._turn_id
        self._current_stream_turn_id = self._active_turn_id
        recording_token = self._allocate_recording_token()
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
        self._start_requested = False
        self._server_connected = False
        self._session_start_attempts = 0
        self._active_turn_backend = self.config.backend_key
        self._active_turn_provider = self.config.provider
        self._active_turn_model = self.config.model
        self._audio_drop_count = 0
        self._audio_backpressure_failure_count = 0
        self._audio_ingestion_failed_reason = None
        self._audio_abort_task = None

        try:
            self._begin_turn_spool()
        except Exception as exc:
            print(f"[IME] Cannot create turn audio spool: {exc}")
            self._invalidate_recording_token()
            self._play_sound(getattr(getattr(self, "config", None), "error_sound", Config.error_sound))
            self._set_state(IMEState.IDLE)
            return

        # Play start cue first so perceived audio feedback does not lag behind state icon.
        self._play_sound(self.config.start_sound)
        self._set_state(IMEState.RECORDING)
        self.transcript = ""
        self._clear_audio_buffer()
        self._last_audio_callback_ts = None
        self._audio_drained.clear()
        self._audio_consumer_paused = False
        print(
            f"[Perf][T{self._active_turn_id}] latency_preset={self.config.latency_preset} "
            f"upload_chunk_ms={self.config.upload_chunk_ms} "
            f"(samples={self._upload_chunk_samples}), callback_chunk_frames={self._chunk_size_frames}"
        )

        stream_turn_id = self._current_stream_turn_id
        enqueue_audio = self._enqueue_audio

        def _audio_callback(in_data, frame_count, time_info, status_flags):
            ts = time.perf_counter()
            if not self._recording_token_is_current(recording_token):
                return (None, pyaudio.paAbort)
            accepted = enqueue_audio(in_data, ts, stream_turn_id)
            return (None, pyaudio.paContinue if accepted else pyaudio.paAbort)

        candidate_stream = None
        try:
            loop = self.loop or asyncio.get_running_loop()
            candidate_stream = await loop.run_in_executor(
                self._pyaudio_executor,
                self._open_audio_stream_with_recovery_blocking,
                recording_token,
                _audio_callback,
            )
            if candidate_stream is None:
                self._finish_abandoned_audio_start()
                return
            if not self._recording_token_is_current(recording_token):
                await loop.run_in_executor(
                    self._pyaudio_executor,
                    self._close_audio_stream_blocking,
                    candidate_stream,
                )
                self._finish_abandoned_audio_start()
                return
            self.audio_stream = candidate_stream
            await loop.run_in_executor(
                self._pyaudio_executor,
                candidate_stream.start_stream,
            )
            if not self._recording_token_is_current(recording_token):
                if self.audio_stream is candidate_stream:
                    self.audio_stream = None
                await loop.run_in_executor(
                    self._pyaudio_executor,
                    self._close_audio_stream_blocking,
                    candidate_stream,
                )
                self._finish_abandoned_audio_start()
                return
            self._recording_started_ts = time.perf_counter()
            if self._hotkey_down_ts is not None:
                hotkey_to_recording_ms = (self._recording_started_ts - self._hotkey_down_ts) * 1000
                print(f"[Perf][T{self._active_turn_id}] hotkey_to_recording_ms={hotkey_to_recording_ms:.1f}")
        except Exception as e:
            print(f"[IME] Audio error: {e}")
            if self.audio_stream is candidate_stream:
                self.audio_stream = None
            if candidate_stream is not None:
                loop = self.loop or asyncio.get_running_loop()
                await loop.run_in_executor(
                    self._pyaudio_executor,
                    self._close_audio_stream_blocking,
                    candidate_stream,
                )
            if self._active_recording_token == recording_token:
                self._invalidate_recording_token()
            self._audio_consumer_paused = True
            self._audio_drained.set()
            self._cleanup_turn_spool()
            self._set_state(IMEState.IDLE)
            return

        self._start_session_task()

    async def _stop_and_drain_audio_capture(self) -> None:
        """Stop the producer, then let the consumer spool every accepted frame."""
        self._invalidate_recording_token()
        await self._close_audio_stream()
        if self._audio_queue is not None:
            self._audio_drained.clear()
            await asyncio.to_thread(self._audio_queue.put, None)
            await self._audio_drained.wait()
        self._audio_consumer_paused = True

    async def _give_up_session_start(self):
        """Fail closed when the session cannot be established for this turn.

        Stops accepting audio, archives the locally-buffered PCM (still intact
        because it was never uploaded), alerts, and returns to idle so the next
        hotkey starts fresh instead of hammering a failing session (0436 S2).
        """
        self._invalidate_recording_token()
        await self._stop_and_drain_audio_capture()
        if self._active_turn_id is not None:
            self._failed_turn_ids.add(self._active_turn_id)
        archive_path = self._archive_failed_turn_audio("session_start_exhausted")
        if archive_path:
            print(f"[IME] Local audio archived: {archive_path}")
        self._archive_recent_turn_audio("session_start_exhausted")
        self._play_sound(getattr(getattr(self, "config", None), "error_sound", Config.error_sound))
        self._set_state(IMEState.IDLE)

    async def _ensure_session_started(self, prompt_mode: str):
        retry_interval_sec = 0.3
        if self._force_ws_refresh_before_turn:
            self._force_ws_refresh_before_turn = False
            await self._refresh_ws_for_idle_hygiene()
        while self.state in (IMEState.RECORDING, IMEState.PROCESSING) and not self._start_requested:
            # Bounded backoff so a confirmation-failure loop during recording
            # cannot become a retry storm (task 0436 S2). Attempts persist across
            # error-triggered restarts and reset on a `connected` ack / new turn.
            self._session_start_attempts += 1
            if self._session_start_attempts > self._max_session_start_attempts:
                print(
                    f"[IME] Session start attempts exhausted "
                    f"({self._max_session_start_attempts}); giving up this turn"
                )
                await self._give_up_session_start()
                return
            if self._session_start_attempts > 1:
                backoff = min(
                    self._session_start_backoff_cap_sec,
                    self._session_start_backoff_base_sec
                    * (2 ** (self._session_start_attempts - 2)),
                )
                await asyncio.sleep(backoff)
                if self.state not in (IMEState.RECORDING, IMEState.PROCESSING):
                    return
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
                self._start_requested = True
                return
            except Exception as exc:
                print(f"[IME] Failed to start recording session, retrying: {exc}")
                self.ws_connected = False
                self.ws = None
                await asyncio.sleep(retry_interval_sec)

    async def _stop_recording(self):
        self._play_sound(self.config.stop_sound)
        if self.state != IMEState.RECORDING:
            return

        self._invalidate_recording_token()
        self._stop_pressed_ts = time.perf_counter()
        # Ensure each processing turn starts from a clean transcript buffer.
        self.transcript = ""
        self._set_state(IMEState.PROCESSING)

        await self._async_stop()

    async def _async_stop(self):
        # 自适应等待尾音：至少等待一个回调周期+保护时间，且保留保守下限。
        tail_wait_sec = self._compute_stop_tail_wait_sec()
        print(f"[Perf][T{self._active_turn_id}] stop_tail_wait_ms={tail_wait_sec * 1000:.1f}")
        await asyncio.sleep(tail_wait_sec)

        await self._close_audio_stream()

        if (
            self._audio_abort_task is not None
            and self._audio_abort_task is not asyncio.current_task()
            and not self._audio_abort_task.done()
        ):
            await self._audio_abort_task
            return

        if self._audio_queue is not None:
            self._audio_drained.clear()
            await asyncio.to_thread(self._audio_queue.put, None)
            await self._audio_drained.wait()
        self._audio_consumer_paused = True

        if self._audio_drop_count != 0 or self._audio_ingestion_failed_reason:
            if self._active_turn_id is not None:
                self._failed_turn_ids.add(self._active_turn_id)
            reason = self._audio_ingestion_failed_reason or "audio_drop_detected"
            archive_path = self._archive_failed_turn_audio(reason)
            if archive_path:
                print(f"[IME] Local audio archived: {archive_path}")
            self._archive_recent_turn_audio(reason)
            self._play_sound(getattr(getattr(self, "config", None), "error_sound", Config.error_sound))
            self._set_state(IMEState.IDLE)
            return

        # Give a slow-but-legitimate session start the confirmation budget it is
        # entitled to before declaring upload failure (task 0436 R3-M1). The old
        # code did `wait_for(self._session_task, 1.5)`, which CANCELS a ~3s start
        # that would have succeeded within the 5s + 5s confirmation contract, and
        # then only polled for 1.0s — losing a whole turn of already-recorded
        # audio. Instead poll the matching-turn `_server_connected` Event (SSOT)
        # up to a budget-aligned deadline. This never touches the concurrently
        # running session task, so a legitimately slow start still reaches its
        # `connected` ack and the buffered PCM (never uploaded) is delivered
        # below. An already-connected turn skips the wait entirely.
        if not self._server_connected and self._stop_session_connect_deadline_sec > 0:
            connect_deadline = time.monotonic() + self._stop_session_connect_deadline_sec
            while (
                not self._server_connected
                and self.state == IMEState.PROCESSING
                and time.monotonic() < connect_deadline
            ):
                await asyncio.sleep(0.05)

        # 发送剩余的缓冲音频
        if self.audio_buffer:
            combined = b''.join(self.audio_buffer)
            if combined and self.ws_connected and self._server_connected:
                try:
                    await self.ws.send(combined)
                except Exception as exc:
                    print(f"[IME] Final audio send error: {exc}")
                    self.ws_connected = False
                else:
                    self._clear_audio_buffer()

        # 立即发送停止信号（不需要额外等待，WebSocket是顺序的）
        if self.ws_connected and self._server_connected:
            await self.ws.send(json.dumps({
                "type": "stop_recording",
                "turn_id": self._active_turn_id,
            }))
        else:
            if self._active_turn_id is not None:
                self._failed_turn_ids.add(self._active_turn_id)
            archive_path = self._archive_failed_turn_audio("upload_unavailable")
            if archive_path:
                print(f"[IME] Local audio archived: {archive_path}")
            self._archive_recent_turn_audio("upload_unavailable")
            self._play_sound(getattr(getattr(self, "config", None), "error_sound", Config.error_sound))
            self._set_state(IMEState.IDLE)

    async def _check_processing_timeout_async(self) -> bool:
        if self.state != IMEState.PROCESSING or self._processing_entered_ts is None:
            return False
        elapsed = time.perf_counter() - self._processing_entered_ts
        warning_limit = self.config.processing_timeout_sec
        if (
            warning_limit > 0
            and elapsed >= warning_limit
            and not self._processing_warning_sent
        ):
            self._processing_warning_sent = True
            print(
                f"[IME] Turn {self._active_turn_id} is still processing after "
                f"{elapsed:.1f}s; keeping the terminal channel open"
            )
            if self.on_processing_delay:
                try:
                    self.on_processing_delay(self._active_turn_id, elapsed)
                except Exception as exc:
                    print(f"[IME] Processing-delay callback error: {exc}")

        hard_limit = self.config.processing_hard_timeout_sec
        if hard_limit > 0 and elapsed >= hard_limit:
            print(
                f"[IME] Processing hard timeout after {elapsed:.1f}s "
                f"(limit={hard_limit}s); failing turn explicitly"
            )
            if self._active_turn_id is not None:
                self._failed_turn_ids.add(self._active_turn_id)
            archive_path = self._archive_failed_turn_audio("processing_hard_timeout")
            if archive_path:
                print(f"[IME] Local audio archived: {archive_path}")
            self._archive_recent_turn_audio("processing_hard_timeout")
            self._play_sound(getattr(getattr(self, "config", None), "error_sound", Config.error_sound))
            await self.disconnect_websocket()
            return True
        return False

    def on_hotkey_down(self):
        """快捷键按下 (兼容旧接口)"""
        if self.loop:
            self.loop.call_soon_threadsafe(
                lambda: asyncio.ensure_future(self._handle_hotkey())
            )

    def on_hotkey_up(self):
        """快捷键释放 (切换模式下不需要处理)"""
        pass

    async def _cleanup_async_resources(self):
        self._invalidate_recording_token()
        self._audio_consumer_paused = True
        await self._close_audio_stream()
        if self._audio_consumer_task and not self._audio_consumer_task.done():
            self._audio_consumer_task.cancel()
            try:
                await self._audio_consumer_task
            except asyncio.CancelledError:
                pass
            except Exception as exc:
                print(f"[IME] Audio consumer cleanup error: {exc}")
        self._audio_consumer_task = None

    def cleanup(self):
        self._invalidate_recording_token()
        if self.loop and self.loop.is_running():
            try:
                future = asyncio.run_coroutine_threadsafe(
                    self._cleanup_async_resources(),
                    self.loop,
                )
                future.result(timeout=2)
            except Exception as exc:
                print(f"[IME] Async cleanup error: {exc}")
        self._cleanup_turn_spool()
        if not self._pyaudio_executor_shutdown:
            terminate_timed_out = False
            try:
                self._pyaudio_executor.submit(
                    self._terminate_pyaudio_instance_blocking
                ).result(timeout=2)
            except concurrent.futures.TimeoutError:
                terminate_timed_out = True
                print(
                    "[IME] PortAudio terminate timed out after 2s; "
                    "not waiting on the executor, but the audio worker thread is "
                    "still blocked and interpreter shutdown will join it "
                    "(ThreadPoolExecutor workers are non-daemon since Python 3.9)"
                )
            finally:
                self._pyaudio_executor.shutdown(
                    wait=not terminate_timed_out,
                    cancel_futures=True,
                )
                self._pyaudio_executor_shutdown = True


if HAS_RUMPS:
    class BrainwaveIMEApp(rumps.App):
        """macOS 菜单栏应用"""

        STATE_ICONS = {
            IMEState.IDLE: "🌊",
            IMEState.RECORDING: "🔵",
            IMEState.PROCESSING: "🌀",
            IMEState.DISCONNECTED: "⚫",
        }
        STATUS_GUIDE = [
            ("idle", IMEState.IDLE, None, "🌊 Idle - Ready"),
            ("recording_optimized", IMEState.RECORDING, None, "🔵 Recording - Listening"),
            ("processing", IMEState.PROCESSING, None, "🌀 Processing - Transcribing"),
            ("disconnected", IMEState.DISCONNECTED, None, "⚫ Disconnected"),
        ]
        PROVIDER_OPTIONS = [
            ("openai", "OpenAI"),
        ]
        MODEL_OPTIONS = [
            ("gpt-realtime-2.1-mini", "GPT Realtime 2.1 Mini"),
            ("gpt-realtime-mini-2025-12-15", "GPT Real Time Mini"),
            ("gpt-realtime", "GPT Realtime"),
        ]

        def __init__(
            self,
            supervisor: Optional[BackendSupervisor] = None,
            selection_store: Optional[BackendSelectionStore] = None,
            startup_decision: Optional[StartupDecision] = None,
        ):
            super().__init__("⚫", quit_button=None)
            descriptors = build_default_descriptors(PROJECT_DIR)
            self._backend_supervisor = supervisor or BackendSupervisor(descriptors)
            self._selection_store = selection_store or BackendSelectionStore()
            self._backend_coordinator = BackendCoordinator(
                self._backend_supervisor,
                self._selection_store,
            )
            self._startup_decision = startup_decision or self._backend_coordinator.start_initial()
            self._current_backend = self._startup_decision.effective_backend
            self._backend_events = queue.Queue()
            self._lifecycle_state_lock = threading.Lock()
            self._lifecycle_operation_generation = 0
            self._active_lifecycle_operation = None
            self._active_lifecycle_kind = None
            self._switch_in_progress = False

            self.config = Config()
            initial_descriptor = self._backend_supervisor.descriptor(self._current_backend)
            self.config.backend_key = initial_descriptor.key
            self.config.server_host = initial_descriptor.host
            self.config.server_port = initial_descriptor.port
            self.config.provider = initial_descriptor.provider
            self.config.model = initial_descriptor.model
            self.core = BrainwaveIMECore(
                self.config,
                on_state_change=self._on_state_change,
                on_transcript=self._on_transcript,
                on_transcript_complete=self._on_transcript_complete,
                on_processing_delay=self._on_processing_delay,
            )

            self.provider_labels = {key: label for key, label in self.PROVIDER_OPTIONS}
            self.model_labels = {key: label for key, label in self.MODEL_OPTIONS}
            self._selected_provider = self.config.provider
            self._selected_model = self.config.model
            self.status_item = rumps.MenuItem("Status: Disconnected", callback=self._noop)
            self.backend_switch_item = rumps.MenuItem(
                self._backend_label(),
                callback=None,
            )
            self._attach_backend_switch_view()
            self.reconnect_item = rumps.MenuItem("Reconnect", callback=self.reconnect)
            self.restart_item = rumps.MenuItem("Restart Service", callback=self._restart_service)

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
                self.backend_switch_item,
                self.status_item,
                None,
                self.provider_menu,
                self.model_menu,
                self.hotkeys_item,
                None,
                self._recent_item,
                self._history_item,
                None,
                self.reconnect_item,
                self.restart_item,
                None,
                rumps.MenuItem("Quit", callback=self.quit_app),
            ]

            self.loop = None
            self.loop_thread = None
            self.event_tap = None
            self.event_tap_thread = None
            self._last_state = IMEState.DISCONNECTED
            self._sync_status_menu(self.core.state)
            self._sync_backend_menu(self.core.state)
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
            _register_timer_for_menu_tracking(self._state_timer)

            print("[App] BrainwaveIMEApp initialized, starting auto-connect in 1 second...")

            # 启动后自动连接
            threading.Timer(1.0, self._auto_connect).start()

        def _auto_connect(self):
            threading.Thread(target=self.core.warm_up_sounds, daemon=True).start()
            self._connect()
            if self._startup_decision.diagnostic:
                print(f"[Backend] {self._startup_decision.diagnostic}")
                rumps.notification(
                    "EchoWave IME",
                    "Backend fallback",
                    (
                        f"Requested {self._startup_decision.requested_backend}; "
                        f"using {self._startup_decision.effective_backend}. "
                        f"{self._startup_decision.diagnostic}"
                    ),
                )
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
                            # 标记为 Null 事件，避免系统默认切换窗口导致光标跳走
                            try:
                                CGEventSetType(event, kCGEventNull)
                            except Exception as exc:
                                print(f"[EventTap] Failed to nullify event: {exc}")
                            if self.loop:
                                self.loop.call_soon_threadsafe(
                                    lambda: asyncio.ensure_future(self.core._handle_hotkey())
                                )
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
                "Open System Settings > Privacy & Security > Accessibility to re-enable EchoWave IME. "
                "If it already looks enabled, toggle it off and back on."
            )
            print(f"[Access] {message}")
            if HAS_RUMPS:
                rumps.notification(
                    "EchoWave IME",
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
                "Open System Settings > Privacy & Security > Input Monitoring to re-enable EchoWave IME."
            )
            print(f"[Access] {message}")
            if HAS_RUMPS:
                rumps.notification(
                    "EchoWave IME",
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
            return self.STATE_ICONS.get(state, "⚫")

        def _backend_label(self, backend: Optional[str] = None) -> str:
            key = backend or self._current_backend
            return self._backend_supervisor.descriptor(key).label

        def _attach_backend_switch_view(self) -> None:
            self.backend_switch_view = None
            if not HAS_APPKIT or BackendSwitchMenuView is None:
                print(
                    "[UI] AppKit custom menu views are unavailable; "
                    "backend switching is disabled"
                )
                return
            native_item = getattr(self.backend_switch_item, "_menuitem", None)
            required_methods = ("setAction_", "action", "setView_", "view")
            missing = [
                name
                for name in required_methods
                if native_item is None or not hasattr(native_item, name)
            ]
            if missing:
                raise RuntimeError(
                    "rumps MenuItem bridge changed; sticky backend switch "
                    f"cannot attach ({', '.join(missing)})"
                )

            native_item.setAction_(None)
            view = BackendSwitchMenuView.alloc().initWithFrame_(
                (
                    (0.0, 0.0),
                    (_BACKEND_SWITCH_VIEW_WIDTH, _BACKEND_SWITCH_VIEW_HEIGHT),
                )
            )
            if view is None:
                raise RuntimeError("AppKit failed to allocate backend switch view")
            view.configureWithOwner_(self)
            native_item.setView_(view)
            if native_item.action() is not None or native_item.view() is not view:
                raise RuntimeError(
                    "rumps MenuItem bridge rejected sticky backend switch view"
                )
            # Retain the PyObjC proxy, its label, and tracking area for the full
            # app lifetime. NSMenuItem also retains the native view.
            self.backend_switch_view = view

        def _set_backend_switch_dismiss_armed(self, armed: bool) -> None:
            view = getattr(self, "backend_switch_view", None)
            if view is None:
                return
            setter = getattr(view, "setDismissOnExitArmed_", None)
            if setter is None:
                raise RuntimeError(
                    "backend switch view bridge changed: "
                    "setDismissOnExitArmed_ is unavailable"
                )
            setter(bool(armed))

        def _cancel_backend_menu_tracking(self) -> None:
            native_item = getattr(self.backend_switch_item, "_menuitem", None)
            menu_getter = getattr(native_item, "menu", None)
            if menu_getter is None:
                raise RuntimeError(
                    "rumps MenuItem bridge changed: native menu is unavailable"
                )
            menu = menu_getter()
            if menu is None:
                return
            cancel_tracking = getattr(menu, "cancelTrackingWithoutAnimation", None)
            if cancel_tracking is None:
                raise RuntimeError(
                    "AppKit menu bridge changed: "
                    "cancelTrackingWithoutAnimation is unavailable"
                )
            cancel_tracking()

        def _model_label(self) -> str:
            return self.model_labels.get(self._selected_model, self._selected_model)

        def _lifecycle_operation_active(self) -> bool:
            lock = getattr(self, "_lifecycle_state_lock", None)
            if lock is None:
                return bool(getattr(self, "_switch_in_progress", False))
            with lock:
                return self._active_lifecycle_operation is not None

        def _begin_lifecycle_operation(self, kind: str) -> Optional[int]:
            with self._lifecycle_state_lock:
                if self._active_lifecycle_operation is not None:
                    return None
                self._lifecycle_operation_generation += 1
                token = self._lifecycle_operation_generation
                self._active_lifecycle_operation = token
                self._active_lifecycle_kind = kind
                self._switch_in_progress = True
                return token

        def _adopt_lifecycle_operation(
            self,
            kind: str,
            token: Optional[int],
        ) -> Optional[int]:
            if token is None:
                return self._begin_lifecycle_operation(kind)
            with self._lifecycle_state_lock:
                if self._active_lifecycle_operation != token:
                    return None
                return token

        def _finish_lifecycle_operation(self, token: int) -> None:
            with self._lifecycle_state_lock:
                if self._active_lifecycle_operation != token:
                    return
                self._active_lifecycle_operation = None
                self._active_lifecycle_kind = None
                self._switch_in_progress = False
            self._backend_events.put(
                ("operation_finished", self._current_backend, None, token)
            )

        def _lifecycle_event_is_current(self, token: Optional[int]) -> bool:
            if token is None:
                return True
            with self._lifecycle_state_lock:
                return token == self._lifecycle_operation_generation

        def _sync_status_menu(self, state: IMEState):
            icon = self._current_state_icon(state)
            self.status_item.title = (
                f"{icon} {state.value.capitalize()} · "
                f"{self._backend_label()} · {self._model_label()}"
            )
            lifecycle_idle = not self._lifecycle_operation_active()
            self.reconnect_item._menuitem.setEnabled_(
                state == IMEState.DISCONNECTED and lifecycle_idle
            )
            if hasattr(self, "restart_item"):
                self.restart_item._menuitem.setEnabled_(
                    state in (IMEState.IDLE, IMEState.DISCONNECTED)
                    and lifecycle_idle
                )

        def _sync_backend_menu(self, state: IMEState):
            label = self._backend_label()
            self.backend_switch_item.title = label
            self.backend_switch_item.state = 0
            enabled = (
                state in (IMEState.IDLE, IMEState.DISCONNECTED)
                and not self._lifecycle_operation_active()
            )
            self.backend_switch_item._menuitem.setEnabled_(enabled)
            view = getattr(self, "backend_switch_view", None)
            if view is not None:
                setter = getattr(view, "setBackendLabel_enabled_", None)
                if setter is None:
                    raise RuntimeError(
                        "backend switch view bridge changed: "
                        "setBackendLabel_enabled_ is unavailable"
                    )
                setter(label, enabled)

        def _sync_provider_menu(self):
            for key, item in self.provider_items.items():
                item.state = 1 if key == self._selected_provider else 0
            self.provider_menu._menuitem.setEnabled_(self._current_backend == BACKEND_ECHO)

        def _sync_model_menu(self):
            for key, item in self.model_items.items():
                item.state = 1 if key == self._selected_model else 0
            self.model_menu._menuitem.setEnabled_(self._current_backend == BACKEND_ECHO)

        def _provider_selected(self, sender):
            self.set_provider(sender)

        def _model_selected(self, sender):
            self.set_model(sender)

        def set_provider(self, sender):
            if self._current_backend != BACKEND_ECHO:
                print("[UI] Echo provider controls are disabled for WhisperWave")
                return
            provider_key = getattr(sender, "_provider_key", None)
            if provider_key not in self.provider_labels:
                print(f"[UI] Unknown provider: {provider_key}")
                return
            if self._selected_provider == provider_key:
                return
            if not self.loop:
                print("[UI] Provider change skipped: asyncio loop is not running")
                return
            self.loop.call_soon_threadsafe(
                lambda: self.core._apply_provider(provider_key)
            )
            self._selected_provider = provider_key
            self._sync_provider_menu()
            self._sync_status_menu(self.core.state)

        def set_model(self, sender):
            if self._current_backend != BACKEND_ECHO:
                print("[UI] Echo model controls are disabled for WhisperWave")
                return
            model_key = getattr(sender, "_model_key", None)
            if model_key not in self.model_labels:
                print(f"[UI] Unknown model: {model_key}")
                return
            if self._selected_model == model_key:
                return
            if not self.loop:
                print("[UI] Model change skipped: asyncio loop is not running")
                return
            self.loop.call_soon_threadsafe(
                lambda: self.core._apply_model(model_key)
            )
            self._selected_model = model_key
            self._sync_model_menu()
            self._sync_status_menu(self.core.state)

        def _backend_switch_selected(self, _):
            if (
                self.core.state not in (IMEState.IDLE, IMEState.DISCONNECTED)
                or not self.loop
            ):
                return
            operation_token = self._begin_lifecycle_operation("switch")
            if operation_token is None:
                return
            self._set_backend_switch_dismiss_armed(False)
            target = (
                BACKEND_WHISPER
                if self._current_backend == BACKEND_ECHO
                else BACKEND_ECHO
            )
            self._sync_backend_menu(self.core.state)
            self._sync_status_menu(self.core.state)
            self.loop.call_soon_threadsafe(
                lambda: asyncio.ensure_future(
                    self._switch_backend_async(target, operation_token)
                )
            )

        async def _switch_backend_async(
            self,
            target: str,
            operation_token: Optional[int] = None,
        ) -> bool:
            operation_token = self._adopt_lifecycle_operation(
                "switch",
                operation_token,
            )
            if operation_token is None:
                return False
            previous_backend = self._current_backend
            try:
                result = await self._backend_coordinator.switch(
                    current_backend=previous_backend,
                    target_backend=target,
                    state=self.core.state.value,
                    repoint=self.core.repoint_backend,
                )
                # Commit the routing fields before releasing the operation token
                # so a newly admitted lifecycle action can never observe the old
                # backend after persistence/core already moved to the target.
                self._current_backend = result.effective_backend
                self._selected_provider = self.core.config.provider
                self._selected_model = self.core.config.model
                self._backend_events.put(
                    (
                        "switched",
                        result.effective_backend,
                        result.cleanup_warning,
                        operation_token,
                    )
                )
                return True
            except Exception as exc:
                self._backend_events.put(
                    ("error", previous_backend, str(exc), operation_token)
                )
                return False
            finally:
                self._finish_lifecycle_operation(operation_token)

        def reconnect(self, _):
            if self.core.state != IMEState.DISCONNECTED or not self.loop:
                return
            operation_token = self._begin_lifecycle_operation("reconnect")
            if operation_token is None:
                return
            self._set_backend_switch_dismiss_armed(False)
            self._sync_backend_menu(self.core.state)
            self._sync_status_menu(self.core.state)
            self.loop.call_soon_threadsafe(
                lambda: asyncio.ensure_future(
                    self._reconnect_current_backend_async(operation_token)
                )
            )

        async def _reconnect_current_backend_async(
            self,
            operation_token: Optional[int] = None,
        ) -> bool:
            operation_token = self._adopt_lifecycle_operation(
                "reconnect",
                operation_token,
            )
            if operation_token is None:
                return False
            descriptor = self._backend_supervisor.descriptor(self._current_backend)
            try:
                await asyncio.to_thread(
                    self._backend_supervisor.start,
                    self._current_backend,
                )
                connected = await self.core.repoint_backend(descriptor)
                if not connected:
                    raise BackendRuntimeError(
                        f"menu client could not reconnect to {descriptor.label}"
                    )
                self._backend_events.put(
                    ("reconnected", self._current_backend, None, operation_token)
                )
                return True
            except Exception as exc:
                self._backend_events.put(
                    ("error", self._current_backend, str(exc), operation_token)
                )
                return False
            finally:
                self._finish_lifecycle_operation(operation_token)

        def _restart_service(self, _):
            """Menu callback. Dispatch to asyncio loop, return immediately."""
            print("[App] Restart Service requested by user")
            if (
                self.core.state not in (IMEState.IDLE, IMEState.DISCONNECTED)
                or not self.loop
            ):
                return
            operation_token = self._begin_lifecycle_operation("restart")
            if operation_token is None:
                return
            self._set_backend_switch_dismiss_armed(False)
            self._sync_backend_menu(self.core.state)
            self._sync_status_menu(self.core.state)
            self.loop.call_soon_threadsafe(
                lambda: asyncio.ensure_future(
                    self._restart_service_async(operation_token)
                )
            )

        async def _restart_service_async(
            self,
            operation_token: Optional[int] = None,
        ) -> bool:
            operation_token = self._adopt_lifecycle_operation(
                "restart",
                operation_token,
            )
            if operation_token is None:
                return False
            try:
                if self.core.state not in (IMEState.IDLE, IMEState.DISCONNECTED):
                    return False
                try:
                    if not await self.core.rebuild_audio_device_table():
                        return False
                except Exception as exc:
                    print(
                        "[App] Restart Service continuing after unexpected "
                        f"PortAudio rebuild error: {exc}"
                    )
                await self.core.disconnect_websocket()
                descriptor = self._backend_supervisor.descriptor(self._current_backend)
                await asyncio.to_thread(
                    self._backend_supervisor.restart,
                    self._current_backend,
                )
                if not await self.core.repoint_backend(descriptor):
                    raise BackendRuntimeError(
                        f"menu client could not reconnect after restarting {descriptor.label}"
                    )
                self._backend_events.put(
                    ("restarted", self._current_backend, None, operation_token)
                )
                return True
            except Exception as exc:
                self._backend_events.put(
                    ("error", self._current_backend, str(exc), operation_token)
                )
                return False
            finally:
                self._finish_lifecycle_operation(operation_token)

        def _poll_state(self, timer):
            """定时器回调 - 在主线程检查并更新 UI"""
            current_state = self.core.state
            if self._accessibility_warning_needed and not self._accessibility_warning_sent:
                self._accessibility_warning_sent = True
                self._show_accessibility_warning()
            if self._input_monitoring_warning_needed and not self._input_monitoring_warning_sent:
                self._input_monitoring_warning_sent = True
                self._show_input_monitoring_warning()
            self._drain_backend_events()
            self._drain_pending_transcripts()
            if current_state != self._last_state:
                self._last_state = current_state
                icon = self._current_state_icon(current_state)
                print(f"[UI] State changed: {current_state.value}, updating icon to: {icon}")
                self.title = icon
                self._sync_status_menu(current_state)
                self._sync_backend_menu(current_state)
                print(f"[UI] Title is now: {self.title}")
            if self.loop:
                self.loop.call_soon_threadsafe(
                    lambda: asyncio.ensure_future(self.core._check_processing_timeout_async())
                )

        def _on_state_change(self, state: IMEState):
            """状态变更回调 - 状态会被定时器轮询更新到 UI"""
            # 不在这里更新 UI，由 _poll_state 定时器处理
            pass

        def _on_transcript(self, text: str):
            pass

        def _on_processing_delay(self, turn_id, elapsed: float):
            self._backend_events.put(
                (
                    "processing_delay",
                    self._current_backend,
                    f"Turn {turn_id} is still processing after {elapsed:.0f}s; waiting for terminal output.",
                )
            )

        def _drain_backend_events(self):
            while True:
                try:
                    item = self._backend_events.get_nowait()
                except queue.Empty:
                    break
                if len(item) == 3:
                    event, backend, detail = item
                    operation_token = None
                else:
                    event, backend, detail, operation_token = item
                if not self._lifecycle_event_is_current(operation_token):
                    if event == "switched" and detail:
                        print(f"[Backend] Cleanup warning from completed operation: {detail}")
                        rumps.notification(
                            "EchoWave IME",
                            "Backend cleanup warning",
                            detail,
                        )
                    print(
                        f"[Backend] Ignoring stale lifecycle event {event} "
                        f"for token {operation_token}"
                    )
                    # A completed worker may already have committed
                    # _current_backend before a newer lifecycle token makes its
                    # queued event stale. Re-render from current authority so
                    # dropping the payload can never strand the old label.
                    self._sync_backend_menu(self.core.state)
                    continue
                arm_dismiss_after_render = False
                reset_dismiss_after_render = False
                if event == "switched":
                    core_backend = getattr(
                        getattr(self.core, "config", None),
                        "backend_key",
                        None,
                    )
                    if (
                        core_backend in (BACKEND_ECHO, BACKEND_WHISPER)
                        and backend != core_backend
                    ):
                        print(
                            "[Backend] Ignoring switched event that disagrees "
                            f"with committed core backend {core_backend}: {backend}"
                        )
                        self._sync_backend_menu(self.core.state)
                        continue
                    self._current_backend = backend
                    self._selected_provider = self.core.config.provider
                    self._selected_model = self.core.config.model
                    print(f"[Backend] Active backend switched to {backend}")
                    arm_dismiss_after_render = True
                    if detail:
                        print(f"[Backend] Cleanup warning: {detail}")
                        rumps.notification(
                            "EchoWave IME",
                            "Backend switched with cleanup warning",
                            detail,
                        )
                elif event == "error":
                    print(f"[Backend] Operation failed: {detail}")
                    reset_dismiss_after_render = True
                    rumps.notification(
                        "EchoWave IME",
                        "Backend operation failed",
                        detail or "The current backend remains selected.",
                    )
                elif event == "processing_delay":
                    rumps.notification(
                        "EchoWave IME",
                        "Still processing",
                        detail or "Waiting for terminal transcription output.",
                    )
                self._sync_backend_menu(self.core.state)
                self._sync_provider_menu()
                self._sync_model_menu()
                self._sync_status_menu(self.core.state)
                if reset_dismiss_after_render:
                    self._set_backend_switch_dismiss_armed(False)
                elif arm_dismiss_after_render:
                    # Arm only after the committed target has been written to
                    # both the native item title and the visible custom label.
                    self._set_backend_switch_dismiss_armed(True)

        def _on_transcript_complete(
            self,
            text: str,
            recording_mode,
            provider: str,
            model: str,
            backend: str,
        ):
            if not text or not text.strip():
                return
            entry = {
                "ts": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "text": text,
                "mode": recording_mode.value if recording_mode else None,
                "backend": backend,
                "provider": provider,
                "model": model,
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
            backend = entry.get("backend")
            backend_prefix = f"[{backend}] " if backend else ""
            return (
                f"{prefix}{backend_prefix}{preview}"
                if preview
                else f"{prefix}{backend_prefix}(empty)"
            )

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

        def _connect(self):
            if self.loop is None:
                self.loop = asyncio.new_event_loop()
                self.loop_thread = threading.Thread(
                    target=self.loop.run_forever,
                    daemon=True
                )
                self.loop_thread.start()
                self.core.loop = self.loop

            self.loop.call_soon_threadsafe(
                lambda: asyncio.ensure_future(self._connect_async())
            )

        async def _connect_async(self):
            self.core._ensure_audio_pipeline()
            await self.core.connect_websocket()

        def quit_app(self, _):
            if self.loop:
                try:
                    future = asyncio.run_coroutine_threadsafe(
                        self.core.disconnect_websocket(),
                        self.loop
                    )
                    future.result(timeout=2)
                except Exception as exc:
                    print(f"[App] Failed to disconnect websocket cleanly: {exc}")
                self.core.cleanup()
                self.loop.call_soon_threadsafe(self.loop.stop)
            else:
                self.core.cleanup()
            self._backend_supervisor.stop_all()
            rumps.quit_application()


def run_menubar(
    supervisor: Optional[BackendSupervisor] = None,
    selection_store: Optional[BackendSelectionStore] = None,
    startup_decision: Optional[StartupDecision] = None,
):
    """运行菜单栏应用"""
    if not HAS_RUMPS:
        print("Error: rumps not installed")
        sys.exit(1)

    app = BrainwaveIMEApp(
        supervisor=supervisor,
        selection_store=selection_store,
        startup_decision=startup_decision,
    )
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

    print("EchoWave IME (CLI Mode)")
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
