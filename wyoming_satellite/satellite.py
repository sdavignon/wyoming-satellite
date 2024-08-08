import array
import asyncio
import logging
import math
import time
import wave
from dataclasses import dataclass
from enum import Enum, auto
from pathlib import Path
from typing import Callable, Dict, Final, List, Optional, Set, Union

from nltk.sentiment.vader import SentimentIntensityAnalyzer
import websockets

from pyring_buffer import RingBuffer
from wyoming.asr import Transcript
from wyoming.audio import AudioChunk, AudioFormat, AudioStart, AudioStop
from wyoming.client import AsyncClient
from wyoming.error import Error
from wyoming.event import Event, async_write_event
from wyoming.info import Describe, Info
from wyoming.mic import MicProcessAsyncClient
from wyoming.ping import Ping, Pong
from wyoming.pipeline import PipelineStage, RunPipeline
from wyoming.satellite import (
    PauseSatellite,
    RunSatellite,
    SatelliteConnected,
    SatelliteDisconnected,
    StreamingStarted,
    StreamingStopped,
)
from wyoming.snd import Played, SndProcessAsyncClient
from wyoming.timer import TimerCancelled, TimerFinished, TimerStarted, TimerUpdated
from wyoming.tts import Synthesize
from wyoming.vad import VoiceStarted, VoiceStopped
from wyoming.wake import Detect, Detection, WakeProcessAsyncClient

from .settings import SatelliteSettings
from .utils import (
    DebugAudioWriter,
    multiply_volume,
    normalize_wake_word,
    run_event_command,
    wav_to_events,
)
from .vad import SileroVad
from .webrtc import WebRtcAudio

_LOGGER = logging.getLogger()

_PONG_TIMEOUT: Final = 5
_PING_SEND_DELAY: Final = 2
_WAKE_INFO_TIMEOUT: Final = 2

class State(Enum):
    NOT_STARTED = auto()
    STARTING = auto()
    STARTED = auto()
    RESTARTING = auto()
    STOPPING = auto()
    STOPPED = auto()

@dataclass
class SoundEvent:
    event: Event
    is_tts: bool

class SatelliteBase:
    """Base class for satellites."""

    def __init__(self, settings: SatelliteSettings) -> None:
        self.settings = settings
        self.server_id: Optional[str] = None
        self._state = State.NOT_STARTED
        self._state_changed = asyncio.Event()
        self._writer: Optional[asyncio.StreamWriter] = None
        self.websocket_sentiment_server = None
        self._mic_task: Optional[asyncio.Task] = None
        self._mic_webrtc: Optional[Callable[[bytes], bytes]] = None
        self._snd_task: Optional[asyncio.Task] = None
        self._snd_queue: "Optional[asyncio.Queue[SoundEvent]]" = None
        self._wake_task: Optional[asyncio.Task] = None
        self._wake_queue: "Optional[asyncio.Queue[Event]]" = None
        self._event_task: Optional[asyncio.Task] = None
        self._event_queue: "Optional[asyncio.Queue[Event]]" = None

        self._ping_server_enabled: bool = False
        self._pong_received_event = asyncio.Event()
        self._ping_server_task: Optional[asyncio.Task] = None

        self.microphone_muted = False
        self._unmute_microphone_task: Optional[asyncio.Task] = None
        self.sentiment_analyzer = SentimentIntensityAnalyzer()

        # Debug audio recording
        self.wake_audio_writer: Optional[DebugAudioWriter] = None
        self.stt_audio_writer: Optional[DebugAudioWriter] = None
        if settings.debug_recording_dir:
            self.wake_audio_writer = DebugAudioWriter(
                settings.debug_recording_dir,
                "wake",
                ring_buffer_size=(2 * 16000 * 2 * 1),  # last 2 sec
            )
            self.stt_audio_writer = DebugAudioWriter(
                settings.debug_recording_dir, "stt"
            )

    @property
    def is_running(self) -> bool:
        """True if not stopping/stopped."""
        return self._state not in (State.STOPPED,)

    @property
    def state(self) -> State:
        """Get current state."""
        return self._state

    @state.setter
    def state(self, value: State) -> None:
        """Set state."""
        self._state = value
        self._state_changed.set()
        self._state_changed.clear()

    async def run(self) -> None:
        """Run main satellite loop."""
        # Start the WebSocket servers
        self.websocket_sentiment_server = await websockets.serve(self.handle_sentiment, "localhost", 6790)

        while self.is_running:
            try:
                if self.state == State.NOT_STARTED:
                    await self._start()
                elif self.state == State.RESTARTING:
                    await self._restart()
                elif self.state == State.STOPPING:
                    await self._stop()
                elif self.state == State.STOPPED:
                    await self.stopped()
                    break
                else:
                    await self._state_changed.wait()
            except Exception:
                if self.is_running:
                    # Automatically restart
                    _LOGGER.exception("Unexpected error running satellite")
                    self.state = State.RESTARTING

    async def stop(self) -> None:
        """Stop the satellite"""
        self.state = State.STOPPING

        # Wait for STOPPED
        while self.state != State.STOPPED:
            await self._state_changed.wait()

    async def set_server(self, server_id: str, writer: asyncio.StreamWriter) -> None:
        """Set event writer."""
        self.server_id = server_id
        self._writer = writer
        _LOGGER.debug("Server set: %s", server_id)
        await self.trigger_server_connected()

    async def clear_server(self) -> None:
        """Remove writer."""
        self.server_id = None
        self._writer = None
        self._disable_ping()

        _LOGGER.debug("Server disconnected")
        await self.trigger_server_disonnected()

    async def event_to_server(self, event: Event) -> None:
        """Send an event to the server."""
        if self._writer is None:
            return

        try:
            await async_write_event(event, self._writer)
        except Exception as err:
            await self.clear_server()

            if isinstance(err, ConnectionResetError):
                _LOGGER.warning("Server disconnected unexpectedly")
            else:
                _LOGGER.exception("Unexpected error sending event to server")

    def _enable_ping(self) -> None:
        self._ping_server_enabled = True
        self._ping_server_task = asyncio.create_task(self._ping_server(), name="ping")

    def _disable_ping(self) -> None:
        self._ping_server_enabled = False
        if self._ping_server_task is not None:
            self._ping_server_task.cancel()
            self._ping_server_task = None

    async def _ping_server(self) -> None:
        try:
            while self.is_running:
                await asyncio.sleep(_PING_SEND_DELAY)
                if (self.server_id is None) or (not self._ping_server_enabled):
                    # No server connected
                    continue

                # Send ping and wait for pong
                self._pong_received_event.clear()
                await self.event_to_server(Ping().event())
                try:
                    await asyncio.wait_for(
                        self._pong_received_event.wait(), timeout=_PONG_TIMEOUT
                    )
                except asyncio.TimeoutError:
                    if self.server_id is None:
                        # No server connected
                        continue

                    _LOGGER.warning("Did not receive ping response within timeout")
                    await self.clear_server()
        except asyncio.CancelledError:
            pass
        except Exception:
            _LOGGER.exception("Unexpected error in ping server task")

    # -------------------------------------------------------------------------

    async def _start(self) -> None:
        """Connect to services."""
        self.state = State.STARTING
        await self._connect_to_services()
        self.state = State.STARTED
        await self.started()

    async def started(self) -> None:
        """Called when satellite has started."""

    async def _stop(self) -> None:
        """Disconnect from services."""
        self.server_id = None
        self._writer = None

        await self._disconnect_from_services()
        self._disable_ping()
        self.state = State.STOPPED

    async def stopped(self) -> None:
        """Called when satellite has stopped."""

    async def event_from_server(self, event: Event) -> None:
        """Called when an event is received from the server."""
        forward_event = True
        if Ping.is_type(event.type):
            # Respond with pong
            ping = Ping.from_event(event)
            await self.event_to_server(Pong(text=ping.text).event())

            if not self._ping_server_enabled:
                # Enable pinging
                self._enable_ping()
                _LOGGER.debug("Ping enabled")

            forward_event = False
        elif Pong.is_type(event.type):
            # Response from our ping
            self._pong_received_event.set()
            forward_event = False
        elif AudioChunk.is_type(event.type):
            # TTS audio
            await self.event_to_snd(event)
            forward_event = False
        elif AudioStart.is_type(event.type):
            # TTS started
            await self.event_to_snd(event)
            await self.trigger_tts_start()
        elif AudioStop.is_type(event.type):
            # TTS stopped
            await self.event_to_snd(event)
            await self.trigger_tts_stop()
        elif Detect.is_type(event.type):
            # Wake word detection started
            await self.trigger_detect()
        elif Detection.is_type(event.type):
            # Wake word detected
            _LOGGER.debug("Wake word detected")
            await self.trigger_detection(Detection.from_event(event))
        elif VoiceStarted.is_type(event.type):
            # STT start
            await self.trigger_stt_start()
        elif VoiceStopped.is_type(event.type):
            # STT stop
            await self.trigger_stt_stop()
        elif Transcript.is_type(event.type):
            # STT text
            _LOGGER.debug(event)
            await self.trigger_transcript(Transcript.from_event(event))
        elif Synthesize.is_type(event.type):
            # TTS request
            _LOGGER.debug(event)
            synthesize = Synthesize.from_event(event)
            await self.trigger_synthesize(synthesize)
            sentiment = self.analyze_sentiment(synthesize.text)
            await self.notify_frontend(f"sentiment:{sentiment}")
        elif Error.is_type(event.type):
            _LOGGER.warning(event)
            await self.trigger_error(Error.from_event(event))
        elif TimerStarted.is_type(event.type):
            _LOGGER.debug(event)
            await self.trigger_timer_started(TimerStarted.from_event(event))
        elif TimerUpdated.is_type(event.type):
            _LOGGER.debug(event)
            await self.trigger_timer_updated(TimerUpdated.from_event(event))
        elif TimerCancelled.is_type(event.type):
            _LOGGER.debug(event)
            await self.trigger_timer_cancelled(TimerCancelled.from_event(event))
        elif TimerFinished.is_type(event.type):
            _LOGGER.debug(event)
            await self.trigger_timer_finished(TimerFinished.from_event(event))

        # Forward everything except audio/ping/pong to event service
        if forward_event:
            await self.forward_event(event)

    async def _send_run_pipeline(self, pipeline_name: Optional[str] = None) -> None:
        """Sends a RunPipeline event with the correct stages."""
        if self.settings.wake.enabled:
            # Local wake word detection
            start_stage = PipelineStage.ASR
            restart_on_end = False
        else:
            # Remote wake word detection
            start_stage = PipelineStage.WAKE
            restart_on_end = not self.settings.vad.enabled

        if self.settings.snd.enabled:
            # Play TTS response
            end_stage = PipelineStage.TTS
        else:
            # No audio output
            end_stage = PipelineStage.HANDLE

        run_pipeline = RunPipeline(
            start_stage=start_stage,
            end_stage=end_stage,
            name=pipeline_name,
            restart_on_end=restart_on_end,
            snd_format=AudioFormat(
                rate=self.settings.snd.rate,
                width=self.settings.snd.width,
                channels=self.settings.snd.channels,
            ),
        ).event()
        _LOGGER.debug(run_pipeline)
        await self.event_to_server(run_pipeline)
        await self.forward_event(run_pipeline)

    async def handle_sentiment(self, websocket, path):
        """Handle incoming sentiment commands."""
        async for message in websocket:
            try:
                text = message
                sentiment = self.analyze_sentiment(text)
                await websocket.send(f'sentiment:{sentiment}')
            except Exception as e:
                _LOGGER.error(f"Error handling sentiment analysis: {e}")

    def analyze_sentiment(self, text):
        """Analyze the sentiment of the text."""
        scores = self.sentiment_analyzer.polarity_scores(text)
        if scores['compound'] >= 0.05:
            return 'happy'
        elif scores['compound'] <= -0.05:
            return 'sad'
        else:
            return 'neutral'

    async def notify_frontend(self, command):
        """Send command to the WebSocket client."""
        uri = "ws://localhost:6790"  # WebSocket server running on your Electron app
        async with websockets.connect(uri) as websocket:
            await websocket.send(command)

    async def trigger_tts_start(self) -> None:
        """Called when text-to-speech audio starts."""
        await run_event_command(self.settings.event.tts_start)

    async def trigger_tts_stop(self) -> None:
        """Called when text-to-speech audio stops."""
        await run_event_command(self.settings.event.tts_stop)

    async def trigger_error(self, error: Error) -> None:
        """Called when an error occurs on the server."""
        await run_event_command(self.settings.event.error, error.text)

    async def trigger_timer_started(self, timer_started: TimerStarted) -> None:
        """Called when timer-started event is received."""
        await run_event_command(self.settings.timer.started, timer_started)

    async def trigger_timer_updated(self, timer_updated: TimerUpdated) -> None:
        """Called when timer-updated event is received."""
        await run_event_command(self.settings.timer.updated, timer_updated)

    async def trigger_timer_cancelled(self, timer_cancelled: TimerCancelled) -> None:
        """Called when timer-cancelled event is received."""
        await run_event_command(self.settings.timer.cancelled, timer_cancelled.id)

    async def trigger_timer_finished(self, timer_finished: TimerFinished) -> None:
        """Called when timer-finished event is received."""
        await run_event_command(self.settings.timer.finished, timer_finished.id)
        for _ in range(self.settings.timer.finished_wav_plays):
            await self._play_wav(
                self.settings.timer.finished_wav,
                mute_microphone=self.settings.mic.mute_during_awake_wav,
            )
            await asyncio.sleep(self.settings.timer.finished_wav_delay)

    async def forward_event(self, event: Event) -> None:
        """Forward an event to the event service."""
        if self._event_queue is not None:
            self._event_queue.put_nowait(event)

    def _make_event_client(self) -> Optional[AsyncClient]:
        """Create client for event service."""
        if self.settings.event.uri:
            return AsyncClient.from_uri(self.settings.event.uri)

        return None

    async def _event_task_proc(self) -> None:
        """Event service loop."""
        event_client: Optional[AsyncClient] = None

        async def _disconnect() -> None:
            try:
                if event_client is not None:
                    await event_client.disconnect()
            except Exception:
                pass  # ignore disconnect errors

        while self.is_running:
            try:
                if self._event_queue is None:
                    self._event_queue = asyncio.Queue()

                event = await self._event_queue.get()

                if event_client is None:
                    event_client = self._make_event_client()
                    assert event_client is not None
                    await event_client.connect()
                    _LOGGER.debug("Connected to event service")

                await event_client.write_event(event)
            except asyncio.CancelledError:
                break
            except Exception:
                _LOGGER.exception("Unexpected error in event read task")
                await _disconnect()
                event_client = None  # reconnect
                self._event_queue = None
                await asyncio.sleep(self.settings.event.reconnect_seconds)

        await _disconnect()

    # -------------------------------------------------------------------------
    # Info
    # -------------------------------------------------------------------------

    async def update_info(self, info: Info) -> None:
        pass


# -----------------------------------------------------------------------------


class AlwaysStreamingSatellite(SatelliteBase):
    """Satellite that always streams audio."""

    def __init__(self, settings: SatelliteSettings) -> None:
        super().__init__(settings)
        self.is_streaming = False

        if settings.vad.enabled:
            _LOGGER.warning("VAD is enabled but will not be used")

        if settings.wake.enabled:
            _LOGGER.warning("Local wake word detection is enabled but will not be used")

    async def event_from_server(self, event: Event) -> None:
        await super().event_from_server(event)

        if RunSatellite.is_type(event.type):
            self.is_streaming = True
            _LOGGER.info("Streaming audio")
            await self._send_run_pipeline()
            await self.trigger_streaming_start()
        elif PauseSatellite.is_type(event.type):
            self.is_streaming = False
            _LOGGER.info("Satellite paused")
        elif Detection.is_type(event.type):
            # Start debug recording
            if self.stt_audio_writer is not None:
                self.stt_audio_writer.start()
        elif Transcript.is_type(event.type) or Error.is_type(event.type):
            # Stop debug recording
            if self.stt_audio_writer is not None:
                self.stt_audio_writer.stop()

            if Transcript.is_type(event.type):
                # We're always streaming
                _LOGGER.info("Streaming audio")

                # Re-trigger streaming start even though we technically don't stop
                # so the event service can reset LEDs, etc.
                await self.trigger_streaming_start()

    async def event_from_mic(self, event: Event, audio_bytes: Optional[bytes] = None) -> None:
        if not self.is_streaming or self.microphone_muted:
            return

        if AudioChunk.is_type(event.type):
            # Forward to server
            await self.event_to_server(event)

            # Debug audio recording
            if self.stt_audio_writer is not None:
                if audio_bytes is None:
                    chunk = AudioChunk.from_event(event)
                    audio_bytes = chunk.audio

                self.stt_audio_writer.write(audio_bytes)


class VadStreamingSatellite(SatelliteBase):
    """Satellite that only streams after speech is detected."""

    def __init__(self, settings: SatelliteSettings) -> None:
        if not settings.vad.enabled:
            raise ValueError("VAD is not enabled")

        super().__init__(settings)
        self.is_streaming = False
        self.vad = SileroVad(
            threshold=settings.vad.threshold, trigger_level=settings.vad.trigger_level
        )

        # Timestamp in the future when we will have timed out (set with time.monotonic())
        self.timeout_seconds: Optional[float] = None

        # Audio from right before speech starts (circular buffer)
        self.vad_buffer: Optional[RingBuffer] = None

        if settings.vad.buffer_seconds > 0:
            # Assume 16Khz, 16-bit mono samples
            vad_buffer_bytes = int(math.ceil(settings.vad.buffer_seconds * 16000 * 2))
            self.vad_buffer = RingBuffer(maxlen=vad_buffer_bytes)

        if settings.wake.enabled:
            _LOGGER.warning("Local wake word detection is enabled but will not be used")

        self._is_paused = False

    async def event_from_server(self, event: Event) -> None:
        await super().event_from_server(event)

        if RunSatellite.is_type(event.type):
            self._is_paused = False
            _LOGGER.info("Waiting for speech")
        elif Detection.is_type(event.type):
            # Start debug recording
            if self.stt_audio_writer is not None:
                self.stt_audio_writer.start()
        elif (
            Transcript.is_type(event.type)
            or Error.is_type(event.type)
            or PauseSatellite.is_type(event.type)
        ):
            if PauseSatellite.is_type(event.type):
                self._is_paused = True
                _LOGGER.debug("Satellite paused")

            self.is_streaming = False

            # Stop debug recording
            if self.stt_audio_writer is not None:
                self.stt_audio_writer.stop()

    async def event_from_mic(self, event: Event, audio_bytes: Optional[bytes] = None) -> None:
        if not AudioChunk.is_type(event.type) or self.microphone_muted or self._is_paused:
            return

        # Only unpack chunk once
        chunk: Optional[AudioChunk] = None

        # Debug audio recording
        if self.stt_audio_writer is not None:
            if audio_bytes is None:
                # Need to unpack
                chunk = AudioChunk.from_event(event)
                audio_bytes = chunk.audio

            self.stt_audio_writer.write(audio_bytes)

        if (
            self.is_streaming
            and self.timeout_seconds is not None
            and time.monotonic() >= self.timeout_seconds
        ):
            # Time out during wake word recognition
            self.is_streaming = False
            self.timeout_seconds = None

            # Stop debug recording
            if self.stt_audio_writer is not None:
                self.stt_audio_writer.stop()

            # Stop pipeline
            await self.event_to_server(AudioStop().event())

            _LOGGER.info("Waiting for speech")
            await self.trigger_streaming_stop()

        if not self.is_streaming:
            # Check VAD
            if audio_bytes is None:
                if chunk is None:
                    # Need to unpack
                    chunk = AudioChunk.from_event(event)

                audio_bytes = chunk.audio

            if not self.vad(audio_bytes):
                # No speech
                if self.vad_buffer is not None:
                    self.vad_buffer.put(audio_bytes)

                return

            # Speech detected
            self.is_streaming = True
            _LOGGER.info("Streaming audio")
            await self._send_run_pipeline()
            await self.trigger_streaming_start()

            if self.settings.vad.wake_word_timeout is not None:
                # Set future time when we'll stop streaming if the wake word
                # hasn't been detected.
                self.timeout_seconds = (
                    time.monotonic() + self.settings.vad.wake_word_timeout
                )
            else:
                # No timeout
                self.timeout_seconds = None

            if self.vad_buffer is not None:
                # Send contents of VAD buffer first. This is the audio that was
                # recorded right before speech was detected.
                if chunk is None:
                    chunk = AudioChunk.from_event(event)

                await self.event_to_server(
                    AudioChunk(
                        rate=chunk.rate,
                        width=chunk.width,
                        channels=chunk.channels,
                        audio=self.vad_buffer.getvalue(),
                    ).event()
                )

            self._reset_vad()

        if self.is_streaming:
            # Forward to server
            await self.event_to_server(event)

    def _reset_vad(self):
        """Reset state of VAD."""
        self.vad(None)

        if self.vad_buffer is not None:
            # Clear buffer
            self.vad_buffer.put(bytes(self.vad_buffer.maxlen))


class WakeStreamingSatellite(SatelliteBase):
    """Satellite that waits for local wake word detection before streaming."""

    def __init__(self, settings: SatelliteSettings) -> None:
        if not settings.wake.enabled:
            raise ValueError("Local wake word detection is not enabled")

        super().__init__(settings)
        self.is_streaming = False

        # Timestamp in the future when the refractory period is over (set with time.monotonic())
        # wake word id -> seconds
        self.refractory_timestamp: Dict[Optional[str], float] = {}

        if settings.vad.enabled:
            _LOGGER.warning("VAD is enabled but will not be used")

        # Used for debug audio recording so both wake and stt WAV files have the same timestamp.
        self._debug_recording_timestamp: Optional[int] = None

        self._is_paused = False

        self._wake_info: Optional[Info] = None
        self._wake_info_ready = asyncio.Event()

    async def event_from_server(self, event: Event) -> None:
        # Only check event types once
        is_run_satellite = False
        is_pause_satellite = False
        is_transcript = False
        is_error = False

        if RunSatellite.is_type(event.type):
            is_run_satellite = True
            self._is_paused = False

        elif PauseSatellite.is_type(event.type):
            is_pause_satellite = True
        elif Transcript.is_type(event.type):
            is_transcript = True
        elif Error.is_type(event.type):
            is_error = True

        if is_transcript or is_pause_satellite:
            # Stop streaming before event_from_server is called because it will play the "done" WAV.
            self.is_streaming = False

            # Stop debug recording (stt)
            if self.stt_audio_writer is not None:
                self.stt_audio_writer.stop()

        await super().event_from_server(event)

        if is_run_satellite or is_transcript or is_error or is_pause_satellite:
            # Stop streaming
            self.is_streaming = False

            if is_pause_satellite:
                self._is_paused = True
                _LOGGER.debug("Satellite is paused")
            else:
                # Go back to wake word detection
                await self.trigger_streaming_stop()

                # It's possible to be paused in the middle of streaming
                if not self._is_paused:
                    await self._send_wake_detect()
                    _LOGGER.info("Waiting for wake word")

                    # Start debug recording (wake)
                    self._debug_recording_timestamp = time.monotonic_ns()
                    if self.wake_audio_writer is not None:
                        self.wake_audio_writer.start(
                            timestamp=self._debug_recording_timestamp
                        )

    async def trigger_server_disconnected(self) -> None:
        await super().trigger_server_disconnected()

        self.is_streaming = False

        # Stop debug recording (stt)
        if self.stt_audio_writer is not None:
            self.stt_audio_writer.stop()

        await self.trigger_streaming_stop()

    async def event_from_mic(self, event: Event, audio_bytes: Optional[bytes] = None) -> None:
        if not AudioChunk.is_type(event.type) or self.microphone_muted or self._is_paused:
            return

        # Debug audio recording
        if self.wake_audio_writer is not None or self.stt_audio_writer is not None:
            if audio_bytes is None:
                chunk = AudioChunk.from_event(event)
                audio_bytes = chunk.audio

            if self.wake_audio_writer is not None:
           if self.wake_audio_writer is not None:
                self.wake_audio_writer.write(audio_bytes)

            if self.stt_audio_writer is not None:
                self.stt_audio_writer.write(audio_bytes)

        if self.is_streaming:
            # Forward to server
            await self.event_to_server(event)
        else:
            # Forward to wake word service
            await self.event_to_wake(event)

    async def event_from_wake(self, event: Event) -> None:
        if Info.is_type(event.type):
            self._wake_info = Info.from_event(event)
            self._wake_info_ready.set()
            return

        if self.is_streaming or self.server_id is None:
            # Not detecting or no server connected
            return

        if Detection.is_type(event.type):
            detection = Detection.from_event(event)

            # Check refractory period to avoid multiple back-to-back detections
            refractory_timestamp = self.refractory_timestamp.get(detection.name)
            if refractory_timestamp is not None and refractory_timestamp > time.monotonic():
                _LOGGER.debug("Wake word detection occurred during refractory period")
                return

            # Stop debug recording (wake)
            if self.wake_audio_writer is not None:
                self.wake_audio_writer.stop()

            # Start debug recording (stt)
            if self.stt_audio_writer is not None:
                self.stt_audio_writer.start(timestamp=self._debug_recording_timestamp)

            _LOGGER.debug(detection)

            self.is_streaming = True
            _LOGGER.debug("Streaming audio")

            if self.settings.wake.refractory_seconds is not None:
                # Another detection may not occur for this wake word until refractory period is over.
                self.refractory_timestamp[detection.name] = time.monotonic() + self.settings.wake.refractory_seconds
            else:
                # No refractory period
                self.refractory_timestamp.pop(detection.name, None)

            # Forward to the server
            await self.event_to_server(event)

            # Match detected wake word name with pipeline name
            pipeline_name: Optional[str] = None
            if self.settings.wake.names:
                detection_name = normalize_wake_word(detection.name)
                for wake_name in self.settings.wake.names:
                    if normalize_wake_word(wake_name.name) == detection_name:
                        pipeline_name = wake_name.pipeline
                        break

            await self._send_run_pipeline(pipeline_name=pipeline_name)
            await self.forward_event(event)  # forward to event service
            await self.trigger_detection(detection)
            await self.trigger_streaming_start()

    async def update_info(self, info: Info) -> None:
        self._wake_info = None
        self._wake_info_ready.clear()
        await self.event_to_wake(Describe().event())

        try:
            await asyncio.wait_for(
                self._wake_info_ready.wait(), timeout=_WAKE_INFO_TIMEOUT
            )

            if self._wake_info is not None:
                # Update wake info only
                info.wake = self._wake_info.wake
        except asyncio.TimeoutError:
            _LOGGER.warning("Failed to get info from wake service")