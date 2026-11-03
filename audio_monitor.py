"""Audio monitoring module for ProctorAI.

Captures real-time audio from the microphone and estimates RMS energy
to detect speech, whispering, or unauthorized ambient noise during an exam.
Gracefully handles missing audio hardware and permission errors.
"""

import math
import threading
import time
from typing import Optional, Tuple

import numpy as np

try:
    import sounddevice as sd
    SOUNDDEVICE_AVAILABLE = True
except Exception:
    SOUNDDEVICE_AVAILABLE = False


class AudioProctor:
    """Monitors microphone stream for suspicious acoustic activity."""

    def __init__(
        self,
        sample_rate: int = 16000,
        chunk_duration: float = 0.5,
        default_threshold: float = 0.04,
        sustained_seconds: float = 1.0,
    ) -> None:
        self.sample_rate = sample_rate
        self.chunk_size = int(sample_rate * chunk_duration)
        self.threshold = default_threshold
        self.sustained_seconds = sustained_seconds

        self.stream: Optional[object] = None
        self.is_running = False
        self.lock = threading.Lock()

        # State tracking
        self.current_rms: float = 0.0
        self.peak_rms: float = 0.0
        self.speech_start_time: Optional[float] = None
        self.is_speech_flagged: bool = False
        self.hardware_available: bool = False

        self._check_hardware()

    def _check_hardware(self) -> None:
        """Verifies if an audio input device is available."""
        if not SOUNDDEVICE_AVAILABLE:
            self.hardware_available = False
            return
        try:
            devices = sd.query_devices()
            input_devices = [d for d in devices if d.get("max_input_channels", 0) > 0]
            self.hardware_available = len(input_devices) > 0
        except Exception:
            self.hardware_available = False

    def _audio_callback(self, indata, frames, time_info, status):
        """Processes each incoming audio buffer."""
        if status:
            pass  # Overflow or underflow

        # Compute Root Mean Square (RMS) amplitude
        audio_data = indata[:, 0] if indata.ndim > 1 else indata
        rms = np.sqrt(np.mean(np.square(audio_data.astype(np.float32))))

        with self.lock:
            self.current_rms = float(rms)
            if self.current_rms > self.peak_rms:
                self.peak_rms = self.current_rms

            now = time.time()
            if self.current_rms >= self.threshold:
                if self.speech_start_time is None:
                    self.speech_start_time = now
                elif (now - self.speech_start_time) >= self.sustained_seconds:
                    self.is_speech_flagged = True
            else:
                self.speech_start_time = None
                self.is_speech_flagged = False

    def start(self) -> bool:
        """Starts the audio capture stream."""
        if not self.hardware_available:
            return False

        if self.is_running:
            return True

        try:
            self.stream = sd.InputStream(
                samplerate=self.sample_rate,
                blocksize=self.chunk_size,
                channels=1,
                dtype="float32",
                callback=self._audio_callback,
            )
            self.stream.start()
            self.is_running = True
            return True
        except Exception:
            self.is_running = False
            self.stream = None
            return False

    def stop(self) -> None:
        """Stops and cleans up the audio capture stream."""
        self.is_running = False
        if self.stream is not None:
            try:
                self.stream.stop()
                self.stream.close()
            except Exception:
                pass
            self.stream = None
        self.speech_start_time = None
        self.is_speech_flagged = False

    def check_status(self) -> Tuple[bool, float, bool]:
        """Returns current acoustic integrity status.

        Returns:
            Tuple of (is_flagged, current_rms, hardware_available).
        """
        with self.lock:
            return self.is_speech_flagged, self.current_rms, self.hardware_available

