"""
video_audio.py
--------------
Generates synchronized voice alert audio tracks for processed videos
and muxes them directly into the final MP4 output using FFmpeg.

Author: CrowdAware AI Team
"""

import os
import wave
import logging
import subprocess
from pathlib import Path
from typing import List, Tuple
import numpy as np

logger = logging.getLogger(__name__)


def synthesize_tts_wav(text: str, wav_path: str) -> bool:
    """
    Synthesize speech text to a WAV file using Windows SAPI (pywin32)
    with pyttsx3 fallback.
    """
    # 1. Try Windows SAPI directly via COM
    try:
        import pythoncom
        pythoncom.CoInitialize()
        import win32com.client
        voice = win32com.client.Dispatch("SAPI.SpVoice")
        stream = win32com.client.Dispatch("SAPI.SpFileStream")
        stream.Open(str(wav_path), 3, False)  # 3 = SSFMCreateForWrite
        voice.AudioOutputStream = stream
        voice.Speak(text)
        stream.Close()
        voice.AudioOutputStream = None
        pythoncom.CoUninitialize()
        if os.path.exists(wav_path) and os.path.getsize(wav_path) > 100:
            return True
    except Exception as e:
        logger.debug(f"[VideoAudio] SAPI synthesis notice: {e}")

    # 2. Try pyttsx3 fallback
    try:
        import pyttsx3
        engine = pyttsx3.init()
        engine.save_to_file(text, str(wav_path))
        engine.runAndWait()
        if os.path.exists(wav_path) and os.path.getsize(wav_path) > 100:
            return True
    except Exception as e:
        logger.warning(f"[VideoAudio] pyttsx3 synthesis notice: {e}")

    return False


def build_video_soundtrack(
    alerts: List[Tuple[float, str]],
    total_duration: float,
    output_wav_path: str,
    sample_rate: int = 22050,
) -> bool:
    """
    Place multiple timed voice alerts along a continuous audio track.

    Args:
        alerts: List of (timestamp_seconds, alert_message)
        total_duration: Total video length in seconds
        output_wav_path: Destination .wav file
        sample_rate: Audio sample rate (22050 Hz standard for SAPI/pyttsx3)
    """
    total_duration = max(float(total_duration), 1.0)
    total_samples = int(np.ceil(total_duration * sample_rate))
    master_audio = np.zeros(total_samples, dtype=np.int16)

    out_p = Path(output_wav_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)

    current_end_time = 0.0

    for idx, (t, text) in enumerate(alerts):
        if not text or not text.strip():
            continue

        temp_wav = str(out_p.parent / f"_tmp_alert_{idx}.wav")
        success = synthesize_tts_wav(text, temp_wav)
        if not success:
            continue

        try:
            with wave.open(temp_wav, "rb") as w:
                n_frames = w.getnframes()
                sr = w.getframerate()
                chunk = np.frombuffer(w.readframes(n_frames), dtype=np.int16)
                dur = n_frames / float(sr)

            # Schedule alert: start at t, but if previous alert is still speaking,
            # wait until it finishes with a slight 0.2s pause so words remain clear.
            start_t = max(float(t), current_end_time)
            if start_t >= total_duration:
                continue

            start_idx = int(start_t * sample_rate)
            end_idx = min(total_samples, start_idx + len(chunk))
            master_audio[start_idx:end_idx] = chunk[: end_idx - start_idx]
            current_end_time = start_t + dur + 0.2
        except Exception as e:
            logger.warning(f"[VideoAudio] Error stitching alert audio: {e}")
        finally:
            if os.path.exists(temp_wav):
                try:
                    os.remove(temp_wav)
                except Exception:
                    pass

    # Save continuous master WAV
    with wave.open(output_wav_path, "wb") as w:
        w.setnchannels(1)       # Mono
        w.setsampwidth(2)       # 16-bit
        w.setframerate(sample_rate)
        w.writeframes(master_audio.tobytes())

    return os.path.exists(output_wav_path) and os.path.getsize(output_wav_path) > 100


def mux_audio_into_video(video_path: str, audio_path: str, output_path: str) -> bool:
    """
    Use FFmpeg to combine video and audio streams into a final MP4 file.
    """
    cmd = [
        "ffmpeg", "-y",
        "-i", str(video_path),
        "-i", str(audio_path),
        "-c:v", "copy",
        "-c:a", "aac",
        "-b:a", "128k",
        "-shortest",
        str(output_path),
    ]

    try:
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode == 0 and os.path.exists(output_path):
            return True
        else:
            logger.error(f"[VideoAudio] FFmpeg error: {res.stderr}")
            return False
    except Exception as e:
        logger.error(f"[VideoAudio] Failed to execute ffmpeg: {e}")
        return False
