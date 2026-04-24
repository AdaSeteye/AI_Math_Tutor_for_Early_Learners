"""
ASR aligned with the brief: **Whisper-tiny** (``faster_whisper`` “tiny”) and optional
**facebook/mms-1b-all** (set ``TUTOR_MMS_ASR=1``; needs ``torch`` + ``transformers`` from ``requirements.txt``).

**Runtime vs training (data vs mic).** Offline data uses ``child_speech_aug``: pitch *up* (+3..+6 st),
tempo, MUSAN. **Live mic** uses :func:`preprocess_child_mic_for_whisper` (or any ``transcribe_*``
here): **peak normalise** → **noisereduce** → **pitch *down*** (default -3 st; env
``TUTOR_CHILD_PITCH_STEPS``); we do *not* add MUSAN or tempo at inference. After ASR, use
:func:`transcribe_and_detect` to run :func:`tutor.lang_detect.detect_language` on the text for
TTS routing.

Install (optional): ``pip install -r requirements.txt`` (``faster-whisper``, ``librosa``, ``noisereduce``).
"""

from __future__ import annotations

import os
import tempfile
import wave
from pathlib import Path
from typing import Any

import numpy as np

from tutor.curriculum_loader import LanguageCode

# Brief reference IDs (for cards / fine-tunes — not all loaded by default)
WHISPER_TINY_HF = "openai/whisper-tiny"
MMS_1B_ALL_HF = "facebook/mms-1b-all"

# Child → adult pitch (semitones down; inverse of train-time +3..+6 st augs, conservative).
# Override: TUTOR_CHILD_PITCH_STEPS="0" to disable.
_DEFAULT_CHILD_PITCH_STEPS = -3.0


def _to_mono_f32(audio: np.ndarray) -> np.ndarray:
    y = np.asarray(audio, dtype=np.float32)
    if y.ndim > 1:
        y = y.mean(axis=1)
    return y.reshape(-1).astype(np.float32, copy=False)


def _peak_normalize(y: np.ndarray, peak: float = 0.99) -> np.ndarray:
    y = np.asarray(y, dtype=np.float32).reshape(-1)
    m = float(np.max(np.abs(y)) + 1e-8)
    if m < 1e-7:
        return y
    return (y * (peak / m)).astype(np.float32)


def _reduce_noise(audio: np.ndarray, sr: int) -> np.ndarray:
    """Light classroom-noise suppression; no-op if ``noisereduce`` is missing."""
    y = np.asarray(audio, dtype=np.float32)
    if y.ndim > 1:
        y = y.mean(axis=1)
    try:
        import noisereduce as nr  # type: ignore[import-not-found]

        return nr.reduce_noise(y=y, sr=int(sr)).astype(np.float32)
    except Exception:
        return y


def _adapt_child_audio(audio: np.ndarray, sr: int) -> np.ndarray:
    """Shift child voice down toward adult range before Whisper sees it."""
    y = np.asarray(audio, dtype=np.float32)
    if y.ndim > 1:
        y = y.mean(axis=1)
    raw = os.environ.get("TUTOR_CHILD_PITCH_STEPS", "").strip()
    if raw:
        try:
            n_steps = float(raw)
        except ValueError:
            n_steps = _DEFAULT_CHILD_PITCH_STEPS
    else:
        n_steps = _DEFAULT_CHILD_PITCH_STEPS
    if abs(n_steps) < 1e-6:
        return y
    try:
        import librosa  # type: ignore[import-not-found]

        n_fft = min(1024, max(2, len(y) // 4))
        return librosa.effects.pitch_shift(
            y, sr=int(sr), n_steps=n_steps, n_fft=n_fft
        ).astype(np.float32)
    except Exception:
        return y


def preprocess_child_mic_for_whisper(
    audio: np.ndarray, sample_rate: int
) -> tuple[np.ndarray, int]:
    """
    **Single entry** for live-mic adaptation before Whisper (or any ASR that expects the same
    front-end as ``transcribe_whisper_tiny_faster`` / ``transcribe_mms_1b_all``).

    **Chain:** stereo/mono float → **peak normalise** → **denoise** → **pitch toward adult** →
    **peak normalise** (stable level for the model). See module docstring for the mapping to
    train-time augs in ``tutor/child_speech_aug.py``.
    """
    sr = int(sample_rate)
    y = _to_mono_f32(audio)
    y = _peak_normalize(y)
    y = _reduce_noise(y, sr)
    y = _adapt_child_audio(y, sr)
    y = _peak_normalize(y)
    return y, sr


def _preprocess_for_asr(audio: np.ndarray, sample_rate: int) -> tuple[np.ndarray, int]:
    """Back-compat alias: same as :func:`preprocess_child_mic_for_whisper`."""
    return preprocess_child_mic_for_whisper(
        np.asarray(audio, dtype=np.float32), int(sample_rate)
    )


def _write_temp_wav(audio: np.ndarray, sample_rate: int) -> Path:
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    audio = np.clip(audio, -1.0, 1.0)
    i16 = (audio * 32767.0).astype(np.int16)
    fd, name = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    path = Path(name)
    try:
        with wave.open(str(path), "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(int(sample_rate))
            w.writeframes(i16.tobytes())
    except OSError:
        path.unlink(missing_ok=True)
        raise
    return path


_fw_by_compute: dict[str, object] = {}


def _whisper_tiny_model(compute_type: str = "int8") -> object:
    if compute_type not in _fw_by_compute:
        from faster_whisper import WhisperModel  # type: ignore[import-untyped]

        _fw_by_compute[compute_type] = WhisperModel(
            "tiny", device="cpu", compute_type=compute_type
        )
    return _fw_by_compute[compute_type]


def transcribe_whisper_tiny_faster(
    audio: np.ndarray,
    sample_rate: int,
    *,
    language: str | None = None,
    compute_type: str = "int8",
) -> str:
    """
    **openai/whisper-tiny**-class model via `faster-whisper` (CT2 build of tiny).
    `language`: ISO-639-1, e.g. ``"en"``, ``"fr"``; ``"rw"`` for Kinyarwanda if supported, or
    ``None`` for auto. Incoming audio is passed through :func:`preprocess_child_mic_for_whisper`
    (normalise, denoise, child→adult pitch) before the temp WAV + VAD.
    """
    a, sr = preprocess_child_mic_for_whisper(
        np.asarray(audio, dtype=np.float32), int(sample_rate)
    )
    model = _whisper_tiny_model(compute_type=compute_type)
    path = _write_temp_wav(a, sr)
    try:
        segs, _info = model.transcribe(
            str(path),
            language=language,
            vad_filter=True,
        )
        return " ".join(s.text for s in segs).strip()
    finally:
        path.unlink(missing_ok=True)


def transcribe_and_detect(
    audio: np.ndarray,
    sample_rate: int,
    *,
    language: str | None = None,
    compute_type: str = "int8",
) -> tuple[str, LanguageCode]:
    """
    1) :func:`transcribe_whisper_tiny_faster` (runs :func:`preprocess_child_mic_for_whisper` +
    VAD in faster-whisper). 2) :func:`tutor.lang_detect.detect_language` on the transcript for
    tutor TTS. Recommended when language may be mixed or unknown (``language=None``).
    """
    from tutor.lang_detect import detect_language

    text = transcribe_whisper_tiny_faster(
        np.asarray(audio, dtype=np.float32),
        int(sample_rate),
        language=language,
        compute_type=compute_type,
    )
    return text, detect_language(text)


def transcribe_gradio_audio(
    audio: Any,
    *,
    language: str | None = None,
    compute_type: str = "int8",
) -> str | None:
    """
    Gradio microphone: ``audio`` is ``(sample_rate, numpy_array)`` or ``None`` if silent.
    Runs the same :func:`preprocess_child_mic_for_whisper` chain as
    :func:`transcribe_whisper_tiny_faster`. For text **and** TTS :class:`LanguageCode`, use
    :func:`transcribe_gradio_and_detect`.
    """
    if audio is None:
        return None
    sr, data = audio[0], audio[1]
    if data is None or len(np.asarray(data)) == 0:
        return None
    return transcribe_whisper_tiny_faster(
        np.asarray(data, dtype=np.float32),
        int(sr),
        language=language,
        compute_type=compute_type,
    )


def transcribe_gradio_and_detect(
    audio: Any, *, compute_type: str = "int8"
) -> tuple[str | None, LanguageCode | None]:
    """
    Gradio **mic** + :func:`transcribe_and_detect` — preprocessed child-friendly audio, Whisper
    (with VAD), then :func:`tutor.lang_detect.detect_language`. If no audio, returns
    ``(None, None)``.
    """
    if audio is None:
        return None, None
    sr, data = audio[0], audio[1]
    if data is None or len(np.asarray(data)) == 0:
        return None, None
    t, code = transcribe_and_detect(
        np.asarray(data, dtype=np.float32), int(sr), language=None, compute_type=compute_type
    )
    return t, code


def transcribe_mms_1b_all(
    audio: np.ndarray,
    sample_rate: int,
    *,
    language: str | None = None,
) -> str:
    """
    **facebook/mms-1b-all** (optional, ~1B parameters — not in the 75 MB on-device cap).

    Set ``TUTOR_MMS_ASR=1`` and install ``transformers`` + ``torch`` + ``librosa``
    (see root ``requirements.txt``). Use a GPU for interactive use.
    Uses the same **preprocess** (denoise + child pitch) as the Whisper path.
    """
    if not (os.environ.get("TUTOR_MMS_ASR", "") or "").lower() in (
        "1",
        "true",
        "yes",
        "on",
    ):
        raise NotImplementedError(
            f"Enable {MMS_1B_ALL_HF} with TUTOR_MMS_ASR=1; install optional deps. "
            "Or use ``transcribe_whisper_tiny_faster`` for the default tiny model."
        )
    a, sr = preprocess_child_mic_for_whisper(
        np.asarray(audio, dtype=np.float32), int(sample_rate)
    )
    from tutor.asr_mms_infer import transcribe_mms_array

    return transcribe_mms_array(a, sr, language=language)


def child_speech_data_recipe() -> str:
    """One-string pointer for your process log / report (train + inference)."""
    return (
        "Train-time: Mozilla Common Voice (en, fr, rw — child age filters) + "
        "DigitalUmuganda Kinyarwanda 8 kHz; augment with +3..+6 semitone pitch shift, "
        "tempo jitter, and MUSAN noise; then fine-tune Whisper-tiny or MMS head. "
        "Inference: preprocess_child_mic_for_whisper in asr_adapt (norm, noisereduce, "
        "pitch down) + transcribe_and_detect for lang_detect."
    )
