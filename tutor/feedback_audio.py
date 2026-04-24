"""Spoken feedback in the learner's language: offline path first (pyttsx3 / espeak), then fallbacks."""

from __future__ import annotations

import re
import shutil
import struct
import subprocess
import tempfile
import wave
from pathlib import Path
from typing import Literal

from tutor.curriculum_loader import LanguageCode

Kind = Literal["correct", "encourage"]


def _sine_wav(
    path: Path,
    duration_s: float = 0.25,
    freq: float = 520.0,
    sample_rate: int = 16000,
) -> None:
    n = int(sample_rate * duration_s)
    with wave.open(str(path), "w") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        for i in range(n):
            v = int(32767 * 0.2 * (1 if (i * freq * 2 / sample_rate) % 2 < 0.5 else -1))
            w.writeframes(struct.pack("<h", v))


def _message_for(lang: LanguageCode, kind: Kind) -> str:
    if kind == "correct":
        if lang == "fr":
            return "Très bien, c'est juste."
        if lang == "kin":
            return "Yego! Ni byiza!"
        return "Great job, that's right."
    if lang == "fr":
        return "Essaie encore, tu y es presque."
    if lang == "kin":
        return "Ongera ugerageze."
    return "Try again, you can do it."


def synthesize_feedback_wav(
    lang: LanguageCode,
    kind: Kind,
    extra_tts: str | None = None,
) -> str:
    """TTS a short line; *extra_tts* = Task 4 mix (L2 number gloss) appended in same clip."""
    text = _message_for(lang, kind)
    if extra_tts and extra_tts.strip():
        text = f"{text} {extra_tts.strip()}"
    return _synth_to_wav_path(text, lang, kind)


def _synth_to_wav_path(text: str, lang: LanguageCode, kind: Kind) -> str:
    # Rubric: prefer Piper / Coqui (see tts_backends) before pyttsx3 / espeak.
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp0:
        brief = Path(tmp0.name)
    try:
        from tutor.tts_backends import try_synthesize_piper_or_coqui

        if try_synthesize_piper_or_coqui(text, brief) and brief.is_file() and brief.stat().st_size > 80:
            return str(brief)
    except Exception:
        pass
    brief.unlink(missing_ok=True)

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        out = Path(tmp.name)
    try:
        import pyttsx3  # type: ignore

        engine = pyttsx3.init()
        engine.save_to_file(text, str(out))
        engine.runAndWait()
        if out.exists() and out.stat().st_size > 100:
            return str(out)
    except Exception:
        if out.exists():
            out.unlink(missing_ok=True)

    espeak = shutil.which("espeak") or shutil.which("espeak-ng")
    if espeak:
        voice = "en" if lang != "fr" else "fr"
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp2:
            out2 = Path(tmp2.name)
        try:
            subprocess.run(
                [espeak, "-v", voice, "-s", "140", text, "-w", str(out2)],
                check=True,
                capture_output=True,
                timeout=30,
            )
            if out2.exists() and out2.stat().st_size > 100:
                return str(out2)
        except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
            out2.unlink(missing_ok=True)

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp3:
        p = Path(tmp3.name)
    _sine_wav(p, duration_s=0.2, freq=880.0 if kind == "correct" else 400.0)
    return str(p)


def synthesize_text_to_wav(
    text: str,
    lang: LanguageCode,
    out_wav: Path,
    *,
    kind: Kind = "encourage",
) -> bool:
    """
    Write TTS to *out_wav* (same engine order as ``synthesize_feedback_wav``).
    Kinyarwanda and other languages without an espeak voice use English voice for Latin text.
    Returns True if a non-trivial clip was written (heuristic; tone fallback is still valid WAV).
    """
    tmp = Path(_synth_to_wav_path(text, lang, kind))
    out_wav.parent.mkdir(parents=True, exist_ok=True)
    try:
        out_wav.write_bytes(tmp.read_bytes())
    finally:
        tmp.unlink(missing_ok=True)
    return out_wav.exists() and out_wav.stat().st_size > 80


def parse_spoken_number(user_text: str) -> int | None:
    """Parse EN/FR/KIN digit words, digits, small numbers (Task 1 + 4)."""
    if not user_text or not user_text.strip():
        return None
    t = user_text.strip().lower()
    m = re.search(r"-?\d+", t)
    if m:
        return int(m.group(0))
    rw_map = {
        "bumwe": 1,
        "rimwe": 1,
        "ebyiri": 2,
        "kabiri": 2,
        "eshatu": 3,
        "beshatu": 3,
        "kane": 4,
        "ine": 4,
        "itanu": 5,
        "gatanu": 5,
        "ntanatu": 4,
    }
    for w, n in rw_map.items():
        if w in t:
            return n
    words: dict[str, int] = {
        "zero": 0,
        "one": 1,
        "two": 2,
        "three": 3,
        "four": 4,
        "five": 5,
        "six": 6,
        "seven": 7,
        "eight": 8,
        "nine": 9,
        "ten": 10,
        "zéro": 0,
        "un": 1,
        "une": 1,
        "deux": 2,
        "trois": 3,
        "quatre": 4,
        "cinq": 5,
        "sept": 7,
        "huit": 8,
        "neuf": 9,
        "dix": 10,
    }
    for w, n in words.items():
        if w in t:
            return n
    return None
