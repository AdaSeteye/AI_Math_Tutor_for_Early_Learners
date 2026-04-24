"""
Brief-aligned TTS: **Piper** or **Coqui TTS** first, then the legacy on-device
stack in ``tutor/feedback_audio`` (pyttsx3 / espeak) as fallback.

Set ``TUTOR_PIPER_MODEL`` to a local ``.onnx`` and install Piper on ``PATH``:
https://github.com/rhasspy/piper
Coqui: install ``TTS`` and set ``TUTOR_COQUI_MODEL`` (see Coqui model zoo).
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from pathlib import Path


def _piper(text: str, out_wav: Path) -> bool:
    """Piper CLI: reads UTF-8 text on stdin, writes RIFF WAV to ``-f`` path."""
    exe = shutil.which("piper")
    m = os.environ.get("TUTOR_PIPER_MODEL", "").strip()
    if not exe or not m:
        return False
    p = Path(m)
    if not p.is_file():
        return False
    out_wav.parent.mkdir(parents=True, exist_ok=True)
    try:
        r = subprocess.run(
            [exe, "--model", str(p.resolve()), "-f", str(out_wav)],
            input=(text or "").encode("utf-8"),
            capture_output=True,
            timeout=120,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return r.returncode == 0 and out_wav.is_file() and out_wav.stat().st_size > 80


def _coqui_tts(text: str, out_wav: Path) -> bool:
    if not (text or "").strip():
        return False
    try:
        from TTS.api import TTS  # type: ignore[import-not-found]
    except Exception:
        return False
    out_wav.parent.mkdir(parents=True, exist_ok=True)
    name = os.environ.get(
        "TUTOR_COQUI_MODEL", "tts_models/en/ljspeech/tacotron2-DDC"
    )
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td) / "c.wav"
        try:
            tts = TTS(name)
            tts.tts_to_file(text=text, file_path=str(tmp))
        except Exception:
            return False
        if not tmp.is_file() or tmp.stat().st_size < 80:
            return False
        out_wav.write_bytes(tmp.read_bytes())
        return True


def try_synthesize_piper_or_coqui(text: str, out_wav: Path) -> str | None:
    """
    Returns ``"piper"`` / ``"coqui"`` if a WAV was written, else ``None``.

    Precedence: Piper (if TUTOR_PIPER_MODEL) → Coqui (if TTS is installed).
    """
    order = os.environ.get("TUTOR_TTS_PRIORITY", "piper,coqui")
    for token in [x.strip() for x in order.split(",") if x.strip()]:
        if token == "piper" and _piper(text, out_wav):
            return "piper"
        if token == "coqui" and _coqui_tts(text, out_wav):
            return "coqui"
    return None
