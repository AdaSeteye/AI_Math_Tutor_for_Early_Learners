"""
Augment *real* child/reader speech for ASR / data rubric: **pitch** (+3..+6 st),
**tempo** stretch, and **noise mix** for ``*_musan`` keys.

**Noise source (brief: MUSAN classroom noise, OpenSLR 17):**

- If ``musan_wav`` points to a real MUSAN (or any) ``.wav`` slice, that file is
  resampled/trimmed and mixed at ``snr_db`` (default 20 dB) with the speech.
- If ``musan_wav`` is ``None`` or missing, we mix **synthetic** band-limited
  pink-ish noise (``_lowpass_pinkish_noise``) at the same SNR — *not* MUSAN, but
  the same *mixing* contract so augs are reproducible without a 12 GB download.

This module is the shared engine for ``scripts/child_speech_prepare.py`` and
``generate_data.py`` (TTS child stand-ins).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

import numpy as np

PITCH_SEMI = (3, 4, 5, 6)
TEMPO_FACTORS: tuple[tuple[Literal["slow_05pct", "fast_10pct"], float], ...] = (
    ("slow_05pct", 0.95),
    ("fast_10pct", 1.05),
)


def resample_to_sr(y: np.ndarray, sr: int, target_sr: int) -> tuple[np.ndarray, int]:
    if target_sr == sr or len(y) < 2:
        return y.astype(np.float32, copy=False), sr
    t_old = np.arange(len(y), dtype=np.float64) / float(sr)
    t_new = np.arange(0.0, t_old[-1], 1.0 / target_sr, dtype=np.float64)
    y2 = np.interp(t_new, t_old, y.astype(np.float64))
    return y2.astype(np.float32), int(target_sr)


def _lowpass_pinkish_noise(
    n: int, sample_rate: int, rng: np.random.Generator
) -> np.ndarray:
    """A few biquads worth of 1/f-ish noise; cheap classroom-noise *stand-in* if MUSAN is absent."""
    x = rng.standard_normal(n).astype(np.float32)
    a = 0.97
    b = 0.03
    y = np.zeros_like(x)
    s = 0.0
    s2 = 0.0
    for i in range(n):
        s = a * s + b * x[i]
        s2 = 0.9 * s2 + 0.1 * s
        y[i] = s2
    m = float(np.max(np.abs(y)) + 1e-6)
    return 0.25 * (y / m).astype(np.float32)


def mix_speech_at_snr(
    clean: np.ndarray, noise: np.ndarray, snr_db: float = 20.0
) -> np.ndarray:
    """Energy-normalized mix (MUSAN-style SNR, scalar speech/noise)."""
    c = clean.astype(np.float32)
    n = noise.astype(np.float32)
    if n.shape[0] < c.shape[0]:
        if len(n) == 0:
            return c
        n = np.tile(n, (c.shape[0] // len(n) + 1,))[: c.shape[0]]
    elif n.shape[0] > c.shape[0]:
        n = n[: c.shape[0]]
    p_c = float(np.mean(c * c) + 1e-8)
    p_n = float(np.mean(n * n) + 1e-8)
    scale = 10.0 ** (-(snr_db) / 20.0) * (p_c / p_n) ** 0.5
    out = c + float(scale) * n
    m = float(np.max(np.abs(out)) + 1e-6)
    return (out * (0.99 / m)).astype(np.float32) if m > 1.0 else out


def build_augmented_family(
    y: np.ndarray,
    sr: int,
    *,
    musan_wav: Path | None = None,
    snr_db: float = 20.0,
    seed: int = 0,
) -> dict[str, np.ndarray]:
    """
    For one clip, return variants aligned with the **brief** (pitch, tempo, noise mix).

    Keys: ``"clean"``, pitch/tempo variants, and ``*_musan`` keys. The ``*_musan`` suffix
    means “speech + noise at ``snr_db``” — **either** a real MUSAN file **or** synthetic
    noise (see module docstring) when ``musan_wav`` is unset.
    """
    y = np.asarray(y, dtype=np.float32)
    y = y.reshape(-1)
    y = y / (float(np.max(np.abs(y))) + 1e-6)
    rng = np.random.default_rng(seed)
    out: dict[str, np.ndarray] = {"clean": y.copy()}

    try:
        import librosa  # type: ignore[import-not-found]
    except Exception as e:  # pragma: no cover
        raise RuntimeError("child_speech_aug requires librosa: pip install librosa") from e

    n_samples = y.shape[0]
    noise: np.ndarray | None = None
    if musan_wav and musan_wav.is_file():
        import soundfile as sf  # type: ignore[import-not-found]

        n2, sr2 = sf.read(musan_wav, always_2d=False)
        if n2.ndim > 1:
            n2 = n2[:, 0]
        n2 = n2.astype(np.float32, copy=False)
        n2, _ = resample_to_sr(n2, int(sr2), int(sr))
        if len(n2) < n_samples:
            n2 = np.tile(n2, (n_samples // len(n2) + 1,))[:n_samples]
        else:
            s0 = int(rng.integers(0, max(1, len(n2) - n_samples + 1)))
            n2 = n2[s0 : s0 + n_samples]
        if len(n2) < n_samples:
            n2 = np.pad(n2, (0, n_samples - len(n2)))
        else:
            n2 = n2[:n_samples]
        noise = n2
    else:
        base = int(os.environ.get("TUTOR_SYNTH_MUSAN_SEED", "0")) or seed
        noise = _lowpass_pinkish_noise(n_samples, int(sr), np.random.default_rng(base))

    assert noise is not None

    def _fit_noise(target_len: int) -> np.ndarray:
        if target_len < 1:
            return np.zeros(0, dtype=np.float32)
        return np.interp(
            np.linspace(0.0, 1.0, target_len, dtype=np.float64),
            np.linspace(0.0, 1.0, len(noise), dtype=np.float64),
            noise.astype(np.float64),
        ).astype(np.float32)

    for s in PITCH_SEMI:
        p = f"p+{s:02d}st"
        p_y = librosa.effects.pitch_shift(
            y, sr=sr, n_steps=float(s), n_fft=min(1024, max(2, n_samples // 4))
        )
        out[p] = p_y
        p_ns = _fit_noise(int(p_y.shape[0]))
        if p_ns.size == p_y.size:
            out[p + "_musan"] = mix_speech_at_snr(p_y, p_ns, snr_db=snr_db)

    for name, f in TEMPO_FACTORS:
        t_y = librosa.effects.time_stretch(y, rate=f)
        k = "tempo_" + name
        out[k] = t_y
        t_ns = _fit_noise(int(t_y.shape[0]))
        if t_ns.size == t_y.size:
            out[k + "_musan"] = mix_speech_at_snr(t_y, t_ns, snr_db=snr_db)

    return out


def write_wav_f32_mono(
    y: np.ndarray, sample_rate: int, path: Path
) -> None:
    """Write 16-bit mono WAV from ``[-1,1]`` float samples."""
    import wave

    y = np.clip(np.asarray(y, dtype=np.float64), -1.0, 1.0)
    i16 = (y * 32767.0).astype(np.int16)
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(int(sample_rate))
        w.writeframes(i16.tobytes())
