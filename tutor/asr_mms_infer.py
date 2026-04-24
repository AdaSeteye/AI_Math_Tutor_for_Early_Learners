"""
**facebook/mms-1b-all** ASR (huge; **excluded** from 75 MB footprint).

**Enable:** set ``TUTOR_MMS_ASR=1`` and install
``transformers`` + ``torch`` + ``librosa`` (see root ``requirements.txt``).
Use a GPU in Colab when possible; CPU is minutes per clip.
"""

from __future__ import annotations

import os
import warnings
from typing import Any

import numpy as np

MMS_FLORES: dict[str | None, str] = {
    None: "eng",
    "en": "eng",
    "fr": "fra",
    "kin": "kin",
    "rw": "kin",
}

_cache: dict[str, Any] = {}


def _as_bool(v: str | None) -> bool:
    return (v or "").lower() in ("1", "true", "yes", "on")


def _resample_16k(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    if int(sample_rate) == 16_000:
        a = np.asarray(audio, dtype=np.float32, order="C")
        if a.ndim > 1:
            a = a.mean(axis=1)
        return a
    import librosa  # type: ignore[import-not-found]

    y = np.asarray(audio, dtype=np.float32)
    if y.ndim > 1:
        y = y.mean(axis=1)
    return librosa.resample(
        y, orig_sr=int(sample_rate), target_sr=16_000, res_type="kaiser_fast"
    )


def _load_mms(
    model_id: str, tlang: str
) -> tuple[Any, Any, str]:
    """(processor, model, device) — reload if model or target language changes."""
    import torch  # type: ignore[import-not-found]
    from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor  # type: ignore[import-not-found]

    key = f"{model_id}|{tlang}"
    if _cache.get("k") == key and _cache.get("proc") is not None:
        return _cache["proc"], _cache["mdl"], str(_cache["dev"])

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        proc = Wav2Vec2Processor.from_pretrained(model_id, trust_remote_code=True)
        mdl = Wav2Vec2ForCTC.from_pretrained(model_id, trust_remote_code=True)
    mdl = mdl.to(dev)  # type: ignore[assignment]
    if hasattr(proc, "tokenizer") and hasattr(proc.tokenizer, "set_target_lang"):
        try:
            proc.tokenizer.set_target_lang(tlang)
        except (TypeError, ValueError, OSError) as e:  # pragma: no cover
            warnings.warn(f"set_target_lang: {e}", UserWarning, stacklevel=1)
    if hasattr(mdl, "load_adapter"):
        try:
            mdl.load_adapter(tlang)  # type: ignore[operator]
        except (OSError, TypeError, ValueError, RuntimeError) as e:  # pragma: no cover
            warnings.warn(f"load_adapter: {e}", UserWarning, stacklevel=1)
    mdl.eval()
    _cache["k"] = key
    _cache["proc"] = proc
    _cache["mdl"] = mdl
    _cache["dev"] = dev
    return proc, mdl, dev


def transcribe_mms_array(
    audio: np.ndarray,
    sample_rate: int,
    *,
    model_id: str = "facebook/mms-1b-all",
    language: str | None = None,
) -> str:
    if not _as_bool(os.environ.get("TUTOR_MMS_ASR", "")):
        raise OSError("Set TUTOR_MMS_ASR=1 to enable facebook/mms-1b-all ASR (optional).")
    tlang = MMS_FLORES.get(
        (language or "").lower() if language else None, MMS_FLORES[None]
    )
    y = _resample_16k(np.asarray(audio, dtype=np.float32), int(sample_rate))
    y = np.clip(y, -1.0, 1.0)
    if y.size < 200:
        return ""

    import torch  # type: ignore[import-not-found]

    proc, mdl, dev = _load_mms(model_id, tlang)
    with torch.inference_mode():
        raw: Any = proc(  # type: ignore[operator]
            y,
            sampling_rate=16_000,
            return_tensors="pt",
            padding="longest",
        )
        d = {k: v.to(dev) for k, v in dict(raw).items()}  # type: ignore[arg-type, call-overload, misc]
        out = mdl(**d)  # type: ignore[operator, arg-type, misc]
        logits = out.logits  # type: ignore[union-attr]
        pred = torch.argmax(logits, dim=-1)
    # CTC: ids → string (MMS may use custom tokenizer)
    p0 = pred[0].cpu().numpy().tolist()
    if hasattr(proc, "batch_decode"):
        try:
            tlist: list[str] = proc.batch_decode(pred)  # type: ignore[union-attr, assignment, misc]
            s = tlist[0] if tlist else ""
        except (AttributeError, TypeError, RuntimeError):
            s = str(proc.tokenizer.decode(p0))  # type: ignore[union-attr]
    else:
        s = str(proc.tokenizer.decode(p0, skip_special_tokens=True))  # type: ignore[union-attr]
    return s.strip() if s else ""
