"""
Download a *small* slice of real speech (Common Voice en/fr/rw + DigitalUmuganda
Kinyarwanda), resample (8 kHz for Kinyarwanda export if needed), then apply the
brief’s **pitch + tempo** aug and **MUSAN** (or synthetic) noise mix.

Does not run on every ``generate_data`` by default — long downloads. Use:

  pip install -r requirements.txt
  python scripts/child_speech_prepare.py --out data/T3.1_Math_Tutor/child_speech_rubric

Requires network, Hugging Face access, and ``huggingface-cli login`` for the dataset pulls.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tutor.child_speech_aug import (  # noqa: E402
    build_augmented_family,
    resample_to_sr,
    write_wav_f32_mono,
)


def _load_musan_path() -> Path | None:
    p = os.environ.get("TUTOR_MUSAN_WAV", "").strip()
    if p and Path(p).is_file():
        return Path(p)
    return None


def _parse_cv_age_filter(s: str) -> frozenset[str] | None:
    """
    Comma-separated Common Voice ``age`` field values to **keep** (rubric: child / youth band).
    A **trailing comma** includes the empty / unknown label (``""``), e.g. ``\"teens,\"`` →
    {``\"teens\"``, ``\"\"``}. CV rarely labels under-12; **teens** is the closest public band.

    Returns None when the string is empty after strip → **no** age filter (unfiltered; debug only).
    """
    if not s.strip():
        return None
    out: set[str] = set()
    for p in s.split(","):
        if p.strip():
            out.add(p.strip().lower())
        else:
            out.add("")
    return frozenset(out)


def _clip_from_common_voice(
    split: str,
    lang: str,
    clip_index: int,
    *,
    allowed_ages: frozenset[str] | None,
) -> tuple[np.ndarray, int, str, str] | None:
    """
    Return the *clip_index*-th **accepted** row (0-based). If ``allowed_ages`` is None, accept
    any age (unfiltered). Otherwise keep only rows whose ``age`` (normalized) is in the set.
    """
    from datasets import load_dataset  # type: ignore[import-not-found]

    ds = load_dataset(
        "mozilla-foundation/common_voice_11_0",
        lang,
        split=split,
        trust_remote_code=True,
        streaming=True,
    )
    n = 0
    for ex in ds:  # type: ignore[union-attr, operator]
        raw_age = ex.get("age")
        age_key = (str(raw_age).strip().lower() if raw_age is not None else "") or ""
        if allowed_ages is not None and age_key not in allowed_ages:
            continue
        if n < clip_index:
            n += 1
            continue
        aud = ex.get("audio")
        if not aud:
            continue
        arr = np.asarray(aud["array"], dtype=np.float32)
        sr0 = int(aud["sampling_rate"])
        sent = (ex.get("sentence") or "")[:200]
        age = age_key or "unknown"
        return arr, sr0, sent, age
    return None


def _clip_from_afrivoice(index: int) -> tuple[np.ndarray, int, str, str] | None:
    from datasets import load_dataset  # type: ignore[import-not-found]

    ds = load_dataset(
        "DigitalUmuganda/Afrivoice_Kinyarwanda",
        split="train",
        trust_remote_code=True,
        streaming=True,
    )
    n = 0
    for ex in ds:  # type: ignore[union-attr, operator]
        if n < index:
            n += 1
            continue
        aud = ex.get("audio")
        if not aud:
            return None
        arr = np.asarray(aud["array"], dtype=np.float32)
        sr0 = int(aud["sampling_rate"])
        txt = (ex.get("text") or ex.get("transcription") or "")[:200]
        return arr, sr0, txt, "afrivoice"
    return None


def main() -> None:
    try:
        import librosa  # noqa: F401
    except ImportError:
        print(
            "This script needs librosa (and soundfile). Run: pip install -r requirements.txt",
            file=sys.stderr,
        )
        raise SystemExit(2) from None
    ap = argparse.ArgumentParser(
        description="Child speech: HF clips + pitch/tempo + MUSAN (rubric data path)"
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=ROOT / "data" / "T3.1_Math_Tutor" / "child_speech_rubric",
        help="Output root for WAVs and manifest",
    )
    ap.add_argument(
        "--cv-samples",
        type=int,
        default=5,
        help="Clips per Common Voice language (en, fr, rw) (rubric: thin if too small)",
    )
    ap.add_argument(
        "--afri-samples",
        type=int,
        default=2,
        help="Clips from DigitalUmuganda/Afrivoice_Kinyarwanda",
    )
    ap.add_argument(
        "--cv-ages",
        type=str,
        default="teens,",
        help=(
            "Comma-separated CV `age` values to keep. Trailing comma adds empty/unknown. "
            "Default `teens,` = **teens** (closest child-youth band in CV) plus unknown. "
            "Stricter: `teens` (drop unknown). Empty string = no age filter (not rubric). "
            "Looser: add more values only if you trust metadata."
        ),
    )
    args = ap.parse_args()
    out: Path = args.out
    out.mkdir(parents=True, exist_ok=True)
    musan = _load_musan_path()
    cv_allowed: frozenset[str] | None = _parse_cv_age_filter(str(args.cv_ages))
    rows: list[dict[str, str]] = []
    key = 0

    for lang, tag in (("en", "cv_en"), ("fr", "cv_fr"), ("rw", "cv_rw")):
        for i in range(int(args.cv_samples)):
            try:
                ex = _clip_from_common_voice(
                    "train", lang, i, allowed_ages=cv_allowed
                )
            except OSError as e:
                print(
                    f"warn: could not open Common Voice split for {lang!r} ({e}). "
                    f"Log in: huggingface-cli login. Skipping {tag}_{i:02d}.",
                    file=sys.stderr,
                )
                continue
            if ex is None:
                filt = "unfiltered" if cv_allowed is None else f"age in {sorted(cv_allowed)!r}"
                print(
                    f"warn: no Common Voice {lang!r} clip {i} ({filt}).",
                    file=sys.stderr,
                )
                continue
            y, sr, sent, age = ex
            if lang == "rw":
                y, sr = resample_to_sr(y, sr, 8_000)
            d = build_augmented_family(
                y, sr, musan_wav=musan, seed=key + 17
            )
            sub = out / f"{tag}_{i:02d}"
            sub.mkdir(parents=True, exist_ok=True)
            for name, ar in d.items():
                p = sub / f"{name}.wav"
                write_wav_f32_mono(ar, sr, p)
            rows.append(
                {
                    "id": f"{tag}_{i:02d}",
                    "source": f"common_voice_11_0/{lang}",
                    "transcript": sent,
                    "age": age,
                }
            )
            key += 1
    for j in range(int(args.afri_samples)):
        try:
            ex = _clip_from_afrivoice(j)
        except OSError as e:
            print(
                f"warn: Afrivoice load failed ({e}). Is `datasets` installed? Network OK?",
                file=sys.stderr,
            )
            ex = None
        if ex is None:
            print("warn: no Afrivoice sample", file=sys.stderr)
            break
        y, sr, sent, _ = ex
        y8, sr8 = resample_to_sr(y, sr, 8_000)
        d = build_augmented_family(y8, sr8, musan_wav=musan, seed=key + 99)
        sub = out / f"dm_kin_{j:02d}"
        sub.mkdir(parents=True, exist_ok=True)
        for name, ar in d.items():
            p = sub / f"{name}.wav"
            write_wav_f32_mono(ar, sr8, p)
        rows.append(
            {
                "id": f"dm_kin_{j:02d}",
                "source": "DigitalUmuganda/Afrivoice_Kinyarwanda@8kHz",
                "transcript": sent,
                "age": "unknown",
            }
        )
        key += 1

    mpath = out / "manifest.csv"
    with mpath.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["id", "source", "transcript", "age"])
        w.writeheader()
        for r in rows:
            w.writerow(r)
    meta = {
        "rows": len(rows),
        "common_voice_age_filter": (
            "unfiltered"
            if cv_allowed is None
            else list(sorted(str(a) for a in cv_allowed))
        ),
        "noise_for_musan_suffix_clips": (
            f"file:{musan}"
            if musan
            else "synthetic_pinkish (see tutor/child_speech_aug._lowpass_pinkish_noise); not MUSAN until TUTOR_MUSAN_WAV set"
        ),
        "defaults_note": "Raised --cv-samples=5 and --afri-samples=2 for a less thin base; augs still multiply counts.",
        "note": "Point child_utt_sample_seed.csv at these dirs after localizing paths.",
    }
    (out / "prepare_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Wrote {len(rows)} source clip(s) under {out} — see manifest.csv.")


if __name__ == "__main__":
    main()
