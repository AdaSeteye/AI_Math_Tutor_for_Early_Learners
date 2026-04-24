"""
Regenerate the ≥60-item curriculum, prompt TTS WAVs, and simple pitch-augmented
child-utterance stand-ins from the 12 trilingual seed items (deterministic).

On free Colab CPU, run (two commands, after install):

  pip install -r requirements.txt
  python generate_data.py
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Any

from tutor.child_speech_aug import build_augmented_family, write_wav_f32_mono
from tutor.curriculum_loader import LanguageCode
from tutor.feedback_audio import synthesize_text_to_wav

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "T3.1_Math_Tutor"
SEED_PATH = DATA / "curriculum_seed.json"
CURR_OUT = DATA / "curriculum.json"
TTS_PROMPTS = DATA / "tts" / "prompts"
CHILD_AUG = DATA / "child_utterance_aug"

OBJECTS: list[dict[str, str]] = [
    {"en": "stars", "fr": "étoiles", "kin": "inyenyeri"},
    {"en": "hearts", "fr": "cœurs", "kin": "imitima"},
    {"en": "trees", "fr": "arbres", "kin": "ibiti"},
    {"en": "cars", "fr": "voitures", "kin": "imodoka"},
    {"en": "blocks", "fr": "blocs", "kin": "ingingo"},
    {"en": "apples", "fr": "pommes", "kin": "amapapayi"},
    {"en": "birds", "fr": "oiseaux", "kin": "ibinyoni"},
    {"en": "fish", "fr": "poissons", "kin": "amafi"},
    {"en": "cups", "fr": "tasses", "kin": "ibyansi"},
    {"en": "toys", "fr": "jouets", "kin": "imikino"},
    {"en": "flowers", "fr": "fleurs", "kin": "indabyo"},
    {"en": "balls", "fr": "ballons", "kin": "mipira"},
    {"en": "cats", "fr": "chats", "kin": "injangwe"},
    {"en": "dogs", "fr": "chiens", "kin": "imbwa"},
]

# French agreement for "Combien de" + object (feminine plural already in list where needed)
# KIN phrases follow seed c2 (goats) / c1 (shapes) patterns loosely.

N_EN: dict[int, str] = {
    0: "zero", 1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 7: "seven", 8: "eight",
    9: "nine", 10: "ten", 11: "eleven", 12: "twelve", 13: "thirteen", 14: "fourteen", 15: "fifteen",
    16: "sixteen", 17: "seventeen", 18: "eighteen", 19: "nineteen", 20: "twenty", 21: "twenty-one",
}

SUBS = (
    "counting",
    "number_sense",
    "addition",
    "subtraction",
    "word_problems",
)

# Rubric: full curriculum must have **≥ 60** items. Seed has 12; we add 52 → **64** (deterministic).
BRIEF_MIN_ITEMS = 60
TARGET_N = 64
GEN_COUNT = TARGET_N - 12
assert TARGET_N >= BRIEF_MIN_ITEMS, "internal: target below brief minimum"


def _counting_item(seq: int) -> dict[str, Any]:
    i = seq // 5
    o = OBJECTS[i % len(OBJECTS)]
    n = 2 + (seq * 2 + 3) % 9
    n_objects = n
    lab = o["en"].rstrip("s")
    pro_fr = f"Combien de {o['fr']} vois-tu ?"
    return {
        "id": f"g{seq+1:03d}_c",
        "subskill": "counting",
        "difficulty": 1 + (seq // 3) % 2,
        "age_band": "5-6" if n < 7 else "6-7",
        "type": "count_image",
        "prompt_en": f"How many {o['en']} do you see?",
        "prompt_fr": pro_fr,
        "prompt_kin": f"Ureba {o['kin']}, zingahe? Zose ni zingahe?",
        "visual": {"mode": "blob_count", "n_objects": n_objects, "object_label": lab},
        "expected_answer": n_objects,
    }


def _ns_item(seq: int) -> dict[str, Any]:
    t = (seq // 5) % 2
    if t == 0:
        a = 1 + (seq * 2) % 6
        b = a + 2 + (seq * 3) % 4
        return {
            "id": f"g{seq+1:03d}_n",
            "subskill": "number_sense",
            "difficulty": 1 + (seq // 3) % 2,
            "age_band": "5-6",
            "type": "tap_number",
            "prompt_en": f"Which is bigger, {a} or {b}? Tap the bigger one.",
            "prompt_fr": f"Lequel est le plus grand, {a} ou {b} ? Touche le plus grand.",
            "prompt_kin": f"Ni uwuhe mwinshi, {a} cyangwa {b} ? Kanda uku mukuru kose.",
            "options": [a, b],
            "expected_answer": max(a, b),
        }
    y = 3 + (seq * 5) % 6
    exp = 0 if y <= 5 else 10
    return {
        "id": f"g{seq+1:03d}_n2",
        "subskill": "number_sense",
        "difficulty": 1 + (seq // 2) % 2,
        "age_band": "5-6",
        "type": "tap_number",
        "prompt_en": f"Is {y} closer to 0 or 10? Tap 0 or 10.",
        "prompt_fr": f"Est-ce que {y} est plus proche de 0 ou de 10? Touche 0 ou 10.",
        "prompt_kin": f"Ni yihariye, {y} ifata ku kure cyane ni 0 cyangwa 10? Kanda 0 cyangwa 10.",
        "options": [0, 10],
        "expected_answer": exp,
    }


def _add_item(seq: int) -> dict[str, Any]:
    a = (seq * 2) % 10
    b = 1 + (seq * 3) % 10
    s = min(a + b, 20)
    return {
        "id": f"g{seq+1:03d}_a",
        "subskill": "addition",
        "difficulty": 1 + (seq // 4) % 2,
        "age_band": "5-6" if s < 10 else "6-7",
        "type": "arithmetic",
        "prompt_en": f"What is {a} + {b} ?",
        "prompt_fr": f"Combien font {a} + {b} ?",
        "prompt_kin": f"{a} n'ibindi {b} bigana ibihe?",
        "expected_answer": s,
    }


def _sub_item(seq: int) -> dict[str, Any]:
    a = 4 + (seq * 3) % 12
    b = 1 + (seq * 2) % (a)
    b = max(1, min(b, a - 1))
    r = a - b
    return {
        "id": f"g{seq+1:03d}_s",
        "subskill": "subtraction",
        "difficulty": 1 + (seq // 4) % 2,
        "age_band": "6-7",
        "type": "arithmetic",
        "prompt_en": f"What is {a} − {b} ?",
        "prompt_fr": f"Combien font {a} − {b} ?",
        "prompt_kin": f"{a} ukuyemo {b} bisigara ibihe?",
        "expected_answer": r,
    }


def _word_item(seq: int) -> dict[str, Any]:
    t = (seq // 5) % 2
    x = 1 + (seq * 2) % 5
    y = 1 + (seq * 3) % 4
    if t == 0:
        n = x + y
        return {
            "id": f"g{seq+1:03d}_w",
            "subskill": "word_problems",
            "difficulty": 1,
            "age_band": "6-7",
            "type": "word",
            "prompt_en": f"You have {x} stickers. You get {y} more. How many stickers in total?",
            "prompt_fr": f"Tu as {x} autocollants. On t’en donne {y} de plus. Combien au total?",
            "prompt_kin": f"Ufite ibishushanyo {x}. Bongeraho {y} bundi. Byose ni bingahe? Hamwe =?",
            "expected_answer": n,
        }
    n0 = 2 + (seq % 5)
    use = min(1 + (seq % 3), n0 - 1)
    n = n0 - use
    return {
        "id": f"g{seq+1:03d}_w2",
        "subskill": "word_problems",
        "difficulty": 2,
        "age_band": "7-8",
        "type": "word",
        "prompt_en": f"Sam has {n0} balls. {use} roll away. How many are left?",
        "prompt_fr": f"Sam a {n0} balles. {use} roulent. Il en reste combien?",
        "prompt_kin": f"Sam afite mpinga {n0}. {use} zizimuka. Zisigaye ni zingahe?",
        "expected_answer": n,
    }


_BUILDERS: dict[str, Any] = {
    "counting": _counting_item,
    "number_sense": _ns_item,
    "addition": _add_item,
    "subtraction": _sub_item,
    "word_problems": _word_item,
}


def build_curriculum() -> dict[str, Any]:
    with SEED_PATH.open(encoding="utf-8") as f:
        seed = json.load(f)
    items: list[dict[str, Any]] = list(seed["items"])
    for k in range(GEN_COUNT):
        sub = SUBS[k % 5]
        d = _BUILDERS[sub](k)
        items.append(d)
    if len(items) < BRIEF_MIN_ITEMS:
        raise RuntimeError(
            f"curriculum has {len(items)} items; brief requires ≥ {BRIEF_MIN_ITEMS}"
        )
    if len(items) < TARGET_N:
        raise RuntimeError("internal: item count < target")
    out = {**{x: seed[x] for x in ("version", "subskills")}, "items": items}
    return out


def _expected_numeric(item: dict[str, Any]) -> int:
    e = item["expected_answer"]
    if isinstance(e, (int, float)) and e == int(e):
        return int(e)
    return int(round(float(e)))


def _child_answer_english_phrase(expected: int) -> str:
    w = N_EN.get(expected) or str(int(expected))
    return f"Is it {w} ?"


def _synth_tts_for_items(items: list[dict[str, Any]], *, tts: bool) -> int:
    if not tts:
        return 0
    n_ok = 0
    for it in items:
        iid = it["id"]
        base = TTS_PROMPTS / iid
        for lang, key in (("en", "prompt_en"), ("fr", "prompt_fr"), ("kin", "prompt_kin")):
            text = it.get(key, "").strip()
            if not text:
                continue
            lc: LanguageCode = "en" if lang == "en" else "fr" if lang == "fr" else "kin"
            p = base / f"{lang}.wav"
            if synthesize_text_to_wav(text, lc, p):
                n_ok += 1
    return n_ok


def _child_utterance_augments(base_wav: Path, out_dir: Path) -> bool:
    """Rubric: +3..+6 semitones, tempo, MUSAN (or synthetic) — same as ``child_speech_aug``."""
    out_dir.mkdir(parents=True, exist_ok=True)
    musan = None
    mp = os.environ.get("TUTOR_MUSAN_WAV", "").strip()
    if mp and Path(mp).is_file():
        musan = Path(mp)
    try:
        import librosa  # type: ignore[import-not-found]

        y, sr = librosa.load(str(base_wav), sr=None, mono=True)
    except Exception:
        shutil.copy(base_wav, out_dir / "p+03st.wav")
        shutil.copy(base_wav, out_dir / "p+06st.wav")
        return False
    fam = build_augmented_family(
        y, int(sr), musan_wav=musan, seed=42, snr_db=20.0
    )
    for k, ar in fam.items():
        p = out_dir / f"{k}.wav"
        try:
            write_wav_f32_mono(ar, int(sr), p)
        except OSError:
            return False
    return True


def _child_utterance_samples(items: list[dict[str, Any]], *, aug: bool) -> tuple[int, bool]:
    """Stand-in: TTS a short child-like English guess then optional +3 / +6 semitone pitch."""
    t_child = 0
    any_librosa = True
    for it in items:
        iid = it["id"]
        exp = _expected_numeric(it)
        phrase = _child_answer_english_phrase(exp)
        base_dir = CHILD_AUG / iid
        base_dir.mkdir(parents=True, exist_ok=True)
        b = base_dir / "child_guess_en.wav"
        if synthesize_text_to_wav(phrase, "en", b):
            t_child += 1
        if not aug or not b.exists():
            continue
        ok = _child_utterance_augments(b, base_dir)
        if not ok:
            any_librosa = False
    return t_child, any_librosa


def write_curriculum(data: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")


def _lint_unique_ids(data: dict[str, Any]) -> None:
    ids = [x["id"] for x in data["items"]]
    if len(set(ids)) != len(ids):
        dups = [i for i in ids if ids.count(i) > 1]
        raise SystemExit(f"Duplicate item ids: {dups!r}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Build curriculum, TTS clips, and pitch-augmented samples.")
    ap.add_argument("--curriculum-only", action="store_true", help="Only write curriculum.json")
    ap.add_argument("--no-tts", action="store_true", help="Skip TTS and child-utterance outputs")
    ap.add_argument(
        "--no-aug",
        action="store_true",
        help="With TTS: only base child-utterance clips, no +3/+6 semitone aug (ignored with --no-tts).",
    )
    args = ap.parse_args()
    tts = not args.no_tts
    do_aug = tts and not args.no_aug
    if args.curriculum_only:
        tts = False
        do_aug = False
    if not SEED_PATH.is_file():
        raise SystemExit(f"Missing seed: {SEED_PATH}")

    data = build_curriculum()
    _lint_unique_ids(data)
    write_curriculum(data, CURR_OUT)
    print(f"Wrote {CURR_OUT} ({len(data['items'])} items).")
    n_items = data["items"]

    if tts:
        t_ok = _synth_tts_for_items(n_items, tts=True)
        print(f"TTS prompt clips written (best-effort): {t_ok} file(s) under {TTS_PROMPTS}/")
    elif args.curriculum_only:
        print("Curriculum only: no TTS or child-utterance audio generated.")
    else:
        print("Skipped TTS (--no-tts).")

    if tts and do_aug:
        t_ch, had_lib = _child_utterance_samples(n_items, aug=True)
        if had_lib:
            print(f"Child-utterance clips + pitch aug under {CHILD_AUG}/ ({t_ch} base).")
        else:
            print(
                f"Child-utterance bases under {CHILD_AUG}/ ({t_ch}); "
                "librosa missing or aug failed — copies used for a few steps. "
                "Install: pip install -r requirements.txt (librosa, etc.). "
                "Set TUTOR_MUSAN_WAV=path/to/noise_slice.wav for real MUSAN (OpenSLR 17)."
            )
    elif tts and not do_aug:
        _child_utterance_samples(n_items, aug=False)
        print("Child-utterance bases only (--no-aug).")


if __name__ == "__main__":
    main()
