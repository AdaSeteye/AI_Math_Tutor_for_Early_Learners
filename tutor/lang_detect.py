"""
Task 4: KIN / FR / EN + code-mix (offline). Keyword heuristics + ``langdetect`` when available.
Tutor = dominant. Append a short L2 number gloss in TTS whenever a **non-dominant** number
word appears, not only when the utterance is labeled *mix*.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

from tutor.curriculum_loader import LanguageCode

Label = Literal["en", "fr", "kin", "mix"]

KIN_KEYWORDS: frozenset[str] = frozenset(
    "ubona kanda zingahe ibihe buri uri oya wongera byiza ari"  # noqa: E501
    .split()
) | frozenset("eshatu ebyiri bumwe rimwe itanu ntandatu inane icumi".split())
FR_KEYWORDS: frozenset[str] = frozenset(
    "combien c'est moins plus grand touché dix un une deux trois quatre"  # noqa: E501
    .split()
)
EN_KEYWORDS: frozenset[str] = frozenset(
    "what how many the bigger smaller dots apples sticks"  # noqa: E501
    .split()
)
KIN_NUM = re.compile(
    r"\b(eshatu|beshatu|ebyiri|bumwe|rimwe|kane|ine|itanu|gatanu|tandatu|inani|icumi)\b",
    re.I,
)
EN_NUM = re.compile(
    r"\b(zero|one|two|three|four|five|six|seven|eight|nine|ten)\b", re.I
)
FR_NUM = re.compile(
    r"\b(z[ée]ro|un|deux|trois|quatre|cinq|six|sept|huit|neuf|dix|une)\b", re.I
)


@dataclass(frozen=True)
class ChildLanguageProfile:
    label: Label
    dominant: LanguageCode
    scores: dict[LanguageCode, float]
    l2_numeral_appendix: str
    child_summary: str


def _ld_probs(text: str) -> dict[LanguageCode, float]:
    p: dict[LanguageCode, float] = {"en": 0.0, "fr": 0.0, "kin": 0.0}
    t = (text or "").strip()
    if len(t) < 3:
        return p
    try:
        from langdetect import detect_langs
    except ImportError:
        return p
    for x in detect_langs(t):
        if x.lang == "en":
            p["en"] = max(p["en"], x.prob)
        elif x.lang == "fr":
            p["fr"] = max(p["fr"], x.prob)
        elif x.lang in ("rw", "rn"):
            p["kin"] = max(p["kin"], 0.55 * x.prob)
    return p


def _lex_score(toks: set[str], voc: frozenset[str]) -> float:
    h = toks & voc
    return min(0.5, 0.15 * max(1, len(h)) / 1.0)


def _kin_boost(t: str) -> float:
    t = t.lower()
    b = 0.0
    if KIN_NUM.search(t):
        b += 0.38
    return b


def _l2_numeral_appendix(text: str, dominant: LanguageCode) -> str:
    t = (text or "").lower()
    parts: list[str] = []
    if KIN_NUM.search(t) and dominant != "kin":
        m = KIN_NUM.search(t)
        w = m.group(0) if m else "eshatu"
        if dominant == "en":
            parts.append(f"Kinyarwanda number word: {w}.")
        if dominant == "fr":
            parts.append(f"Kinyarwanda (votre nombre) : {w}.")

    if EN_NUM.search(t) and dominant == "kin":
        m = EN_NUM.search(t)
        w = m.group(0) if m else "three"
        parts.append(f"English number you used: {w}.")

    if EN_NUM.search(t) and dominant == "fr":
        parts.append("I hear English in your number words; OK.")

    if FR_NUM.search(t) and dominant in ("en", "kin"):
        m = FR_NUM.search(t)
        w = m.group(0) if m else "trois"
        parts.append(f"French number word: {w}.")

    if not parts:
        return ""
    return " ".join(parts)[: 350]


def detect_language(user_text: str) -> LanguageCode:
    """
    Dominant tutor language (``en`` / ``fr`` / ``kin``) from transcribed or typed child text.
    Use after ASR when the spoken language is unknown (e.g. code-switched); aligns with
    :func:`detect_child_utterance` for routing TTS, without returning the full profile.
    """
    return detect_child_utterance(user_text).dominant


def detect_child_utterance(user_text: str) -> ChildLanguageProfile:
    raw = (user_text or "").strip()
    if not raw:
        return ChildLanguageProfile(
            label="en",
            dominant="en",
            scores={"en": 0.0, "fr": 0.0, "kin": 0.0},
            l2_numeral_appendix="",
            child_summary="(empty input)",
        )

    toks = {w for w in re.split(r"[^\w']+", raw.lower()) if w}
    ld = _ld_probs(raw)
    s_kin = _lex_score(toks, KIN_KEYWORDS) + _kin_boost(raw) + 0.5 * ld["kin"]
    s_fr = _lex_score(toks, FR_KEYWORDS) + 0.6 * ld["fr"]
    s_en = _lex_score(toks, EN_KEYWORDS) + 0.6 * ld["en"]

    scores: dict[LanguageCode, float] = {
        "en": min(1.0, s_en),
        "fr": min(1.0, s_fr),
        "kin": min(1.0, s_kin),
    }

    strong = sum(1 for v in scores.values() if v > 0.2)
    numl: set[LanguageCode] = set()
    if KIN_NUM.search(raw):
        numl.add("kin")
    if EN_NUM.search(raw):
        numl.add("en")
    if FR_NUM.search(raw):
        numl.add("fr")
    is_mix = (strong >= 2) or (len(numl) > 1)

    if is_mix:
        label = "mix"
    else:
        if max(scores.values()) < 0.1 and len(raw) > 1:
            label = "en"  # type: ignore[assignment, arg-type]
        else:
            label = max(scores, key=scores.__getitem__)  # type: ignore[assignment, arg-type]
    if is_mix:
        dominant = max(scores, key=scores.__getitem__)  # type: ignore[assignment, arg-type]
    else:
        dominant = max(scores, key=scores.__getitem__)  # type: ignore[assignment, arg-type]
        if max(scores.values()) < 0.1 and len(raw) > 1:
            dominant = "en"
    apx = _l2_numeral_appendix(raw, dominant)

    summ = (
        f"**{label}** | main **{dominant}** | en/fr/kin scores "
        f"{scores['en']:.2f}/{scores['fr']:.2f}/{scores['kin']:.2f}"
    )
    return ChildLanguageProfile(
        label=label,  # type: ignore[arg-type]
        dominant=dominant,  # type: ignore[arg-type]
        scores=scores,
        l2_numeral_appendix=apx,
        child_summary=summ,
    )
