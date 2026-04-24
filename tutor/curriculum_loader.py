from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

LanguageCode = Literal["en", "fr", "kin"]


@dataclass(frozen=True)
class CurriculumItem:
    id: str
    subskill: str
    difficulty: int
    age_band: str
    type: str
    visual: dict[str, Any] | None
    options: list[int] | None
    expected_answer: int | float
    prompts: dict[LanguageCode, str]
    raw: dict[str, Any]

    def prompt_for(self, lang: LanguageCode) -> str:
        return self.prompts.get(lang) or self.prompts["en"]


def _load_item(d: dict[str, Any]) -> CurriculumItem:
    prompts: dict[LanguageCode, str] = {
        "en": d.get("prompt_en", ""),
        "fr": d.get("prompt_fr", ""),
        "kin": d.get("prompt_kin", ""),
    }
    return CurriculumItem(
        id=d["id"],
        subskill=d["subskill"],
        difficulty=int(d["difficulty"]),
        age_band=d["age_band"],
        type=d["type"],
        visual=d.get("visual"),
        options=d.get("options"),
        expected_answer=d["expected_answer"],
        prompts=prompts,
        raw=d,
    )


def load_curriculum(path: str | Path) -> list[CurriculumItem]:
    p = Path(path)
    with p.open(encoding="utf-8") as f:
        data = json.load(f)
    return [_load_item(x) for x in data["items"]]


def default_curriculum_path() -> Path:
    """Prefer generated ``curriculum.json``; fall back to 12-item ``curriculum_seed.json``."""
    base = Path(__file__).resolve().parents[1] / "data" / "T3.1_Math_Tutor"
    gen = base / "curriculum.json"
    if gen.is_file():
        return gen
    return base / "curriculum_seed.json"
