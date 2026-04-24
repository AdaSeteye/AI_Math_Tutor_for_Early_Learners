"""On-device inference loop: present item, accept tap or voice-derived answer, score, feedback audio."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Literal

from tutor.curriculum_loader import CurriculumItem, LanguageCode, default_curriculum_path, load_curriculum
from tutor.feedback_audio import parse_spoken_number, synthesize_feedback_wav
from tutor.vision_grounding import grounded_count
from tutor.visuals import render_count_image


@dataclass
class PresentResult:
    item: CurriculumItem
    prompt: str
    image_png: bytes | None
    language: LanguageCode
    vision_count: int | None = None
    """Model guess at object count (Task 5: blob or optional OWL-ViT)."""
    vision_method: str | None = None


@dataclass
class ScoreResult:
    correct: bool
    expected: int | float
    given: int | float | None
    latency_s: float  # time from score() entry → feedback wav ready (not think time)
    feedback_wav_path: str
    feedback_kind: Literal["correct", "encourage"]
    child_language_note: str | None = None


@dataclass
class TutorSession:
    """
    Sequenced curriculum play. :meth:`reset_to_demo_start` is used by ``demo.py`` on first load
    (seeks a friendly ``count_image`` like ``c2``).
    """

    items: list[CurriculumItem]
    language: LanguageCode = "en"
    learner_id: str = "demo-child-1"
    _index: int = 0

    @classmethod
    def from_default(cls, language: LanguageCode = "en") -> TutorSession:
        path = default_curriculum_path()
        return cls(load_curriculum(path), language=language)

    def set_language(self, lang: LanguageCode) -> None:
        self.language = lang

    def current_item(self) -> CurriculumItem | None:
        if 0 <= self._index < len(self.items):
            return self.items[self._index]
        return None

    def present(self) -> PresentResult | None:
        item = self.current_item()
        if item is None:
            return None
        prompt = item.prompt_for(self.language)
        image_png: bytes | None = None
        v_c: int | None = None
        v_m: str | None = None
        if item.type == "count_image" and item.visual:
            n = int(item.visual.get("n_objects", 0))
            label = str(item.visual.get("object_label", "circle"))
            seed = hash(item.id) % (2**31)
            image_png = render_count_image(n, object_label=label, seed=seed, with_caption=False)
            v_c, v_m = grounded_count(image_png, label, method="auto")
        return PresentResult(
            item=item,
            prompt=prompt,
            image_png=image_png,
            language=self.language,
            vision_count=v_c,
            vision_method=v_m,
        )

    def _normalize_answer(self, item: CurriculumItem, response: Any) -> int | float | None:
        if isinstance(response, (int, float)) and not isinstance(response, bool):
            return int(response) if item.expected_answer == int(item.expected_answer) else float(response)
        if isinstance(response, str):
            p = parse_spoken_number(response)
            return p
        return None

    def score(
        self,
        response: int | float | str | None,
        mode: Literal["tap", "voice"] = "tap",
        feedback_lang: LanguageCode | None = None,
        feedback_extra_tts: str | None = None,
        child_language_note: str | None = None,
    ) -> ScoreResult:
        """``latency_s`` = wall time inside this call (parse answer + TTS feedback); use for the under-2.5s target."""
        item = self.current_item()
        if item is None:
            raise RuntimeError("No active item")
        t0 = time.perf_counter()
        given = self._normalize_answer(item, response)
        expected = item.expected_answer
        if isinstance(expected, float):
            expected = float(expected)
        else:
            expected = int(expected)
        cor = given is not None and int(given) == int(expected)
        kind: Literal["correct", "encourage"] = "correct" if cor else "encourage"
        t_lang = feedback_lang if feedback_lang is not None else self.language
        extra = (feedback_extra_tts or "").strip() or None
        feedback_path = synthesize_feedback_wav(t_lang, kind, extra)
        t1 = time.perf_counter()
        latency = t1 - t0
        try:
            from tutor.progress_store import get_progress_db

            get_progress_db().log_attempt(
                self.learner_id, item.id, item.subskill, cor
            )
        except Exception:
            pass
        return ScoreResult(
            correct=cor,
            expected=expected,
            given=given,
            latency_s=latency,
            feedback_wav_path=feedback_path,
            feedback_kind=kind,
            child_language_note=child_language_note,
        )

    def advance(self) -> None:
        self._index += 1

    def reset(self) -> None:
        self._index = 0

    def seek_item_id(self, item_id: str) -> bool:
        for i, it in enumerate(self.items):
            if it.id == item_id:
                self._index = i
                return True
        return False

    def reset_to_demo_start(self) -> None:
        """Start session on a friendly first counting item (goats) when available."""
        self._index = 0
        if not self.seek_item_id("c2"):
            for i, it in enumerate(self.items):
                if it.type == "count_image" and it.visual and it.difficulty <= 1:
                    self._index = i
                    return

    def jump_to_simpler_count(self) -> bool:
        """
        After sustained silence, move to an easier count (fewer objects) or a different easy count.
        No score / no 'wrong' signal.
        """
        cur = self.current_item()
        if not cur or cur.type != "count_image" or not cur.visual:
            return False
        cur_n = int(cur.visual.get("n_objects", 99))
        cur_id = cur.id
        best_i: int | None = None
        best_n = 10**6
        for i, it in enumerate(self.items):
            if it.type != "count_image" or not it.visual:
                continue
            n = int(it.visual.get("n_objects", 99))
            if n < cur_n and n < best_n:
                best_n = n
                best_i = i
        if best_i is not None:
            self._index = best_i
            return True
        for i, it in enumerate(self.items):
            if it.type != "count_image" or not it.visual:
                continue
            n = int(it.visual.get("n_objects", 99))
            if n == cur_n and it.id != cur_id and it.difficulty <= 1:
                self._index = i
                return True
        if self.seek_item_id("g031_c"):
            return True
        return False
