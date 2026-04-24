# S2.T3.1 — full project reference (tasks & rubric)

Internal / course reference. The main project overview is in the root `README.md`.

## Task 1 (done): on-device inference pipeline

- **Curriculum** — `tutor/curriculum_loader.py` loads `data/T3.1_Math_Tutor/curriculum.json` when present (**64** trilingual items), else the **12**-item `curriculum_seed.json` starter. **Regenerate** the full set: `python generate_data.py` (from repo root; see below).
- **Loop** — `tutor/pipeline.py`: show item (image for counting tasks, text for others), accept **tap** (number) or **text** (or **mic**), score, **feedback audio** in **en / fr / kin** via `tutor/feedback_audio.py`. TTS order matches the **brief**: **Piper** / **Coqui TTS** if you set `TUTOR_PIPER_MODEL` + `piper` on `PATH` or `TUTOR_COQUI_MODEL` + the `TTS` package (`tutor/tts_backends.py`); else `pyttsx3` → `espeak` → short tone.
- **ASR (optional)** — `tutor/asr_adapt.py`: `faster-whisper` **tiny**; **live mic** audio is first adapted with `preprocess_child_mic_for_whisper` (normalise, `noisereduce`, child→adult pitch) before transcription; `transcribe_and_detect` then runs `lang_detect` on the text. For **MMS-1B-All** set `TUTOR_MMS_ASR=1` (large `torch`+`transformers` download; CPU slow). In `demo.py` set `TUTOR_ENABLE_MIC_ASR=1` and use **Check (mic ASR)**; for MMS also set `TUTOR_ASR_ENGINE=mms` if you use the optional path. **Train-time** child speech aug: `tutor/child_speech_aug.py`, `scripts/child_speech_prepare.py` (MUSAN: set `TUTOR_MUSAN_WAV`).
- **Child speech + aug (rubric data)** — `scripts/child_speech_prepare.py` downloads a small Common Voice (en / fr / rw) + `DigitalUmuganda/Afrivoice_Kinyarwanda` slice, resamples Kinyarwanda to **8 kHz**, then pitch / tempo + MUSAN or synthetic noise (`tutor/child_speech_aug.py`). Needs `librosa` from `requirements.txt`, network, and `huggingface-cli login` for dataset access.
- **Visuals** — `tutor/visuals.py` draws count-the-objects scenes (e.g. “How many goats?”) as PNG; no large detector weights in Task 1.
- **Demo** — `demo.py` (Gradio). Auto-start, tap or mic, feedback and auto-advance; optional **microphone** when `TUTOR_ENABLE_MIC_ASR=1`.
- **Latency** — `TutorSession.score` measures **answer → feedback audio ready** (excludes think time). On a warm run (e.g. Colab CPU), aim **&lt; 2.5 s**; first TTS init can be slower.

### Regenerate data (Colab: ≤ 2 commands after install)

1. `pip install -r requirements.txt`  
2. `python generate_data.py && python demo.py`

`generate_data.py` writes **`curriculum.json`** (≥ 60 items from the 12 **-item** seed + deterministic generators), optional **TTS** WAVs under `data/T3.1_Math_Tutor/tts/prompts/`, and **augmented** child-utterance stand-ins under `data/T3.1_Math_Tutor/child_utterance_aug/` using the **same** pitch / tempo / noise chain as `child_speech_aug` (set `TUTOR_MUSAN_WAV` to a MUSAN noise clip for a real **classroom noise** mix). For JSON only: `python generate_data.py --curriculum-only`. Generated audio dirs are **gitignored**; commit **`curriculum.json`** so a clone has the full item set without re-running TTS.

- **Model files** — see `tutor/MODELS.txt` (e.g. optional `model.gguf` with an external download link; not committed by default).

### Task 2 (done): knowledge tracing + Elo baseline

- `tutor/adaptive.py` — **BKT** (shared p_T, p_g, p_s, per-skill p_L0, L-BFGS-B on train NLL) and **1PL / Elo** (item difficulties b from train, online θ per session).
- **AUC** of *next-response* correctness (held-out sessions); **Brier** score in `kt_eval.ipynb`.
- **Next-item selection** vs. Elo: from a **pool of 5** (true next + 4 decoys), BKT uses **max response entropy**; Elo picks the item with **p(correct) closest to 0.5**. **Top-1** hit rate reported in the notebook.
- `kt_eval.ipynb` — end-to-end numerics; swap in a CSV replay if you have real (item_id, y) rows (keep `item_skill` mapping consistent).

### Task 3 (done, minimal)

- `tutor/llm_qlora.py` — LoRA (QLoRA when bitsandbytes is available) on `data/T3.1_Math_Tutor/numeracy_instruct.jsonl`, saves `adapter` + **merged** `merged_f16`.
- **CPU smoke (fast):** `python -m tutor.llm_qlora --smoke` (after `pip install -r requirements.txt`)
- **int4:** merge is FP16; convert to **GGUF Q4** or **AWQ** with external tools (steps in `process_log.md` § *Int4 GGUF*).

### Task 4 (done)

- `tutor/lang_detect.py` — **KIN / FR / EN / mix**; **dominant** drives tutor feedback TTS; **mix** appends an **L2 number gloss** on the same clip.
- Voice path in `demo.py` updates session language from speech.

### Task 5 (done)

- `tutor/vision_grounding.py` — **baseline** counts on PNG (distance-transform local maxima; no access to `n` from the curriculum). Optional **OWL-ViT** (`TUTOR_OWLVIT=1`).

### Task 6 (done)

- `tutor/progress_store.py` — SQLite; optional **Fernet** encryption; `tutor/dp_sync.py` — Laplace noise; `parent_report.py` — weekly JSON. See code and `parent_report_schema.json`.

### Task 7 (done): on-device footprint **≤ 75 MiB** (TTS cache excluded)

- `footprint_report.md` — `python scripts/measure_footprint.py` (repo root). **TTS cache** and generated audio dirs are out of scope for the 75 MB line.

### Product and business (brief)

- **`docs/PRODUCT_ADAPTATION.md`**, **`process_log.md`**

### Submission (brief)

- **`process_log.md`**, **`SIGNED.md`**

### Optional

- **GGUF** — `tutor/MODELS.txt`, `process_log.md` § *Int4 GGUF*

## License

See `LICENSE` in the repo root.
