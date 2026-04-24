# process_log.md — S2.T3.1 (AI Math Tutor for Early Learners)


## Hour-by-hour timeline

| When | What I did |
|------|------------|
| **Session 1** | Read the candidate brief; scaffolded `data/T3.1_Math_Tutor/`, `tutor/curriculum_loader.py`, `tutor/pipeline.py`, `tutor/feedback_audio.py`, `tutor/visuals.py`, and `demo.py` (Gradio). Set `127.0.0.1` default for the server on Windows. |
| **Session 2** | Implemented `tutor/adaptive.py` (BKT + Elo), synthetic replay, `kt_eval.ipynb`, metrics (AUC, Top-1 pool). |
| **Session 3** | LoRA path: `tutor/llm_qlora.py`, `numeracy_instruct.jsonl` (GGUF export notes below); decided smoke train on small LM vs full TinyLlama on GPU later. |
| **Session 4** | `tutor/lang_detect.py` (KIN/FR/EN/mix) + TTS appendix for mixed; extended `pipeline` and `demo` voice path. |
| **Session 5** | `tutor/vision_grounding.py` (distance-transform baseline + optional OWL); wired `PresentResult.vision_count`. |
| **Session 6** | `tutor/progress_store.py`, `tutor/dp_sync.py`, `parent_report.py` (HMAC learner key, Laplace export). |
| **Session 7** | `scripts/measure_footprint.py` + `footprint_report.md` vs 75 MiB; README updates. |
| **Session 8** | Added `process_log.md`, `SIGNED.md`, final README checks. |


## Tools I used (LLM / coding assistants) and why

| Tool | Why |
|------|-----|
| **Cursor** | Primary editor; AI-assisted refactors, debugging, and file generation against the task list and the repo layout. |
| **Python / pip** | Local runs: Gradio, scipy/numpy, sklearn, `langdetect`, training smoke tests. |
| **Git**  | Version control and pushing to a public host before Live Defense. |


## Three sample prompts I actually used

1. **"Complete this BKT and Elo baseline, held-out AUC, kt_eval."**  
   

## One prompt I started and discarded (and why)

- **Discarded idea:**  
- **Why I dropped it:** It would have broken the `tutor/` package layout the brief asks for, made the footprint and imports harder to defend, and would not have improved the child UX compared to a clean Gradio layout.

## The single hardest decision (one paragraph)

The hardest choice was how to keep **on-device and offline** constraints while still **looking credible** on the LLM and vision parts. Full **Phi-3 / TinyLlama** QLoRA and **OWL-ViT** in the same repo as a **&lt; 75 MB** install is not realistic without splitting training artifacts, document-only GGUF steps, and optional second-stage downloads. I chose to ship a **small, testable** LoRA smoke path and a **hand-tuned** vision baseline (distance-transform peaks) with an **opt-in** OWL path behind an environment variable, and to be explicit in the README and `footprint_report.md` about what ships on the tablet vs what stays on a laptop or Hugging Face. That trades polish on every optional component for a defensible story and runnable code in Live Defense.

## Int4 GGUF (or AWQ) after LoRA merge

1. Merged HF folder: `tutor/outputs/lora/merged_f16/`.
2. **GGUF:** use `llama.cpp` (or `ggml-org/llama.cpp`) `convert_hf_to_gguf.py` on `merged_f16/`, then quantize (e.g. `Q4_K_M`).
3. **AWQ:** `optimum` / `autoawq` on `merged_f16` if targeting GPU int4.
4. Record size in `footprint_report.md`.

## Product & business (rubric)

- **Full narrative (required for scoring):** `docs/PRODUCT_ADAPTATION.md` — 90s UX, community-centre tablet for 3 children, non-literate parent report.
- **First 90s:** Kinyarwanda-first TTS, large Start, first task tap-only; if **10s silence** before Start, repeat short Kinyarwanda + pulse the button; no free-text/ASR on first run.
- **Shared tablet (3 children):** picture profile grid, `TUTOR_HMAC_SECRET` for stable learner keys, per-child `progress.db` rows, **Next child** handover, **reboot** → return to profile grid, DB still local.
- **Parent 1-pager (non-literate):** 5 sub-skill “battery” icons, one big number, **QR → ~60s Kinyarwanda** audio, lock icon; DP noise explained on teacher view only.

## Child speech rubric data (`scripts/child_speech_prepare.py`)

- **Common Voice age field:** The script defaults to `--cv-ages teens,` (keep **teens** plus **empty/unknown** metadata; CV’s public `age` labels are coarse — **teens** is the closest public band to a “youth/child” story). Pass `--cv-ages ""` only to disable filtering (e.g. debugging; not aligned with a strict “child age band” defense).
- **Sample defaults:** Defaults were raised to `--cv-samples 5` per en/fr/rw and `--afri-samples 2` for Kinyarwanda so the **base** set is less thin before pitch/tempo/noise augs; increase further if the evaluator wants more source diversity.
- **MUSAN vs synthetic:** If `TUTOR_MUSAN_WAV` points at a real WAV (MUSAN slice), the `*_musan` family clips use that file. If unset or missing, those mixes use **synthetic** pinkish noise in `tutor/child_speech_aug` — this is written to `child_speech_rubric/prepare_meta.json` as `noise_for_musan_suffix_clips` for audit.

