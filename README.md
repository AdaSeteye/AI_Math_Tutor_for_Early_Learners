# AI Math Tutor for Early Learners

An on-device **math learning prototype** for young learners: trilingual (English, French, Kinyarwanda) counting and numeracy items, **spoken** feedback, optional **voice** input (ASR), and a **Gradio** child-facing demo. Includes optional **knowledge-tracing** experiments, a **LoRA** numeracy-tuned language model, and **privacy** tooling (progress store, optional DP-aggregated reports).

**License:** MIT — see [`LICENSE`](LICENSE).

---

## Hugging Face model

Merged numeracy-tuned weights (for inference or research) are published on the Hub:

**[AddisuSeteye/AI_Math_Tutor_for_Early_Learners](https://huggingface.co/AddisuSeteye/AI_Math_Tutor_for_Early_Learners)**

The model card (`README` on that page) describes the merged checkpoint, limitations, and a short loading example. This repo is the **full application and training** codebase; the Hub model is the **exported** `merged_f16` artifactory only.

---

## Quick start (run the Gradio demo)

```bash
pip install -r requirements.txt
python demo.py
```

Then open the URL the terminal prints (commonly `http://127.0.0.1:7860/`).  
Use `http://127.0.0.1:...` in the browser even if the server binds to `0.0.0.0` (set with `GRADIO_SERVER_NAME=0.0.0.0` for LAN access). On Windows, if port `7860` is busy, the demo picks the next free port and prints it.

**Optional (Linux / Colab):** for better offline TTS when `pyttsx3` is weak, install `espeak-ng` and ensure `espeak` or `espeak-ng` is on your `PATH`.

**Microphone / ASR:** set `TUTOR_ENABLE_MIC_ASR=1` (on by default in the demo) so the voice check path is enabled. Heavy engines (e.g. MMS) are opt-in via environment variables; see the reference doc below.

**Regenerate curriculum JSON** (after editing generators): `python generate_data.py` — or `python generate_data.py --curriculum-only` for JSON without audio side effects.

---

## What’s in this repository

| Area | Description |
|------|-------------|
| **Child demo** | `demo.py` — Gradio UI, curriculum loop, TTS, optional mic |
| **Tutor engine** | `tutor/pipeline.py`, `feedback_audio.py`, `asr_adapt.py`, `visuals.py`, `lang_detect.py`, etc. |
| **Curriculum & data** | `data/T3.1_Math_Tutor/curriculum.json` (and seed / JSONL for training) |
| **QLoRA training** | `tutor/llm_qlora.py` — outputs under `tutor/outputs/lora/` (not required to run the child demo) |
| **KT / adaptive** | `tutor/adaptive.py`, `kt_eval.ipynb` |
| **Product & process** | `docs/PRODUCT_ADAPTATION.md`, `process_log.md` |

A detailed **task-by-task rubric** and module callouts live in **[`docs/PROJECT_REFERENCE.md`](docs/PROJECT_REFERENCE.md)** for coursework and internal use.

---

## System notes

- **Dependencies** are in [`requirements.txt`](requirements.txt) (`torch` / `transformers` are large; use a venv on modest machines).  
- **On-disk footprint** for a “shipping” slice (excluding local training outputs and TTS cache) is documented in [`footprint_report.md`](footprint_report.md) and `scripts/measure_footprint.py`.  
- **Child speech / dataset download scripts** may need `huggingface-cli login` and a network; see `scripts/child_speech_prepare.py` and the reference doc.  
- **Model weights in Git:** do not commit large `tutor/outputs/` trees if you can avoid it; use the Hub model for sharing weights.

---

## Citation

If you use the code or the Hugging Face model, cite the model page and this repository. See the Hub **README** for a sample BibTeX block.

---

## More documentation

- [`docs/PRODUCT_ADAPTATION.md`](docs/PRODUCT_ADAPTATION.md) — product / UX narrative  
- [`docs/HF_MODEL_README.md`](docs/HF_MODEL_README.md) — copy of the model card text for the Hub (optional)  
- [`process_log.md`](process_log.md) — development / submission log (edit for your run)  
- [`docs/PROJECT_REFERENCE.md`](docs/PROJECT_REFERENCE.md) — full S2.T3.1 task reference  

**Submission (course):** `process_log.md`, `SIGNED.md` as required by your cohort.
