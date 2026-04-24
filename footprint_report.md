# On-device footprint — `tutor/`

**Generated:** 2026-04-24 10:45 UTC

**Constraint (brief):** total on-device footprint **≤ 75 MiB** (TTS cache excluded from this budget).

## `du` equivalent (repo root)

```
# Unix / Git Bash
du -sh tutor/

# PowerShell (bytes)
Get-ChildItem -Path tutor -Recurse -File | Measure-Object -Property Length -Sum

# This run (Python walk): **175.77 MiB**
```

## Total `tutor/`: **175.77 MiB** — vs **75 MiB** — **OVER**

### Shipping-sized estimates (dev artifacts removed)

| Slice | Size | vs 75 MiB |
|-------|------|-----------|
| `tutor/` **without** `outputs/lora/` (LoRA/GGUF dev tree) | 279.9 KiB | **OK** |
| `tutor/` **without** entire `outputs/` (code + no local DB: you’d re-create empty dirs on device) | 205.9 KiB | **OK** |
| `data/T3.1_Math_Tutor/` (curriculum JSON, etc.) | 277.1 KiB | (separate from `tutor/`) |

| Component | Size | Notes |
|-----------|------|-------|
| `tutor/__init__.py` | 72 B (0.00 MiB) |  |
| `tutor/adaptive.py` | 10.8 KiB (0.01 MiB) |  |
| `tutor/asr_adapt.py` | 9.8 KiB (0.01 MiB) |  |
| `tutor/asr_mms_infer.py` | 4.4 KiB (0.00 MiB) |  |
| `tutor/child_speech_aug.py` | 6.1 KiB (0.01 MiB) |  |
| `tutor/curriculum_loader.py` | 1.7 KiB (0.00 MiB) |  |
| `tutor/dp_sync.py` | 2.7 KiB (0.00 MiB) |  |
| `tutor/feedback_audio.py` | 5.4 KiB (0.01 MiB) |  |
| `tutor/lang_detect.py` | 5.7 KiB (0.01 MiB) |  |
| `tutor/llm_qlora.py` | 5.5 KiB (0.01 MiB) |  |
| `tutor/outputs/` | 175.57 MiB (175.57 MiB) | Trim `outputs/lora/` for shipping. |
| `tutor/pipeline.py` | 4.8 KiB (0.00 MiB) |  |
| `tutor/progress_store.py` | 7.2 KiB (0.01 MiB) |  |
| `tutor/tts_backends.py` | 2.5 KiB (0.00 MiB) |  |
| `tutor/vision_grounding.py` | 3.9 KiB (0.00 MiB) |  |
| `tutor/visuals.py` | 2.9 KiB (0.00 MiB) |  |
| `(sum) tutor/*.py only` | 73.4 KiB (0.07 MiB) | Core package source |

## Excluded from budget (per brief)

- **TTS cache** — keep under a separate path (e.g. `~/.cache/tts/`) and do not count toward the 75 MiB app bundle.

## How to meet the budget

- Do **not** ship `tutor/outputs/lora/`, Hugging Face cache, or merged FP16 weights inside `tutor/`.
- Load **int4 GGUF** from user storage or download once; keep the **Python package** to source + small assets only.
- Re-run: `python scripts/measure_footprint.py` after pruning.
