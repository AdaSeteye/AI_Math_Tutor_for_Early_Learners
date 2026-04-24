# AI Math Tutor for Early Learners

**Numeracy-tuned small language model** for short, child-friendly math explanations.  
This repository contains the **merged float16** checkpoint (all weights in one folder: `model.safetensors` + config + tokenizer files).

| | |
|---|---|
| **Model page** | [https://huggingface.co/AddisuSeteye/AI_Math_Tutor_for_Early_Learners](https://huggingface.co/AddisuSeteye/AI_Math_Tutor_for_Early_Learners) |
| **Source code** | [https://github.com/AdaSeteye/AI_Math_Tutor_for_Early_Learners](https://github.com/AdaSeteye/AI_Math_Tutor_for_Early_Learners) |
| **License** | MIT |

## Model details

- **Architecture:** `GPT2LMHeadModel` (6 layers, 768 hidden size, 12 attention heads) — same family as [distilgpt2](https://huggingface.co/distilgpt2) used as the base for the training pipeline in this project.
- **Format:** Merged full model in **FP16** (`model.safetensors`), with `config.json`, `generation_config.json`, and tokenizer files.
- **Training:** Supervised fine-tuning with **LoRA** (QLoRA when 4-bit is available), then **merge** into a single Hugging Face–compatible directory (`merged_f16`).  
- **Training data:** `numeracy_instruct.jsonl` — short user/assistant **chat-style** turns (counting, addition, subtraction, simple word problems) for early numeracy.
- **Intended use:** Prototype / teaching demo for an **AI Math Tutor** context; not a production safety-critical system for unsupervised use with children without human oversight.

## Files in this repo (upload from `merged_f16`)

| File | Role |
|------|------|
| `model.safetensors` | Merged model weights (FP16) |
| `config.json` | Model configuration |
| `generation_config.json` | Default generation settings |
| `tokenizer.json` | Tokenizer (JSON) |
| `tokenizer_config.json` | Tokenizer config |

## How to use

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_id = "AddisuSeteye/AI_Math_Tutor_for_Early_Learners"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    device_map="auto",
)

# Example: single-turn style prompt (match your training format in practice)
text = "You are a math helper. How many is 2 + 1?"
inputs = tokenizer(text, return_tensors="pt").to(model.device)
out = model.generate(**inputs, max_new_tokens=80, do_sample=True, top_p=0.9)
print(tokenizer.decode(out[0], skip_special_tokens=True))
```

### Requirements

- `transformers` (version compatible with your `config.json`, e.g. 4.36+)
- `torch`
- `safetensors` (for loading `.safetensors`)

## Limitations

- Trained for **short** numeracy-style responses; may **hallucinate** or be incorrect on out-of-distribution questions.
- **Not** a replacement for a teacher or parent; early-learner products should add safety, privacy, and pedagogy review.
- If you re-train on a different base (e.g. a larger chat model) and re-upload, update this card to match the new `config.json`.

## Citation

If you use this model, please cite the project repository and the Hugging Face model page. Example:

```bibtex
@misc{aimathtutor2026,
  title   = {AI Math Tutor for Early Learners},
  author  = {Addisu Seteye},
  year    = {2026},
  howpublished = {\url{https://huggingface.co/AddisuSeteye/AI_Math_Tutor_for_Early_Learners}}
}
```

## Project context

This checkpoint is one deliverable of the **S2.T3.1** numeracy tutor work: instruction data in `data/T3.1_Math_Tutor/`, training script `tutor/llm_qlora.py` (see the GitHub repo for the full app, curriculum, and on-device tutor pipeline). The **Gradio child demo** in that repo does not load this merged checkpoint by default; the main product loop uses a separate pipeline (TTS, ASR, curriculum) described in the GitHub `README.md`.

---

*Model card generated for upload of `tutor/outputs/lora/merged_f16` contents. If your local `config.json` shows a different `model_type` (e.g. you merged from another base), edit the “Model details” section to match your files.*
