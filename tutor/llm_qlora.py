"""
LoRA (QLoRA when bitsandbytes is available), merge, save merged HF for GGUF export.
Run smoke (CPU, small, no big download):  python -m tutor.llm_qlora --smoke
Real run (GPU, 4-bit):  python -m tutor.llm_qlora --model TinyLlama/TinyLlama-1.1B-Chat-v1.0
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

def _load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _fmt_text(tok, row: dict) -> str:
    m = row.get("messages", [])
    if not m:
        return ""
    parts: list[str] = []
    for t in m:
        role, c = t.get("role", ""), t.get("content", "")
        parts.append(f"### {role}\n{c}")
    return "\n\n".join(parts) + (f"\n{tok.eos_token}" if tok.eos_token else "")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--data",
        type=Path,
        default=Path("data/T3.1_Math_Tutor/numeracy_instruct.jsonl"),
    )
    ap.add_argument("--out", type=Path, default=Path("tutor/outputs/lora"))
    ap.add_argument(
        "--model",
        type=str,
        default="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    )
    ap.add_argument(
        "--smoke",
        action="store_true",
        help="Use distilgpt2, 2 steps, CPU, no 4bit.",
    )
    ap.add_argument("--max_steps", type=int, default=30)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--no_4bit", action="store_true")
    args = ap.parse_args()

    import torch
    from datasets import Dataset
    from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
        DataCollatorForLanguageModeling,
        Trainer,
        TrainingArguments,
    )

    if args.smoke:
        model_id = "distilgpt2"
        use_4bit = False
        target_modules = ["c_attn", "c_proj", "c_fc"]
    else:
        model_id = args.model
        use_4bit = not args.no_4bit
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

    bnb = None
    if use_4bit:
        try:
            bnb = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
        except Exception:
            use_4bit = False

    load_kw: dict = {"trust_remote_code": True}
    if not args.smoke:
        load_kw["device_map"] = "auto"
    if use_4bit and bnb is not None:
        load_kw["quantization_config"] = bnb

    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(model_id, **load_kw)
    if use_4bit:
        model = prepare_model_for_kbit_training(model)

    lora = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=target_modules,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora)

    raw = _load_jsonl(args.data)
    if args.smoke:
        raw = raw[:2]
    texts = [_fmt_text(tok, r) for r in raw if r.get("messages")]
    if not texts:
        raise SystemExit("No rows with messages in data file")

    def tok_one(ex: dict) -> dict:
        o = tok(
            ex["text"],
            truncation=True,
            max_length=256,
            padding="max_length",
        )
        o["labels"] = o["input_ids"].copy()
        return o

    ds = Dataset.from_dict({"text": texts}).map(tok_one, remove_columns=["text"])
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    steps = 2 if args.smoke else min(args.max_steps, max(3, len(ds) * 2))
    targs = TrainingArguments(
        output_dir=str(out / "trainer"),
        max_steps=steps,
        per_device_train_batch_size=1,
        learning_rate=args.lr,
        logging_steps=1,
        save_steps=steps,
        save_total_limit=1,
        report_to="none",
    )
    collator = DataCollatorForLanguageModeling(tok, mlm=False)
    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=ds,
        data_collator=collator,
    )
    trainer.train()
    (out / "adapter").mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out / "adapter")
    tok.save_pretrained(out / "adapter")

    # merge -> FP16 (for llama.cpp convert; optional AWQ on GPU elsewhere)
    mrg = out / "merged_f16"
    mrg.mkdir(parents=True, exist_ok=True)
    base2 = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map="auto" if not args.smoke else None,
        trust_remote_code=True,
    )
    peft = PeftModel.from_pretrained(base2, str(out / "adapter"))
    merged = peft.merge_and_unload()
    merged.save_pretrained(mrg, safe_serialization=True)
    tok.save_pretrained(mrg)

    (out / "EXPORT_NEXT.txt").write_text(
        "Merged: merged_f16/\nInt4 GGUF: see process_log.md (Int4 GGUF section) and llama.cpp convert script.\n",
        encoding="utf-8",
    )
    print("OK adapter:", out / "adapter")
    print("OK merged: ", mrg)


if __name__ == "__main__":
    main()
