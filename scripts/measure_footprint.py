"""Generate footprint_report.md from disk sizes (Windows/macOS/Linux). Run from repo root."""

from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TUTOR = ROOT / "tutor"
BUDGET_MB = 75.0


def dir_size(path: Path) -> int:
    if not path.exists():
        return 0
    total = 0
    for root, _dirs, files in os.walk(path):
        for f in files:
            fp = Path(root) / f
            try:
                total += fp.stat().st_size
            except OSError:
                pass
    return total


def human(n: int) -> str:
    if n < 1024:
        return f"{n} B"
    if n < 1024 * 1024:
        return f"{n / 1024:.1f} KiB"
    return f"{n / (1024 * 1024):.2f} MiB"


def status_ok(n: int) -> str:
    return "OK" if n <= BUDGET_MB * 1024 * 1024 else "OVER"


def main() -> None:
    total = dir_size(TUTOR)
    out_lora = TUTOR / "outputs" / "lora"
    out_all = TUTOR / "outputs"
    sz_lora = dir_size(out_lora) if out_lora.is_dir() else 0
    sz_outputs = dir_size(out_all) if out_all.is_dir() else 0
    without_lora = total - sz_lora
    without_outputs = total - sz_outputs

    rows: list[tuple[str, int, str]] = []
    for p in sorted(TUTOR.iterdir()):
        if p.name == "__pycache__":
            continue
        if p.is_file() and p.name.endswith(".py"):
            rows.append((f"tutor/{p.name}", p.stat().st_size, ""))
        elif p.is_dir():
            s = dir_size(p)
            note = ""
            if p.name == "outputs":
                note = "Trim `outputs/lora/` for shipping."
            rows.append((f"tutor/{p.name}/", s, note))

    code_only = sum(p.stat().st_size for p in TUTOR.glob("*.py"))
    rows.append(("(sum) tutor/*.py only", code_only, "Core package source"))

    when = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    rel = TUTOR.relative_to(ROOT)
    data_sz = dir_size(ROOT / "data" / "T3.1_Math_Tutor")

    out = ROOT / "footprint_report.md"
    lines = [
        f"# On-device footprint — `tutor/`",
        "",
        f"**Generated:** {when}",
        "",
        f"**Constraint (brief):** total on-device footprint **≤ {BUDGET_MB:.0f} MiB** (TTS cache excluded from this budget).",
        "",
        "## `du` equivalent (repo root)",
        "",
        "```",
        "# Unix / Git Bash",
        f"du -sh {rel.as_posix()}/",
        "",
        f"# PowerShell (bytes)",
        f'Get-ChildItem -Path {rel.as_posix()} -Recurse -File | Measure-Object -Property Length -Sum',
        "",
        f"# This run (Python walk): **{human(total)}**",
        "```",
        "",
        f"## Total `tutor/`: **{human(total)}** — vs **{BUDGET_MB:.0f} MiB** — **{status_ok(total)}**",
        "",
        "### Shipping-sized estimates (dev artifacts removed)",
        "",
        "| Slice | Size | vs 75 MiB |",
        "|-------|------|-----------|",
        f"| `tutor/` **without** `outputs/lora/` (LoRA/GGUF dev tree) | {human(without_lora)} | **{status_ok(without_lora)}** |",
        f"| `tutor/` **without** entire `outputs/` (code + no local DB: you’d re-create empty dirs on device) | {human(without_outputs)} | **{status_ok(without_outputs)}** |",
        f"| `data/T3.1_Math_Tutor/` (curriculum JSON, etc.) | {human(data_sz)} | (separate from `tutor/`) |",
        "",
        "| Component | Size | Notes |",
        "|-----------|------|-------|",
    ]
    for name, s, note in rows:
        lines.append(
            f"| `{name}` | {human(s)} ({s / (1024*1024):.2f} MiB) | {note} |"
        )
    lines += [
        "",
        "## Excluded from budget (per brief)",
        "",
        "- **TTS cache** — keep under a separate path (e.g. `~/.cache/tts/`) and do not count toward the 75 MiB app bundle.",
        "",
        "## How to meet the budget",
        "",
        "- Do **not** ship `tutor/outputs/lora/`, Hugging Face cache, or merged FP16 weights inside `tutor/`.",
        "- Load **int4 GGUF** from user storage or download once; keep the **Python package** to source + small assets only.",
        "- Re-run: `python scripts/measure_footprint.py` after pruning.",
        "",
    ]
    out.write_text("\n".join(lines), encoding="utf-8")
    print(out, human(total), "without_lora", human(without_lora))


if __name__ == "__main__":
    main()
