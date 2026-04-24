"""
Build a weekly parent report (JSON) from the local progress DB, schema: data/T3.1_Math_Tutor/parent_report_schema.json
Usage:  python parent_report.py --learner demo-child-1
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from tutor.progress_store import ProgressDB, _anon_tag

ROOT = Path(__file__).resolve().parent
SCHEMA_PATH = ROOT / "data" / "T3.1_Math_Tutor" / "parent_report_schema.json"


def _week_window(week_ending: date) -> tuple[float, float]:
    """Inclusive 7-day window ending on *week_ending* (local UTC)."""
    start = datetime.combine(week_ending - timedelta(days=6), datetime.min.time(), tzinfo=timezone.utc)
    end = datetime.combine(week_ending, datetime.max.time(), tzinfo=timezone.utc)
    return (start.timestamp(), end.timestamp())


def build_report(
    learner_id: str,
    week_ending: date,
    db: ProgressDB,
    t_range: tuple[float, float] | None = None,
) -> dict:
    t0, t1 = t_range if t_range is not None else _week_window(week_ending)
    attempts = list(db.iter_attempts_for_learner(learner_id, t0, t1))
    n = len(attempts)
    correct = sum(1 for a in attempts if a.correct)
    by_skill: dict[str, dict[str, int]] = {}
    for a in attempts:
        by_skill.setdefault(a.subskill, {"n": 0, "ok": 0})
        by_skill[a.subskill]["n"] += 1
        by_skill[a.subskill]["ok"] += a.correct
    skills = [
        {
            "name": k,
            "attempts": v["n"],
            "correct_rate": round(v["ok"] / v["n"], 2) if v["n"] else 0.0,
        }
        for k, v in sorted(by_skill.items())
    ]
    report = {
        "learner_id": _anon_tag(learner_id),
        "week_ending": week_ending.isoformat(),
        "summary": {
            "total_attempts": n,
            "total_correct": correct,
            "accuracy": round(correct / n, 2) if n else 0.0,
        },
        "skills": skills,
        "streaks": {
            "week_label": f"Week ending {week_ending}",
            "note": "Icons and audio QR can be added in a front-end; this JSON is the data layer.",
        },
    }
    return report


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--learner", default="demo-child-1", help="local learner id (HMAC-keyed in DB)")
    ap.add_argument("--db", type=Path, default=None)
    ap.add_argument("--week-ending", type=str, default=None, help="YYYY-MM-DD (default: last Sunday UTC)")
    ap.add_argument("-o", "--out", type=Path, default=ROOT / "tutor" / "outputs" / "parent_report_last.json")
    ap.add_argument(
        "--rolling-days",
        type=int,
        default=None,
        help="use last N calendar days of attempts (ignores --week-ending window math)",
    )
    ap.add_argument(
        "--all-time",
        action="store_true",
        help="include all stored attempts (for debug / clock skew)",
    )
    args = ap.parse_args()

    t_override: tuple[float, float] | None = None
    we = date.today()
    if args.all_time:
        t_override = (0.0, time.time() + 86400.0 * 365 * 50)
    elif args.rolling_days is not None and args.rolling_days > 0:
        t1 = time.time()
        t0 = t1 - float(args.rolling_days) * 86400.0
        t_override = (t0, t1)
        we = datetime.fromtimestamp(t1, tz=timezone.utc).date()
    elif args.week_ending:
        we = date.fromisoformat(args.week_ending)
    else:
        we = we - timedelta(days=(we.weekday() + 1) % 7)
        if we > date.today():
            we -= timedelta(days=7)

    db = ProgressDB(args.db)
    rep = build_report(args.learner, we, db, t_range=t_override)
    db.close()

    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    for k in schema.get("required", []):
        if k not in rep:
            print(f"missing {k}", file=sys.stderr)
            sys.exit(1)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(rep, indent=2), encoding="utf-8")
    print(args.out)
    print(json.dumps(rep, indent=2))


if __name__ == "__main__":
    main()
