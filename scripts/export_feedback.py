"""Export saved feedback (MySQL) for analysis / retraining.

    python scripts/export_feedback.py

Writes data/feedback_export.csv and data/feedback_retrain.jsonl. The JSONL uses the same
{"src_lang", "tgt_lang", "src", "tgt"} format as the Phase 1 `directional/*.jsonl` files,
with the human correction as the target.
"""
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from backend.config import Settings  # noqa: E402
from backend.main import mysql_config  # noqa: E402
from database import DatabaseUnavailable, FeedbackRepository  # noqa: E402


def main() -> None:
    repo = FeedbackRepository(mysql_config(Settings()))
    try:
        rows = repo.list_all()
    except DatabaseUnavailable as exc:
        sys.exit(f"Could not read feedback: {exc}")
    out_dir = ROOT / "data"
    out_dir.mkdir(exist_ok=True)
    with open(out_dir / "feedback_export.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", "created_at", "source_text", "source_lang", "target_lang",
                    "generated_translation", "corrected_translation", "explanation", "model_version"])
        for r in rows:
            w.writerow([r["id"], r["created_at"].isoformat() if r["created_at"] else "", r["source_text"], r["source_lang"],
                        r["target_lang"], r["generated_translation"], r["corrected_translation"], r["explanation"] or "",
                        r["model_version"] or ""])
    with open(out_dir / "feedback_retrain.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps({"src_lang": r["source_lang"], "tgt_lang": r["target_lang"],
                                "src": r["source_text"], "tgt": r["corrected_translation"]}, ensure_ascii=False) + "\n")
    print(f"Exported {len(rows)} feedback rows to {out_dir}")


if __name__ == "__main__":
    main()
