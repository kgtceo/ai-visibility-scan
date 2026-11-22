"""Judge agreement against hand labels.

Usage:
  aivs label-template runs/<scan_id>            # writes evals/labels.csv with the judge's stances
  # blind sheet: fill human_stance (recommended | mentioned) for 20-30 rows
  # verify sheet (label-template --show-judge): write ok to confirm the judge, or the corrected word
  python evals/run_evals.py runs/<scan_id> evals/labels.csv

Reports agreement, and the confusion between recommended and mentioned. The report's
recommendation numbers are only worth quoting once agreement is where you want it
(the target here is 85%+ on a 20-30 row sample; below that, tighten the judge prompt
and re-score, which is cheap because the raw answers are cached).
"""

from __future__ import annotations

import csv
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from aivs.scan import load_scan  # noqa: E402


def main(scan_path: str, labels_path: str) -> int:
    result = load_scan(scan_path)
    client = result.brief.client.name
    judge = {}
    for s in result.scores:
        if s.judgement:
            bj = next((b for b in s.judgement.brands if b.brand == client), None)
            if bj:
                judge[(s.engine, s.question_id, int(s.run))] = bj.stance
    rows = [r for r in csv.DictReader(open(labels_path)) if r.get("human_stance", "").strip()]
    if not rows:
        print("No labelled rows (fill human_stance).")
        return 1
    agree = 0
    verified = 0
    confusion: Counter[tuple[str, str]] = Counter()
    for r in rows:
        key = (r["engine"], r["question_id"], int(r["run"]))
        j = judge.get(key, "missing")
        h = r["human_stance"].strip().lower()
        if h in ("ok", "agree", "y", "yes"):  # verify mode: the human confirmed the judge's call
            h = j
            verified += 1
        confusion[(h, j)] += 1
        agree += j == h
    n = len(rows)
    mode = "verified (judge shown, 'ok' = agreed)" if verified else "blind (judge hidden)"
    print(f"Judge: {result.judge_model}  rows: {n}  mode: {mode}  agreement: {agree}/{n} = {agree / n:.0%}")
    root = Path(scan_path) if Path(scan_path).is_dir() else Path(scan_path).parent
    check = {"rows": n, "agree": agree, "corrected": n - agree, "mode": "verified" if verified else "blind", "judge_model": result.judge_model, "labels": str(labels_path)}
    (root / "judge_check.json").write_text(json.dumps(check, indent=1))
    print(f"Saved {root / 'judge_check.json'}; `aivs report {root}` puts it in the report's method section.")
    print("human → judge:")
    for (h, j), c in sorted(confusion.items()):
        print(f"  {h:>12} → {j:<12} {c}")
    return 0 if agree / n >= 0.85 else 2


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(1)
    sys.exit(main(sys.argv[1], sys.argv[2]))
