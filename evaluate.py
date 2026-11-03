"""Run every guardrail on the test set and save raw scores to results/scores.json."""
import json
import time
from pathlib import Path

from guardrails import CHECKS, GUARDRAILS, PII_GUARDRAILS, THRESHOLDS
from scoring import missed_names_by_group, summarize

ROOT = Path(__file__).parent


def main():
    cases = [json.loads(line) for line in (ROOT / "data" / "cases.jsonl").read_text(encoding="utf-8").splitlines()]
    flags_by_id = {case["id"]: [] for case in cases}
    latency_ms = {}

    for check in CHECKS.values():
        check("warm up")  # model loading shouldn't count towards latency
    for name, check in CHECKS.items():
        start = time.perf_counter()
        for case in cases:
            flags_by_id[case["id"]].extend(check(case["text"]))
        latency_ms[name] = round((time.perf_counter() - start) * 1000 / len(cases), 1)

    output = ROOT / "results" / "scores.json"
    output.parent.mkdir(exist_ok=True)
    output.write_text(json.dumps({"cases": cases, "flags": flags_by_id, "latency_ms": latency_ms}, indent=1), encoding="utf-8")

    for split in ("dev", "test", "fresh"):
        subset = [c for c in cases if c["split"] == split]
        print(f"\n{split}: {len(subset)} cases")
        for name, title in GUARDRAILS.items():
            s = summarize(subset, flags_by_id, name, THRESHOLDS[name])
            print(f"  {title:32} at {THRESHOLDS[name]:.2f}  false alarms {s['false_alarms']:>3}/{s['clean']:<3} misses {s['misses']:>3}/{s['attacks']:<3} {latency_ms[name]:>7} ms")
        for name in PII_GUARDRAILS:
            groups = missed_names_by_group(subset, flags_by_id, name, THRESHOLDS[name])
            print(f"  Missed names, {name}: " + ", ".join(f"{group} {missed}/{total}" for group, (missed, total) in groups.items()))


if __name__ == "__main__":
    main()
