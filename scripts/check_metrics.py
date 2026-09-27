#!/usr/bin/env python3
"""Compares the result files a reproduction wrote with the reference files in expected_metrics/.

The comparison is a deep recursive walk, so no metric path is hard-coded and a change anywhere in
a result file is caught. Numeric leaves must match within --tol, 1e-6 by default; strings and
booleans must match exactly; a missing or extra key, or a list of a different length, is a
failure.

The exit code is 0 only if every reproduced file matches. A file that was not written, because
its step was skipped, is reported as SKIP and does not fail the run unless --require-all is given.

Usage:
  python scripts/check_metrics.py                 # compare all reproduced JSONs
  python scripts/check_metrics.py --only weblog_edgar   # one table
  python scripts/check_metrics.py --tol 1e-9 --require-all
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# label -> the path the reproduction writes, then its reference file in expected_metrics/
MAPPING = {
    "lab_testbed":        ("models/lab_testbed/eval_results.json",          "lab_testbed.json"),
    "lab_external":       ("models/lab_testbed/lab_external_results.json",  "lab_external.json"),
    "single_source_fail": ("models/single_source/single_source_results.json","single_source_fail.json"),
    "loso":               ("models/loso/loso_results.json",                  "loso.json"),
    "in_domain":          ("models/in_domain/in_domain_results.json",        "in_domain.json"),
    "ml_vs_rules":        ("models/in_domain/ml_vs_rules_results.json",      "ml_vs_rules.json"),
    "family":             ("models/in_domain/family_results.json",           "family.json"),
    "multidomain":        ("models/multidomain/multidomain_results.json",    "multidomain.json"),
    "adversarial":        ("models/adversarial/adversarial_results.json",    "adversarial.json"),
    "weblog_edgar":       ("models/weblog_edgar/weblog_edgar_results.json",  "weblog_edgar.json"),
    "multisite_coverage": ("models/weblog_edgar/multisite_coverage_results.json", "multisite_coverage.json"),
    "representation_coverage": ("models/coverage/representation_coverage_results.json", "representation_coverage.json"),
    "ablation_positives": ("models/in_domain/ablation_positives_results.json", "ablation_positives.json"),
}


def _diffs(exp, got, path, tol, out):
    """Recurse, appending path, expected, got and the absolute delta for each mismatch to `out`."""
    if isinstance(exp, bool) or isinstance(got, bool):
        if exp != got:
            out.append((path, exp, got, None))
        return
    if isinstance(exp, (int, float)) and isinstance(got, (int, float)):
        if exp is None or got is None or math.isnan(exp) != math.isnan(got):
            out.append((path, exp, got, None)); return
        d = abs(float(exp) - float(got))
        if not (math.isnan(exp) and math.isnan(got)) and d > tol:
            out.append((path, exp, got, d))
        return
    if isinstance(exp, dict) and isinstance(got, dict):
        for k in exp.keys() | got.keys():
            if k not in exp or k not in got:
                out.append((f"{path}.{k}", exp.get(k, "<missing>"), got.get(k, "<missing>"), None))
            else:
                _diffs(exp[k], got[k], f"{path}.{k}", tol, out)
        return
    if isinstance(exp, list) and isinstance(got, list):
        if len(exp) != len(got):
            out.append((f"{path}[len]", len(exp), len(got), None)); return
        for i, (a, b) in enumerate(zip(exp, got)):
            _diffs(a, b, f"{path}[{i}]", tol, out)
        return
    if exp != got:
        out.append((path, exp, got, None))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tol", type=float, default=1e-6, help="max abs difference for numeric leaves")
    ap.add_argument("--only", default=None, help="check a single table by label")
    ap.add_argument("--require-all", action="store_true", help="a SKIP (unreproduced file) fails the run")
    ap.add_argument("--expected-dir", default=os.path.join(ROOT, "expected_metrics"))
    args = ap.parse_args()

    labels = [args.only] if args.only else list(MAPPING)
    rows, any_fail, any_skip = [], False, False
    for label in labels:
        if label not in MAPPING:
            print(f"unknown label: {label}", file=sys.stderr); sys.exit(2)
        fresh_rel, golden_name = MAPPING[label]
        fresh = os.path.join(ROOT, fresh_rel)
        golden = os.path.join(args.expected_dir, golden_name)
        if not os.path.exists(golden):
            rows.append((label, "MISSING-GOLD", 0, "")); any_fail = True; continue
        if not os.path.exists(fresh):
            rows.append((label, "SKIP", 0, "not reproduced")); any_skip = True; continue
        exp = json.load(open(golden)); got = json.load(open(fresh))
        diffs = []
        _diffs(exp, got, label, args.tol, diffs)
        if diffs:
            any_fail = True
            worst = max((d for *_, d in diffs if d is not None), default=None)
            note = diffs[0][0].split(".", 1)[-1]
            rows.append((label, "FAIL", len(diffs), f"worstΔ={worst:.2e} @ {note}" if worst else f"struct @ {note}"))
        else:
            rows.append((label, "PASS", 0, ""))

    w = max(len(r[0]) for r in rows)
    print(f"\n{'table':{w}}  result   ndiff  detail")
    print("-" * (w + 30))
    for label, status, nd, detail in rows:
        print(f"{label:{w}}  {status:7} {nd:5}  {detail}")
    npass = sum(1 for r in rows if r[1] == "PASS")
    print("-" * (w + 30))
    print(f"{npass}/{len(rows)} PASS  (tol={args.tol:g})")

    if any_fail:
        # list every mismatching value for the tables that failed
        for label in labels:
            fresh_rel, golden_name = MAPPING.get(label, ("", ""))
            fresh = os.path.join(ROOT, fresh_rel); golden = os.path.join(args.expected_dir, golden_name)
            if not (os.path.exists(fresh) and os.path.exists(golden)):
                continue
            diffs = []
            _diffs(json.load(open(golden)), json.load(open(fresh)), label, args.tol, diffs)
            for p, e, g, d in diffs[:20]:
                print(f"  DIFF {p}: expected={e} got={g}" + (f" (Δ={d:.3e})" if d else ""))
        sys.exit(1)
    if any_skip and args.require_all:
        sys.exit(1)
    sys.exit(0)


if __name__ == "__main__":
    main()
