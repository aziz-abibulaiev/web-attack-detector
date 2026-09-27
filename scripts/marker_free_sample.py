#!/usr/bin/env python3
"""Draws the sample of 100 marker-free testbed attacks, stratified by family with seed 42, that
the authors classified by hand for Section 5.9.1, and prints marked and marker-free counts per
family. Writes the unclassified sample; the classification is docs/review/marker_free_sample_classified.csv.

Usage:  python scripts/marker_free_sample.py [--out data/corpus/samples/marker_free_sample.csv]
"""
import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import pandas as pd  # noqa: E402

from src.content_detector.normalize import decode_fixed_point  # noqa: E402
from src.content_detector.predict import _EXPLAIN_RULES  # noqa: E402

TESTBED = os.path.join(ROOT, "data/corpus/labelled/testbed.parquet")


def dec(r):
    return decode_fixed_point(f"{r.get('url_path_raw','')} {r.get('url_query','')} {r.get('request_body','')}")[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(ROOT, "data/corpus/samples/marker_free_sample.csv"))
    args = ap.parse_args()

    df = pd.read_parquet(TESTBED)
    atk = df[df.label == "attack"].copy()
    atk["decoded"] = [dec(r) for r in atk[["url_path_raw", "url_query", "request_body"]].to_dict("records")]
    atk["marked"] = atk.decoded.apply(lambda d: any(p.search(d) for p in _EXPLAIN_RULES.values()))
    print("attacks:", len(atk), "marked:", int(atk.marked.sum()), "marker-free:", int((~atk.marked).sum()))
    print(atk.groupby("attack_family").marked.agg(["size", "sum"])
          .rename(columns={"size": "n", "sum": "marked"}).assign(marker_free=lambda t: t.n - t.marked))
    mf = atk[~atk.marked]
    n_by_fam = (mf.attack_family.value_counts(normalize=True) * 100).round().astype(int)
    # the largest family absorbs the rounding difference, so the sample has exactly 100 rows
    diff = 100 - n_by_fam.sum(); n_by_fam.iloc[0] += diff
    parts = [mf[mf.attack_family == f].sample(n=min(k, (mf.attack_family == f).sum()), random_state=42)
             for f, k in n_by_fam.items() if k > 0]
    s = pd.concat(parts)
    cols = [c for c in ["attack_family", "attack_tool", "method", "url_path_raw", "url_query", "request_body",
                        "headers", "decoded"] if c in s.columns]
    out = s[cols].copy()
    for c in ["request_body", "headers", "decoded"]:
        if c in out:
            out[c] = out[c].astype(str).str.slice(0, 400)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    out.to_csv(args.out, index=True, index_label="testbed_row")
    print("wrote", args.out, len(out))


if __name__ == "__main__":
    main()
