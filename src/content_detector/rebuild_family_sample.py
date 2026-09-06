"""Rebuilds the 100-row SR-BH sample that family_eval scores, by matching each decoded row of
models/in_domain/family_labels.csv back to the SR-BH attacks. Writes
data/corpus/samples/srbh_family_sample.parquet. Duplicate decoded contents resolve to the first
SR-BH occurrence in parquet order, and a round-trip assert checks that the rebuilt sample
decodes back to the CSV row for row.
"""

from __future__ import annotations

import os

import pandas as pd

from .normalize import decode_fixed_point

CSV = "models/in_domain/family_labels.csv"
SRBH = "data/corpus/external/srbh.parquet"
OUT = "data/corpus/samples/srbh_family_sample.parquet"


def main():
    ref = pd.read_csv(CSV, index_col=0)
    assert len(ref) == 100, f"expected 100 labelled rows, got {len(ref)}"
    ref_keys = list(zip(ref.http_method.astype(str), ref.dec.astype(str)))
    wanted = set(ref_keys)

    srbh = pd.read_parquet(SRBH)
    atk = srbh[srbh.label == "attack"].reset_index(drop=True)
    for c in ["url_query", "request_body"]:
        if c not in atk:
            atk[c] = ""

    # first-occurrence lookup, keyed by method and decoded content
    lookup = {}
    remaining = set(wanted)
    for i in range(len(atk)):
        r = atk.iloc[i]
        dec = decode_fixed_point(f"{r.url_path_raw} {r.url_query} {r.request_body}")[0]
        key = (str(r.http_method), dec)
        if key in remaining:
            lookup[key] = i
            remaining.discard(key)
            if not remaining:
                break
    assert not remaining, f"{len(remaining)} CSV rows not found in SR-BH attacks: {sorted(remaining)[:2]}"

    sample = atk.iloc[[lookup[k] for k in ref_keys]].reset_index(drop=True)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    sample.to_parquet(OUT, index=False, compression="zstd")

    # round-trip verification: rebuilt sample must decode exactly to the frozen CSV, in order
    dec2 = sample.apply(
        lambda r: decode_fixed_point(f"{r.url_path_raw} {r.url_query} {r.request_body}")[0], axis=1)
    assert (dec2.values == ref.dec.astype(str).values).all(), "round-trip decode mismatch"
    assert (sample.http_method.astype(str).values == ref.http_method.astype(str).values).all()
    print(f"[rebuild_family_sample] wrote {OUT}: {len(sample)} rows, order verified against {CSV}")


if __name__ == "__main__":
    main()
