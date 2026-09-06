"""Split the testbed corpus by session, never by row.

A session is the atomic unit and never straddles train, calib and test: a login session, or
a scan campaign, where one attacker source IP is one session. The external sets zanbil and
srbh are held out and never passed here. Asserts zero session_id overlap across splits.
"""

import argparse
import hashlib
import json
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True, help="testbed labelled parquet (vampi+crapi)")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--train", type=float, default=0.6)
    ap.add_argument("--calib", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    df = pd.read_parquet(args.corpus)
    os.makedirs(args.out_dir, exist_ok=True)

    # deterministic session -> split by hashing session_id with the seed
    def bucket(sid: str) -> str:
        h = int(hashlib.sha256(f"{args.seed}:{sid}".encode()).hexdigest(), 16) % 10_000 / 10_000.0
        if h < args.train:
            return "train"
        if h < args.train + args.calib:
            return "calib"
        return "test"

    sessions = df["session_id"].unique()
    assign = {s: bucket(s) for s in sessions}
    df["_split"] = df["session_id"].map(assign)

    splits = {}
    for name in ("train", "calib", "test"):
        part = df[df["_split"] == name].drop(columns=["_split"]).reset_index(drop=True)
        part.to_parquet(os.path.join(args.out_dir, f"testbed_{name}.parquet"), index=False, compression="zstd")
        splits[name] = part

    # assert zero session overlap
    sids = {n: set(p["session_id"]) for n, p in splits.items()}
    overlaps = {}
    names = list(sids)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            inter = sids[a] & sids[b]
            if inter:
                overlaps[f"{a}∩{b}"] = len(inter)

    stats = {
        "seed": args.seed,
        "total_rows": int(len(df)),
        "total_sessions": int(len(sessions)),
        "splits": {n: {"rows": int(len(p)),
                       "sessions": int(p["session_id"].nunique()),
                       "benign": int((p.label == "benign").sum()),
                       "attack": int((p.label == "attack").sum())}
                   for n, p in splits.items()},
        "session_overlaps": overlaps,
        "disjoint": not overlaps,
    }
    with open(os.path.join(args.out_dir, "split_stats.json"), "w") as f:
        json.dump(stats, f, indent=2)
    print(json.dumps(stats, indent=2))
    if overlaps:
        print("[split] FAIL: session overlap across splits", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
